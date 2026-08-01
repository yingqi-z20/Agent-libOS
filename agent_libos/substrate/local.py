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
import json
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
    McpConnectionInfo,
    McpExchangePhase,
    McpExchangeReceipt,
    McpProtocolEra,
    McpProtocolMode,
    McpProviderCallResult,
    McpProviderDiscoveryResult,
    McpProviderTool,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
)
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.models.mcp import mcp_runtime_secret_values
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
from agent_libos.utils.redaction import redact_sensitive_text

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_SHELL_DEFAULTS = DEFAULT_CONFIG.shell
_MCP_LOCAL_HTTP_HOSTS = {"localhost", "127.0.0.1", "::1"}
_MCP_FORBIDDEN_HOSTS = {"metadata.google.internal"}
_MCP_MODERN_PROTOCOL_REVISION = "2026-07-28"
_MCP_SUPPORTED_MODERN_PROTOCOL_REVISIONS = (_MCP_MODERN_PROTOCOL_REVISION,)
_MCP_LEGACY_PROTOCOL_REVISION = "2025-11-25"
_MCP_SUPPORTED_LEGACY_PROTOCOL_REVISIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    _MCP_LEGACY_PROTOCOL_REVISION,
)
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


class _McpAbsoluteDeadlineExceeded(TimeoutError):
    """The adapter's own absolute deadline elapsed, not a provider timeout."""


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
        if sys.platform == "darwin":
            self.root = self._darwin_existing_descriptor_path(
                lexical_root,
                require_directory=True,
                purpose="filesystem adapter root",
            )
        elif os.name == "nt":
            # pathlib preserves DOS 8.3 aliases and caller casing. Pin the
            # provider root to the spelling returned by the opened directory
            # handle so later containment checks use one Windows identity.
            guard = _WindowsDirectoryGuard.open(lexical_root)
            try:
                self.root = self._windows_final_path_from_handle(guard.handle)
            finally:
                guard.close()
        else:
            self.root = lexical_root
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
        guard_target = target if target == self.root else target.parent
        while guard_target != self.root and not guard_target.exists():
            guard_target = guard_target.parent
        guard = self._windows_directory_guard(guard_target)
        if guard is None:
            raise CapabilityDenied(
                "filesystem state requires a guarded Windows path"
            )
        try:
            self._before_path_sink_checked("state", target)
            self._reject_reparse_components(target)
            try:
                observed = target.lstat()
            except (FileNotFoundError, NotADirectoryError):
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
            # Floating-point addition/subtraction can otherwise reconstruct a
            # duration a few ulps above the caller's exact timeout budget.
            remaining_timeout = min(timeout_s, deadline - time.monotonic())
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
                timed_out = (
                    isinstance(exc, TimeoutError)
                    or (
                        deadline_guard is not None
                        and deadline_guard.expired.is_set()
                    )
                    or time.monotonic() >= deadline
                )
                last_error = (
                    "TimeoutError: JSON-RPC pinned request timed out"
                    if timed_out
                    else f"{type(exc).__name__}: {exc}"
                )
                if request_dispatch_started or timed_out:
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


class _McpToolCatalogValidationError(RuntimeError):
    """A received tools/list catalog cannot safely authorize tools/call."""


class _McpIncompleteToolCatalog(_McpToolCatalogValidationError):
    """Manifest v1 received a partial catalog that cannot be trusted."""


class _McpEnteredTransport:
    """Adapt a stream pair already owned by an outer strict transport scope."""

    def __init__(self, streams: Any) -> None:
        self.streams = streams

    async def __aenter__(self) -> Any:
        return self.streams

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _McpToolsOnlyReadStream:
    """Drop server notifications after wire accounting, before SDK dispatch."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    async def __aenter__(self) -> "_McpToolsOnlyReadStream":
        await self.stream.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        return await self.stream.__aexit__(*exc)

    def __aiter__(self) -> "_McpToolsOnlyReadStream":
        return self

    async def __anext__(self) -> Any:
        try:
            return await self.receive()
        except Exception as exc:
            if type(exc).__name__ == "EndOfStream":
                raise StopAsyncIteration from None
            raise

    async def receive(self) -> Any:
        while True:
            item = await self.stream.receive()
            message = getattr(item, "message", None)
            if type(message).__name__ == "JSONRPCNotification":
                continue
            return item

    @property
    def last_context(self) -> Any:
        return getattr(self.stream, "last_context", None)

    async def aclose(self) -> None:
        await self.stream.aclose()


@contextlib.asynccontextmanager
async def _mcp_tools_only_streamable_http_client(
    url: str,
    *,
    http_client: Any,
    terminate_on_close: bool = True,
    legacy_listen: bool = False,
):
    """Bounded SDK v2 HTTP transport with era-specific listen behavior.

    Manifest v1 preserves the released SDK-v1 background GET after the
    initialized notification.  Manifest v2 is Tools-only and suppresses that
    deprecated listen/subscription surface and its SSE resumption GET.  The
    Manifest-v1 SDK reconnect loop remains governed by this operation's strict
    HTTP client and task scope.
    """

    try:
        import anyio
        from mcp.client.streamable_http import StreamableHTTPTransport
        from mcp.shared._compat import resync_tracer
        from mcp.shared._context_streams import create_context_streams
    except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover
        raise ValidationError("MCP Python SDK v2 HTTP transport is unavailable") from exc

    transport = StreamableHTTPTransport(url)
    if not legacy_listen:
        async def reject_sse_resumption(
            context: Any,
            _last_event_id: str,
            _retry_interval_ms: int | None = None,
            _attempt: int = 0,
        ) -> None:
            message = getattr(context.session_message, "message", None)
            request_id = getattr(message, "id", None)
            await transport._resolve_abandoned_request(  # noqa: SLF001
                context.read_stream_writer,
                request_id,
                "MCP SSE resumption is unsupported by the Tools-only client",
            )

        # SDK v2 otherwise turns an event-id-bearing POST disconnect into an
        # implicit GET+Last-Event-ID replay.  Manifest v2 forbids that hidden
        # resume/retry path; Manifest v1 keeps its released behavior.
        transport._handle_reconnection = reject_sse_resumption  # type: ignore[method-assign]  # noqa: SLF001
    read_stream_writer, read_stream = create_context_streams(0)
    write_stream, write_stream_reader = create_context_streams(0)
    async with (
        read_stream_writer,
        read_stream,
        write_stream,
        write_stream_reader,
        anyio.create_task_group() as task_group,
    ):

        def start_listen_stream() -> None:
            if legacy_listen:
                task_group.start_soon(
                    transport.handle_get_stream,
                    http_client,
                    read_stream_writer,
                )

        task_group.start_soon(
            transport.post_writer,
            http_client,
            write_stream_reader,
            read_stream_writer,
            write_stream,
            start_listen_stream,
            task_group,
        )
        try:
            yield read_stream, write_stream
        finally:
            if transport.session_id and terminate_on_close:
                await transport.terminate_session(http_client)
            task_group.cancel_scope.cancel()
    await resync_tracer()


@dataclass
class _McpWireExchange:
    """One operation-local protocol exchange measured at the strict wire edge."""

    phase: McpExchangePhase | None
    method: str | None
    request_id: object | None = None
    request_bytes: int = 0
    response_bytes: int = 0
    started_at: float | None = None
    completed_at: float | None = None
    call_started: bool = False
    request_body: bytearray | None = None
    response_body: bytearray | None = None
    response_declared_bytes: int | None = None
    merged: bool = False


def _mcp_wire_phase(method: str | None) -> McpExchangePhase | None:
    return {
        "server/discover": McpExchangePhase.SERVER_DISCOVER,
        "initialize": McpExchangePhase.INITIALIZE,
        "notifications/initialized": McpExchangePhase.INITIALIZE,
        "tools/list": McpExchangePhase.TOOLS_LIST,
        "tools/call": McpExchangePhase.TOOLS_CALL,
    }.get(method)


def _mcp_wire_request_id_key(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    return (type(value).__name__, repr(value))


class _McpWireLedger:
    """Bounded operation-local accounting of raw protocol body/frame bytes.

    The strict transports enforce the aggregate limits.  This ledger only
    attributes bytes already accepted at those transport edges to an MCP
    phase; it never serializes a second, projected representation.
    """

    def __init__(self) -> None:
        self._exchanges: list[_McpWireExchange] = []
        self._pending_stdio: dict[tuple[str, object], _McpWireExchange] = {}

    def begin_http(self, method: str | None) -> _McpWireExchange:
        exchange = _McpWireExchange(
            phase=_mcp_wire_phase(method),
            method=method,
            request_body=bytearray(),
        )
        self._exchanges.append(exchange)
        return exchange

    def record_http_request(self, exchange: _McpWireExchange, chunk: bytes) -> None:
        if not exchange.call_started:
            exchange.call_started = True
            exchange.started_at = time.monotonic()
        exchange.request_bytes += len(chunk)
        if exchange.request_body is not None:
            exchange.request_body.extend(chunk)

    def finish_http_request(self, exchange: _McpWireExchange) -> None:
        self._bind_request_body(exchange)

    def record_http_response(self, exchange: _McpWireExchange, size: int) -> None:
        exchange.response_bytes += size

    def record_http_response_body(
        self,
        exchange: _McpWireExchange,
        chunk: bytes,
    ) -> None:
        self.record_http_response(exchange, len(chunk))
        if exchange.response_body is not None:
            exchange.response_body.extend(chunk)

    def finish_http_response(self, exchange: _McpWireExchange) -> None:
        self._bind_request_body(exchange)
        if exchange.completed_at is None:
            exchange.completed_at = time.monotonic()

    def record_stdio_request(self, encoded: bytes) -> None:
        payload = self._parse_payload(encoded)
        method = payload.get("method") if isinstance(payload, dict) else None
        if not isinstance(method, str):
            target = self._latest_active_exchange()
            if target is not None:
                target.request_bytes += len(encoded)
            return
        phase = _mcp_wire_phase(method)
        if method == "notifications/initialized":
            target = self._latest_phase(McpExchangePhase.INITIALIZE)
            if target is not None:
                target.request_bytes += len(encoded)
            return
        if phase is None:
            target = self._latest_active_exchange()
            if target is not None:
                target.request_bytes += len(encoded)
            return
        exchange = _McpWireExchange(
            phase=phase,
            method=method,
            request_id=payload.get("id"),
            request_bytes=len(encoded),
            started_at=time.monotonic(),
            call_started=True,
        )
        self._exchanges.append(exchange)
        if "id" in payload:
            self._pending_stdio[_mcp_wire_request_id_key(payload["id"])] = exchange

    def record_stdio_response(self, encoded: bytes, message: Any) -> None:
        if isinstance(getattr(message, "method", None), str):
            target = self._latest_active_exchange()
            if target is not None:
                target.response_bytes += len(encoded)
            return
        response_id = getattr(message, "id", None)
        if response_id is None:
            payload = self._parse_payload(encoded)
            response_id = payload.get("id") if isinstance(payload, dict) else None
        if response_id is None:
            return
        exchange = self._pending_stdio.pop(
            _mcp_wire_request_id_key(response_id),
            None,
        )
        if exchange is None:
            # An unmatched response is still accepted protocol input and must
            # count against the active exchange's bounded response budget.  It
            # must not complete that exchange: JSON-RPC string and number ids
            # are distinct identities (for example, "1" is not 1).
            target = self._latest_active_exchange()
            if target is not None:
                target.response_bytes += len(encoded)
            return
        exchange.response_bytes += len(encoded)
        exchange.completed_at = time.monotonic()

    def record_stdio_partial_response(self, encoded: bytes) -> None:
        target = self._latest_active_exchange()
        if target is not None:
            target.response_bytes += len(encoded)

    def receipts(self) -> tuple[McpExchangeReceipt, ...]:
        now = time.monotonic()
        receipts: list[McpExchangeReceipt] = []
        for exchange in self._exchanges:
            if exchange.merged or exchange.phase is None or not exchange.call_started:
                continue
            started_at = exchange.started_at or now
            completed_at = exchange.completed_at or now
            receipts.append(
                McpExchangeReceipt(
                    phase=exchange.phase,
                    request_bytes=exchange.request_bytes,
                    response_bytes=exchange.response_bytes,
                    duration_s=max(0.0, completed_at - started_at),
                    call_started=True,
                )
            )
        return tuple(receipts)

    def attach(self, error: BaseException) -> None:
        with contextlib.suppress(Exception):
            setattr(error, "_agent_libos_mcp_receipts", self.receipts())

    def _bind_request_body(self, exchange: _McpWireExchange) -> None:
        if exchange.request_body is None:
            return
        payload = self._parse_payload(bytes(exchange.request_body))
        exchange.request_body = None
        if not isinstance(payload, dict):
            return
        method = payload.get("method")
        if not isinstance(method, str):
            target = self._latest_active_exchange(before=exchange)
            if target is not None:
                target.request_bytes += exchange.request_bytes
                exchange.merged = True
            return
        exchange.method = method
        exchange.phase = _mcp_wire_phase(method)
        exchange.request_id = payload.get("id")
        if method != "notifications/initialized":
            return
        target = self._latest_phase(McpExchangePhase.INITIALIZE, before=exchange)
        if target is not None:
            target.request_bytes += exchange.request_bytes
            exchange.merged = True

    def _latest_phase(
        self,
        phase: McpExchangePhase,
        *,
        before: _McpWireExchange | None = None,
    ) -> _McpWireExchange | None:
        for candidate in reversed(self._exchanges):
            if candidate is before:
                continue
            if not candidate.merged and candidate.phase is phase:
                return candidate
        return None

    def _latest_active_exchange(
        self,
        *,
        before: _McpWireExchange | None = None,
    ) -> _McpWireExchange | None:
        for candidate in reversed(self._exchanges):
            if candidate is before or candidate.merged or candidate.phase is None:
                continue
            if candidate.call_started and candidate.completed_at is None:
                return candidate
        return None

    @staticmethod
    def _parse_payload(encoded: bytes) -> Any:
        try:
            return json.loads(encoded.decode("utf-8").strip())
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


def _mcp_wire_receipts(value: Any) -> tuple[McpExchangeReceipt, ...] | None:
    ledger = getattr(value, "_agent_libos_wire_ledger", None)
    if isinstance(ledger, _McpWireLedger):
        return ledger.receipts()
    return None


def _mcp_attach_wire_evidence(
    error: BaseException,
    *,
    wire_ledger: _McpWireLedger | None,
    connection: McpConnectionInfo | None,
) -> None:
    """Bind exact operation-local phase evidence to a propagated failure."""

    if wire_ledger is None:
        return
    with contextlib.suppress(Exception):
        setattr(error, "_agent_libos_mcp_wire_evidence", True)
    wire_ledger.attach(error)
    if connection is not None:
        with contextlib.suppress(Exception):
            setattr(error, "_agent_libos_mcp_connection", connection)


def _mcp_legacy_wire_bytes(encoded: bytes, *, newline: bool) -> bytes:
    """Reproduce SDK-v1 framing without reserializing JSON value tokens.

    SDK v1 used a different top-level member order and omitted an empty
    request ``params._meta``.  Rebuilding the entire value through
    ``json.dumps`` is not byte-compatible: it changes valid number spellings
    such as ``1e-7`` and can rewrite Unicode escapes.  Parse only member
    boundaries, then splice the original key/value tokens.
    """

    try:
        source = encoded.decode("utf-8").strip()
    except UnicodeDecodeError:
        return encoded
    members = _mcp_raw_json_object_members(source)
    if members is None:
        return encoded
    if len({name for name, _key, _value in members}) != len(members):
        return encoded

    rewritten: list[tuple[str, str, str]] = []
    for name, raw_key, raw_value in members:
        if name != "params":
            rewritten.append((name, raw_key, raw_value))
            continue
        params = _mcp_raw_json_object_members(raw_value)
        if params is None:
            rewritten.append((name, raw_key, raw_value))
            continue
        retained = [
            item
            for item in params
            if not (
                item[0] == "_meta"
                and _mcp_raw_json_object_members(item[2]) == []
            )
        ]
        if retained:
            rewritten.append(
                (name, raw_key, _mcp_rebuild_raw_json_object(retained))
            )

    by_name = {name: item for name, *item in rewritten}
    ordered: list[tuple[str, str, str]] = []
    for name in ("method", "params", "jsonrpc", "id", "result", "error"):
        selected = by_name.pop(name, None)
        if selected is not None:
            ordered.append((name, selected[0], selected[1]))
    ordered.extend(
        item for item in rewritten if item[0] in by_name
    )
    suffix = "\n" if newline else ""
    return (_mcp_rebuild_raw_json_object(ordered) + suffix).encode("utf-8")


def _mcp_raw_json_object_members(
    source: str,
) -> list[tuple[str, str, str]] | None:
    """Return decoded names plus untouched JSON member tokens for one object."""

    decoder = json.JSONDecoder()
    length = len(source)

    def skip_space(index: int) -> int:
        while index < length and source[index] in " \t\r\n":
            index += 1
        return index

    index = skip_space(0)
    if index >= length or source[index] != "{":
        return None
    index = skip_space(index + 1)
    if index < length and source[index] == "}":
        return [] if skip_space(index + 1) == length else None

    members: list[tuple[str, str, str]] = []
    while index < length:
        key_start = index
        try:
            key, key_end = decoder.raw_decode(source, index)
        except json.JSONDecodeError:
            return None
        if not isinstance(key, str):
            return None
        raw_key = source[key_start:key_end]
        index = skip_space(key_end)
        if index >= length or source[index] != ":":
            return None
        index = skip_space(index + 1)
        value_start = index
        try:
            _value, value_end = decoder.raw_decode(source, index)
        except json.JSONDecodeError:
            return None
        members.append((key, raw_key, source[value_start:value_end]))
        index = skip_space(value_end)
        if index >= length:
            return None
        if source[index] == "}":
            return members if skip_space(index + 1) == length else None
        if source[index] != ",":
            return None
        index = skip_space(index + 1)
    return None


def _mcp_rebuild_raw_json_object(
    members: list[tuple[str, str, str]],
) -> str:
    return "{" + ",".join(
        f"{raw_key}:{raw_value}"
        for _name, raw_key, raw_value in members
    ) + "}"


def _mcp_protocol_mode(server: McpServerSpec) -> McpProtocolMode:
    if server.schema_version == 1:
        return McpProtocolMode.LEGACY
    if server.schema_version != 2 or server.protocol_mode is None:
        raise ValidationError(
            "Manifest v2 requires protocol_mode legacy, auto, or 2026-07-28"
        )
    try:
        return McpProtocolMode(server.protocol_mode)
    except ValueError as exc:  # defensive for hand-built public model values
        raise ValidationError(f"unsupported MCP protocol mode: {server.protocol_mode}") from exc


@contextlib.contextmanager
def _mcp_sanitized_otel_context() -> Iterator[None]:
    """Prevent ambient trace or baggage values from entering MCP wire metadata."""

    token: object | None = None
    detach: Callable[[object], object] | None = None
    try:
        from opentelemetry.context import Context, attach
        from opentelemetry.context import detach as otel_detach

        token = attach(Context())
        detach = otel_detach
    except ModuleNotFoundError:
        pass
    try:
        yield
    finally:
        if token is not None and detach is not None:
            detach(token)


@contextlib.asynccontextmanager
async def _mcp_sdk_v2_client(
    client_session_type: Any,
    transport: Any,
    *,
    server: McpServerSpec,
    mode: McpProtocolMode,
    sdk_mode: str,
    deadline: float,
    max_response_bytes: int,
    http_policy_transport: Any = None,
    wire_ledger: _McpWireLedger | None = None,
    mcp_config: Any = None,
    sensitive_values: tuple[str, ...] = (),
):
    """Enter one SDK v2 client session with fail-closed era negotiation."""

    try:
        import mcp.types as mcp_types
    except ModuleNotFoundError as exc:  # pragma: no cover - imported above
        raise ValidationError("MCP Python SDK v2 is unavailable") from exc

    del sdk_mode
    client_info = mcp_types.Implementation(
        name=("mcp" if server.schema_version == 1 else "agent-libos"),
        version=("0.1.0" if server.schema_version == 1 else "1.2.1"),
    )
    negotiation_started = time.monotonic()
    session: Any = None
    entered_session = False
    entered_transport = False
    connection_evidence: McpConnectionInfo | None = None
    with _mcp_sanitized_otel_context():
        try:
            streams = await transport.__aenter__()
            entered_transport = True
            read, write = streams[:2]
            read = _McpToolsOnlyReadStream(read)
            session = client_session_type(
                read,
                write,
                sampling_callback=None,
                elicitation_callback=None,
                list_roots_callback=None,
                logging_callback=None,
                client_info=client_info,
                log_level=None,
                extensions=None,
            )
            if server.schema_version == 1:
                dispatcher = getattr(session, "_dispatcher", None)
                if hasattr(dispatcher, "_next_id"):
                    # SDK v1 minted request id 0 first; SDK v2 starts at 1.
                    # Preserve Manifest-v1 raw-wire identity exactly.
                    dispatcher._next_id = -1
            await session.__aenter__()
            entered_session = True
            await _mcp_negotiate_sdk_v2_session(
                session,
                server=server,
                mode=mode,
                deadline=deadline,
                negotiation_started=negotiation_started,
                http_policy_transport=http_policy_transport,
                mcp_types=mcp_types,
                protocol_probe_timeout_s=(
                    DEFAULT_CONFIG.mcp.protocol_probe_timeout_s
                    if mcp_config is None
                    else mcp_config.protocol_probe_timeout_s
                ),
            )

            protocol_revision = str(session.protocol_version)
            if protocol_revision in _MCP_SUPPORTED_MODERN_PROTOCOL_REVISIONS:
                protocol_era = McpProtocolEra.MODERN
            elif protocol_revision in _MCP_SUPPORTED_LEGACY_PROTOCOL_REVISIONS:
                protocol_era = McpProtocolEra.LEGACY
            else:
                # ClientSession.adopt() follows the installed SDK's supported
                # revision set.  A future SDK minor must not silently widen the
                # product's release-locked protocol surface.
                raise ValidationError(
                    "MCP negotiation selected a protocol revision outside the "
                    "release-locked supported set"
                )
            if (
                mode is McpProtocolMode.REVISION_2026_07_28
                and protocol_revision != _MCP_MODERN_PROTOCOL_REVISION
            ):
                raise ValidationError(
                    "MCP server does not support required protocol revision 2026-07-28"
                )
            connected = _McpSdkV2ClientAdapter(session)
            connection = _mcp_connection_info(
                connected,
                mode=mode,
                protocol_era=protocol_era,
                protocol_revision=protocol_revision,
                sensitive_values=sensitive_values,
            )
            connection_evidence = connection
            receipts = (
                wire_ledger.receipts()
                if wire_ledger is not None
                else (() if server.schema_version == 1 else _mcp_negotiation_receipts(
                    connection,
                    duration_s=max(0.0, time.monotonic() - negotiation_started),
                ))
            )
            setattr(connected, "_agent_libos_sdk_v2", True)
            setattr(connected, "_agent_libos_connection", connection)
            setattr(connected, "_agent_libos_receipts", list(receipts))
            setattr(connected, "_agent_libos_manifest_version", server.schema_version)
            setattr(
                connected,
                "_agent_libos_mcp_config",
                DEFAULT_CONFIG.mcp if mcp_config is None else mcp_config,
            )
            if wire_ledger is not None:
                setattr(connected, "_agent_libos_wire_ledger", wire_ledger)
            _mcp_check_receipt_budget(
                receipts,
                server=server,
                max_response_bytes=max_response_bytes,
            )
            yield connected
        except BaseException as error:
            _mcp_attach_wire_evidence(
                error,
                wire_ledger=wire_ledger,
                connection=connection_evidence,
            )
            raise
        finally:
            await _mcp_close_sdk_v2_client(
                session=session,
                entered_session=entered_session,
                transport=transport,
                entered_transport=entered_transport,
                wire_ledger=wire_ledger,
                connection=connection_evidence,
            )


async def _mcp_close_sdk_v2_client(
    *,
    session: Any,
    entered_session: bool,
    transport: Any,
    entered_transport: bool,
    wire_ledger: _McpWireLedger | None,
    connection: McpConnectionInfo | None,
) -> None:
    """Close both SDK scopes while binding cleanup failures to wire evidence."""

    try:
        if entered_session and session is not None:
            try:
                await session.__aexit__(None, None, None)
            except BaseException as error:
                _mcp_attach_wire_evidence(
                    error,
                    wire_ledger=wire_ledger,
                    connection=connection,
                )
                raise
    finally:
        if entered_transport:
            try:
                await transport.__aexit__(None, None, None)
            except BaseException as error:
                _mcp_attach_wire_evidence(
                    error,
                    wire_ledger=wire_ledger,
                    connection=connection,
                )
                raise


async def _mcp_negotiate_sdk_v2_session(
    session: Any,
    *,
    server: McpServerSpec,
    mode: McpProtocolMode,
    deadline: float,
    negotiation_started: float,
    http_policy_transport: Any,
    mcp_types: Any,
    protocol_probe_timeout_s: float,
) -> None:
    """Negotiate exactly one SDK v2 session without widening fallback policy."""

    if mode is McpProtocolMode.LEGACY:
        await _mcp_await_with_deadline(
            _mcp_initialize_locked(session, mcp_types),
            deadline=deadline,
            stage="initialize",
        )
        return

    fallback = False
    probe_deadline = min(
        deadline,
        negotiation_started + protocol_probe_timeout_s,
    )
    try:
        raw_discover = await _mcp_await_with_deadline(
            session.send_discover(_MCP_MODERN_PROTOCOL_REVISION),
            deadline=probe_deadline,
            stage="server/discover probe",
        )
        discover = mcp_types.DiscoverResult.model_validate(raw_discover)
        supported = tuple(discover.supported_versions)
        if _MCP_MODERN_PROTOCOL_REVISION in supported:
            session.adopt(discover)
        elif _mcp_stdio_legacy_versions(supported, mode=mode, server=server):
            fallback = True
        else:
            raise ValidationError(
                "MCP server/discover returned no supported modern protocol revision"
            )
    except _McpAbsoluteDeadlineExceeded:
        if mode is McpProtocolMode.AUTO and server.transport == "stdio":
            fallback = True
        else:
            raise
    except Exception as exc:
        retry_version = _mcp_mutual_modern_retry_version(exc)
        if retry_version is not None:
            raw_discover = await _mcp_await_with_deadline(
                session.send_discover(retry_version),
                deadline=probe_deadline,
                stage="server/discover version retry",
            )
            discover = mcp_types.DiscoverResult.model_validate(raw_discover)
            if retry_version not in discover.supported_versions:
                raise ValidationError(
                    "MCP server/discover retry did not confirm the requested revision"
                )
            session.adopt(discover)
        elif _mcp_stdio_legacy_error_fallback(exc, mode=mode, server=server):
            fallback = True
        elif _mcp_auto_fallback_allowed(
            exc,
            mode=mode,
            transport=server.transport,
            http_policy_transport=http_policy_transport,
        ):
            fallback = True
        else:
            raise
    if fallback:
        await _mcp_await_with_deadline(
            _mcp_initialize_locked(session, mcp_types),
            deadline=deadline,
            stage="initialize fallback",
        )


async def _mcp_initialize_locked(session: Any, mcp_types: Any) -> Any:
    """Perform a legacy handshake without trusting an SDK minor's latest."""

    result = await session.send_request(
        mcp_types.InitializeRequest(
            params=mcp_types.InitializeRequestParams(
                protocol_version=_MCP_LEGACY_PROTOCOL_REVISION,
                capabilities=session._build_capabilities(  # noqa: SLF001
                    _MCP_LEGACY_PROTOCOL_REVISION
                ),
                client_info=session._client_info,  # noqa: SLF001
            )
        ),
        mcp_types.InitializeResult,
    )
    if result.protocol_version not in _MCP_SUPPORTED_LEGACY_PROTOCOL_REVISIONS:
        raise ValidationError(
            "MCP initialize returned a protocol revision outside the "
            "release-locked legacy set"
        )
    session.adopt(result)
    await session.send_notification(mcp_types.InitializedNotification())
    return result


class _McpSdkV2ClientAdapter:
    """Minimal Tools-only view over the official SDK v2 ClientSession."""

    def __init__(self, session: Any) -> None:
        self.session = session

    @property
    def protocol_version(self) -> Any:
        return self.session.protocol_version

    @property
    def server_info(self) -> Any:
        return self.session.server_info

    @property
    def server_capabilities(self) -> Any:
        return self.session.server_capabilities

    async def list_tools(
        self,
        *,
        cursor: str | None = None,
        cache_mode: str | None = None,
    ) -> Any:
        del cache_mode
        params = None
        if cursor is not None:
            try:
                import mcp.types as mcp_types
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ValidationError("MCP Python SDK v2 is unavailable") from exc
            params = mcp_types.PaginatedRequestParams(cursor=cursor)
        return await self.session.list_tools(params=params)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Issue tools/call without SDK outputSchema enforcement.

        Server-provided outputSchema is diagnostic-only in Agent libOS.  Use
        the SDK dispatcher and typed response parsing, but intentionally skip
        ClientSession.call_tool(), whose post-response validation would turn a
        diagnostic schema into an execution gate.
        """

        try:
            import mcp.types as mcp_types
            from pydantic import TypeAdapter
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ValidationError("MCP Python SDK v2 is unavailable") from exc
        return await self.session.send_request(
            mcp_types.CallToolRequest(
                params=mcp_types.CallToolRequestParams(
                    name=name,
                    arguments=arguments,
                )
            ),
            TypeAdapter(mcp_types.CallToolResult | mcp_types.InputRequiredResult),
        )


def _mcp_auto_fallback_allowed(
    error: Exception,
    *,
    mode: McpProtocolMode,
    transport: str,
    http_policy_transport: Any,
) -> bool:
    """Return only positive, transport-specific evidence of a legacy peer."""

    if mode is not McpProtocolMode.AUTO:
        return False
    code = getattr(error, "code", None)
    if code is None:
        nested = getattr(error, "error", None)
        code = getattr(nested, "code", None)
    if code in {-32020, -32021, -32022}:
        return False
    if transport == "stdio":
        return code == -32601
    if transport == "streamable_http":
        return bool(
            http_policy_transport is not None
            and getattr(http_policy_transport, "last_response_status", None) == 400
            and getattr(http_policy_transport, "last_request_method", None)
            == "server/discover"
            and getattr(
                http_policy_transport,
                "last_legacy_400_signal",
                # Test doubles predating raw-body classification keep their
                # established behavior; production policy transports always
                # expose the fail-closed flag.
                True,
            )
        )
    return False


def _mcp_error_supported_versions(error: Exception) -> tuple[str, ...] | None:
    code = getattr(error, "code", None)
    nested = getattr(error, "error", None)
    if code is None:
        code = getattr(nested, "code", None)
    if code != -32022:
        return None
    data = getattr(nested, "data", None)
    if data is None:
        data = getattr(error, "data", None)
    supported = data.get("supported") if isinstance(data, dict) else getattr(data, "supported", None)
    if (
        not isinstance(supported, list)
        or not supported
        or any(not isinstance(item, str) or not item for item in supported)
    ):
        return None
    return tuple(supported)


def _mcp_stdio_legacy_versions(
    supported: tuple[str, ...],
    *,
    mode: McpProtocolMode,
    server: McpServerSpec,
) -> bool:
    return bool(
        mode is McpProtocolMode.AUTO
        and server.transport == "stdio"
        and supported
        and all(
            item in _MCP_SUPPORTED_LEGACY_PROTOCOL_REVISIONS
            for item in supported
        )
    )


def _mcp_stdio_legacy_error_fallback(
    error: Exception,
    *,
    mode: McpProtocolMode,
    server: McpServerSpec,
) -> bool:
    supported = _mcp_error_supported_versions(error)
    return bool(
        supported is not None
        and _mcp_stdio_legacy_versions(supported, mode=mode, server=server)
    )


def _mcp_mutual_modern_retry_version(error: Exception) -> str | None:
    """Return the one pinned modern revision named by a -32022 response."""

    supported = _mcp_error_supported_versions(error)
    if supported is not None and _MCP_MODERN_PROTOCOL_REVISION in supported:
        return _MCP_MODERN_PROTOCOL_REVISION
    return None


def _mcp_connection_info(
    client: Any,
    *,
    mode: McpProtocolMode,
    protocol_era: McpProtocolEra,
    protocol_revision: str,
    sensitive_values: tuple[str, ...] = (),
) -> McpConnectionInfo:
    server_info = getattr(client, "server_info", None)
    capabilities_value = _jsonable_mcp_value(
        getattr(client, "server_capabilities", None)
    )
    advertised: list[str] = []
    if isinstance(capabilities_value, dict):
        advertised = sorted(
            str(name)
            for name, value in capabilities_value.items()
            if value not in (None, False, {}, [])
        )
    sanitized_capabilities = tuple(
        dict.fromkeys(
            redact_sensitive_text(
                name,
                sensitive_values=sensitive_values,
            )
            for name in advertised
        )
    )
    unsupported = tuple(
        dict.fromkeys(
            redact_sensitive_text(
                name,
                sensitive_values=sensitive_values,
            )
            for name in advertised
            if name != "tools"
        )
    )
    raw_server_name = str(getattr(server_info, "name", "")) or None
    raw_server_version = str(getattr(server_info, "version", "")) or None
    return McpConnectionInfo(
        protocol_mode=mode,
        protocol_era=protocol_era,
        protocol_revision=protocol_revision,
        sessionless=protocol_era is McpProtocolEra.MODERN,
        fallback_used=(mode is McpProtocolMode.AUTO and protocol_era is McpProtocolEra.LEGACY),
        server_name=(
            redact_sensitive_text(
                raw_server_name,
                sensitive_values=sensitive_values,
            )
            if raw_server_name is not None
            else None
        ),
        server_version=(
            redact_sensitive_text(
                raw_server_version,
                sensitive_values=sensitive_values,
            )
            if raw_server_version is not None
            else None
        ),
        capabilities=sanitized_capabilities,
        unsupported_capabilities=unsupported,
    )


def _mcp_negotiation_receipts(
    connection: McpConnectionInfo,
    *,
    duration_s: float,
) -> tuple[McpExchangeReceipt, ...]:
    connection_payload = dumps(to_jsonable(connection)).encode("utf-8")
    if connection.protocol_mode is McpProtocolMode.LEGACY:
        return (
            McpExchangeReceipt(
                phase=McpExchangePhase.INITIALIZE,
                request_bytes=len(dumps({"method": "initialize"}).encode("utf-8")),
                response_bytes=len(connection_payload),
                duration_s=duration_s,
                call_started=True,
            ),
        )
    discover = McpExchangeReceipt(
        phase=McpExchangePhase.SERVER_DISCOVER,
        request_bytes=len(dumps({"method": "server/discover"}).encode("utf-8")),
        response_bytes=(0 if connection.fallback_used else len(connection_payload)),
        duration_s=duration_s,
        call_started=True,
    )
    if not connection.fallback_used:
        return (discover,)
    return (
        discover,
        McpExchangeReceipt(
            phase=McpExchangePhase.INITIALIZE,
            request_bytes=len(dumps({"method": "initialize"}).encode("utf-8")),
            response_bytes=len(connection_payload),
            duration_s=0.0,
            call_started=True,
        ),
    )


class SdkMcpProvider:
    """MCP client provider backed by the optional official Python SDK."""

    supports_executable_snapshots = True
    supports_runtime_environment_snapshots = True
    supports_subprocess_limits = True
    supports_mcp_modern_protocol = True

    def __init__(
        self,
        workspace_root: str | Path | None = None,
        *,
        mcp_config: Any = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
        self.mcp_config = DEFAULT_CONFIG.mcp if mcp_config is None else mcp_config

    def discover(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
        runtime_environment: Mapping[str, str] | None = None,
        limits: SubprocessLimits | None = None,
    ) -> McpProviderDiscoveryResult:
        """Negotiate one Manifest v2 connection without caching its result."""

        mode = _mcp_protocol_mode(server)
        if server.schema_version != 2 or mode is McpProtocolMode.LEGACY:
            raise ValidationError(
                "MCP discovery requires a Manifest v2 server in auto or 2026-07-28 mode"
            )
        deadline = time.monotonic() + timeout_s
        started = time.monotonic()
        with self._stdio_dispatch_snapshot(
            server,
            executable_snapshot,
            runtime_environment=runtime_environment,
        ) as selected_snapshot:
            async def run() -> McpProviderDiscoveryResult:
                async with self._session(
                    server,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                    executable_snapshot=selected_snapshot,
                    runtime_environment=runtime_environment,
                    limits=limits,
                ) as client:
                    connection = _mcp_session_connection(client)
                    if connection is None:  # pragma: no cover - real SDK invariant
                        raise RuntimeError("MCP SDK v2 did not expose negotiated connection metadata")
                    receipts = _mcp_session_receipts(client)
                    return McpProviderDiscoveryResult(
                        connection=connection,
                        request_bytes=sum(item.request_bytes for item in receipts),
                        response_bytes=sum(item.response_bytes for item in receipts),
                        duration_s=max(0.0, time.monotonic() - started),
                        receipts=receipts,
                    )

            try:
                return _run_mcp_async(
                    _mcp_await_with_deadline(
                        run(),
                        deadline=deadline,
                        stage="server/discover",
                    )
                )
            except BaseExceptionGroup as exc:
                self._raise_mcp_transport_limit_error(exc)
                raise

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
                pre_call = self._mcp_v2_pre_call_exception_result(
                    server,
                    error=exc,
                    started_at=deadline - timeout_s,
                )
                if pre_call is not None:
                    return pre_call
                message = self._mcp_transport_limit_message(exc)
                if message is not None:
                    _, receipts, connection = self._mcp_wire_failure_evidence(exc)
                    return self._mcp_transport_failure_result(
                        server,
                        tool,
                        arguments,
                        message=message,
                        started_at=deadline - timeout_s,
                        max_response_bytes=max_response_bytes,
                        receipts=receipts,
                        connection=connection,
                    )
                raise
            except RuntimeError as exc:
                pre_call = self._mcp_v2_pre_call_exception_result(
                    server,
                    error=exc,
                    started_at=deadline - timeout_s,
                )
                if pre_call is not None:
                    return pre_call
                message = self._mcp_transport_limit_message(exc)
                if message is None:
                    raise
                _, receipts, connection = self._mcp_wire_failure_evidence(exc)
                return self._mcp_transport_failure_result(
                    server,
                    tool,
                    arguments,
                    message=message,
                    started_at=deadline - timeout_s,
                    max_response_bytes=max_response_bytes,
                    receipts=receipts,
                    connection=connection,
                )
            except Exception as exc:
                pre_call = self._mcp_v2_pre_call_exception_result(
                    server,
                    error=exc,
                    started_at=deadline - timeout_s,
                )
                if pre_call is not None:
                    return pre_call
                raise

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
                    "MCP HTTP operation exceeded max_response_bytes=",
                    "MCP HTTP request exceeded max_request_bytes=",
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
    def _mcp_transport_receipts(
        error: BaseException,
    ) -> tuple[McpExchangeReceipt, ...]:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        selected: tuple[McpExchangeReceipt, ...] = ()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            candidate = getattr(current, "_agent_libos_mcp_receipts", ())
            if (
                isinstance(candidate, tuple)
                and all(isinstance(item, McpExchangeReceipt) for item in candidate)
                and len(candidate) > len(selected)
            ):
                selected = candidate
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return selected

    @staticmethod
    def _mcp_wire_failure_evidence(
        error: BaseException,
    ) -> tuple[
        bool,
        tuple[McpExchangeReceipt, ...],
        McpConnectionInfo | None,
    ]:
        pending: list[BaseException] = [error]
        seen: set[int] = set()
        certified = False
        selected: tuple[McpExchangeReceipt, ...] = ()
        connection: McpConnectionInfo | None = None
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            if getattr(current, "_agent_libos_mcp_wire_evidence", False) is True:
                certified = True
                candidate = getattr(current, "_agent_libos_mcp_receipts", ())
                if (
                    isinstance(candidate, tuple)
                    and all(
                        isinstance(item, McpExchangeReceipt)
                        for item in candidate
                    )
                    and len(candidate) >= len(selected)
                ):
                    selected = candidate
                candidate_connection = getattr(
                    current,
                    "_agent_libos_mcp_connection",
                    None,
                )
                if isinstance(candidate_connection, McpConnectionInfo):
                    connection = candidate_connection
            if isinstance(current, BaseExceptionGroup):
                pending.extend(current.exceptions)
            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return certified, selected, connection

    @staticmethod
    def _mcp_v2_pre_call_exception_result(
        server: McpServerSpec,
        *,
        error: BaseException,
        started_at: float,
    ) -> McpProviderCallResult | None:
        if server.schema_version != 2:
            return None
        certified, receipts, connection = SdkMcpProvider._mcp_wire_failure_evidence(
            error
        )
        if not certified or any(
            item.phase is McpExchangePhase.TOOLS_CALL
            for item in receipts
        ):
            return None
        list_receipts = tuple(
            item
            for item in receipts
            if item.phase is McpExchangePhase.TOOLS_LIST
        )
        return McpProviderCallResult(
            error="MCP operation failed before tools/call dispatch",
            error_type="McpPreCallFailure",
            correlation_id=new_id("corr"),
            duration_s=max(0.0, time.monotonic() - started_at),
            list_request_bytes=sum(item.request_bytes for item in list_receipts),
            list_response_bytes=sum(item.response_bytes for item in list_receipts),
            call_request_bytes=0,
            call_response_bytes=0,
            call_started=False,
            connection=connection,
            receipts=receipts,
        )

    @staticmethod
    def _mcp_transport_error_type(message: str) -> str:
        prefixes = (
            ("MCP stdio frame exceeded", "McpStdioFrameTooLarge"),
            ("MCP stdio stdout exceeded", "McpStdioStdoutTooLarge"),
            ("MCP stdio stderr exceeded", "McpStdioStderrTooLarge"),
            ("MCP HTTP SSE frame exceeded", "McpHttpSseFrameTooLarge"),
            ("MCP HTTP response exceeded", "McpHttpResponseTooLarge"),
            ("MCP HTTP operation exceeded", "McpHttpOperationTooLarge"),
            ("MCP HTTP request exceeded", "McpHttpRequestTooLarge"),
            (
                "MCP HTTP response uses unsupported Content-Encoding=",
                "McpHttpContentEncodingDenied",
            ),
        )
        return next(
            (
                error_type
                for prefix, error_type in prefixes
                if message.startswith(prefix)
            ),
            "McpTransportLimitError",
        )

    @staticmethod
    def _mcp_transport_failure_result(
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        message: str,
        started_at: float,
        max_response_bytes: int,
        receipts: tuple[McpExchangeReceipt, ...] = (),
        connection: McpConnectionInfo | None = None,
    ) -> McpProviderCallResult:
        legacy = server.schema_version == 1
        if legacy:
            receipts = ()
            connection = None
        list_receipts = tuple(
            item for item in receipts if item.phase is McpExchangePhase.TOOLS_LIST
        )
        call_receipts = tuple(
            item for item in receipts if item.phase is McpExchangePhase.TOOLS_CALL
        )
        list_request_bytes = sum(item.request_bytes for item in list_receipts)
        list_response_bytes = sum(item.response_bytes for item in list_receipts)
        call_request_bytes = sum(item.request_bytes for item in call_receipts)
        call_response_bytes = sum(item.response_bytes for item in call_receipts)
        if legacy:
            list_request_bytes = len(
                dumps({"method": "tools/list", "server_id": server.server_id}).encode(
                    "utf-8"
                )
            )
            call_request_bytes = len(
                dumps({"name": tool.mcp_name, "arguments": arguments}).encode("utf-8")
            )
            call_response_bytes = max_response_bytes
        error_type = SdkMcpProvider._mcp_transport_error_type(message)
        return McpProviderCallResult(
            error="bounded MCP transport failure",
            error_type=error_type,
            correlation_id=new_id("corr"),
            response_bytes=call_response_bytes,
            duration_s=max(0.0, time.monotonic() - started_at),
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_request_bytes=call_request_bytes,
            call_response_bytes=call_response_bytes,
            call_started=(
                True
                if legacy
                else any(item.call_started for item in call_receipts)
            ),
            connection=connection,
            receipts=receipts,
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
            tools, response_bytes = await _mcp_collect_tools(
                session,
                server,
                deadline=deadline,
                max_response_bytes=max_response_bytes,
                stage="tools/list response",
            )
            connection = _mcp_session_connection(session)
            receipts = _mcp_session_receipts(session)
        return McpToolListResult(
            server_id=server.server_id,
            tools=tools,
            response_bytes=response_bytes,
            duration_s=time.monotonic() - started,
            connection=connection,
            receipts=receipts,
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
            receipt_count = len(_mcp_session_receipts(session))
            call_started = time.monotonic()
            result = await _mcp_await_with_deadline(
                _mcp_session_call_tool(session, tool.mcp_name, arguments),
                deadline=deadline,
                stage=f"tools/call response {tool.mcp_name}",
            )
            connection = _mcp_session_connection(session)
            if _mcp_is_input_required_result(result):
                call_response_bytes = _mcp_result_size(result)
                _mcp_append_receipt(
                    session,
                    McpExchangeReceipt(
                        phase=McpExchangePhase.TOOLS_CALL,
                        request_bytes=_mcp_call_request_size(tool.mcp_name, arguments),
                        response_bytes=min(call_response_bytes, max_response_bytes),
                        duration_s=max(0.0, time.monotonic() - call_started),
                        call_started=True,
                    ),
                    server=server,
                    max_response_bytes=max_response_bytes,
                )
                call_request_bytes, call_response_bytes = _mcp_phase_wire_sizes(
                    session,
                    phase=McpExchangePhase.TOOLS_CALL,
                    after=receipt_count,
                    request_bytes=_mcp_call_request_size(tool.mcp_name, arguments),
                    response_bytes=min(call_response_bytes, max_response_bytes),
                )
                return _mcp_input_required_failure(
                    started=started,
                    connection=connection,
                    receipts=_mcp_session_receipts(session),
                    call_request_bytes=call_request_bytes,
                    call_response_bytes=call_response_bytes,
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
        _mcp_append_receipt(
            session,
            McpExchangeReceipt(
                phase=McpExchangePhase.TOOLS_CALL,
                request_bytes=_mcp_call_request_size(tool.mcp_name, arguments),
                response_bytes=call_response_bytes,
                duration_s=max(0.0, time.monotonic() - call_started),
                call_started=True,
            ),
            server=server,
            max_response_bytes=max_response_bytes,
        )
        call_request_bytes, call_response_bytes = _mcp_phase_wire_sizes(
            session,
            phase=McpExchangePhase.TOOLS_CALL,
            after=receipt_count,
            request_bytes=_mcp_call_request_size(tool.mcp_name, arguments),
            response_bytes=call_response_bytes,
        )
        return McpProviderCallResult(
            content=payload["content"],
            structured_content=payload["structured_content"],
            is_error=bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
            response_bytes=call_response_bytes,
            duration_s=time.monotonic() - started,
            too_large=too_large,
            call_request_bytes=call_request_bytes,
            call_response_bytes=call_response_bytes,
            call_started=True,
            connection=connection,
            receipts=_mcp_session_receipts(session),
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
        async with self._session(
            server,
            deadline=deadline,
            max_response_bytes=max_response_bytes,
            executable_snapshot=executable_snapshot,
            runtime_environment=runtime_environment,
            limits=limits,
        ) as session:
            list_receipt_start = len(_mcp_session_receipts(session))
            try:
                live_tools, list_response_bytes = await _mcp_collect_tools(
                    session,
                    server,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                    stage="validated tools/list response",
                )
            except _McpToolCatalogValidationError:
                if server.schema_version == 2:
                    return _mcp_pre_call_validation_failure(
                        session,
                        started=started,
                        list_receipt_start=list_receipt_start,
                    )
                # Preserve the Manifest-v1 provider projection byte-for-byte.
                return _mcp_v1_pre_call_validation_failure(
                    session,
                    tool=tool,
                    arguments=arguments,
                    started=started,
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=max_response_bytes,
                )
            list_receipts = tuple(
                item
                for item in _mcp_session_receipts(session)[list_receipt_start:]
                if item.phase is McpExchangePhase.TOOLS_LIST
            )
            if list_receipts:
                list_request_bytes = sum(item.request_bytes for item in list_receipts)
                list_response_bytes = sum(item.response_bytes for item in list_receipts)
            list_encoded = dumps([to_jsonable(item) for item in live_tools]).encode("utf-8")
            if list_response_bytes > max_response_bytes:
                if server.schema_version == 2:
                    return _mcp_pre_call_validation_failure(
                        session,
                        started=started,
                        list_receipt_start=list_receipt_start,
                        error="MCP tools/list response exceeded limit",
                        error_type="ResponseTooLarge",
                    )
                return _mcp_v1_pre_call_validation_failure(
                    session,
                    tool=tool,
                    arguments=arguments,
                    started=started,
                    error="MCP tools/list response exceeded limit",
                    error_type="ResponseTooLarge",
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=max_response_bytes,
                )
            live = next((item for item in live_tools if item.name == tool.mcp_name), None)
            if live is None or (
                tool.input_schema
                and not _mcp_canonical_json_equal(
                    live.input_schema,
                    tool.input_schema,
                )
            ):
                if server.schema_version == 2:
                    return _mcp_pre_call_validation_failure(
                        session,
                        started=started,
                        list_receipt_start=list_receipt_start,
                    )
                return _mcp_v1_pre_call_validation_failure(
                    session,
                    tool=tool,
                    arguments=arguments,
                    started=started,
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=len(list_encoded),
                )
            call_request_bytes = _mcp_call_request_size(
                tool.mcp_name,
                arguments,
            )
            call_receipt_start = len(_mcp_session_receipts(session))
            call_started_at = time.monotonic()
            result = await _mcp_await_with_deadline(
                _mcp_session_call_tool(session, tool.mcp_name, arguments),
                deadline=deadline,
                stage=f"validated tools/call response {tool.mcp_name}",
            )
            connection = _mcp_session_connection(session)
            if _mcp_is_input_required_result(result):
                call_response_bytes = min(_mcp_result_size(result), max_response_bytes)
                _mcp_append_receipt(
                    session,
                    McpExchangeReceipt(
                        phase=McpExchangePhase.TOOLS_CALL,
                        request_bytes=call_request_bytes,
                        response_bytes=call_response_bytes,
                        duration_s=max(0.0, time.monotonic() - call_started_at),
                        call_started=True,
                    ),
                    server=server,
                    max_response_bytes=max_response_bytes,
                )
                call_request_bytes, call_response_bytes = _mcp_phase_wire_sizes(
                    session,
                    phase=McpExchangePhase.TOOLS_CALL,
                    after=call_receipt_start,
                    request_bytes=call_request_bytes,
                    response_bytes=call_response_bytes,
                )
                return _mcp_input_required_failure(
                    started=started,
                    connection=connection,
                    receipts=_mcp_session_receipts(session),
                    list_request_bytes=list_request_bytes,
                    list_response_bytes=list_response_bytes,
                    call_request_bytes=call_request_bytes,
                    call_response_bytes=call_response_bytes,
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
        _mcp_append_receipt(
            session,
            McpExchangeReceipt(
                phase=McpExchangePhase.TOOLS_CALL,
                request_bytes=call_request_bytes,
                response_bytes=call_response_bytes,
                duration_s=max(0.0, time.monotonic() - call_started_at),
                call_started=True,
            ),
            server=server,
            max_response_bytes=max_response_bytes,
        )
        call_request_bytes, call_response_bytes = _mcp_phase_wire_sizes(
            session,
            phase=McpExchangePhase.TOOLS_CALL,
            after=call_receipt_start,
            request_bytes=call_request_bytes,
            response_bytes=call_response_bytes,
        )
        return McpProviderCallResult(
            content=payload["content"],
            structured_content=payload["structured_content"],
            is_error=bool(getattr(result, "isError", False) or getattr(result, "is_error", False)),
            response_bytes=call_response_bytes,
            duration_s=time.monotonic() - started,
            too_large=too_large,
            list_request_bytes=list_request_bytes,
            list_response_bytes=list_response_bytes,
            call_request_bytes=call_request_bytes,
            call_response_bytes=call_response_bytes,
            call_started=True,
            connection=connection,
            receipts=_mcp_session_receipts(session),
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
            from mcp.client import ClientSession
            from mcp.client.stdio import StdioServerParameters
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP provider requires the optional dependency; install with `uv sync --extra mcp --all-groups`"
            ) from exc
        except ImportError:
            # Compatibility seam for tests and environments still carrying the
            # pre-v2 SDK. The released MCP extra requires SDK v2, but keeping
            # this path avoids changing the established Manifest v1 Provider
            # SPI and makes the upgrade failure explicit at the package edge.
            async with self._legacy_sdk_session(
                server,
                deadline=deadline,
                max_response_bytes=max_response_bytes,
                executable_snapshot=executable_snapshot,
                runtime_environment=runtime_environment,
                limits=limits,
            ) as session:
                yield session
            return

        mode = _mcp_protocol_mode(server)
        sdk_mode = "legacy" if mode is McpProtocolMode.LEGACY else "auto"
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
                wire_ledger = (
                    _McpWireLedger() if server.schema_version == 2 else None
                )
                stdio_environment = self._stdio_dispatch_env(
                    server,
                    executable_snapshot,
                    runtime_environment=runtime_environment,
                )
                params = StdioServerParameters(
                    command=command,
                    args=list(server.stdio.args),
                    env=stdio_environment,
                    cwd=dispatch_cwd,
                )
                transport = _strict_stdio_client(
                    params,
                    max_frame_bytes=max_response_bytes,
                    max_request_bytes=server.max_request_bytes,
                    cwd_fd=cwd_fd,
                    deadline=deadline,
                    limits=limits,
                    stdout_limit_bytes=(
                        max_response_bytes if server.schema_version == 2 else None
                    ),
                    stdin_limit_bytes=(
                        server.max_request_bytes if server.schema_version == 2 else None
                    ),
                    wire_ledger=wire_ledger,
                    legacy_wire_compat=server.schema_version == 1,
                )
                async with _mcp_sdk_v2_client(
                    ClientSession,
                    transport,
                    server=server,
                    mode=mode,
                    sdk_mode=sdk_mode,
                    deadline=deadline,
                    max_response_bytes=max_response_bytes,
                    wire_ledger=wire_ledger,
                    mcp_config=self.mcp_config,
                    sensitive_values=mcp_runtime_secret_values(
                        server,
                        stdio_environment,
                    ),
                ) as client:
                    yield client
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
                async with _mcp_tools_only_streamable_http_client(
                    server.http.url,
                    http_client=http_client,
                    terminate_on_close=True,
                    legacy_listen=server.schema_version == 1,
                ) as transport:
                    # ``streamable_http_client`` has already been entered here,
                    # while Client expects the context manager itself. Wrap the
                    # live stream pair in a no-op transport context so Client
                    # owns only its dispatcher/session and this scope continues
                    # to own the strict HTTP transport.
                    async with _mcp_sdk_v2_client(
                        ClientSession,
                        _McpEnteredTransport(transport),
                        server=server,
                        mode=mode,
                        sdk_mode=sdk_mode,
                        deadline=deadline,
                        max_response_bytes=max_response_bytes,
                        http_policy_transport=getattr(
                            http_client,
                            "_agent_libos_policy_transport",
                            None,
                        ),
                        wire_ledger=(
                            getattr(
                                getattr(
                                    http_client,
                                    "_agent_libos_policy_transport",
                                    None,
                                ),
                                "wire_ledger",
                                None,
                            )
                            if server.schema_version == 2
                            else None
                        ),
                        mcp_config=self.mcp_config,
                        sensitive_values=getattr(
                            http_client,
                            "_agent_libos_sensitive_values",
                            (),
                        ),
                    ) as client:
                        yield client
            return
        raise RuntimeError(f"unsupported MCP transport: {server.transport}")

    @contextlib.asynccontextmanager
    async def _legacy_sdk_session(
        self,
        server: McpServerSpec,
        *,
        deadline: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None,
        runtime_environment: Mapping[str, str] | None,
        limits: SubprocessLimits | None,
    ):
        if server.schema_version != 1 or _mcp_protocol_mode(server) is not McpProtocolMode.LEGACY:
            raise ValidationError("Manifest v2 requires MCP Python SDK v2")
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters

        if server.transport == "stdio":
            if server.stdio is None:
                raise RuntimeError("MCP stdio transport is missing stdio configuration")
            command = (
                str(executable_snapshot.executable_path)
                if executable_snapshot is not None
                else str(
                    self._stdio_command_candidate(
                        server,
                        runtime_environment=runtime_environment,
                    )
                )
            )
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
                async with _mcp_tools_only_streamable_http_client(
                    server.http.url,
                    http_client=http_client,
                ) as streams:
                    read, write = streams[:2]
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
            import httpx2 as httpx
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP provider requires httpx2 from the optional MCP dependency; "
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
        sensitive_values = mcp_runtime_secret_values(server, headers)
        headers["Accept-Encoding"] = "identity"
        transport = _McpPolicyAsyncHTTPTransport(
            max_response_bytes=max_response_bytes,
            max_request_bytes=server.max_request_bytes,
            deadline=selected_deadline,
            legacy_wire_compat=server.schema_version == 1,
        )
        try:
            async with httpx.AsyncClient(
                headers=headers,
                follow_redirects=False,
                timeout=timeout,
                transport=transport,
                trust_env=False,
            ) as client:
                setattr(client, "_agent_libos_policy_transport", transport)
                setattr(
                    client,
                    "_agent_libos_sensitive_values",
                    sensitive_values,
                )
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
        if operation in {"discover", "list_tools"}:
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
        raise _McpAbsoluteDeadlineExceeded(
            f"MCP absolute deadline exhausted during {stage}"
        )
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
    task = asyncio.ensure_future(awaitable)
    try:
        done, _pending = await asyncio.wait({task}, timeout=timeout)
    except BaseException as error:
        # A task-group/transport failure can cancel this parent while the SDK
        # request task is still pending.  Always retrieve the child outcome so
        # it cannot later surface as an unhandled MCPError, while preserving
        # the original parent exception.
        child_error = await _mcp_cancel_and_retrieve(task)
        _mcp_copy_wire_evidence(child_error, error)
        raise
    if task in done:
        return task.result()
    child_error = await _mcp_cancel_and_retrieve(task)
    timeout_error = _McpAbsoluteDeadlineExceeded(
        f"MCP absolute deadline exhausted during {stage}"
    )
    _mcp_copy_wire_evidence(child_error, timeout_error)
    raise timeout_error


async def _mcp_anext_with_deadline(
    iterator: Any,
    *,
    deadline: float,
    stage: str,
) -> Any:
    """Advance and later close an HTTP response stream in the same task."""

    timeout = _mcp_remaining_timeout(deadline, stage=stage)
    timeout_scope = asyncio.timeout(timeout)
    try:
        async with timeout_scope:
            return await anext(iterator)
    except TimeoutError as error:
        if not timeout_scope.expired():
            raise
        raise _McpAbsoluteDeadlineExceeded(
            f"MCP absolute deadline exhausted during {stage}"
        ) from error


async def _mcp_cancel_and_retrieve(
    task: asyncio.Future[Any],
) -> BaseException | None:
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException as error:
        return error
    return None


def _mcp_copy_wire_evidence(
    source: BaseException | None,
    target: BaseException,
) -> None:
    if source is None:
        return
    for name in (
        "_agent_libos_mcp_wire_evidence",
        "_agent_libos_mcp_receipts",
        "_agent_libos_mcp_connection",
    ):
        with contextlib.suppress(Exception):
            if hasattr(source, name):
                setattr(target, name, getattr(source, name))


def _mcp_session_connection(session: Any) -> McpConnectionInfo | None:
    value = getattr(session, "_agent_libos_connection", None)
    return value if isinstance(value, McpConnectionInfo) else None


def _mcp_session_receipts(session: Any) -> tuple[McpExchangeReceipt, ...]:
    wire_receipts = _mcp_wire_receipts(session)
    if wire_receipts is not None:
        return wire_receipts
    value = getattr(session, "_agent_libos_receipts", ())
    return tuple(item for item in value if isinstance(item, McpExchangeReceipt))


def _mcp_check_receipt_budget(
    receipts: tuple[McpExchangeReceipt, ...] | list[McpExchangeReceipt],
    *,
    server: McpServerSpec,
    max_response_bytes: int,
) -> None:
    request_bytes = sum(item.request_bytes for item in receipts)
    response_bytes = sum(item.response_bytes for item in receipts)
    if request_bytes > server.max_request_bytes:
        raise RuntimeError(
            f"MCP operation exceeded max_request_bytes={server.max_request_bytes}"
        )
    if response_bytes > max_response_bytes:
        raise RuntimeError(
            f"MCP operation exceeded max_response_bytes={max_response_bytes}"
        )


def _mcp_append_receipt(
    session: Any,
    receipt: McpExchangeReceipt,
    *,
    server: McpServerSpec,
    max_response_bytes: int,
) -> None:
    if getattr(session, "_agent_libos_manifest_version", None) == 1:
        return
    wire_receipts = _mcp_wire_receipts(session)
    if wire_receipts is not None:
        _mcp_check_receipt_budget(
            wire_receipts,
            server=server,
            max_response_bytes=max_response_bytes,
        )
        setattr(session, "_agent_libos_receipts", list(wire_receipts))
        return
    receipts = list(_mcp_session_receipts(session))
    receipts.append(receipt)
    _mcp_check_receipt_budget(
        receipts,
        server=server,
        max_response_bytes=max_response_bytes,
    )
    if getattr(session, "_agent_libos_sdk_v2", False):
        setattr(session, "_agent_libos_receipts", receipts)


def _mcp_phase_wire_sizes(
    session: Any,
    *,
    phase: McpExchangePhase,
    after: int,
    request_bytes: int,
    response_bytes: int,
) -> tuple[int, int]:
    recorded = _mcp_session_receipts(session)
    for receipt in reversed(recorded[after:]):
        if receipt.phase is phase:
            return receipt.request_bytes, receipt.response_bytes
    return request_bytes, response_bytes


async def _mcp_session_list_tools(session: Any, cursor: str | None) -> Any:
    if getattr(session, "_agent_libos_sdk_v2", False):
        return await session.list_tools(cursor=cursor, cache_mode="bypass")
    if cursor is None:
        return await session.list_tools()
    return await session.list_tools(cursor=cursor)


async def _mcp_session_call_tool(
    session: Any,
    name: str,
    arguments: dict[str, Any],
) -> Any:
    if getattr(session, "_agent_libos_sdk_v2", False):
        return await session.call_tool(name, arguments)
    return await session.call_tool(name, arguments)


def _mcp_provider_tool(item: Any) -> McpProviderTool:
    return McpProviderTool(
        name=str(getattr(item, "name", "")),
        description=getattr(item, "description", None),
        input_schema=dict(
            getattr(item, "inputSchema", None)
            or getattr(item, "input_schema", None)
            or {}
        ),
        metadata=_mcp_metadata(item),
    )


def _mcp_tools_list_cursor(result: Any) -> tuple[bool, str | None]:
    for field in ("nextCursor", "next_cursor"):
        if hasattr(result, field):
            value = getattr(result, field)
            if value is not None:
                return True, value if isinstance(value, str) else None
    return False, None


async def _mcp_collect_tools(
    session: Any,
    server: McpServerSpec,
    *,
    deadline: float,
    max_response_bytes: int,
    stage: str,
) -> tuple[list[McpProviderTool], int]:
    """Collect one legacy page or a bounded Manifest v2 catalog."""

    is_v2 = server.schema_version == 2
    mcp_config = getattr(session, "_agent_libos_mcp_config", DEFAULT_CONFIG.mcp)
    max_pages = mcp_config.list_max_pages if is_v2 else 1
    max_items = mcp_config.list_limit
    cursor: str | None = None
    seen_cursors: set[str] = set()
    seen_names: set[str] = set()
    tools: list[McpProviderTool] = []
    total_response_bytes = 0

    for page_index in range(max_pages):
        receipt_count = len(_mcp_session_receipts(session))
        request_payload: dict[str, Any] = {"method": "tools/list"}
        if cursor is not None:
            request_payload["params"] = {"cursor": cursor}
        request_bytes = len(dumps(request_payload).encode("utf-8"))
        prospective = (*_mcp_session_receipts(session), McpExchangeReceipt(
            phase=McpExchangePhase.TOOLS_LIST,
            request_bytes=request_bytes,
        ))
        _mcp_check_receipt_budget(
            prospective,
            server=server,
            max_response_bytes=max_response_bytes,
        )
        page_started = time.monotonic()
        result = await _mcp_await_with_deadline(
            _mcp_session_list_tools(session, cursor),
            deadline=deadline,
            stage=f"{stage} page {page_index + 1}",
        )
        try:
            page_tools = [
                _mcp_provider_tool(item)
                for item in list(getattr(result, "tools", []) or [])
            ]
        except (TypeError, ValueError) as exc:
            raise _McpToolCatalogValidationError(
                "MCP tools/list returned a malformed tool catalog"
            ) from exc
        page_encoded = dumps([to_jsonable(item) for item in page_tools]).encode("utf-8")
        page_response_bytes = len(page_encoded)
        receipt = McpExchangeReceipt(
            phase=McpExchangePhase.TOOLS_LIST,
            request_bytes=request_bytes,
            response_bytes=page_response_bytes,
            duration_s=max(0.0, time.monotonic() - page_started),
            call_started=True,
        )
        _mcp_append_receipt(
            session,
            receipt,
            server=server,
            max_response_bytes=max_response_bytes,
        )
        recorded = _mcp_session_receipts(session)
        if len(recorded) > receipt_count:
            wire_receipt = recorded[-1]
            if wire_receipt.phase is McpExchangePhase.TOOLS_LIST:
                request_bytes = wire_receipt.request_bytes
                page_response_bytes = wire_receipt.response_bytes
        total_response_bytes += page_response_bytes

        for item in page_tools:
            if is_v2 and item.name in seen_names:
                raise _McpToolCatalogValidationError(
                    f"MCP tools/list returned duplicate tool name: {item.name}"
                )
            seen_names.add(item.name)
            tools.append(item)
            if is_v2 and len(tools) > max_items:
                raise _McpToolCatalogValidationError(
                    f"MCP tools/list exceeded maximum tool count={max_items}"
                )

        has_cursor, next_cursor = _mcp_tools_list_cursor(result)
        if not has_cursor:
            return tools, total_response_bytes
        if not is_v2:
            raise _McpIncompleteToolCatalog(
                "MCP tools/list returned an unsupported continuation cursor"
            )
        if not next_cursor:
            raise _McpToolCatalogValidationError(
                "MCP tools/list returned a malformed continuation cursor"
            )
        if next_cursor in seen_cursors:
            raise _McpToolCatalogValidationError(
                "MCP tools/list returned a repeated continuation cursor"
            )
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    raise _McpToolCatalogValidationError(
        f"MCP tools/list exceeded maximum page count={max_pages}"
    )


def _mcp_v1_pre_call_validation_failure(
    session: Any,
    *,
    tool: McpToolSpec,
    arguments: dict[str, Any],
    started: float,
    list_request_bytes: int,
    list_response_bytes: int,
    error: str = "MCP live tool validation failed",
    error_type: str = "LiveToolValidationError",
) -> McpProviderCallResult:
    return McpProviderCallResult(
        error=error,
        error_type=error_type,
        correlation_id=new_id("corr"),
        duration_s=max(0.0, time.monotonic() - started),
        list_request_bytes=list_request_bytes,
        list_response_bytes=list_response_bytes,
        call_request_bytes=_mcp_call_request_size(tool.mcp_name, arguments),
        call_started=False,
        connection=_mcp_session_connection(session),
        receipts=_mcp_session_receipts(session),
    )


def _mcp_pre_call_validation_failure(
    session: Any,
    *,
    started: float,
    list_receipt_start: int,
    error: str = "MCP live tool validation failed",
    error_type: str = "LiveToolValidationError",
) -> McpProviderCallResult:
    """Project a v2 live-validation denial without implying tools/call dispatch."""

    receipts = _mcp_session_receipts(session)
    list_receipts = tuple(
        receipt
        for receipt in receipts[list_receipt_start:]
        if receipt.phase is McpExchangePhase.TOOLS_LIST
    )
    return McpProviderCallResult(
        error=error,
        error_type=error_type,
        correlation_id=new_id("corr"),
        duration_s=max(0.0, time.monotonic() - started),
        list_request_bytes=sum(item.request_bytes for item in list_receipts),
        list_response_bytes=sum(item.response_bytes for item in list_receipts),
        call_request_bytes=0,
        call_response_bytes=0,
        call_started=False,
        connection=_mcp_session_connection(session),
        receipts=receipts,
    )


def _mcp_canonical_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    try:
        return dumps(left).encode("utf-8") == dumps(right).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeEncodeError):
        return False


def _mcp_call_request_size(name: str, arguments: dict[str, Any]) -> int:
    return len(dumps({"name": name, "arguments": arguments}).encode("utf-8"))


def _mcp_result_size(result: Any) -> int:
    return len(dumps(_jsonable_mcp_value(result)).encode("utf-8"))


def _mcp_is_input_required_result(result: Any) -> bool:
    if type(result).__name__ == "InputRequiredResult":
        return True
    value = getattr(result, "result_type", None)
    return value in {"input_required", "inputRequired"}


def _mcp_input_required_failure(
    *,
    started: float,
    connection: McpConnectionInfo | None,
    receipts: tuple[McpExchangeReceipt, ...],
    list_request_bytes: int = 0,
    list_response_bytes: int = 0,
    call_request_bytes: int,
    call_response_bytes: int,
) -> McpProviderCallResult:
    return McpProviderCallResult(
        is_error=True,
        error="MCP server requested unsupported multi-round input",
        error_type="mcp_input_required_unsupported",
        correlation_id=new_id("corr"),
        response_bytes=call_response_bytes,
        duration_s=max(0.0, time.monotonic() - started),
        list_request_bytes=list_request_bytes,
        list_response_bytes=list_response_bytes,
        call_request_bytes=call_request_bytes,
        call_response_bytes=call_response_bytes,
        call_started=True,
        connection=connection,
        receipts=receipts,
    )


class _McpPolicyAsyncHTTPTransport:
    """MCP address policy plus pre-materialization HTTP response bounds."""

    def __init__(
        self,
        *,
        max_response_bytes: int,
        max_request_bytes: int | None = None,
        deadline: float | None = None,
        wire_ledger: _McpWireLedger | None = None,
        legacy_wire_compat: bool = False,
    ) -> None:
        try:
            import httpcore2 as httpcore
            import httpx2 as httpx  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP HTTP transport requires httpx2/httpcore2 from the optional MCP dependency; "
                "install with `uv sync --extra mcp --all-groups`"
            ) from exc
        if isinstance(max_response_bytes, bool) or max_response_bytes < 1:
            raise ValidationError("MCP HTTP max_response_bytes must be a positive integer")
        selected_request_bytes = (
            max_response_bytes if max_request_bytes is None else max_request_bytes
        )
        if isinstance(selected_request_bytes, bool) or selected_request_bytes < 1:
            raise ValidationError("MCP HTTP max_request_bytes must be a positive integer")
        self.max_response_bytes = max_response_bytes
        self.max_request_bytes = selected_request_bytes
        self.request_bytes = 0
        self.response_bytes = 0
        self.last_request_method: str | None = None
        self.last_response_status: int | None = None
        self.last_legacy_400_signal = False
        self.deadline = float("inf") if deadline is None else deadline
        self.limit_error: RuntimeError | None = None
        self.wire_ledger = wire_ledger or _McpWireLedger()
        self._agent_libos_wire_ledger = self.wire_ledger
        self.legacy_wire_compat = legacy_wire_compat
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
        import httpcore2 as httpcore
        import httpx2 as httpx

        request.headers["Accept-Encoding"] = "identity"
        self.last_request_method = request.headers.get("Mcp-Method")
        self.last_legacy_400_signal = False
        exchange = self.wire_ledger.begin_http(self.last_request_method)
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
        request_stream = request.stream
        request_headers = request.headers
        if self.legacy_wire_compat:
            buffered = bytearray()
            async for chunk in request.stream:
                buffered.extend(chunk)
                if len(buffered) > self.max_request_bytes + 1024:
                    raise self._limit_failure(
                        f"MCP HTTP request exceeded max_request_bytes={self.max_request_bytes}"
                    )
            encoded = _mcp_legacy_wire_bytes(bytes(buffered), newline=False)
            if len(encoded) > self.max_request_bytes:
                raise self._limit_failure(
                    f"MCP HTTP request exceeded max_request_bytes={self.max_request_bytes}"
                )
            request_stream = _mcp_one_chunk_stream(encoded)
            request_headers = httpx.Headers(request.headers)
            request_headers["Content-Length"] = str(len(encoded))
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request_headers.raw,
            content=self._bounded_request_stream(request_stream, exchange),
            extensions=extensions,
        )
        try:
            with _map_mcp_httpcore_exceptions():
                core_response = await self._pool.handle_async_request(core_request)
        except BaseException as exc:
            self.wire_ledger.finish_http_request(exchange)
            self.wire_ledger.finish_http_response(exchange)
            self.wire_ledger.attach(exc)
            raise
        self.last_response_status = int(core_response.status)
        if self.last_response_status == 400:
            exchange.response_body = bytearray()
        headers = httpx.Headers(core_response.headers)
        content_length = headers.get("content-length")
        if content_length is not None and content_length.isdigit():
            exchange.response_declared_bytes = int(content_length)
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
                count_response_bytes=lambda size: self._count_response_bytes(
                    size,
                    exchange=exchange,
                ),
                observe_response_chunk=lambda chunk: self._observe_response_chunk(
                    exchange,
                    chunk,
                ),
                response_complete=lambda: self._finish_http_response(exchange),
            ),
            extensions=core_response.extensions,
        )

    def _limit_failure(self, message: str) -> RuntimeError:
        error = RuntimeError(message)
        self.wire_ledger.attach(error)
        self.limit_error = error
        return error

    async def _bounded_request_stream(
        self,
        stream: Any,
        exchange: _McpWireExchange,
    ):
        async for chunk in stream:
            prospective = (
                exchange.request_bytes + len(chunk)
                if self.legacy_wire_compat
                else self.request_bytes + len(chunk)
            )
            if prospective > self.max_request_bytes:
                raise self._limit_failure(
                    f"MCP HTTP request exceeded max_request_bytes={self.max_request_bytes}"
                )
            self.request_bytes += len(chunk)
            self.wire_ledger.record_http_request(exchange, chunk)
            yield chunk
        self.wire_ledger.finish_http_request(exchange)

    def _count_response_bytes(
        self,
        size: int,
        *,
        exchange: _McpWireExchange,
    ) -> None:
        current_response_bytes = (
            exchange.response_bytes
            if self.legacy_wire_compat
            else self.response_bytes
        )
        if current_response_bytes + size > self.max_response_bytes:
            raise self._limit_failure(
                (
                    f"MCP HTTP response exceeded max_response_bytes={self.max_response_bytes}"
                    if self.legacy_wire_compat
                    else f"MCP HTTP operation exceeded max_response_bytes={self.max_response_bytes}"
                )
            )
        if size:
            self.wire_ledger.record_http_response(exchange, size)
            self.response_bytes += size

    def _observe_response_chunk(
        self,
        exchange: _McpWireExchange,
        chunk: bytes,
    ) -> None:
        if exchange.response_body is not None:
            exchange.response_body.extend(chunk)

    def _finish_http_response(self, exchange: _McpWireExchange) -> None:
        self.wire_ledger.finish_http_response(exchange)
        if (
            self.last_response_status != 400
            or exchange.method != "server/discover"
            or exchange.response_body is None
            or self.limit_error is not None
        ):
            return
        body = bytes(exchange.response_body)
        if not body:
            self.last_legacy_400_signal = exchange.response_declared_bytes == 0
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
            return
        code = payload["error"].get("code")
        self.last_legacy_400_signal = bool(
            isinstance(code, int) and code not in {-32020, -32021, -32022}
        )

    async def aclose(self) -> None:
        with _map_mcp_httpcore_exceptions():
            await self._pool.aclose()


@contextlib.contextmanager
def _map_mcp_httpcore_exceptions() -> Iterator[None]:
    """Translate public httpcore transport errors to their httpx equivalents."""

    try:
        yield
    except Exception as exc:
        import httpcore2 as httpcore
        import httpx2 as httpx

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


async def _mcp_one_chunk_stream(content: bytes):
    yield content


def _bounded_mcp_http_stream(
    stream: Any,
    *,
    max_response_bytes: int,
    is_sse: bool,
    fail: Callable[[str], RuntimeError],
    deadline: float = float("inf"),
    count_response_bytes: Callable[[int], None] | None = None,
    observe_response_chunk: Callable[[bytes], None] | None = None,
    response_complete: Callable[[], None] | None = None,
) -> Any:
    import httpx2 as httpx

    limiter = _McpHttpResponseLimiter(max_response_bytes=max_response_bytes, is_sse=is_sse)

    class BoundedMcpHttpStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            iterator = stream.__aiter__()
            while True:
                try:
                    with _map_mcp_httpcore_exceptions():
                        chunk = await _mcp_anext_with_deadline(
                            iterator,
                            deadline=deadline,
                            stage="HTTP response body",
                        )
                except StopAsyncIteration:
                    if response_complete is not None:
                        response_complete()
                    return
                else:
                    message = limiter.feed(chunk)
                    if message is not None:
                        raise fail(message)
                    if count_response_bytes is not None:
                        count_response_bytes(len(chunk))
                    if observe_response_chunk is not None:
                        observe_response_chunk(chunk)
                    yield chunk

        async def aclose(self) -> None:
            try:
                with _map_mcp_httpcore_exceptions():
                    await stream.aclose()
            finally:
                if response_complete is not None:
                    response_complete()

    return BoundedMcpHttpStream()


class _McpPolicyNetworkBackend:
    """Resolve, validate, then connect to the exact MCP HTTP address."""

    def __init__(self, *, deadline: float | None = None) -> None:
        try:
            import httpcore2 as httpcore
        except ModuleNotFoundError as exc:
            raise ValidationError(
                "MCP HTTP transport requires httpcore2 from the optional MCP dependency; "
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
    stdin_limit_bytes: int
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
    stdin_limit_bytes: int | None,
    deadline: float | None,
    limits: SubprocessLimits | None,
    stdout_limit_bytes: int | None,
    stderr_limit_bytes: int | None,
) -> _McpStdioConfig:
    selected_request_limit = (
        max_frame_bytes if max_request_bytes is None else max_request_bytes
    )
    selected_stdin_limit = (
        selected_request_limit * _MCP_STDIO_PROTOCOL_OUTPUT_MULTIPLIER
        if stdin_limit_bytes is None
        else stdin_limit_bytes
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
        ("stdin_limit_bytes", selected_stdin_limit),
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
        stdin_limit_bytes=selected_stdin_limit,
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
        wire_ledger: _McpWireLedger | None,
        legacy_wire_compat: bool,
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
        self.wire_ledger = wire_ledger
        self.legacy_wire_compat = legacy_wire_compat
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
        if self.wire_ledger is not None:
            self.wire_ledger.attach(error)
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
                    if self.wire_ledger is not None:
                        self.wire_ledger.record_stdio_partial_response(bytes(buffer))
                    await self._send_stdout_failure(self._frame_limit_error())
                    return False
                return True
            if newline > self.config.max_frame_bytes:
                if self.wire_ledger is not None:
                    self.wire_ledger.record_stdio_partial_response(
                        bytes(buffer[: newline + 1])
                    )
                await self._send_stdout_failure(self._frame_limit_error())
                return False
            line = bytes(buffer[:newline])
            del buffer[: newline + 1]
            try:
                adapter = getattr(
                    self.mcp_types,
                    "jsonrpc_message_adapter",
                    None,
                )
                if adapter is not None:
                    message = adapter.validate_json(line, by_name=False)
                else:
                    message = self.mcp_types.JSONRPCMessage.model_validate_json(line)
            except Exception as exc:
                await self.read_stream_writer.send(exc)
                continue
            if self.wire_ledger is not None:
                self.wire_ledger.record_stdio_response(line + b"\n", message)
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
                        self.config.stdin_limit_bytes
                    )
                    if self.stdin_bytes > aggregate_limit:
                        self.record_failure(
                            RuntimeError(
                                "MCP stdio stdin exceeded "
                                f"max_output_bytes={aggregate_limit}"
                            )
                        )
                        return
                    if self.wire_ledger is not None:
                        # Mark immediately before the transport send.  A send
                        # interrupted after this point may still have delivered
                        # the complete stdio frame.
                        self.wire_ledger.record_stdio_request(encoded)
                    await self.process.stdin.send(encoded)
        except self.anyio.ClosedResourceError:
            await self.anyio.lowlevel.checkpoint()

    def _encode_request(self, session_message: Any) -> bytes:
        raw = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
        encoded = (raw + "\n").encode(
            encoding=self.server.encoding,
            errors=self.server.encoding_error_handler,
        )
        if self.legacy_wire_compat:
            return _mcp_legacy_wire_bytes(encoded, newline=True)
        return encoded

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
    stdin_limit_bytes: int | None = None,
    cwd_fd: int | None = None,
    deadline: float | None = None,
    limits: SubprocessLimits | None = None,
    stdout_limit_bytes: int | None = None,
    stderr_limit_bytes: int | None = None,
    wire_ledger: _McpWireLedger | None = None,
    legacy_wire_compat: bool = False,
):
    anyio, mcp_types, session_message_type = _mcp_stdio_dependencies()
    config = _validated_mcp_stdio_config(
        max_frame_bytes=max_frame_bytes,
        max_request_bytes=max_request_bytes,
        stdin_limit_bytes=stdin_limit_bytes,
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
        wire_ledger=wire_ledger,
        legacy_wire_compat=legacy_wire_compat,
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


def _mcp_tools_list_has_continuation(result: Any) -> bool:
    """Return whether an MCP tools/list result is only one page of a catalog."""

    for field in ("nextCursor", "next_cursor"):
        cursor = getattr(result, field, None)
        if cursor is not None:
            return True
    return False


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
