from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
import unicodedata
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Iterator, Protocol

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    JsonRpcTransportResult,
    McpProviderCallResult,
    McpServerSpec,
    McpToolListResult,
    McpToolSpec,
)
from agent_libos.models.external_effect import ExternalEffectClassification
from agent_libos.models.exceptions import ValidationError

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools


class HierarchicalPathLock:
    """A fair, re-entrant lock for overlapping lexical path scopes.

    The same path and ancestor/descendant scopes exclude each other while
    unrelated siblings can proceed concurrently.  A queued overlapping scope
    cannot be bypassed by later arrivals, preventing directory operations from
    starving behind a stream of child operations.  Re-entrancy may keep or
    narrow an owned scope; widening an owned child to an ancestor is rejected
    instead of risking a lock-upgrade deadlock.

    Lock keys use conservative Unicode normalization and case folding.  That
    may serialize distinct names on a case-sensitive filesystem, but it never
    allows aliases on case-insensitive Host filesystems to bypass exclusion.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._active: dict[int, tuple[tuple[str, ...], int]] = {}
        self._waiters: list[tuple[int, tuple[str, ...], int]] = []
        self._next_token = 0
        self._root_tokens = threading.local()

    @staticmethod
    def _parts(path: str) -> tuple[str, ...]:
        normalized = str(path).replace("\\", "/").strip("/")
        if normalized in {"", "."}:
            return ()
        return tuple(
            unicodedata.normalize("NFC", part).casefold()
            for part in normalized.split("/")
            if part not in {"", "."}
        )

    @staticmethod
    def creation_scope(path: str) -> str:
        """Return a scope covering every parent a create may materialize.

        The workspace root already exists.  Locking the first path component
        therefore covers all possibly missing descendants shared by concurrent
        sibling creates without turning every workspace write back into one
        global lock.
        """

        normalized = str(path).replace("\\", "/").strip("/")
        if normalized in {"", "."}:
            return "."
        if "/" not in normalized:
            return normalized
        return normalized.split("/", 1)[0] or "."

    @staticmethod
    def _overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
        shared = min(len(left), len(right))
        return left[:shared] == right[:shared]

    @staticmethod
    def _contains(
        ancestor: tuple[str, ...],
        descendant: tuple[str, ...],
    ) -> bool:
        return (
            len(ancestor) <= len(descendant)
            and descendant[: len(ancestor)] == ancestor
        )

    def _owned_parts(self, owner: int) -> tuple[tuple[str, ...], ...]:
        return tuple(
            parts
            for parts, active_owner in self._active.values()
            if active_owner == owner
        )

    def _reentrant_narrow(
        self,
        parts: tuple[str, ...],
        owner: int,
    ) -> bool:
        return any(
            self._contains(owned, parts)
            for owned in self._owned_parts(owner)
        )

    def _reject_widening_upgrade(
        self,
        parts: tuple[str, ...],
        owner: int,
    ) -> None:
        owned_parts = self._owned_parts(owner)
        if any(self._contains(owned, parts) for owned in owned_parts):
            return
        if any(self._contains(parts, owned) for owned in owned_parts):
            raise RuntimeError(
                "cannot widen an owned filesystem path lock; "
                "acquire the ancestor scope first"
            )

    def _can_acquire(
        self,
        token: int,
        parts: tuple[str, ...],
        owner: int,
        *,
        reentrant_narrow: bool,
    ) -> bool:
        if any(
            active_owner != owner and self._overlap(parts, active_parts)
            for active_parts, active_owner in self._active.values()
        ):
            return False
        if reentrant_narrow:
            return True
        for waiter_token, waiter_parts, _waiter_owner in self._waiters:
            if waiter_token == token:
                return True
            if self._overlap(parts, waiter_parts):
                return False
        return False

    def _acquire(
        self,
        parts: tuple[str, ...],
        *,
        blocking: bool,
        timeout: float,
    ) -> int | None:
        owner = threading.get_ident()
        deadline = None if timeout < 0 else time.monotonic() + timeout
        with self._condition:
            self._reject_widening_upgrade(parts, owner)
            reentrant_narrow = self._reentrant_narrow(parts, owner)
            self._next_token += 1
            token = self._next_token
            waiter = (token, parts, owner)
            self._waiters.append(waiter)
            try:
                while not self._can_acquire(
                    token,
                    parts,
                    owner,
                    reentrant_narrow=reentrant_narrow,
                ):
                    if not blocking:
                        return None
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        return None
                    self._condition.wait(remaining)
                self._active[token] = (parts, owner)
                return token
            finally:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()

    def _release(self, token: int) -> None:
        owner = threading.get_ident()
        with self._condition:
            active = self._active.get(token)
            if active is None or active[1] != owner:
                raise RuntimeError("cannot release an unowned filesystem path lock")
            self._active.pop(token)
            self._condition.notify_all()

    @contextmanager
    def hold(self, path: str) -> Iterator[None]:
        token = self._acquire(self._parts(path), blocking=True, timeout=-1)
        assert token is not None
        try:
            yield
        finally:
            self._release(token)

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not blocking and timeout != -1:
            raise ValueError("can't specify a timeout for a non-blocking acquire")
        token = self._acquire((), blocking=blocking, timeout=timeout)
        if token is None:
            return False
        tokens = getattr(self._root_tokens, "tokens", None)
        if tokens is None:
            tokens = []
            self._root_tokens.tokens = tokens
        tokens.append(token)
        return True

    def release(self) -> None:
        tokens = getattr(self._root_tokens, "tokens", None)
        if not tokens:
            raise RuntimeError("cannot release an un-acquired filesystem root lock")
        self._release(tokens.pop())

    def __enter__(self) -> "HierarchicalPathLock":
        self.acquire()
        return self

    def __exit__(self, _exc_type: Any, _exc: BaseException | None, _tb: Any) -> None:
        self.release()


def _executable_stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    change_identity = int(value.st_ctime_ns)
    if os.name == "nt":
        # Python 3.14 exposes Windows creation time as st_birthtime_ns while
        # st_ctime_ns from a descriptor and from its path can disagree for the
        # same file.  Birth time remains stable across both APIs and, together
        # with the file ID, size, mtime, and content hash, identifies the file
        # without treating every Windows executable as a replacement race.
        change_identity = int(
            getattr(value, "st_birthtime_ns", value.st_ctime_ns)
        )
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        change_identity,
    )


def executable_content_sha256(executable: str | Path) -> str:
    """Hash one stable regular-file executable without following a final link."""

    selected = Path(executable)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValidationError(
            f"executable identity cannot be opened without following links: {selected}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"executable identity is not a regular file: {selected}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        current = os.stat(selected, follow_symlinks=False)
    except OSError as exc:
        raise ValidationError(
            f"executable identity changed while it was fingerprinted: {selected}"
        ) from exc
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise ValidationError(f"executable identity changed to a non-regular file: {selected}")

    if (
        _executable_stat_identity(before) != _executable_stat_identity(after)
        or _executable_stat_identity(after) != _executable_stat_identity(current)
    ):
        raise ValidationError(
            f"executable identity changed while it was fingerprinted: {selected}"
        )
    return digest.hexdigest()


class ExecutableSnapshot:
    """Host-owned immutable copy of one validated executable file.

    The copy lives in a private directory outside the model-visible workspace.
    Callers pass ``executable_path`` to the provider and keep this object alive
    until the process has consumed the executable (or, for scripts, until the
    interpreter no longer needs to reopen it).
    """

    def __init__(
        self,
        *,
        source_path: Path,
        executable_path: Path,
        content_sha256: str,
        directory: Path,
    ) -> None:
        self.source_path = source_path
        self.executable_path = executable_path
        self.content_sha256 = content_sha256
        self._directory = directory
        self._closed = False

    def verify(self) -> None:
        if self._closed:
            raise ValidationError("executable snapshot is already closed")
        try:
            actual = executable_content_sha256(self.executable_path)
        except (OSError, ValidationError) as exc:
            raise ValidationError("Host executable snapshot is no longer available") from exc
        if actual != self.content_sha256:
            raise ValidationError("Host executable snapshot content changed")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            os.chmod(self._directory, 0o700)
        except OSError:
            pass
        shutil.rmtree(self._directory, ignore_errors=True)

    def __enter__(self) -> "ExecutableSnapshot":
        return self

    def __exit__(self, _exc_type: Any, _exc: BaseException | None, _tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def snapshot_executable(
    executable: str | Path,
    *,
    sibling_limit: int = _TOOL_DEFAULTS.executable_snapshot_sibling_limit,
) -> ExecutableSnapshot:
    """Copy one stable executable into a private Host-owned dispatch object."""

    selected = Path(executable)
    source_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    source_flags |= getattr(os, "O_CLOEXEC", 0)
    source_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(selected, source_flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValidationError(
            f"executable snapshot cannot open source without following links: {selected}"
        ) from exc

    directory = Path(tempfile.mkdtemp(prefix="agent-libos-executable-"))
    destination = directory / (selected.name or "executable")
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValidationError(f"executable snapshot source is not a regular file: {selected}")
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        destination_flags |= getattr(os, "O_BINARY", 0)
        destination_flags |= getattr(os, "O_CLOEXEC", 0)
        destination_fd = os.open(destination, destination_flags, 0o700)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_fd, remaining)
                if written <= 0:
                    raise OSError("executable snapshot write made no progress")
                remaining = remaining[written:]
        os.fsync(destination_fd)
        executable_mode = stat.S_IMODE(before.st_mode) & 0o555
        if hasattr(os, "fchmod"):
            os.fchmod(destination_fd, executable_mode)
        else:  # pragma: no cover - Windows has no descriptor chmod.
            os.chmod(destination, executable_mode)
        after = os.fstat(source_fd)
        try:
            current = os.stat(selected, follow_symlinks=False)
        except OSError as exc:
            raise ValidationError(
                f"executable identity changed while it was snapshotted: {selected}"
            ) from exc
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
            raise ValidationError(
                f"executable identity changed to a non-regular file: {selected}"
            )
        if (
            _executable_stat_identity(before) != _executable_stat_identity(after)
            or _executable_stat_identity(after) != _executable_stat_identity(current)
        ):
            raise ValidationError(
                f"executable identity changed while it was snapshotted: {selected}"
            )
        os.close(destination_fd)
        destination_fd = None
        _mirror_executable_siblings(
            selected,
            directory,
            sibling_limit=sibling_limit,
        )
        os.chmod(directory, 0o500)
        snapshot = ExecutableSnapshot(
            source_path=selected.resolve(strict=True),
            executable_path=destination,
            content_sha256=digest.hexdigest(),
            directory=directory,
        )
        snapshot.verify()
        return snapshot
    except BaseException:
        if destination_fd is not None:
            try:
                os.close(destination_fd)
            except OSError:
                pass
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        os.close(source_fd)


def _mirror_executable_siblings(
    source: Path,
    snapshot_directory: Path,
    *,
    sibling_limit: int,
) -> None:
    """Expose sibling resources beside a private executable snapshot.

    Scripts commonly resolve resources relative to ``$0`` or ``__file__``.
    Executing the stable copy from a private directory must not make those
    sibling resources disappear.  The links preserve the original live
    resource semantics while the executable bytes themselves remain pinned.
    The direct sibling set is bounded and all-or-nothing.  Silently omitting a
    configuration file, plugin, or directory can change the executable's
    behavior after its Sink identity was authorized, so any enumeration/link
    failure aborts before final provider dispatch.
    """

    if (
        isinstance(sibling_limit, bool)
        or not isinstance(sibling_limit, int)
        or sibling_limit <= 0
    ):
        raise ValidationError("executable snapshot sibling limit must be a positive integer")
    siblings: list[Path] = []
    try:
        for sibling in source.parent.iterdir():
            if sibling.name == source.name:
                continue
            if len(siblings) >= sibling_limit:
                raise ValidationError(
                    "executable snapshot sibling count exceeds configured limit "
                    f"{sibling_limit}"
                )
            siblings.append(sibling)
    except ValidationError:
        raise
    except OSError as exc:
        raise ValidationError(
            "executable snapshot cannot enumerate sibling resources"
        ) from exc
    for sibling in siblings:
        mirror = snapshot_directory / sibling.name
        try:
            mirror.symlink_to(
                sibling.absolute(),
                target_is_directory=sibling.is_dir(),
            )
        except OSError as symlink_error:
            # Windows may not permit unprivileged symlink creation. A hard
            # link preserves ordinary sibling-file access without copying
            # content; directory mirroring instead fails closed on that Host.
            if os.name != "nt" or not sibling.is_file():
                raise ValidationError(
                    "executable snapshot cannot expose sibling resource "
                    f"{sibling.name!r}"
                ) from symlink_error
            try:
                os.link(sibling, mirror)
            except OSError as link_error:
                raise ValidationError(
                    "executable snapshot cannot expose sibling resource "
                    f"{sibling.name!r}"
                ) from link_error


def resolve_runtime_python_alias(
    command: str,
    *,
    workspace_root: str | Path,
) -> str | None:
    """Resolve supported bare Python aliases to the Host runtime interpreter.

    A workspace-local virtualenv entry is never returned. Its resolved target
    must be outside the workspace so a model-writable checkout cannot replace
    the executable selected for the Sink identity or provider dispatch.
    """

    raw_command = Path(command)
    if raw_command.is_absolute() or "/" in command or "\\" in command:
        return None
    normalized = command.casefold()
    if normalized.endswith(".exe"):
        normalized = normalized[:-4]
    aliases = {
        "python",
        "python3",
        f"python{sys.version_info.major}",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }
    if normalized not in aliases:
        return None
    root = Path(workspace_root).resolve(strict=False)
    candidates = (
        getattr(sys, "_base_executable", None),
        sys.executable,
    )
    for candidate in candidates:
        if not candidate:
            continue
        raw_executable = Path(candidate).expanduser()
        if not raw_executable.is_absolute():
            continue
        # Reject workspace-local aliases before following symlinks. A writable
        # virtualenv link can otherwise be retargeted to a different external
        # binary after the runtime starts.
        if raw_executable == root or root in raw_executable.parents:
            continue
        executable = raw_executable.resolve(strict=False)
        if executable == root or root in executable.parents:
            continue
        return str(executable)
    return None


@dataclass(frozen=True)
class ResolvedPath:
    relative: str
    display: str
    is_root: bool = False


@dataclass(frozen=True)
class PathState:
    exists: bool
    kind: str
    size_bytes: int | None = None
    modified_at: str | None = None


@dataclass(frozen=True)
class DirectoryEntrySnapshot:
    name: str
    path: str
    kind: str
    size_bytes: int | None
    modified_at: str


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    metrics: "CommandMetrics | None" = None


@dataclass(frozen=True)
class GitCommandResult:
    """Byte-preserving result returned by the Host Git provider."""

    argv: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_sha256: str
    stderr_sha256: str
    metrics: "CommandMetrics"


@dataclass(frozen=True)
class GitRepositoryLayout:
    """Validated identity of one worktree belonging to the pinned repository."""

    root: Path
    git_dir: Path
    common_dir: Path
    object_format: str
    linked_worktree: bool
    repository_id: str
    worktree_id: str
    git_version: str


@dataclass(frozen=True)
class GitRepositoryState:
    """Complete bounded repository state used to derive opaque CAS tokens."""

    layout: GitRepositoryLayout
    head_ref: str | None
    head_oid: str | None
    index_sha256: str
    config_sha256: str
    refs_sha256: str
    worktrees_sha256: str
    pull_requests_sha256: str
    worktree_sha256: str
    status_porcelain: bytes
    status_sha256: str


@dataclass(frozen=True)
class CommandMetrics:
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    killed: bool = False
    limit_kind: str | None = None


@dataclass(frozen=True)
class SubprocessLimits:
    wall_seconds: float | None = None
    cpu_seconds: float | None = None
    memory_bytes: int | None = None


class SubprocessLimitExceeded(Exception):
    def __init__(self, message: str, *, metrics: CommandMetrics, result: CommandResult | None = None) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.result = result


class SubprocessTimeoutExpired(TimeoutError):
    def __init__(self, message: str, *, metrics: CommandMetrics, result: CommandResult | None = None) -> None:
        super().__init__(message)
        self.metrics = metrics
        self.result = result


class ProviderEffectNotStarted(RuntimeError):
    """Provider-certified failure before an externally visible effect began.

    Primitives treat every other exception from an effectful provider call as
    an ambiguous outcome: one-shot authority remains consumed and an unknown
    external-effect record is persisted.  A provider may raise this exception
    only when it can guarantee that no mutation, delivery, remote request, or
    other externally visible operation was attempted.
    """


class FilesystemProvider(Protocol):
    namespace: str
    root_display: str

    def resolve(self, path: Any) -> ResolvedPath: ...

    def state(self, path: ResolvedPath) -> PathState: ...

    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes: ...

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = "\n",
        *,
        overwrite: bool = True,
    ) -> None: ...

    def make_directory(self, path: ResolvedPath, *, parents: bool, exist_ok: bool) -> None: ...

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None) -> Sequence[DirectoryEntrySnapshot]: ...

    def delete_file(self, path: ResolvedPath) -> None: ...

    def delete_directory(self, path: ResolvedPath, *, recursive: bool) -> None: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class ClockProvider(Protocol):
    def now(self, timezone: tzinfo) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...

    async def asleep(self, seconds: float) -> None: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class ShellProvider(Protocol):
    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]: ...

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
    ) -> CommandResult: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class GitProvider(Protocol):
    def validate_operation(
        self,
        operation: str,
        *,
        worktree: str | Path | None = None,
        remote: str | None = None,
    ) -> dict[str, str]: ...

    def validate_read_only_operation(
        self,
        operation: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, str]: ...

    def repository_layout(
        self,
        *,
        worktree: str | Path | None = None,
    ) -> GitRepositoryLayout: ...

    def repository_state(
        self,
        *,
        worktree: str | Path | None = None,
    ) -> GitRepositoryState: ...

    def run(
        self,
        args: Sequence[str],
        *,
        worktree: str | Path | None = None,
        timeout: float | None = None,
        stdin: bytes | None = None,
        max_output_bytes: int | None = None,
        read_only: bool = True,
        remote: str | None = None,
        expected_remote_fingerprint: str | None = None,
        verify_after: bool = True,
    ) -> GitCommandResult: ...

    def repository_lock(
        self,
        *,
        worktree: str | Path | None = None,
        timeout: float | None = None,
    ) -> Iterator[GitRepositoryLayout]: ...

    def reconcile_external_effect(self, effect: Any) -> dict[str, Any]: ...

    def remote_fingerprint(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, Any]: ...

    def preflight_remote_fingerprint(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, Any]: ...

    def remote_configuration(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> tuple[str, str, dict[str, Any]]: ...

    def prepare_managed_worktree(self, worktree_id: str) -> Path: ...

    def path_content_sha256(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str | None: ...

    def path_kind(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str: ...

    def preflight_path_kind(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str: ...

    def read_pull_request_metadata(self, pr_id: str) -> tuple[bytes, str] | None: ...

    def list_pull_request_metadata(self, *, limit: int) -> tuple[tuple[str, bytes, str], ...]: ...

    def write_pull_request_metadata(
        self,
        pr_id: str,
        data: bytes,
        *,
        expected_sha256: str | None,
        create: bool = False,
    ) -> str: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class HumanProvider(Protocol):
    def write(self, message: str) -> None: ...

    def read(self, prompt: str) -> str: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class JsonRpcProvider(Protocol):
    def call(
        self,
        endpoint: JsonRpcEndpointSpec,
        method: JsonRpcMethodSpec,
        request_body: bytes,
        *,
        timeout_s: float,
        max_response_bytes: int,
        resolved_addresses: tuple[str, ...] | None = None,
    ) -> JsonRpcTransportResult: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class McpProvider(Protocol):
    def validate_and_call(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult: ...

    def list_tools(
        self,
        server: McpServerSpec,
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpToolListResult: ...

    def call_tool(
        self,
        server: McpServerSpec,
        tool: McpToolSpec,
        arguments: dict[str, Any],
        *,
        timeout_s: float,
        max_response_bytes: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> McpProviderCallResult: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class ResourceProviderSubstrate(Protocol):
    """Collection of host-effect providers behind libOS primitives.

    Providers implement concrete filesystem, clock, shell, and human I/O
    backends. Primitive managers remain responsible for process identity,
    capability checks, approval, events, and audit before calling providers.
    """

    filesystem: FilesystemProvider
    clock: ClockProvider
    shell: ShellProvider
    git: GitProvider
    human: HumanProvider
    jsonrpc: JsonRpcProvider
    mcp: McpProvider
    workspace_display: str
