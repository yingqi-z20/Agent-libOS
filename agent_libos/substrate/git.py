from __future__ import annotations

import contextlib
import hashlib
import math
import mmap
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import psutil

from agent_libos.config import DEFAULT_CONFIG, GitDefaults
from agent_libos.models import (
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    GitErrorCode,
)
from agent_libos.models.exceptions import GitError, ValidationError
from agent_libos.substrate.base import (
    CommandMetrics,
    GitCommandResult,
    GitRepositoryLayout,
    GitRepositoryState,
    ProviderEffectNotStarted,
    SubprocessLimitExceeded,
    SubprocessLimits,
    executable_content_sha256,
)
from agent_libos.utils.secure_host_files import (
    SecureFileDescriptor,
    SecureFileChanged,
    SecureFileLimitExceeded,
    SecureFileReadUnavailable,
    StablePathSnapshot,
    open_secure_directory,
    open_secure_file,
    open_secure_readwrite_child,
    read_stable_file_limited,
    snapshot_from_stat,
    stable_identity_available,
)

_VERSION_RE = re.compile(rb"(?:git version\s+)?(\d+)\.(\d+)(?:\.(\d+))?")
_REMOTE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANAGED_WORKTREE_ID_RE = re.compile(r"wt_[A-Za-z0-9_-]{1,96}\Z")
_PULL_REQUEST_ID_RE = re.compile(r"pr_[A-Za-z0-9_-]{1,96}\Z")
_SCP_REMOTE_RE = re.compile(
    r"(?:(?P<user>[A-Za-z0-9._-]+)@)?(?P<host>[A-Za-z0-9.-]+):(?P<path>[^\x00\r\n]+)\Z"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTRIBUTE_LINE_RE = re.compile(
    rb'(?P<pattern>"(?:[^"\\]|\\.)*"|[^ \t\r\n]+)(?:[ \t]+(?P<attributes>.*))?\Z'
)
_ATTRIBUTE_DRIVER_RE = re.compile(
    rb"(?:^|[ \t])(?P<kind>filter|diff|merge)=(?P<name>[^ \t\r\n]+)(?=$|[ \t\r\n])"
)
_DANGEROUS_HELPER_CHARS = frozenset(";&|`$<>(){}\n\r")
_REMOTE_OPERATIONS = frozenset({"fetch", "pull", "push", "ls-remote"})
_READ_OPERATIONS = frozenset(
    {
        "repository_info",
        "status",
        "diff",
        "log",
        "show",
        "blame",
        "list_refs",
        "list_remotes",
        "list_worktrees",
    }
)
_RAW_READ_OPERATIONS = frozenset({"repository_info", "status", "diff", "list_refs"})
_CONTENT_FILTER_OPERATIONS = frozenset(
    {
        "add",
        "apply",
        "checkout",
        "cherry-pick",
        "commit",
        "diff",
        "merge",
        "pull",
        "rebase",
        "reset",
        "restore",
        "show",
        "stash",
        "status",
        "switch",
        "worktree",
    }
)
_DIFF_OPERATIONS = frozenset({"blame", "diff", "log", "show"})
_MERGE_OPERATIONS = frozenset({"cherry-pick", "merge", "pull", "rebase", "revert"})
_MINIMUM_SPAWNED_PROCESS_RSS_BYTES = max(1, mmap.PAGESIZE)


@dataclass(slots=True)
class _GitSubprocessScope:
    limits: SubprocessLimits | None
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    killed: bool = False
    limit_kind: str | None = None

    @property
    def metrics(self) -> CommandMetrics:
        return CommandMetrics(
            wall_seconds=self.wall_seconds,
            cpu_seconds=self.cpu_seconds,
            peak_memory_bytes=self.peak_memory_bytes,
            killed=self.killed,
            limit_kind=self.limit_kind,
        )

    def remaining_limits(self) -> SubprocessLimits | None:
        if self.limits is None:
            return None
        return SubprocessLimits(
            wall_seconds=(
                None
                if self.limits.wall_seconds is None
                else max(0.0, self.limits.wall_seconds - self.wall_seconds)
            ),
            cpu_seconds=(
                None
                if self.limits.cpu_seconds is None
                else max(0.0, self.limits.cpu_seconds - self.cpu_seconds)
            ),
            memory_bytes=self.limits.memory_bytes,
        )

    def record(self, metrics: CommandMetrics) -> None:
        self.wall_seconds += max(0.0, metrics.wall_seconds)
        self.cpu_seconds += max(0.0, metrics.cpu_seconds)
        self.peak_memory_bytes = max(
            self.peak_memory_bytes,
            max(0, metrics.peak_memory_bytes),
        )
        self.killed = self.killed or metrics.killed
        if metrics.limit_kind is not None:
            self.limit_kind = metrics.limit_kind


@dataclass(slots=True)
class _GitProcessSupervision:
    cpu_seconds: float = 0.0
    peak_memory_bytes: int = 0
    resource_limit_kind: str | None = None
    output_limit_kind: str | None = None
    timed_out: bool = False
    terminated_for_limit: bool = False


@dataclass(slots=True)
class _GitStdinDelivery:
    thread: threading.Thread
    errors: list[BaseException]

    @property
    def failure(self) -> BaseException | None:
        return self.errors[0] if self.errors else None


@dataclass(frozen=True, slots=True)
class _GitInvocationObservation:
    returncode: int
    stdout: bytes
    stderr: bytes
    metrics: CommandMetrics
    resource_limit_kind: str | None
    output_limit_kind: str | None
    timed_out: bool


_FILTER_DRIVER_SUFFIXES = frozenset({"clean", "smudge", "process"})
_DIFF_DRIVER_SUFFIXES = frozenset({"command", "textconv"})
_MERGE_DRIVER_SUFFIXES = frozenset({"driver"})
_SAFE_REF_SUFFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,500}\Z")
_REPOSITORY_HTTP_SECURITY_SUFFIXES = frozenset(
    {
        "cookiefile",
        "curloptresolve",
        "delegation",
        "emptyauth",
        "extraheader",
        "followredirects",
        "pinnedpubkey",
        "proxy",
        "proxyauthmethod",
        "savecookies",
        "schannelcheckrevoke",
        "schannelusesslcainfo",
        "sslbackend",
        "sslcapath",
        "sslcainfo",
        "sslcert",
        "sslcertpasswordprotected",
        "sslkey",
        "sslverify",
    }
)


def _configured_driver_is_active(
    key: str,
    *,
    kind: str,
    suffixes: frozenset[str],
    driver_is_active: Any,
) -> bool:
    return (
        key.startswith(f"{kind}.")
        and key.rsplit(".", 1)[-1] in suffixes
        and driver_is_active(kind, key)
    )


def _is_credential_helper_key(key: str) -> bool:
    return key == "credential.helper" or (
        key.startswith("credential.") and key.endswith(".helper")
    )


def _is_safe_fetch_refspec(value: str, remote: str) -> bool:
    selected = value[1:] if value.startswith("+") else value
    source, separator, destination = selected.partition(":")
    source_prefix = "refs/heads/"
    destination_prefix = f"refs/remotes/{remote}/"
    if (
        not separator
        or not source.startswith(source_prefix)
        or not destination.startswith(destination_prefix)
    ):
        return False
    source_suffix = source[len(source_prefix) :]
    destination_suffix = destination[len(destination_prefix) :]
    if source_suffix != destination_suffix:
        return False
    if source_suffix == "*":
        return True
    return bool(
        _SAFE_REF_SUFFIX_RE.fullmatch(source_suffix)
        and not source_suffix.endswith(("/", ".", ".lock"))
        and "//" not in source_suffix
        and ".." not in source_suffix
        and "@{" not in source_suffix
        and not any(
            part.startswith(".") or part.endswith(".lock")
            for part in source_suffix.split("/")
        )
    )


class GitProviderEffectNotStarted(GitError, ProviderEffectNotStarted):
    """Stable Git error certifying that no protected Git effect started."""


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(
        int(getattr(metadata, "st_file_attributes", 0) or 0) & reparse_attribute
    )


def _validate_git_metadata_file_snapshot(
    snapshot: StablePathSnapshot,
    _after_read: bool,
) -> StablePathSnapshot:
    if (
        snapshot.is_reparse_point
        or not stat.S_ISREG(snapshot.mode)
        or snapshot.links < 1
        or not stable_identity_available(snapshot)
    ):
        raise OSError("Git metadata file is not a stable regular file")
    return snapshot


def _path_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _directory_identity(path: Path) -> tuple[int, int]:
    value = path.stat()
    snapshot = snapshot_from_stat(value)
    if (
        snapshot.is_reparse_point
        or not stat.S_ISDIR(snapshot.mode)
        or snapshot.links < 1
        or not stable_identity_available(snapshot)
    ):
        raise NotADirectoryError(path)
    return snapshot.device, snapshot.inode


def _digest_fields(*fields: str) -> str:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _is_positive_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return value > 0 and math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


class LocalGitProvider:
    """Pinned, byte-preserving system-Git provider.

    This provider deliberately has no public arbitrary-argv compatibility
    surface.  ``run`` is the narrow Host boundary used only by GitPrimitive,
    which constructs every accepted argument.  Repository discovery is also
    avoided: the workspace's lexical ``.git`` entry is validated first and is
    then supplied to Git explicitly.
    """

    supports_subprocess_limits = os.name != "nt"

    def __init__(
        self,
        workspace_root: str | Path,
        *,
        config: GitDefaults | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve(strict=False)
        self.config = config or DEFAULT_CONFIG.git
        raw_managed = Path(self.config.worktree_root)
        if raw_managed.is_absolute():
            managed = raw_managed.resolve(strict=False)
        else:
            managed = (self.workspace_root / raw_managed).resolve(strict=False)
        if not _is_within(managed, self.workspace_root) or managed == self.workspace_root:
            raise ValueError("git.worktree_root must resolve below the workspace root")
        if any(part.casefold() == ".git" for part in managed.relative_to(self.workspace_root).parts):
            raise ValueError("git.worktree_root must not be inside Git metadata")
        self.managed_worktree_root = managed
        self._thread_lock = threading.RLock()
        self._repository_lock_owner: int | None = None
        self._repository_lock_depth = 0
        self._subprocess_scope_state = threading.local()
        self._executable_cache: tuple[Path, tuple[int, int, int, int, int], str] | None = None
        self._version_cache: tuple[
            str,
            Path,
            tuple[int, int, int, int, int],
            str,
        ] | None = None
        # Creating the Host-owned empty hooks directory is deliberately lazy:
        # Git availability (including a usable temporary directory) must not
        # become a Runtime-startup prerequisite.  Retaining the owner object
        # also lets Python reclaim the directory with the provider instead of
        # leaking one directory per Runtime construction.
        self._hooks_tempdir: tempfile.TemporaryDirectory[str] | None = None

    @contextmanager
    def subprocess_scope(
        self,
        *,
        limits: SubprocessLimits | None = None,
    ) -> Iterator[_GitSubprocessScope]:
        if limits is not None:
            if not self.supports_subprocess_limits:
                raise ValidationError(
                    "Git provider cannot enforce SubprocessLimits on this platform"
                )
            for field in ("wall_seconds", "cpu_seconds", "memory_bytes"):
                value = getattr(limits, field)
                if value is None:
                    continue
                if not _is_positive_finite_number(value):
                    raise ValidationError(
                        f"Git provider {field} limit must be finite and > 0"
                    )
        if getattr(self._subprocess_scope_state, "current", None) is not None:
            raise ValidationError("Git provider subprocess scopes must not be nested")
        scope = _GitSubprocessScope(limits=limits)
        self._subprocess_scope_state.current = scope
        try:
            yield scope
        finally:
            with contextlib.suppress(AttributeError):
                del self._subprocess_scope_state.current

    def _active_subprocess_scope(self) -> _GitSubprocessScope | None:
        scope = getattr(self._subprocess_scope_state, "current", None)
        return scope if isinstance(scope, _GitSubprocessScope) else None

    def _disabled_hooks_path(self) -> Path:
        with self._thread_lock:
            if self._hooks_tempdir is None:
                try:
                    selected = tempfile.TemporaryDirectory(
                        prefix="agent-libos-git-hooks-"
                    )
                    path = Path(selected.name)
                    os.chmod(path, 0o500)
                except OSError as exc:
                    raise self._error(
                        GitErrorCode.COMMAND_FAILED,
                        "Git hook isolation could not be initialized",
                    ) from exc
                self._hooks_tempdir = selected
            return Path(self._hooks_tempdir.name)

    def _error(
        self,
        code: GitErrorCode,
        message: str,
        *,
        operation: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> GitError:
        return GitError(
            code.value,
            message,
            operation=operation,
            retryable=retryable,
            details=details,
        )

    def _safe_path(self) -> str:
        selected: list[str] = []
        for raw in os.environ.get("PATH", os.defpath).split(os.pathsep):
            if not raw:
                continue
            candidate = Path(raw).expanduser().resolve(strict=False)
            if _is_within(candidate, self.workspace_root):
                continue
            selected.append(str(candidate))
        if not selected:
            selected = os.defpath.split(os.pathsep)
        return os.pathsep.join(dict.fromkeys(selected))

    def _safe_env(self, *, read_only: bool) -> dict[str, str]:
        env: dict[str, str] = {
            "PATH": self._safe_path(),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "PAGER": "cat",
            "GIT_EDITOR": "true",
            "GIT_SEQUENCE_EDITOR": "true",
            "GIT_MERGE_AUTOEDIT": "no",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_NOSYSTEM": "0",
            "GIT_TRACE": "0",
            "GIT_TRACE2": "0",
            "GIT_TRACE2_EVENT": "0",
            "GIT_TRACE2_PERF": "0",
        }
        for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TMP", "TEMP"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        home = os.environ.get("HOME")
        if home:
            resolved_home = Path(home).expanduser().resolve(strict=False)
            if not _is_within(resolved_home, self.workspace_root):
                env["HOME"] = str(resolved_home)
        if self.config.inherit_ssh_agent:
            agent = os.environ.get("SSH_AUTH_SOCK")
            if agent:
                resolved_agent = Path(agent).expanduser().resolve(strict=False)
                if not _is_within(resolved_agent, self.workspace_root):
                    env["SSH_AUTH_SOCK"] = str(resolved_agent)
        if os.name == "nt":
            env["GCM_INTERACTIVE"] = "Never"
        return env

    @staticmethod
    def _expand_git_user_path(path: str | Path) -> Path:
        """Expand ``~/`` using the HOME value inherited by the Git child."""

        raw = os.fspath(path)
        if raw == "~" or raw.startswith(("~/", "~\\")):
            home = os.environ.get("HOME")
            if home:
                return Path(home) if raw == "~" else Path(home) / raw[2:]
        return Path(raw).expanduser()

    def _trusted_host_executable(self, name: str) -> tuple[Path, str]:
        selected = shutil.which(name, path=self._safe_path())
        if not selected and os.name == "nt" and not name.casefold().endswith(".exe"):
            selected = shutil.which(f"{name}.exe", path=self._safe_path())
        if not selected:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, f"required Host executable is unavailable: {name}")
        try:
            resolved = Path(selected).resolve(strict=True)
            metadata = resolved.stat()
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, f"Host executable could not be inspected: {name}") from exc
        if _is_within(resolved, self.workspace_root) or not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
            raise self._error(GitErrorCode.UNSAFE_CONFIG, f"Host executable is not trusted: {name}")
        return resolved, executable_content_sha256(resolved)

    @staticmethod
    def _uses_ssh(url: str) -> bool:
        return bool(_SCP_REMOTE_RE.fullmatch(url) and "://" not in url) or urlsplit(url).scheme.casefold() == "ssh"

    def _remote_dispatch_environment(self, fetch_url: str, push_url: str) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
        if not (self._uses_ssh(fetch_url) or self._uses_ssh(push_url)):
            return {}, ()
        executable, digest = self._trusted_host_executable("ssh")
        null_config = "NUL" if os.name == "nt" else "/dev/null"
        arguments = [
            str(executable),
            "-F",
            null_config,
            "-oBatchMode=yes",
            "-oClearAllForwardings=yes",
            "-oForwardAgent=no",
            "-oPermitLocalCommand=no",
            "-oProxyCommand=none",
            "-oProxyJump=none",
            "-oControlMaster=no",
            "-oCanonicalizeHostname=no",
        ]
        command = (
            subprocess.list2cmdline(arguments)
            if os.name == "nt"
            else " ".join(shlex.quote(argument) for argument in arguments)
        )
        return {"GIT_SSH_COMMAND": command}, ((str(executable), digest),)

    def _resolve_executable(self) -> tuple[Path, tuple[int, int, int, int, int], str]:
        configured = self.config.executable
        selected: str | None
        if Path(configured).is_absolute():
            selected = configured
        elif "/" in configured or "\\" in configured:
            raise self._error(
                GitErrorCode.GIT_UNAVAILABLE,
                "configured Git executable must be an absolute Host path or a bare name",
            )
        else:
            selected = shutil.which(configured, path=self._safe_path())
        if not selected:
            raise self._error(GitErrorCode.GIT_UNAVAILABLE, "system Git is unavailable")
        try:
            executable = Path(selected).resolve(strict=True)
            identity = _path_identity(executable)
        except OSError as exc:
            raise self._error(GitErrorCode.GIT_UNAVAILABLE, "system Git is unavailable") from exc
        if _is_within(executable, self.workspace_root):
            raise self._error(
                GitErrorCode.GIT_UNAVAILABLE,
                "workspace-controlled Git executables are not trusted",
            )
        mode = executable.stat().st_mode
        if not stat.S_ISREG(mode) or not os.access(executable, os.X_OK):
            raise self._error(GitErrorCode.GIT_UNAVAILABLE, "system Git is not executable")
        cached = self._executable_cache
        if cached is not None and cached[0] == executable and cached[1] == identity:
            return cached
        content_sha256 = executable_content_sha256(executable)
        resolved = (executable, identity, content_sha256)
        self._executable_cache = resolved
        return resolved

    @staticmethod
    def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
        try:
            parent = psutil.Process(process.pid)
            descendants = parent.children(recursive=True)
        except (psutil.Error, OSError):
            descendants = []
        for child in reversed(descendants):
            try:
                child.kill()
            except (psutil.Error, OSError):
                pass
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except (OSError, ProcessLookupError):
            try:
                process.kill()
            except OSError:
                pass

    @staticmethod
    def _sample_process_tree(
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
                    "Git provider cannot enforce CPU/memory SubprocessLimits "
                    "because complete process metrics are unavailable"
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
                        "Git provider cannot enforce CPU/memory SubprocessLimits "
                        "because complete process metrics are unavailable"
                    ) from exc
        return cpu_seconds, max(peak_memory, memory_bytes)

    @staticmethod
    def _subprocess_limit_kind(
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

    @staticmethod
    def _metrics_payload(metrics: CommandMetrics) -> dict[str, Any]:
        return {
            "wall_seconds": metrics.wall_seconds,
            "cpu_seconds": metrics.cpu_seconds,
            "peak_memory_bytes": metrics.peak_memory_bytes,
            "killed": metrics.killed,
            "limit_kind": metrics.limit_kind,
        }

    def _validate_invoke_bounds(
        self,
        *,
        timeout: float,
        max_output_bytes: int,
        operation: str,
    ) -> None:
        timeout_hard_limit = self.config.timeout_hard_limit_s
        if not _is_positive_finite_number(timeout_hard_limit):
            raise self._error(
                GitErrorCode.TIMEOUT,
                "invalid configured Git timeout hard limit",
                operation=operation,
            )
        if (
            not _is_positive_finite_number(timeout)
            or timeout > timeout_hard_limit
        ):
            raise self._error(GitErrorCode.TIMEOUT, "invalid Git timeout", operation=operation)
        output_hard_limit = self.config.output_hard_limit_bytes
        if (
            isinstance(output_hard_limit, bool)
            or not isinstance(output_hard_limit, int)
            or output_hard_limit <= 0
        ):
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "invalid configured Git output hard limit",
                operation=operation,
            )
        if (
            isinstance(max_output_bytes, bool)
            or not isinstance(max_output_bytes, int)
            or max_output_bytes <= 0
            or max_output_bytes > output_hard_limit
        ):
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "requested Git output limit exceeds the configured hard limit",
                operation=operation,
            )

    @staticmethod
    def _exhausted_subprocess_limit_kind(
        limits: SubprocessLimits,
    ) -> str | None:
        for kind, value in (
            ("subprocess_wall_seconds", limits.wall_seconds),
            ("subprocess_cpu_seconds", limits.cpu_seconds),
            ("subprocess_memory_bytes", limits.memory_bytes),
        ):
            if value is not None and value <= 0:
                return kind
        return None

    def _remaining_invoke_limits(
        self,
        active_scope: _GitSubprocessScope | None,
    ) -> SubprocessLimits | None:
        if active_scope is None:
            return None
        limits = active_scope.remaining_limits()
        if limits is None:
            return None
        exhausted_kind = self._exhausted_subprocess_limit_kind(limits)
        if exhausted_kind is None:
            return limits
        metrics = CommandMetrics(limit_kind=exhausted_kind)
        active_scope.limit_kind = exhausted_kind
        raise SubprocessLimitExceeded(
            f"Git subprocess exceeded {exhausted_kind}",
            metrics=metrics,
        )

    def _prepare_git_command(
        self,
        argv: Sequence[str],
    ) -> tuple[Path, tuple[int, int, int, int, int], str, list[str]]:
        executable, identity_before, content_before = self._resolve_executable()
        full_argv = [str(executable), *map(str, argv)]
        if any("\x00" in item for item in full_argv):
            raise self._error(GitErrorCode.COMMAND_FAILED, "Git argv contains a NUL byte")
        return executable, identity_before, content_before, full_argv

    def _launch_git_process(
        self,
        full_argv: list[str],
        *,
        stdin: bytes | None,
        stdout_file: Any,
        stderr_file: Any,
        read_only: bool,
        env_overrides: dict[str, str] | None,
    ) -> subprocess.Popen[bytes]:
        environment = self._safe_env(read_only=read_only)
        environment.update(dict(env_overrides or {}))
        try:
            process = subprocess.Popen(
                full_argv,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=self.workspace_root,
                env=environment,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError as exc:
            raise self._error(
                GitErrorCode.GIT_UNAVAILABLE,
                "system Git is unavailable",
            ) from exc
        except OSError as exc:
            raise self._error(
                GitErrorCode.COMMAND_FAILED,
                "system Git could not start",
            ) from exc
        return process

    @staticmethod
    def _start_git_stdin_delivery(
        process: subprocess.Popen[bytes],
        payload: bytes | None,
    ) -> _GitStdinDelivery | None:
        if payload is None or process.stdin is None:
            return None
        errors: list[BaseException] = []
        stdin_stream = process.stdin

        def deliver() -> None:
            try:
                stdin_stream.write(payload)
            except BrokenPipeError:
                pass
            except BaseException as exc:
                errors.append(exc)
            finally:
                try:
                    stdin_stream.close()
                except BrokenPipeError:
                    pass
                except BaseException as exc:
                    if not errors:
                        errors.append(exc)

        thread = threading.Thread(
            target=deliver,
            name=f"agent-libos-git-stdin-{process.pid}",
            daemon=True,
        )
        delivery = _GitStdinDelivery(thread=thread, errors=errors)
        thread.start()
        return delivery

    def _finish_git_stdin_delivery(
        self,
        delivery: _GitStdinDelivery | None,
        *,
        operation: str,
    ) -> BaseException | None:
        if delivery is None:
            return None
        delivery.thread.join(timeout=2.0)
        if delivery.thread.is_alive():
            return self._error(
                GitErrorCode.COMMAND_FAILED,
                "Git stdin delivery did not terminate after child cleanup",
                operation=operation,
            )
        return delivery.failure

    def _cleanup_git_process_after_failure(
        self,
        process: subprocess.Popen[bytes],
        delivery: _GitStdinDelivery | None,
    ) -> None:
        # This cleanup owns the child immediately after Popen succeeds.  It
        # deliberately tolerates control-flow BaseExceptions from individual
        # cleanup steps so that kill, reap, and writer join are all attempted
        # before the original failure is re-raised.
        with contextlib.suppress(BaseException):
            if process.poll() is None:
                self._kill_process_tree(process)
        with contextlib.suppress(BaseException):
            self._wait_for_git_process(process)
        if delivery is not None:
            with contextlib.suppress(BaseException):
                delivery.thread.join(timeout=2.0)

    def _attach_process_metrics(
        self,
        process: subprocess.Popen[bytes],
        *,
        require_complete: bool,
    ) -> psutil.Process | None:
        try:
            return psutil.Process(process.pid)
        except (psutil.Error, OSError) as exc:
            if require_complete:
                self._kill_process_tree(process)
                with contextlib.suppress(Exception):
                    process.wait(timeout=1.0)
                raise ValidationError(
                    "Git provider cannot enforce CPU/memory SubprocessLimits "
                    "because process metrics are unavailable"
                ) from exc
        return None

    @staticmethod
    def _output_limit_kind_from_files(
        stdout_file: Any,
        stderr_file: Any,
        max_output_bytes: int,
    ) -> str | None:
        if os.fstat(stdout_file.fileno()).st_size > max_output_bytes:
            return "subprocess_stdout_bytes"
        if os.fstat(stderr_file.fileno()).st_size > max_output_bytes:
            return "subprocess_stderr_bytes"
        return None

    @staticmethod
    def _output_limit_kind_from_bytes(
        stdout: bytes,
        stderr: bytes,
        max_output_bytes: int,
    ) -> str | None:
        if len(stdout) > max_output_bytes:
            return "subprocess_stdout_bytes"
        if len(stderr) > max_output_bytes:
            return "subprocess_stderr_bytes"
        return None

    def _supervise_git_process(
        self,
        process: subprocess.Popen[bytes],
        ps_process: psutil.Process | None,
        *,
        started: float,
        timeout: float,
        max_output_bytes: int,
        subprocess_limits: SubprocessLimits | None,
        require_complete_metrics: bool,
        stdout_file: Any,
        stderr_file: Any,
        stdin_delivery: _GitStdinDelivery | None,
    ) -> _GitProcessSupervision:
        # RSS is page-granular, so every successfully spawned process has at
        # least one resident page. Seed that safe lower bound when enforcing a
        # memory limit so a short-lived Git process cannot exit before psutil's
        # first sample and incorrectly appear to have consumed zero memory.
        state = _GitProcessSupervision(
            peak_memory_bytes=(
                _MINIMUM_SPAWNED_PROCESS_RSS_BYTES
                if subprocess_limits is not None
                and subprocess_limits.memory_bytes is not None
                else 0
            )
        )
        try:
            while True:
                if stdin_delivery is not None and stdin_delivery.failure is not None:
                    state.terminated_for_limit = True
                    self._kill_process_tree(process)
                    break
                wall_seconds = max(0.0, time.monotonic() - started)
                if ps_process is not None:
                    state.cpu_seconds, state.peak_memory_bytes = (
                        self._sample_process_tree(
                            ps_process,
                            state.peak_memory_bytes,
                            require_complete=require_complete_metrics,
                        )
                    )
                state.resource_limit_kind = self._subprocess_limit_kind(
                    wall_seconds=wall_seconds,
                    cpu_seconds=state.cpu_seconds,
                    peak_memory=state.peak_memory_bytes,
                    limits=subprocess_limits,
                )
                state.output_limit_kind = self._output_limit_kind_from_files(
                    stdout_file,
                    stderr_file,
                    max_output_bytes,
                )
                if (
                    state.resource_limit_kind is not None
                    or state.output_limit_kind is not None
                ):
                    state.terminated_for_limit = True
                    self._kill_process_tree(process)
                    break
                if wall_seconds >= timeout:
                    state.timed_out = True
                    state.terminated_for_limit = True
                    self._kill_process_tree(process)
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.01)
        finally:
            if process.poll() is None:
                self._kill_process_tree(process)
        return state

    def _wait_for_git_process(self, process: subprocess.Popen[bytes]) -> None:
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._kill_process_tree(process)
            process.wait()

    def _complete_git_observation(
        self,
        process: subprocess.Popen[bytes],
        ps_process: psutil.Process | None,
        state: _GitProcessSupervision,
        *,
        started: float,
        timeout: float,
        max_output_bytes: int,
        subprocess_limits: SubprocessLimits | None,
        require_complete_metrics: bool,
        stdout_file: Any,
        stderr_file: Any,
    ) -> _GitInvocationObservation:
        self._wait_for_git_process(process)
        elapsed = max(0.0, time.monotonic() - started)
        if ps_process is not None:
            final_cpu, state.peak_memory_bytes = self._sample_process_tree(
                ps_process,
                state.peak_memory_bytes,
                require_complete=require_complete_metrics,
            )
            state.cpu_seconds = max(state.cpu_seconds, final_cpu)
        if state.resource_limit_kind is None:
            state.resource_limit_kind = self._subprocess_limit_kind(
                wall_seconds=elapsed,
                cpu_seconds=state.cpu_seconds,
                peak_memory=state.peak_memory_bytes,
                limits=subprocess_limits,
            )
        if not state.timed_out and elapsed >= timeout:
            state.timed_out = True
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read(max_output_bytes + 1)
        stderr = stderr_file.read(max_output_bytes + 1)
        if state.output_limit_kind is None:
            state.output_limit_kind = self._output_limit_kind_from_bytes(
                stdout,
                stderr,
                max_output_bytes,
            )
        limit_kind = (
            state.resource_limit_kind
            or state.output_limit_kind
            or ("subprocess_timeout" if state.timed_out else None)
        )
        metrics = CommandMetrics(
            wall_seconds=elapsed,
            cpu_seconds=state.cpu_seconds,
            peak_memory_bytes=state.peak_memory_bytes,
            killed=state.terminated_for_limit,
            limit_kind=limit_kind,
        )
        return _GitInvocationObservation(
            returncode=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            metrics=metrics,
            resource_limit_kind=state.resource_limit_kind,
            output_limit_kind=state.output_limit_kind,
            timed_out=state.timed_out,
        )

    def _verify_git_executable_after_invoke(
        self,
        executable: Path,
        identity_before: tuple[int, int, int, int, int],
        content_before: str,
        *,
        operation: str,
    ) -> None:
        try:
            executable_after, identity_after, content_after = self._resolve_executable()
        except GitError as exc:
            raise self._error(
                GitErrorCode.UNKNOWN_EFFECT,
                "Git executable identity could not be revalidated after dispatch",
                operation=operation,
            ) from exc
        if (
            executable_after != executable
            or identity_after != identity_before
            or content_after != content_before
        ):
            raise self._error(
                GitErrorCode.UNKNOWN_EFFECT,
                "Git executable identity changed during dispatch",
                operation=operation,
            )

    def _raise_for_invoke_failure(
        self,
        observation: _GitInvocationObservation,
        *,
        operation: str,
        read_only: bool,
    ) -> None:
        metrics = observation.metrics
        if observation.resource_limit_kind is not None:
            raise SubprocessLimitExceeded(
                f"Git subprocess exceeded {observation.resource_limit_kind}",
                metrics=metrics,
            )
        if observation.timed_out:
            raise self._error(
                GitErrorCode.TIMEOUT,
                "Git operation timed out; its effect may be unknown",
                operation=operation,
                retryable=read_only,
                details={
                    "effect": "none" if read_only else "unknown",
                    "limit_kind": "subprocess_timeout",
                    "metrics": self._metrics_payload(metrics),
                },
            )
        if observation.output_limit_kind is not None:
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "Git output exceeded the configured hard limit",
                operation=operation,
                details={
                    "effect": "none" if read_only else "unknown",
                    "limit_kind": observation.output_limit_kind,
                    "metrics": self._metrics_payload(metrics),
                },
            )

    def _invoke(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        stdin: bytes | None,
        max_output_bytes: int,
        read_only: bool,
        operation: str,
        env_overrides: dict[str, str] | None = None,
    ) -> GitCommandResult:
        self._validate_invoke_bounds(
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            operation=operation,
        )
        active_scope = self._active_subprocess_scope()
        subprocess_limits = self._remaining_invoke_limits(active_scope)
        executable, identity_before, content_before, full_argv = (
            self._prepare_git_command(argv)
        )
        started = time.monotonic()
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process: subprocess.Popen[bytes] | None = None
            stdin_delivery: _GitStdinDelivery | None = None
            try:
                process = self._launch_git_process(
                    full_argv,
                    stdin=stdin,
                    stdout_file=stdout_file,
                    stderr_file=stderr_file,
                    read_only=read_only,
                    env_overrides=env_overrides,
                )
                stdin_delivery = self._start_git_stdin_delivery(process, stdin)
                require_complete_metrics = bool(
                    subprocess_limits is not None
                    and (
                        subprocess_limits.cpu_seconds is not None
                        or subprocess_limits.memory_bytes is not None
                    )
                )
                ps_process = self._attach_process_metrics(
                    process,
                    require_complete=require_complete_metrics,
                )
                state = self._supervise_git_process(
                    process,
                    ps_process,
                    started=started,
                    timeout=timeout,
                    max_output_bytes=max_output_bytes,
                    subprocess_limits=subprocess_limits,
                    require_complete_metrics=require_complete_metrics,
                    stdout_file=stdout_file,
                    stderr_file=stderr_file,
                    stdin_delivery=stdin_delivery,
                )
                observation = self._complete_git_observation(
                    process,
                    ps_process,
                    state,
                    started=started,
                    timeout=timeout,
                    max_output_bytes=max_output_bytes,
                    subprocess_limits=subprocess_limits,
                    require_complete_metrics=require_complete_metrics,
                    stdout_file=stdout_file,
                    stderr_file=stderr_file,
                )
                stdin_failure = self._finish_git_stdin_delivery(
                    stdin_delivery,
                    operation=operation,
                )
            except BaseException:
                if process is not None:
                    self._cleanup_git_process_after_failure(process, stdin_delivery)
                raise
        if active_scope is not None:
            active_scope.record(observation.metrics)
        if stdin_failure is not None:
            raise stdin_failure
        self._verify_git_executable_after_invoke(
            executable,
            identity_before,
            content_before,
            operation=operation,
        )
        self._raise_for_invoke_failure(
            observation,
            operation=operation,
            read_only=read_only,
        )
        return GitCommandResult(
            argv=tuple(full_argv),
            returncode=observation.returncode,
            stdout=observation.stdout,
            stderr=observation.stderr,
            stdout_sha256=hashlib.sha256(observation.stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(observation.stderr).hexdigest(),
            metrics=observation.metrics,
        )

    def _git_version(self) -> tuple[str, Path]:
        cached = self._version_cache
        if cached is not None:
            executable, identity, digest = self._resolve_executable()
            if (executable, identity, digest) == cached[1:]:
                return cached[0], cached[1]
        result = self._invoke(
            ["--version"],
            timeout=min(5.0, self.config.local_timeout_s),
            stdin=None,
            max_output_bytes=4096,
            read_only=True,
            operation="version",
        )
        match = _VERSION_RE.search(result.stdout.strip())
        if result.returncode != 0 or match is None:
            raise self._error(
                GitErrorCode.UNSUPPORTED_GIT_VERSION,
                "system Git returned an unsupported version string",
            )
        parts = tuple(int(value or b"0") for value in match.groups())
        minimum_match = re.fullmatch(r"(\d+)\.(\d+)(?:\.(\d+))?", self.config.minimum_version)
        if minimum_match is None:
            raise ValueError("git.minimum_version must be a dotted numeric version")
        minimum = tuple(int(value or "0") for value in minimum_match.groups())
        if parts < minimum:
            raise self._error(
                GitErrorCode.UNSUPPORTED_GIT_VERSION,
                f"system Git {parts[0]}.{parts[1]}.{parts[2]} is older than the required {self.config.minimum_version}",
            )
        executable, identity, digest = self._resolve_executable()
        selected = (
            f"{parts[0]}.{parts[1]}.{parts[2]}",
            executable,
            identity,
            digest,
        )
        self._version_cache = selected
        return selected[0], selected[1]

    def _resolve_worktree(self, worktree: str | Path | None) -> Path:
        if worktree is None:
            return self.workspace_root
        selected = Path(worktree)
        if not selected.is_absolute():
            selected = self.workspace_root / selected
        lexical = Path(os.path.abspath(selected))
        if lexical != self.workspace_root and not _is_within(
            lexical,
            self.managed_worktree_root,
        ):
            raise self._error(
                GitErrorCode.INVALID_PATH,
                "worktree is outside the Runtime-managed worktree root",
            )
        self._reject_path_indirection(
            lexical,
            anchor=self.workspace_root,
            description="Git worktree path",
        )
        resolved = lexical.resolve(strict=False)
        if resolved != lexical:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "Git worktree path traverses an untrusted link or reparse point",
            )
        return lexical

    def _reject_path_indirection(
        self,
        path: Path,
        *,
        anchor: Path,
        description: str,
    ) -> None:
        """Reject link/reparse components between a trusted anchor and path.

        ``Path.resolve`` cannot be used as the security check because it erases
        the lexical component that selected the target.  Missing suffixes are
        allowed here so callers such as worktree creation can validate their
        existing parents; the operation-specific existence check still owns the
        final error classification.
        """

        selected = Path(os.path.abspath(path))
        trusted = Path(os.path.abspath(anchor))
        if not _is_within(selected, trusted):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                f"{description} escaped its trusted root",
            )
        paths = [trusted]
        for component in selected.relative_to(trusted).parts:
            paths.append(paths[-1] / component)
        for current in paths:
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise self._error(
                    GitErrorCode.UNSAFE_REPOSITORY,
                    f"{description} could not be inspected",
                ) from exc
            if _is_link_or_reparse(metadata):
                raise self._error(
                    GitErrorCode.UNSAFE_REPOSITORY,
                    f"{description} traverses an untrusted link or reparse point",
                )
            if current != selected and not stat.S_ISDIR(metadata.st_mode):
                raise self._error(
                    GitErrorCode.UNSAFE_REPOSITORY,
                    f"{description} has a non-directory ancestor",
                )

    def _trusted_metadata_directory(self, candidate: Path) -> Path:
        lexical = Path(os.path.abspath(candidate))
        primary_entry = self.workspace_root / ".git"
        try:
            primary_state = primary_entry.lstat()
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "primary Git metadata entry could not be inspected",
            ) from exc
        if _is_link_or_reparse(primary_state):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "primary Git metadata entry is a link or reparse point",
            )
        primary = primary_entry.resolve(strict=False)
        trusted_roots = self._trusted_metadata_roots(primary)
        matching_roots = tuple(
            root
            for root in trusted_roots
            if _is_within(lexical, root)
        )
        if not matching_roots:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree metadata is outside trusted metadata roots",
            )
        trusted_root = max(matching_roots, key=lambda item: len(item.parts))
        self._reject_path_indirection(
            lexical,
            anchor=trusted_root,
            description="Git metadata path",
        )
        try:
            resolved = lexical.resolve(strict=True)
            target_state = lexical.lstat()
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "Git metadata directory is unavailable",
            ) from exc
        if (
            resolved != lexical
            or _is_link_or_reparse(target_state)
            or not stat.S_ISDIR(target_state.st_mode)
        ):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "Git metadata path is not a trusted directory",
            )
        return lexical

    @staticmethod
    def _read_small_file(path: Path, *, limit: int = 65536) -> bytes:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"metadata file exceeds {limit} bytes")
        return data

    @staticmethod
    def _read_trusted_metadata_file(path: Path, *, limit: int) -> bytes:
        secure_file = open_secure_file(path)
        return read_stable_file_limited(
            secure_file,
            max_bytes=limit,
            chunk_bytes=min(limit + 1, 64 * 1024),
            validate_snapshot=_validate_git_metadata_file_snapshot,
        )

    def _trusted_metadata_roots(self, primary_git_dir: Path) -> tuple[Path, ...]:
        roots = [primary_git_dir.resolve(strict=False)]
        for raw in self.config.trusted_metadata_roots:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            roots.append(candidate.resolve(strict=False))
        return tuple(dict.fromkeys(roots))

    def _git_dir_from_entry(self, worktree: Path) -> tuple[Path, bool]:
        entry = worktree / ".git"
        try:
            entry_state = entry.lstat()
        except FileNotFoundError as exc:
            raise self._error(
                GitErrorCode.NOT_REPOSITORY,
                "workspace root is not an existing Git worktree",
            ) from exc
        if _is_link_or_reparse(entry_state):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "the worktree .git entry must not be a symlink",
            )
        if stat.S_ISDIR(entry_state.st_mode):
            if worktree != self.workspace_root:
                raise self._error(
                    GitErrorCode.UNSAFE_REPOSITORY,
                    "managed linked worktrees must use a validated gitfile",
                )
            return entry.resolve(strict=True), False
        if not stat.S_ISREG(entry_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid worktree .git entry")
        try:
            raw = self._read_small_file(entry, limit=8192).strip()
        except (OSError, ValueError) as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid worktree gitfile") from exc
        prefix = b"gitdir: "
        if not raw.startswith(prefix) or b"\x00" in raw or b"\n" in raw or b"\r" in raw:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid worktree gitfile")
        raw_path = Path(os.fsdecode(raw[len(prefix) :]))
        candidate = raw_path if raw_path.is_absolute() else entry.parent / raw_path
        git_dir = self._trusted_metadata_directory(candidate)
        self._validate_linked_worktree_backpointer(
            worktree=worktree,
            entry=entry,
            git_dir=git_dir,
        )
        return git_dir, True

    def _validate_linked_worktree_backpointer(
        self,
        *,
        worktree: Path,
        entry: Path,
        git_dir: Path,
    ) -> None:
        primary_git_dir = (self.workspace_root / ".git").resolve(strict=False)
        if git_dir == primary_git_dir:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree cannot reuse primary Git metadata",
            )
        marker = git_dir / "gitdir"
        try:
            raw = self._read_trusted_metadata_file(marker, limit=8192).strip()
        except (
            OSError,
            SecureFileChanged,
            SecureFileLimitExceeded,
            SecureFileReadUnavailable,
        ) as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree backpointer is unavailable or unsafe",
            ) from exc
        if not raw or b"\x00" in raw or b"\n" in raw or b"\r" in raw:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree backpointer is malformed",
            )
        selected = Path(os.fsdecode(raw))
        candidate = selected if selected.is_absolute() else git_dir / selected
        backpointer = Path(os.path.abspath(candidate))
        expected = Path(os.path.abspath(entry))
        if backpointer != expected or expected != worktree / ".git":
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree backpointer does not match its worktree",
            )

    def _common_dir(self, git_dir: Path) -> Path:
        marker = git_dir / "commondir"
        try:
            marker_state = marker.lstat()
        except FileNotFoundError:
            return git_dir
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid Git common directory") from exc
        try:
            if _is_link_or_reparse(marker_state) or not stat.S_ISREG(marker_state.st_mode):
                raise ValueError("invalid commondir marker")
            raw = self._read_small_file(marker, limit=8192).strip()
            if not raw or b"\x00" in raw or b"\n" in raw or b"\r" in raw:
                raise ValueError("invalid commondir marker")
            selected = Path(os.fsdecode(raw))
            candidate = selected if selected.is_absolute() else git_dir / selected
            return self._trusted_metadata_directory(candidate)
        except (GitError, OSError, ValueError) as exc:
            if isinstance(exc, GitError):
                raise
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid Git common directory") from exc

    def _repo_prefix(
        self,
        layout: GitRepositoryLayout | tuple[Path, Path, Path],
        *,
        literal_pathspecs: bool = True,
    ) -> list[str]:
        if isinstance(layout, GitRepositoryLayout):
            worktree, git_dir = layout.root, layout.git_dir
        else:
            worktree, git_dir, _common = layout
        protocol_file = "always" if self.config.allow_file_remotes else "never"
        prefix = [
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            # Git for Windows still gates Win32 long-path support behind this
            # setting. Runtime-managed worktrees can legitimately cross the
            # legacy MAX_PATH boundary even when the primary checkout does not.
            "-c",
            "core.longpaths=true",
            "-c",
            f"core.hooksPath={self._disabled_hooks_path()}",
            "-c",
            "diff.external=",
            "-c",
            "color.ui=false",
            "-c",
            "trace2.eventTarget=",
            "-c",
            "trace2.normalTarget=",
            "-c",
            "trace2.perfTarget=",
            "-c",
            "commit.gpgSign=false",
            "-c",
            "tag.gpgSign=false",
            "-c",
            "push.gpgSign=false",
            "-c",
            "merge.autoStash=false",
            "-c",
            "submodule.recurse=false",
            "-c",
            "fetch.recurseSubmodules=false",
            "-c",
            "credential.interactive=never",
            "-c",
            "protocol.ext.allow=never",
            "-c",
            f"protocol.file.allow={protocol_file}",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            f"--git-dir={git_dir}",
            f"--work-tree={worktree}",
        ]
        if literal_pathspecs:
            prefix.insert(1, "--literal-pathspecs")
        return prefix

    def _raw_repo(
        self,
        raw_layout: tuple[Path, Path, Path],
        args: Sequence[str],
        *,
        operation: str,
        max_output_bytes: int = 65536,
    ) -> GitCommandResult:
        return self._invoke(
            [*self._repo_prefix(raw_layout), *args],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=max_output_bytes,
            read_only=True,
            operation=operation,
        )

    def _reject_alternates(self, common_dir: Path) -> None:
        for name in ("alternates", "http-alternates"):
            candidate = common_dir / "objects" / "info" / name
            try:
                state = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git alternates could not be inspected") from exc
            if _is_link_or_reparse(state) or not stat.S_ISREG(state.st_mode):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "untrusted Git alternates metadata")
            try:
                if self._read_small_file(candidate).strip():
                    raise self._error(
                        GitErrorCode.UNSAFE_REPOSITORY,
                        "external Git object alternates are not supported",
                    )
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git alternates could not be inspected") from exc

    def repository_layout(
        self,
        *,
        worktree: str | Path | None = None,
    ) -> GitRepositoryLayout:
        if not self.config.enabled:
            raise self._error(GitErrorCode.GIT_UNAVAILABLE, "Git integration is disabled")
        version, _executable = self._git_version()
        selected = self._resolve_worktree(worktree)
        try:
            selected_state = selected.lstat()
        except FileNotFoundError as exc:
            raise self._error(GitErrorCode.NOT_REPOSITORY, "Git worktree does not exist") from exc
        if _is_link_or_reparse(selected_state) or not stat.S_ISDIR(selected_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git worktree root is not a trusted directory")
        git_dir, linked = self._git_dir_from_entry(selected)
        common_dir = self._common_dir(git_dir)
        if linked and git_dir == common_dir:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree must use distinct per-worktree Git metadata",
            )
        primary_git_dir = (self.workspace_root / ".git").resolve(strict=False)
        if not any(_is_within(common_dir, root) for root in self._trusted_metadata_roots(primary_git_dir)):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git common directory is outside trusted metadata roots")
        raw_layout = (selected, git_dir, common_dir)
        identity = self._raw_repo(
            raw_layout,
            [
                "rev-parse",
                "--is-bare-repository",
                "--show-toplevel",
                "--absolute-git-dir",
                "--git-common-dir",
            ],
            operation="repository_info",
        )
        identity_lines = identity.stdout.rstrip(b"\r\n").splitlines()
        if identity.returncode != 0 or len(identity_lines) != 4:
            raise self._error(GitErrorCode.NOT_REPOSITORY, "workspace root is not an existing non-bare Git repository")
        bare_value, top_value, git_value, common_value = identity_lines
        if bare_value != b"false":
            raise self._error(GitErrorCode.NOT_REPOSITORY, "bare Git repositories are not supported")
        try:
            actual_top = Path(os.fsdecode(top_value)).resolve(strict=True)
            actual_git_dir = Path(os.fsdecode(git_value)).resolve(strict=True)
            raw_common_path = Path(os.fsdecode(common_value))
            actual_common_dir = (
                raw_common_path
                if raw_common_path.is_absolute()
                else selected / raw_common_path
            ).resolve(strict=True)
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git repository identity is unstable") from exc
        if actual_top != selected or actual_git_dir != git_dir or actual_common_dir != common_dir:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git repository identity did not match pinned metadata")
        object_result = self._raw_repo(raw_layout, ["rev-parse", "--show-object-format"], operation="repository_info")
        if object_result.returncode == 0:
            object_format = object_result.stdout.strip().decode("ascii", errors="strict")
        else:
            object_format = "sha1"
        if object_format not in {"sha1", "sha256"}:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unsupported Git object format")
        self._reject_alternates(common_dir)
        try:
            common_identity = _directory_identity(common_dir)
            git_identity = _directory_identity(git_dir)
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git metadata identity could not be read") from exc
        repository_id = _digest_fields(str(common_dir), repr(common_identity), object_format)[:32]
        worktree_id = _digest_fields(str(selected), str(git_dir), repr(git_identity))[:32]
        return GitRepositoryLayout(
            root=selected,
            git_dir=git_dir,
            common_dir=common_dir,
            object_format=object_format,
            linked_worktree=linked,
            repository_id=repository_id,
            worktree_id=worktree_id,
            git_version=version,
        )

    def _config_entries(self, layout: GitRepositoryLayout) -> list[tuple[str, str, str, str]]:
        result = self._invoke(
            [*self._repo_prefix(layout), "config", "--null", "--show-origin", "--show-scope", "--list"],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="config_inspection",
        )
        if result.returncode != 0:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "effective Git configuration could not be inspected")
        fields = result.stdout.split(b"\0")
        if fields and fields[-1] == b"":
            fields.pop()
        if len(fields) % 3:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "effective Git configuration has an unsupported format")
        entries: list[tuple[str, str, str, str]] = []
        for offset in range(0, len(fields), 3):
            try:
                scope = fields[offset].decode("utf-8", errors="strict")
                origin = fields[offset + 1].decode("utf-8", errors="strict")
                key_value = fields[offset + 2].decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "effective Git configuration is not UTF-8") from exc
            key, separator, value = key_value.partition("\n")
            entries.append((scope, origin, key.casefold(), value if separator else ""))
        return entries

    def _resolve_helper(self, value: str) -> tuple[str, str]:
        if not value or value.startswith("!") or any(char in value for char in _DANGEROUS_HELPER_CHARS):
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "shell credential helpers are not allowed")
        try:
            words = shlex.split(value, posix=os.name != "nt")
        except ValueError as exc:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "invalid credential helper") from exc
        if not words:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "invalid credential helper")
        helper = words[0]
        candidates: list[Path] = []

        def add_candidate(candidate: Path) -> None:
            candidates.append(candidate)
            if os.name == "nt" and candidate.suffix.casefold() != ".exe":
                candidates.append(Path(f"{candidate}.exe"))

        if Path(helper).is_absolute():
            add_candidate(Path(helper))
        else:
            executable = shutil.which(f"git-credential-{helper}", path=self._safe_path())
            if executable:
                add_candidate(Path(executable))
            git_path, _identity, _digest = self._resolve_executable()
            exec_path = self._invoke(
                ["--exec-path"],
                timeout=min(5.0, self.config.local_timeout_s),
                stdin=None,
                max_output_bytes=8192,
                read_only=True,
                operation="credential_helper_inspection",
            )
            if exec_path.returncode == 0:
                git_exec_dir = Path(os.fsdecode(exec_path.stdout.strip()))
                add_candidate(git_exec_dir / f"git-credential-{helper}")
                if os.name == "nt" and len(git_exec_dir.parents) >= 2:
                    # Git for Windows keeps Git Credential Manager in
                    # <install>/mingw64/bin while `git --exec-path` reports
                    # <install>/mingw64/libexec/git-core.
                    add_candidate(
                        git_exec_dir.parent.parent
                        / "bin"
                        / f"git-credential-{helper}"
                    )
            add_candidate(git_path.parent / f"git-credential-{helper}")
            if os.name == "nt":
                for parent in tuple(git_path.parents)[:3]:
                    add_candidate(
                        parent
                        / "mingw64"
                        / "bin"
                        / f"git-credential-{helper}"
                    )
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
                if _is_within(resolved, self.workspace_root):
                    continue
                if not stat.S_ISREG(resolved.stat().st_mode) or not os.access(resolved, os.X_OK):
                    continue
                return str(resolved), executable_content_sha256(resolved)
            except OSError:
                continue
        raise self._error(GitErrorCode.UNSAFE_CONFIG, "credential helper executable is not a trusted Host executable")

    @staticmethod
    def _config_selects_executable_extension(
        *,
        key: str,
        value: str,
        remote: str | None,
        operation: str,
        driver_is_active: Callable[[str, str], bool],
    ) -> bool:
        if not value:
            return False
        if key == "core.alternaterefscommand":
            return True
        if remote is not None and key in {"core.askpass", "core.sshcommand"}:
            return True
        if operation in _DIFF_OPERATIONS and key == "diff.external":
            return True
        driver_checks = (
            (_CONTENT_FILTER_OPERATIONS, "filter", _FILTER_DRIVER_SUFFIXES),
            (_DIFF_OPERATIONS, "diff", _DIFF_DRIVER_SUFFIXES),
            (_MERGE_OPERATIONS, "merge", _MERGE_DRIVER_SUFFIXES),
        )
        if any(
            operation in operations
            and _configured_driver_is_active(
                key,
                kind=kind,
                suffixes=suffixes,
                driver_is_active=driver_is_active,
            )
            for operations, kind, suffixes in driver_checks
        ):
            return True
        return bool(
            remote is not None
            and key.startswith("remote.")
            and key.rsplit(".", 1)[-1] in {"uploadpack", "receivepack", "vcs"}
        )

    def _validate_remote_config_entry(
        self,
        *,
        scope: str,
        key: str,
        value: str,
        remote: str | None,
    ) -> None:
        if remote is None:
            return
        remote_key = f"remote.{remote.casefold()}."
        if scope in {"local", "worktree"}:
            suffix = key.rsplit(".", 1)[-1]
            repository_http_override = (
                key.startswith("http.")
                and (
                    suffix in _REPOSITORY_HTTP_SECURITY_SUFFIXES
                    or suffix.startswith(("proxyssl", "schannel", "ssl"))
                )
            )
            remote_proxy_override = (
                key.startswith(remote_key)
                and suffix in {"proxy", "proxyauthmethod"}
            )
            if repository_http_override or remote_proxy_override:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "repository config cannot override Git HTTP transport security",
                )
        if key == f"{remote_key}fetch" and not _is_safe_fetch_refspec(value, remote):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "configured Git fetch refspec escapes the selected remote-tracking namespace",
            )
        mirror_enabled = (
            key == f"{remote_key}mirror"
            and value.casefold() not in {"", "false", "no", "off", "0"}
        )
        if (key == "push.pushoption" and bool(value)) or mirror_enabled:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "repository config expands the typed Git remote effect",
            )

    def _credential_helper_identity(
        self,
        *,
        scope: str,
        origin_path: Path | None,
        key: str,
        value: str,
        remote: str | None,
        allowed_local_origins: set[Path],
    ) -> tuple[str, str] | None:
        if not value or not _is_credential_helper_key(key) or remote is None:
            return None
        if scope not in {"system", "global"} and (
            origin_path is None or origin_path in allowed_local_origins
        ):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "repository credential helpers are not allowed",
            )
        if not self.config.inherit_credential_helpers:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "Host credential helpers are disabled",
            )
        return self._resolve_helper(value)

    def _trusted_local_config_origins(
        self,
        layout: GitRepositoryLayout,
    ) -> set[Path]:
        local_config_paths = (
            layout.common_dir / "config",
            layout.git_dir / "config.worktree",
        )
        allowed_local_origins = {
            candidate.resolve(strict=False)
            for candidate in local_config_paths
        }
        for candidate in local_config_paths:
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "repository Git config could not be inspected",
                ) from exc
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "repository Git config is not a trusted regular file",
                )
        return allowed_local_origins

    def _repository_config_origin_path(
        self,
        *,
        scope: str,
        origin: str,
        allowed_local_origins: set[Path],
    ) -> Path | None:
        if not origin.startswith("file:"):
            return None
        origin_path = Path(origin[5:]).expanduser().resolve(strict=False)
        if scope in {"local", "worktree"} and origin_path not in allowed_local_origins:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "repository Git config includes are not allowed",
            )
        if (
            _is_within(origin_path, self.workspace_root)
            and origin_path not in allowed_local_origins
        ):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "workspace-controlled Git config includes are not allowed",
            )
        return origin_path

    def _validate_repository_config_entry(
        self,
        *,
        scope: str,
        origin: str,
        key: str,
        value: str,
        remote: str | None,
        operation: str,
        allowed_local_origins: set[Path],
        driver_is_active: Callable[[str, str], bool],
    ) -> tuple[str, str] | None:
        if (
            operation == "merge"
            and value.strip()
            and key.startswith("branch.")
            and key.endswith(".mergeoptions")
        ):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "branch mergeOptions can change typed merge semantics",
            )
        origin_path = self._repository_config_origin_path(
            scope=scope,
            origin=origin,
            allowed_local_origins=allowed_local_origins,
        )
        if self._config_selects_executable_extension(
            key=key,
            value=value,
            remote=remote,
            operation=operation,
            driver_is_active=driver_is_active,
        ):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "repository config contains an executable Git extension",
            )
        self._validate_remote_config_entry(
            scope=scope,
            key=key,
            value=value,
            remote=remote,
        )
        if bool(value) and (
            key == "extensions.partialclone"
            or (
                key.startswith("remote.")
                and key.rsplit(".", 1)[-1] in {"promisor", "partialclonefilter"}
            )
        ):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "partial-clone repositories are unavailable because reads must never lazy-fetch",
            )
        return self._credential_helper_identity(
            scope=scope,
            origin_path=origin_path,
            key=key,
            value=value,
            remote=remote,
            allowed_local_origins=allowed_local_origins,
        )

    def _validate_repository_config(
        self,
        layout: GitRepositoryLayout,
        *,
        remote: str | None,
        operation: str,
        treeish_targets: Sequence[str] = (),
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        entries = self._config_entries(layout)
        normalized: list[str] = []
        helper_identities: list[tuple[str, str]] = []
        allowed_local_origins = self._trusted_local_config_origins(layout)
        active_drivers: dict[str, set[str]] | None = None

        def driver_is_active(kind: str, key: str) -> bool:
            nonlocal active_drivers
            if active_drivers is None:
                active_drivers = self._active_attribute_drivers(
                    layout,
                    entries,
                    treeish_targets=treeish_targets,
                )
            key_without_suffix, separator, _suffix = key.rpartition(".")
            prefix = f"{kind}."
            configured_name = key_without_suffix[len(prefix) :]
            return (
                bool(separator)
                and key_without_suffix.startswith(prefix)
                and bool(configured_name)
                and configured_name in active_drivers[kind]
            )
        for scope, origin, key, value in entries:
            if scope == "command":
                continue
            normalized.append(f"{scope}\0{origin}\0{key}\0{value}")
            helper_identity = self._validate_repository_config_entry(
                scope=scope,
                origin=origin,
                key=key,
                value=value,
                remote=remote,
                operation=operation,
                driver_is_active=driver_is_active,
                allowed_local_origins=allowed_local_origins,
            )
            if helper_identity is not None:
                helper_identities.append(helper_identity)
        digest = hashlib.sha256("\n".join(sorted(normalized)).encode("utf-8")).hexdigest()
        return digest, tuple(sorted(helper_identities))

    def validate_read_only_operation(
        self,
        operation: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, str]:
        """Validate a Host-owned legacy Git read without dispatching it.

        Shell, PTY, and benchmark provenance call this at their final Host
        boundary so their narrow compatibility reads share the typed
        provider's repository identity, executable-config, and no-lazy-fetch
        checks.
        """

        if operation not in _RAW_READ_OPERATIONS:
            raise self._error(
                GitErrorCode.UNSUPPORTED,
                "unsupported legacy Git read operation",
                operation=operation,
            )
        layout = self.repository_layout(worktree=worktree)
        config_sha256, _helpers = self._validate_repository_config(
            layout,
            remote=None,
            operation=operation,
        )
        return {
            "repository_id": layout.repository_id,
            "worktree_id": layout.worktree_id,
            "config_sha256": config_sha256,
        }

    def validate_operation(
        self,
        operation: str,
        *,
        worktree: str | Path | None = None,
        remote: str | None = None,
    ) -> dict[str, str]:
        if not isinstance(operation, str) or not operation or "\x00" in operation:
            raise self._error(
                GitErrorCode.COMMAND_FAILED,
                "invalid Git operation",
            )
        layout = self.repository_layout(worktree=worktree)
        config_sha256, _helpers = self._validate_repository_config(
            layout,
            remote=remote,
            operation=operation,
        )
        return {
            "repository_id": layout.repository_id,
            "worktree_id": layout.worktree_id,
            "config_sha256": config_sha256,
        }

    def _active_attribute_drivers(
        self,
        layout: GitRepositoryLayout,
        entries: Sequence[tuple[str, str, str, str]],
        *,
        treeish_targets: Sequence[str] = (),
    ) -> dict[str, set[str]]:
        """Inspect attribute declarations as data, without invoking Git drivers."""

        files: list[Path] = []
        seen: set[Path] = set()

        def include(candidate: Path) -> None:
            # Keep the lexical entry so the later lstat rejects an attribute
            # symlink instead of silently following it outside the repository.
            selected = Path(os.path.abspath(self._expand_git_user_path(candidate)))
            if selected not in seen:
                seen.add(selected)
                files.append(selected)

        include(layout.common_dir / "info" / "attributes")
        for scope, _origin, key, value in entries:
            if scope != "command" and key == "core.attributesfile" and value:
                selected = self._expand_git_user_path(value)
                if not selected.is_absolute():
                    # Git resolves relative core.attributesFile values from
                    # the command cwd, which is pinned to workspace_root.
                    selected = self.workspace_root / selected
                include(selected)
        for selected in self._working_tree_attribute_paths(layout):
            include(selected)
        drivers = {"filter": set(), "diff": set(), "merge": set()}
        total = 0

        def inspect(raw: bytes) -> None:
            nonlocal total
            total += len(raw)
            if total > 1_048_576:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git attributes exceed their aggregate safety bound",
                )
            for raw_line in raw.splitlines():
                # Git ignores horizontal whitespace around an attributes line.
                # Strip it before separating the pattern from its attributes so
                # an indented driver declaration cannot evade inspection.
                line = raw_line.rstrip(b"\r").strip(b" \t")
                if not line or line.startswith(b"#"):
                    continue
                line_match = _ATTRIBUTE_LINE_RE.fullmatch(line)
                if line_match is None:
                    continue
                attributes = line_match.group("attributes") or b""
                for match in _ATTRIBUTE_DRIVER_RE.finditer(attributes):
                    kind = match.group("kind").decode("ascii")
                    try:
                        name = match.group("name").decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exc:
                        raise self._error(
                            GitErrorCode.UNSAFE_CONFIG,
                            "Git attribute driver name is not UTF-8",
                        ) from exc
                    drivers[kind].add(name.casefold())

        for selected in files:
            try:
                metadata = selected.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes could not be inspected") from exc
            if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes source is not a regular file")
            try:
                raw = self._read_small_file(selected, limit=1_048_576)
            except (OSError, ValueError) as exc:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes exceed their safety bound") from exc
            inspect(raw)
        seen_blobs: set[bytes] = set()
        for raw in self._index_attribute_sources(layout, seen_blobs=seen_blobs):
            inspect(raw)
        for raw in self._tree_attribute_sources(
            layout,
            treeish_targets,
            seen_blobs=seen_blobs,
        ):
            inspect(raw)
        return drivers

    def _working_tree_attribute_paths(
        self,
        layout: GitRepositoryLayout,
    ) -> Iterator[Path]:
        """List only tracked or non-ignored worktree attribute sources.

        Walking the whole checkout lets a large ignored virtual environment or
        dependency cache exhaust the attribute-discovery bound even though Git
        will never consult it. ``ls-files`` is run with hooks, fsmonitor, lazy
        fetches, and executable diff extensions disabled by ``_repo_prefix``;
        it does not invoke content filters.
        """

        result = self._invoke(
            [
                *self._repo_prefix(layout, literal_pathspecs=False),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
                "--",
                ":(top,glob)**/.gitattributes",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="attribute_worktree_inspection",
        )
        if result.returncode != 0:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "working-tree Git attributes could not be enumerated",
            )
        paths = [raw for raw in result.stdout.split(b"\0") if raw]
        if len(paths) > 100_000:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "attribute discovery exceeded its safety bound",
            )
        for raw in paths:
            relative = Path(os.fsdecode(raw))
            if relative.is_absolute() or ".." in relative.parts:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "working-tree Git attributes escaped the repository",
                )
            selected = Path(os.path.abspath(layout.root / relative))
            if not _is_within(selected, layout.root):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "working-tree Git attributes escaped the repository",
                )
            yield selected

    def _index_attribute_sources(
        self,
        layout: GitRepositoryLayout,
        *,
        seen_blobs: set[bytes],
    ) -> Iterator[bytes]:
        result = self._invoke(
            [
                *self._repo_prefix(layout, literal_pathspecs=False),
                "ls-files",
                "--stage",
                "-z",
                "--",
                ":(top,glob)**/.gitattributes",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=min(
                self.config.output_hard_limit_bytes,
                1_048_576,
            ),
            read_only=True,
            operation="attribute_inspection",
        )
        if result.returncode != 0:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "Git index attributes could not be inspected",
            )
        for record in result.stdout.split(b"\0"):
            if not record:
                continue
            header, separator, path = record.partition(b"\t")
            fields = header.split(b" ")
            if (
                not separator
                or len(fields) != 3
                or fields[2] not in {b"0", b"1", b"2", b"3"}
                or not (
                    path == b".gitattributes"
                    or path.endswith(b"/.gitattributes")
                )
            ):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git index attributes have an unsupported format",
                )
            mode, oid, _stage = fields
            # Git does not follow a symlinked attributes file. Intent-to-add
            # entries have a null object id and are represented by the
            # worktree file, which was inspected above.
            if mode == b"120000" or not oid.strip(b"0"):
                continue
            if not re.fullmatch(rb"[0-9a-f]+", oid):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git index attributes have an unsupported entry",
                )
            yield from self._attribute_blob_source(
                layout,
                oid,
                seen_blobs=seen_blobs,
            )

    def _tree_attribute_sources(
        self,
        layout: GitRepositoryLayout,
        treeish_targets: Sequence[str],
        *,
        seen_blobs: set[bytes],
    ) -> Iterator[bytes]:
        for target in dict.fromkeys(treeish_targets):
            # ``ls-tree`` does not support glob-magic pathspecs. Grepping for
            # the empty pattern lists only non-empty attributes blobs, which
            # are the only blobs capable of declaring a driver, without
            # enumerating every ordinary file in a large target tree.
            matched = self._invoke(
                [
                    *self._repo_prefix(layout, literal_pathspecs=False),
                    "grep",
                    "-z",
                    "-l",
                    "--full-name",
                    "-e",
                    "",
                    target,
                    "--",
                    ":(top,glob)**/.gitattributes",
                ],
                timeout=self.config.local_timeout_s,
                stdin=None,
                max_output_bytes=min(
                    self.config.output_hard_limit_bytes,
                    1_048_576,
                ),
                read_only=True,
                operation="attribute_inspection",
            )
            if matched.returncode == 1:
                continue
            if matched.returncode != 0:
                raise self._error(
                    GitErrorCode.STALE_STATE,
                    "Git attribute target could not be inspected",
                    operation="attribute_inspection",
                    retryable=True,
                )
            target_prefix = os.fsencode(target) + b":"
            paths: list[bytes] = []
            for record in matched.stdout.split(b"\0"):
                if not record:
                    continue
                if not record.startswith(target_prefix):
                    raise self._error(
                        GitErrorCode.UNSAFE_CONFIG,
                        "Git attribute tree has an unsupported format",
                    )
                path = record[len(target_prefix) :]
                if not (
                    path == b".gitattributes"
                    or path.endswith(b"/.gitattributes")
                ):
                    raise self._error(
                        GitErrorCode.UNSAFE_CONFIG,
                        "Git attribute tree has an unsupported format",
                    )
                paths.append(path)
                if len(paths) > 10_000:
                    raise self._error(
                        GitErrorCode.UNSAFE_CONFIG,
                        "Git attribute discovery exceeded its safety bound",
                    )
            for offset in range(0, len(paths), 128):
                selected = paths[offset : offset + 128]
                result = self._invoke(
                    [
                        *self._repo_prefix(layout),
                        "ls-tree",
                        "-z",
                        "--full-tree",
                        target,
                        "--",
                        *(os.fsdecode(path) for path in selected),
                    ],
                    timeout=self.config.local_timeout_s,
                    stdin=None,
                    max_output_bytes=min(
                        self.config.output_hard_limit_bytes,
                        1_048_576,
                    ),
                    read_only=True,
                    operation="attribute_inspection",
                )
                if result.returncode != 0:
                    raise self._error(
                        GitErrorCode.STALE_STATE,
                        "Git attribute target could not be inspected",
                        operation="attribute_inspection",
                        retryable=True,
                    )
                remaining = set(selected)
                for record in result.stdout.split(b"\0"):
                    if not record:
                        continue
                    header, separator, path = record.partition(b"\t")
                    fields = header.split(b" ")
                    if (
                        not separator
                        or len(fields) != 3
                        or path not in remaining
                    ):
                        raise self._error(
                            GitErrorCode.UNSAFE_CONFIG,
                            "Git attribute tree has an unsupported format",
                        )
                    remaining.remove(path)
                    mode, object_type, oid = fields
                    # Git deliberately does not follow a symlinked attributes
                    # file in a worktree, so its blob content is not active.
                    if mode == b"120000":
                        continue
                    if object_type != b"blob" or not re.fullmatch(
                        rb"[0-9a-f]+", oid
                    ):
                        raise self._error(
                            GitErrorCode.UNSAFE_CONFIG,
                            "Git attribute tree has an unsupported entry",
                        )
                    yield from self._attribute_blob_source(
                        layout,
                        oid,
                        seen_blobs=seen_blobs,
                    )
                if remaining:
                    raise self._error(
                        GitErrorCode.STALE_STATE,
                        "Git attribute target changed during inspection",
                        operation="attribute_inspection",
                        retryable=True,
                    )

    def _attribute_blob_source(
        self,
        layout: GitRepositoryLayout,
        oid: bytes,
        *,
        seen_blobs: set[bytes],
    ) -> Iterator[bytes]:
        if oid in seen_blobs:
            return
        seen_blobs.add(oid)
        if len(seen_blobs) > 10_000:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "Git attribute discovery exceeded its safety bound",
            )
        blob = self._invoke(
            [*self._repo_prefix(layout), "cat-file", "blob", os.fsdecode(oid)],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=min(
                self.config.output_hard_limit_bytes,
                1_048_576,
            ),
            read_only=True,
            operation="attribute_inspection",
        )
        if blob.returncode != 0:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "Git attribute blob could not be inspected",
            )
        yield blob.stdout

    def _rebase_attribute_targets(
        self,
        layout: GitRepositoryLayout,
        upstream: str,
    ) -> tuple[str, ...]:
        result = self._invoke(
            [
                *self._repo_prefix(layout),
                "rev-list",
                "--reverse",
                f"{upstream}..HEAD",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=min(
                self.config.output_hard_limit_bytes,
                1_048_576,
            ),
            read_only=True,
            operation="attribute_inspection",
        )
        if result.returncode != 0:
            raise self._error(
                GitErrorCode.STALE_STATE,
                "Git rebase attribute targets could not be inspected",
                operation="attribute_inspection",
                retryable=True,
            )
        replayed: list[str] = []
        for raw_oid in result.stdout.splitlines():
            if not re.fullmatch(rb"[0-9a-f]+", raw_oid):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git rebase attribute targets have an unsupported format",
                )
            replayed.append(raw_oid.decode("ascii"))
            if len(replayed) > 10_000:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git rebase attribute discovery exceeded its safety bound",
                )
        return tuple(dict.fromkeys((upstream, *replayed)))

    def _attribute_treeish_targets(
        self,
        layout: GitRepositoryLayout,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        checked = list(args)
        if not checked:
            return ()
        operation = checked[0]
        boundary = checked.index("--") if "--" in checked else len(checked)
        prefix = checked[1:boundary]
        if operation in {"checkout", "switch"}:
            for option in ("--detach", "-b", "-B", "-c", "-C"):
                if option in prefix:
                    index = prefix.index(option)
                    offset = 1 if option == "--detach" else 2
                    return tuple(prefix[index + offset : index + offset + 1])
            positional = [token for token in prefix if not token.startswith("-")]
            return tuple(positional[-1:])
        if operation == "rebase":
            if prefix == ["--abort"]:
                return ("ORIG_HEAD",)
            if any(
                option in prefix
                for option in ("--continue", "--skip", "--quit")
            ):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "incremental Git rebase modes are not supported",
                )
            positional = [token for token in prefix if not token.startswith("-")]
            if len(positional) != 1:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "Git rebase target form is not supported",
                )
            return self._rebase_attribute_targets(layout, positional[0])
        if operation in {"merge", "cherry-pick", "revert", "reset"}:
            positional = [token for token in prefix if not token.startswith("-")]
            return tuple(positional[-1:])
        if operation == "worktree" and prefix[:1] == ["add"]:
            positional = [token for token in prefix[1:] if not token.startswith("-")]
            return tuple(positional[-1:])
        return ()

    def _remote_urls(self, layout: GitRepositoryLayout, remote: str) -> tuple[str, str]:
        if not _REMOTE_NAME_RE.fullmatch(remote) or remote.startswith("-"):
            raise self._error(GitErrorCode.INVALID_REF, "invalid Git remote name")
        fetch = self._invoke(
            [*self._repo_prefix(layout), "remote", "get-url", "--all", "--", remote],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=65536,
            read_only=True,
            operation="remote_inspection",
        )
        push = self._invoke(
            [
                *self._repo_prefix(layout),
                "remote",
                "get-url",
                "--push",
                "--all",
                "--",
                remote,
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=65536,
            read_only=True,
            operation="remote_inspection",
        )
        if fetch.returncode != 0 or push.returncode != 0:
            raise self._error(GitErrorCode.NOT_FOUND, "configured Git remote was not found")
        fetch_urls = fetch.stdout.rstrip(b"\r\n").splitlines()
        push_urls = push.stdout.rstrip(b"\r\n").splitlines()
        if len(fetch_urls) != 1 or len(push_urls) != 1:
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "Git remotes must resolve to exactly one fetch URL and one push URL",
            )
        try:
            fetch_url = fetch_urls[0].decode("utf-8", errors="strict")
            push_url = push_urls[0].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git remote URL is not UTF-8") from exc
        self._validate_remote_url(fetch_url)
        self._validate_remote_url(push_url)
        return fetch_url, push_url

    def _remote_fetch_refspecs(
        self,
        layout: GitRepositoryLayout,
        remote: str,
    ) -> tuple[str, ...]:
        key = f"remote.{remote.casefold()}.fetch"
        refspecs = tuple(
            value
            for scope, _origin, entry_key, value in self._config_entries(layout)
            if scope != "command" and entry_key == key
        )
        if any(not _is_safe_fetch_refspec(value, remote) for value in refspecs):
            raise self._error(
                GitErrorCode.UNSAFE_CONFIG,
                "configured Git fetch refspec escapes the selected remote-tracking namespace",
            )
        return refspecs

    def _validate_remote_url(self, url: str) -> None:
        if not url or any(char in url for char in "\x00\r\n") or url.startswith("-"):
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "invalid Git remote URL")
        if "::" in url and "://" not in url:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git remote helper syntax is not allowed")
        scp = _SCP_REMOTE_RE.fullmatch(url)
        if scp and "://" not in url:
            if not self.config.allow_scp_style_ssh:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "scp-style SSH remotes are disabled")
            if scp.group("user") not in {None, "git"}:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "SSH remote user is not allowed")
            return
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        if parsed.query or parsed.fragment:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git remote URL query and fragment data are not allowed")
        if scheme == "file":
            if not self.config.allow_file_remotes:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "file Git remotes are disabled")
            if parsed.username is not None or parsed.password is not None or parsed.hostname not in {None, "", "localhost"}:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "file Git remotes must be local and contain no user information")
            return
        if scheme not in {
            allowed.casefold() for allowed in self.config.allowed_remote_schemes
        }:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git remote protocol is not allowed")
        if not parsed.hostname or parsed.password is not None:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git remote URL contains invalid user information")
        if scheme == "https" and parsed.username is not None:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "HTTPS remote URL user information is not allowed")
        if scheme == "ssh" and parsed.username not in {None, "git"}:
            raise self._error(GitErrorCode.UNSAFE_CONFIG, "SSH remote user is not allowed")

    def remote_fingerprint(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, Any]:
        layout = self.repository_layout(worktree=worktree)
        fetch_url, push_url = self._remote_urls(layout, remote)
        config_sha256, helper_identities = self._validate_repository_config(
            layout,
            remote=remote,
            operation="remote_inspection",
        )
        _remote_env, ssh_identities = self._remote_dispatch_environment(fetch_url, push_url)
        refs = self._invoke(
            [
                *self._repo_prefix(layout),
                "for-each-ref",
                "--format=%(refname)%00%(objectname)",
                f"refs/remotes/{remote}/",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="remote_inspection",
        )
        if refs.returncode != 0:
            raise self._error(GitErrorCode.COMMAND_FAILED, "remote-tracking refs could not be inspected")
        return {
            "remote": remote,
            "fetch_url_sha256": hashlib.sha256(fetch_url.encode("utf-8")).hexdigest(),
            "push_url_sha256": hashlib.sha256(push_url.encode("utf-8")).hexdigest(),
            "config_sha256": config_sha256,
            "helper_identities": helper_identities,
            "ssh_identities": ssh_identities,
            "refs_sha256": hashlib.sha256(refs.stdout).hexdigest(),
            "fingerprint": _digest_fields(
                remote,
                hashlib.sha256(fetch_url.encode("utf-8")).hexdigest(),
                hashlib.sha256(push_url.encode("utf-8")).hexdigest(),
                config_sha256,
                hashlib.sha256(refs.stdout).hexdigest(),
                repr(helper_identities),
                repr(ssh_identities),
            ),
        }

    def preflight_remote_fingerprint(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> dict[str, Any]:
        """Inspect only local security inputs needed to bind remote approval.

        The primitive gates Task Authority and the remote capability before
        this call.  Network dispatch still uses ``remote_fingerprint`` inside
        the protected provider phase and compares the two digests.
        """

        return self.remote_fingerprint(remote, worktree=worktree)

    def remote_configuration(
        self,
        remote: str,
        *,
        worktree: str | Path | None = None,
    ) -> tuple[str, str, dict[str, Any]]:
        layout = self.repository_layout(worktree=worktree)
        config_sha256, _helpers = self._validate_repository_config(
            layout,
            remote=remote,
            operation="list_remotes",
        )
        fetch_url, push_url = self._remote_urls(layout, remote)
        fetch_refspecs = self._remote_fetch_refspecs(layout, remote)
        after = self.repository_layout(worktree=layout.root)
        if not self._same_layout(layout, after):
            raise self._error(GitErrorCode.STALE_STATE, "Git repository identity changed during remote inspection")
        return fetch_url, push_url, {
            "config_sha256": config_sha256,
            "fetch_url_sha256": hashlib.sha256(fetch_url.encode("utf-8")).hexdigest(),
            "push_url_sha256": hashlib.sha256(push_url.encode("utf-8")).hexdigest(),
            "fetch_refspecs": fetch_refspecs,
        }

    def prepare_managed_worktree(self, worktree_id: str) -> Path:
        if not isinstance(worktree_id, str) or not _MANAGED_WORKTREE_ID_RE.fullmatch(worktree_id):
            raise self._error(GitErrorCode.INVALID_PATH, "invalid managed worktree id")
        layout = self.repository_layout()
        current = self.workspace_root
        try:
            relative_root = self.managed_worktree_root.relative_to(self.workspace_root)
        except ValueError as exc:
            raise self._error(GitErrorCode.INVALID_PATH, "managed worktree root escaped workspace") from exc
        self._ensure_managed_worktree_excluded(layout, relative_root)
        for part in relative_root.parts:
            current = current / part
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "managed worktree root could not be created") from exc
            try:
                metadata = current.lstat()
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "managed worktree root could not be inspected") from exc
            if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "managed worktree root is not a trusted directory")
        target = self.managed_worktree_root / worktree_id
        try:
            target_state = target.lstat()
        except FileNotFoundError:
            return target
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "managed worktree target could not be inspected",
            ) from exc
        if _is_link_or_reparse(target_state) or not (
            stat.S_ISREG(target_state.st_mode)
            or stat.S_ISDIR(target_state.st_mode)
        ):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "managed worktree target is an unsafe existing filesystem entry",
            )
        raise self._error(
            GitErrorCode.ALREADY_EXISTS,
            "managed worktree target already exists",
        )

    @staticmethod
    def _validate_repository_lock_snapshot(
        snapshot: StablePathSnapshot,
    ) -> StablePathSnapshot:
        if (
            snapshot.is_reparse_point
            or not stat.S_ISREG(snapshot.mode)
            or snapshot.links != 1
            or not stable_identity_available(snapshot)
        ):
            raise OSError("repository lock is not a stable regular file")
        return snapshot

    @staticmethod
    def _require_same_repository_lock_identity(
        opened: StablePathSnapshot,
        linked: StablePathSnapshot,
    ) -> None:
        opened_available = stable_identity_available(opened)
        linked_available = stable_identity_available(linked)
        if not opened_available or not linked_available:
            raise OSError("repository lock identity is unavailable")
        if opened.inode > 0 and linked.inode > 0:
            if (opened.device, opened.inode) != (linked.device, linked.inode):
                raise OSError("repository lock path does not match opened file")
            return
        # Win32 can report a zero legacy file index.  This branch is safe only
        # because both snapshots come from the same held no-delete target
        # handle; ``linked_snapshot`` deliberately reuses that handle.
        if not (opened.replacement_locked and linked.replacement_locked):
            raise OSError("repository lock identity requires a held replacement lock")

    def _validate_open_repository_lock_file(
        self,
        secure_file: SecureFileDescriptor,
    ) -> None:
        opened = self._validate_repository_lock_snapshot(secure_file.snapshot())
        linked = self._validate_repository_lock_snapshot(
            secure_file.linked_snapshot()
        )
        self._require_same_repository_lock_identity(opened, linked)

    @contextmanager
    def _open_repository_lock_file(
        self,
        common_dir: Path,
    ) -> Iterator[SecureFileDescriptor]:
        """Open the lock below held directory guards without path traversal."""

        resources = contextlib.ExitStack()
        try:
            common_guard = resources.enter_context(
                open_secure_directory(
                    common_dir,
                    allow_child_mutation=True,
                )
            )
            lock_directory = common_dir / "agent-libos"
            try:
                if os.name == "nt":
                    lock_directory.mkdir(mode=0o700)
                else:
                    if common_guard.descriptor is None:
                        raise OSError("secure common directory descriptor is unavailable")
                    os.mkdir(
                        "agent-libos",
                        mode=0o700,
                        dir_fd=common_guard.descriptor,
                    )
            except FileExistsError:
                pass
            common_guard.verify_path_guard()
            lock_directory_guard = resources.enter_context(
                common_guard.open_child_directory("agent-libos")
            )
            lock_path = lock_directory / "repository.lock"
            secure_file = open_secure_readwrite_child(
                lock_path,
                parent=lock_directory_guard,
                relative_name=lock_path.name,
            )
            resources.callback(secure_file.close)
            self._validate_open_repository_lock_file(secure_file)
            lock_directory_guard.linked_snapshot()
            common_guard.linked_snapshot()
        except OSError as exc:
            resources.close()
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "repository lock could not be opened",
            ) from exc
        try:
            yield secure_file
        finally:
            resources.close()

    def _ensure_managed_worktree_excluded(
        self,
        layout: GitRepositoryLayout,
        relative_root: Path,
    ) -> None:
        """Keep Runtime-owned worktrees invisible to the primary worktree status.

        The ignore is repository-local metadata, written only through this
        provider.  It is deliberately not placed in the tracked ``.gitignore``
        because creating a managed worktree must not edit user content.
        """

        if not relative_root.parts or any(part in {"", ".", ".."} for part in relative_root.parts):
            raise self._error(GitErrorCode.INVALID_PATH, "managed worktree ignore path is invalid")
        info = layout.common_dir / "info"
        try:
            info_state = info.lstat()
        except FileNotFoundError:
            try:
                info.mkdir(mode=0o700)
                info_state = info.lstat()
            except OSError as exc:
                raise self._error(
                    GitErrorCode.UNSAFE_REPOSITORY,
                    "Git info directory could not be created",
                ) from exc
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "Git info directory could not be inspected",
            ) from exc
        if _is_link_or_reparse(info_state) or not stat.S_ISDIR(info_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git info directory is not trusted")

        exclude = info / "exclude"
        try:
            exclude_state = exclude.lstat()
        except FileNotFoundError:
            current = b""
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git exclude file could not be inspected") from exc
        else:
            if _is_link_or_reparse(exclude_state) or not stat.S_ISREG(exclude_state.st_mode):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git exclude file is not trusted")
            try:
                current = self._read_small_file(exclude, limit=1_048_576)
            except (OSError, ValueError) as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git exclude file exceeds its safety bound") from exc

        pattern = f"/{relative_root.as_posix().rstrip('/')}/".encode("utf-8")
        existing = {line.strip() for line in current.splitlines()}
        if pattern in existing:
            return
        updated = current
        if updated and not updated.endswith(b"\n"):
            updated += b"\n"
        updated += pattern + b"\n"
        temporary = info / f".agent-libos-exclude-{os.getpid()}-{time.time_ns()}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            remaining = memoryview(updated)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("Git exclude write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, exclude)
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git exclude file could not be updated") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def path_content_sha256(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str | None:
        if not isinstance(path, bytes) or not path or b"\x00" in path:
            raise self._error(GitErrorCode.INVALID_PATH, "invalid repository path")
        decoded = os.fsdecode(path)
        lexical = Path(decoded)
        if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
            raise self._error(GitErrorCode.INVALID_PATH, "invalid repository path")
        layout = self.repository_layout(worktree=worktree)
        selected = layout.root.joinpath(*lexical.parts)
        try:
            metadata = selected.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "repository path could not be inspected") from exc
        if stat.S_ISREG(metadata.st_mode):
            digest, _consumed = self._hash_regular_file(
                selected,
                remaining=self.config.state_content_hard_limit_bytes,
            )
            return digest
        if stat.S_ISLNK(metadata.st_mode):
            try:
                return hashlib.sha256(os.fsencode(os.readlink(selected))).hexdigest()
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "repository symlink could not be inspected") from exc
        if stat.S_ISDIR(metadata.st_mode):
            return hashlib.sha256(b"<agent-libos-directory>").hexdigest()
        raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unsupported repository path type")

    def path_kind(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str:
        if not isinstance(path, bytes) or not path or b"\x00" in path:
            raise self._error(GitErrorCode.INVALID_PATH, "invalid repository path")
        decoded = os.fsdecode(path)
        lexical = Path(decoded)
        if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
            raise self._error(GitErrorCode.INVALID_PATH, "invalid repository path")
        layout = self.repository_layout(worktree=worktree)
        selected = layout.root.joinpath(*lexical.parts)
        try:
            metadata = selected.lstat()
        except FileNotFoundError:
            return "missing"
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "repository path could not be inspected",
            ) from exc
        if stat.S_ISREG(metadata.st_mode):
            return "file"
        if stat.S_ISLNK(metadata.st_mode):
            return "symlink"
        if stat.S_ISDIR(metadata.st_mode):
            return "directory"
        raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unsupported repository path type")

    def preflight_path_kind(
        self,
        path: bytes,
        *,
        worktree: str | Path | None = None,
    ) -> str:
        """Inspect only path type for capability-scope selection.

        The primitive repeats this observation inside the protected mutation
        phase before dispatch, so a file-to-directory race is fail-closed.
        """

        return self.path_kind(path, worktree=worktree)

    def _pull_request_directory(self, layout: GitRepositoryLayout, *, create: bool) -> Path:
        base = layout.common_dir / "agent-libos"
        selected = base / "pull_requests"
        if create:
            for directory in (base, selected):
                try:
                    directory.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory could not be created") from exc
                try:
                    metadata = directory.lstat()
                except OSError as exc:
                    raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory could not be inspected") from exc
                if _is_link_or_reparse(metadata) or not stat.S_ISDIR(metadata.st_mode):
                    raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory is not trusted")
        return selected

    def _pull_request_path(self, layout: GitRepositoryLayout, pr_id: str, *, create_directory: bool) -> Path:
        if not isinstance(pr_id, str) or not _PULL_REQUEST_ID_RE.fullmatch(pr_id):
            raise self._error(GitErrorCode.INVALID_REF, "invalid pull request id")
        return self._pull_request_directory(layout, create=create_directory) / f"{pr_id}.json"

    def read_pull_request_metadata(self, pr_id: str) -> tuple[bytes, str] | None:
        layout = self.repository_layout()
        selected = self._pull_request_path(layout, pr_id, create_directory=False)
        try:
            metadata = selected.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be inspected") from exc
        if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata is not a trusted regular file")
        try:
            data = self._read_small_file(selected, limit=self.config.output_hard_limit_bytes)
        except (OSError, ValueError) as exc:
            raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata exceeds its hard limit") from exc
        return data, hashlib.sha256(data).hexdigest()

    def _bounded_pull_request_metadata_paths(
        self,
        directory: Path,
    ) -> tuple[Path, ...]:
        paths: list[Path] = []
        try:
            for selected in directory.iterdir():
                if len(paths) >= self.config.status_entry_hard_limit:
                    raise self._error(
                        GitErrorCode.OUTPUT_TOO_LARGE,
                        "pull request metadata count exceeds its hard limit",
                    )
                if (
                    selected.suffix != ".json"
                    or not _PULL_REQUEST_ID_RE.fullmatch(selected.stem)
                ):
                    raise self._error(
                        GitErrorCode.UNSAFE_REPOSITORY,
                        "unexpected pull request metadata entry",
                    )
                paths.append(selected)
        except GitError:
            raise
        except OSError as exc:
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "pull request metadata could not be listed",
            ) from exc
        paths.sort(key=lambda path: path.name)
        return tuple(paths)

    def list_pull_request_metadata(self, *, limit: int) -> tuple[tuple[str, bytes, str], ...]:
        if isinstance(limit, bool) or limit <= 0 or limit > self.config.status_entry_hard_limit:
            raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request list limit is invalid")
        layout = self.repository_layout()
        directory = self._pull_request_directory(layout, create=False)
        try:
            directory_state = directory.lstat()
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be listed") from exc
        if _is_link_or_reparse(directory_state) or not stat.S_ISDIR(directory_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory is not trusted")
        rows: list[tuple[str, bytes, str]] = []
        names = self._bounded_pull_request_metadata_paths(directory)[:limit]
        total = 0
        for selected in names:
            item = self.read_pull_request_metadata(selected.stem)
            if item is None:
                continue
            data, digest = item
            total += len(data)
            if total > self.config.output_hard_limit_bytes:
                raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata list exceeds its hard limit")
            rows.append((selected.stem, data, digest))
        return tuple(rows)

    def write_pull_request_metadata(
        self,
        pr_id: str,
        data: bytes,
        *,
        expected_sha256: str | None,
        create: bool = False,
    ) -> str:
        if not isinstance(data, bytes) or not data or len(data) > self.config.output_hard_limit_bytes:
            raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata exceeds its hard limit")
        if expected_sha256 is not None and not _SHA256_RE.fullmatch(expected_sha256):
            raise self._error(GitErrorCode.STALE_STATE, "pull request metadata CAS digest is invalid")
        layout = self.repository_layout()
        selected = self._pull_request_path(layout, pr_id, create_directory=True)
        current = self.read_pull_request_metadata(pr_id)
        if create:
            if current is not None:
                raise self._error(GitErrorCode.ALREADY_EXISTS, "pull request already exists")
            if expected_sha256 is not None:
                raise self._error(GitErrorCode.STALE_STATE, "new pull request metadata must not have an old digest")
        else:
            if current is None:
                raise self._error(GitErrorCode.NOT_FOUND, "pull request was not found")
            if expected_sha256 is None or current[1] != expected_sha256:
                raise self._error(GitErrorCode.STALE_STATE, "pull request metadata changed before update", retryable=True)
        temporary = selected.parent / f".{pr_id}.{time.time_ns()}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary, flags, 0o600)
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("metadata write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            if create and selected.exists():
                raise self._error(GitErrorCode.ALREADY_EXISTS, "pull request already exists")
            os.replace(temporary, selected)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(selected.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except GitError:
            raise
        except OSError as exc:
            raise self._error(GitErrorCode.UNKNOWN_EFFECT, "pull request metadata write outcome is unknown") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return hashlib.sha256(data).hexdigest()

    def _pull_request_metadata_digest(self, layout: GitRepositoryLayout) -> str:
        directory = self._pull_request_directory(layout, create=False)
        try:
            directory_state = directory.lstat()
        except FileNotFoundError:
            return hashlib.sha256(b"no-pull-requests").hexdigest()
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be inspected") from exc
        if _is_link_or_reparse(directory_state) or not stat.S_ISDIR(directory_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory is not trusted")
        digest = hashlib.sha256()
        total = 0
        names = self._bounded_pull_request_metadata_paths(directory)
        for selected in names:
            try:
                metadata = selected.lstat()
                if _is_link_or_reparse(metadata) or not stat.S_ISREG(metadata.st_mode):
                    raise OSError("unsafe pull request metadata")
                data = self._read_small_file(selected, limit=self.config.output_hard_limit_bytes)
            except (OSError, ValueError) as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be inspected") from exc
            total += len(data)
            if total > self.config.output_hard_limit_bytes:
                raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata exceeds its aggregate limit")
            digest.update(selected.name.encode("ascii"))
            digest.update(hashlib.sha256(data).digest())
        return digest.hexdigest()

    @staticmethod
    def _status_paths(raw: bytes) -> tuple[bytes, ...]:
        records = raw.split(b"\0")
        selected: list[bytes] = []
        offset = 0
        while offset < len(records):
            record = records[offset]
            offset += 1
            if not record or record.startswith(b"# ") or record.startswith(b"! "):
                continue
            prefix = record[:2]
            if prefix == b"1 ":
                parts = record.split(b" ", 8)
                if len(parts) == 9:
                    selected.append(parts[8])
            elif prefix == b"2 ":
                parts = record.split(b" ", 9)
                if len(parts) == 10:
                    selected.append(parts[9])
                    if offset < len(records):
                        selected.append(records[offset])
                        offset += 1
            elif prefix == b"u ":
                parts = record.split(b" ", 10)
                if len(parts) == 11:
                    selected.append(parts[10])
            elif prefix == b"? ":
                selected.append(record[2:])
        return tuple(dict.fromkeys(selected))

    def _hash_regular_file(
        self,
        selected: Path,
        *,
        remaining: int,
    ) -> tuple[str, int]:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(selected, flags)
        except FileNotFoundError:
            return hashlib.sha256(b"missing").hexdigest(), 0
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "repository file could not be read safely") from exc
        digest = hashlib.sha256()
        consumed = 0
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "repository file changed type during state capture")
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, remaining - consumed + 1))
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > remaining:
                    raise self._error(
                        GitErrorCode.OUTPUT_TOO_LARGE,
                        "repository state content exceeds the configured hard limit",
                    )
                digest.update(chunk)
        finally:
            os.close(descriptor)
        return digest.hexdigest(), consumed

    def _index_digest(self, layout: GitRepositoryLayout) -> str:
        selected = layout.git_dir / "index"
        try:
            state = selected.lstat()
        except FileNotFoundError:
            return hashlib.sha256(b"missing-index").hexdigest()
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git index could not be inspected") from exc
        if _is_link_or_reparse(state) or not stat.S_ISREG(state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git index is not a trusted regular file")
        digest, _consumed = self._hash_regular_file(
            selected,
            remaining=self.config.state_content_hard_limit_bytes,
        )
        return digest

    def _worktree_digest(self, layout: GitRepositoryLayout, status: bytes) -> str:
        digest = hashlib.sha256()
        total = 0
        status_paths = self._status_paths(status)
        if len(status_paths) > self.config.status_entry_hard_limit:
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "repository status exceeds the configured entry hard limit",
            )
        for raw_path in status_paths:
            if not raw_path or b"\x00" in raw_path:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid path in Git status")
            decoded = os.fsdecode(raw_path)
            lexical = Path(decoded)
            if lexical.is_absolute() or any(part in {"", ".", ".."} for part in lexical.parts):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unsafe path in Git status")
            selected = layout.root.joinpath(*lexical.parts)
            digest.update(len(raw_path).to_bytes(8, "big"))
            digest.update(raw_path)
            try:
                metadata = selected.lstat()
            except FileNotFoundError:
                digest.update(b"missing")
                continue
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "worktree state could not be inspected") from exc
            digest.update(str(stat.S_IFMT(metadata.st_mode)).encode("ascii"))
            digest.update(str(stat.S_IMODE(metadata.st_mode)).encode("ascii"))
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.fsencode(os.readlink(selected))
                except OSError as exc:
                    raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "worktree symlink could not be inspected") from exc
                digest.update(hashlib.sha256(target).digest())
                total += len(target)
            elif stat.S_ISREG(metadata.st_mode):
                content_sha256, consumed = self._hash_regular_file(
                    selected,
                    remaining=self.config.state_content_hard_limit_bytes - total,
                )
                total += consumed
                digest.update(bytes.fromhex(content_sha256))
            elif stat.S_ISDIR(metadata.st_mode):
                digest.update(b"directory")
            else:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unsupported worktree file type")
            if total > self.config.state_content_hard_limit_bytes:
                raise self._error(
                    GitErrorCode.OUTPUT_TOO_LARGE,
                    "repository state content exceeds the configured hard limit",
                )
        return digest.hexdigest()

    def repository_state(
        self,
        *,
        worktree: str | Path | None = None,
    ) -> GitRepositoryState:
        layout = self.repository_layout(worktree=worktree)
        config_sha256, _helpers = self._validate_repository_config(
            layout,
            remote=None,
            operation="status",
        )
        status = self._invoke(
            [
                *self._repo_prefix(layout),
                "status",
                "--porcelain=v2",
                "-z",
                "--branch",
                "--untracked-files=all",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="status",
        )
        if status.returncode != 0:
            raise self._error(GitErrorCode.COMMAND_FAILED, "Git status could not be read", operation="status")
        head_ref_result = self._invoke(
            [*self._repo_prefix(layout), "symbolic-ref", "-q", "HEAD"],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=65536,
            read_only=True,
            operation="status",
        )
        head_oid_result = self._invoke(
            [*self._repo_prefix(layout), "rev-parse", "--verify", "HEAD"],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=65536,
            read_only=True,
            operation="status",
        )
        refs_result = self._invoke(
            [
                *self._repo_prefix(layout),
                "for-each-ref",
                "--format=%(refname)%00%(objectname)%00%(symref)",
            ],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="status",
        )
        worktrees_result = self._invoke(
            [*self._repo_prefix(layout), "worktree", "list", "--porcelain", "-z"],
            timeout=self.config.local_timeout_s,
            stdin=None,
            max_output_bytes=self.config.output_hard_limit_bytes,
            read_only=True,
            operation="status",
        )
        if refs_result.returncode != 0 or worktrees_result.returncode != 0:
            raise self._error(GitErrorCode.COMMAND_FAILED, "Git refs or worktrees could not be read")
        head_ref = (
            head_ref_result.stdout.rstrip(b"\r\n").decode("utf-8", errors="strict")
            if head_ref_result.returncode == 0
            else None
        )
        head_oid = (
            head_oid_result.stdout.rstrip(b"\r\n").decode("ascii", errors="strict")
            if head_oid_result.returncode == 0
            else None
        )
        after = self.repository_layout(worktree=layout.root)
        if not self._same_layout(layout, after):
            raise self._error(GitErrorCode.STALE_STATE, "Git repository identity changed during state capture")
        return GitRepositoryState(
            layout=layout,
            head_ref=head_ref,
            head_oid=head_oid,
            index_sha256=self._index_digest(layout),
            config_sha256=config_sha256,
            refs_sha256=refs_result.stdout_sha256,
            worktrees_sha256=worktrees_result.stdout_sha256,
            pull_requests_sha256=self._pull_request_metadata_digest(layout),
            worktree_sha256=self._worktree_digest(layout, status.stdout),
            status_porcelain=status.stdout,
            status_sha256=status.stdout_sha256,
        )

    @staticmethod
    def _same_layout(before: GitRepositoryLayout, after: GitRepositoryLayout) -> bool:
        return (
            before.repository_id == after.repository_id
            and before.worktree_id == after.worktree_id
            and before.root == after.root
            and before.git_dir == after.git_dir
            and before.common_dir == after.common_dir
            and before.object_format == after.object_format
        )

    def _remote_environment_for_run(
        self,
        before: GitRepositoryLayout,
        *,
        remote: str | None,
        operation: str,
        expected_remote_fingerprint: str | None,
        treeish_targets: Sequence[str],
    ) -> dict[str, str]:
        if remote is None:
            return {}
        current_fingerprint = self.remote_fingerprint(remote, worktree=before.root)
        if (
            expected_remote_fingerprint is not None
            and current_fingerprint["fingerprint"] != expected_remote_fingerprint
        ):
            raise self._error(
                GitErrorCode.STALE_STATE,
                "Git remote configuration or refs changed before provider dispatch",
                operation=operation,
                retryable=True,
            )
        fetch_url, push_url = self._remote_urls(before, remote)
        config_sha256, _helpers = self._validate_repository_config(
            before,
            remote=remote,
            operation=operation,
            treeish_targets=treeish_targets,
        )
        if (
            hashlib.sha256(fetch_url.encode("utf-8")).hexdigest()
            != current_fingerprint["fetch_url_sha256"]
            or hashlib.sha256(push_url.encode("utf-8")).hexdigest()
            != current_fingerprint["push_url_sha256"]
            or config_sha256 != current_fingerprint["config_sha256"]
        ):
            raise self._error(
                GitErrorCode.STALE_STATE,
                "Git remote configuration changed during provider preflight",
                operation=operation,
                retryable=True,
            )
        environment, _ssh_identities = self._remote_dispatch_environment(
            fetch_url,
            push_url,
        )
        return environment

    def _verify_run_layout(
        self,
        before: GitRepositoryLayout,
        *,
        worktree: str | Path | None,
        operation: str,
        read_only: bool,
        verify_after: bool,
    ) -> None:
        if not verify_after:
            return
        try:
            after = self.repository_layout(worktree=worktree)
        except GitError as exc:
            if read_only:
                raise
            raise self._error(
                GitErrorCode.UNKNOWN_EFFECT,
                "Git repository identity could not be revalidated after mutation",
                operation=operation,
                details={"effect": "unknown"},
            ) from exc
        if not self._same_layout(before, after):
            raise self._error(
                (
                    GitErrorCode.UNSAFE_REPOSITORY
                    if read_only
                    else GitErrorCode.UNKNOWN_EFFECT
                ),
                "Git repository identity changed during operation",
                operation=operation,
                details={"effect": "none" if read_only else "unknown"},
            )

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
    ) -> GitCommandResult:
        if not args:
            raise self._error(GitErrorCode.COMMAND_FAILED, "Git operation is missing")
        operation = str(args[0])
        if any(not isinstance(item, str) or "\x00" in item for item in args):
            raise self._error(GitErrorCode.COMMAND_FAILED, "invalid Git argument", operation=operation)
        before = self.repository_layout(worktree=worktree)
        treeish_targets = self._attribute_treeish_targets(before, args)
        self._validate_repository_config(
            before,
            remote=remote,
            operation=operation,
            treeish_targets=treeish_targets,
        )
        remote_env = self._remote_environment_for_run(
            before,
            remote=remote,
            operation=operation,
            expected_remote_fingerprint=expected_remote_fingerprint,
            treeish_targets=treeish_targets,
        )
        selected_timeout = timeout if timeout is not None else (
            self.config.remote_timeout_s if remote is not None else self.config.local_timeout_s
        )
        selected_output = max_output_bytes or self.config.output_max_bytes
        result = self._invoke(
            [*self._repo_prefix(before), *args],
            timeout=selected_timeout,
            stdin=stdin,
            max_output_bytes=selected_output,
            read_only=read_only,
            operation=operation,
            env_overrides=remote_env,
        )
        self._verify_run_layout(
            before,
            worktree=worktree,
            operation=operation,
            read_only=read_only,
            verify_after=verify_after,
        )
        return result

    def run_with_limits(
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
        limits: SubprocessLimits,
    ) -> GitCommandResult:
        with self.subprocess_scope(limits=limits):
            return self.run(
                args,
                worktree=worktree,
                timeout=timeout,
                stdin=stdin,
                max_output_bytes=max_output_bytes,
                read_only=read_only,
                remote=remote,
                expected_remote_fingerprint=expected_remote_fingerprint,
                verify_after=verify_after,
            )

    @contextmanager
    def repository_lock(
        self,
        *,
        worktree: str | Path | None = None,
        timeout: float | None = None,
    ) -> Iterator[GitRepositoryLayout]:
        selected_timeout = self.config.lock_timeout_s if timeout is None else timeout
        if selected_timeout <= 0 or selected_timeout > self.config.timeout_hard_limit_s:
            raise self._error(GitErrorCode.REPOSITORY_BUSY, "invalid repository lock timeout")
        deadline = time.monotonic() + selected_timeout
        thread_lock_acquired = self._thread_lock.acquire(
            timeout=max(0.0, deadline - time.monotonic())
        )
        if not thread_lock_acquired:
            raise self._error(
                GitErrorCode.REPOSITORY_BUSY,
                "Git repository is busy",
                retryable=True,
            )
        try:
            owner = threading.get_ident()
            depth = self._repository_lock_depth
            if self._repository_lock_owner == owner and depth:
                self._repository_lock_depth = depth + 1
                try:
                    yield self.repository_layout(worktree=worktree)
                finally:
                    self._repository_lock_depth = depth
                return
            layout = self.repository_layout(worktree=worktree)
            with self._open_repository_lock_file(layout.common_dir) as lock_file:
                descriptor = lock_file.descriptor
                acquired = False
                try:
                    while not acquired:
                        if time.monotonic() >= deadline:
                            raise self._error(
                                GitErrorCode.REPOSITORY_BUSY,
                                "Git repository is busy",
                                retryable=True,
                            )
                        try:
                            if os.name == "nt":
                                import msvcrt

                                os.lseek(descriptor, 0, os.SEEK_SET)
                                if os.fstat(descriptor).st_size == 0:
                                    os.write(descriptor, b"\0")
                                os.lseek(descriptor, 0, os.SEEK_SET)
                                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                            else:
                                import fcntl

                                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            acquired = True
                        except (BlockingIOError, OSError):
                            if time.monotonic() >= deadline:
                                raise self._error(
                                    GitErrorCode.REPOSITORY_BUSY,
                                    "Git repository is busy",
                                    retryable=True,
                                )
                            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
                    try:
                        self._validate_open_repository_lock_file(lock_file)
                    except OSError as exc:
                        raise self._error(
                            GitErrorCode.UNSAFE_REPOSITORY,
                            "repository lock identity changed before dispatch",
                        ) from exc
                    self._repository_lock_owner = owner
                    self._repository_lock_depth = 1
                    current = self.repository_layout(worktree=worktree)
                    if not self._same_layout(layout, current):
                        raise self._error(GitErrorCode.STALE_STATE, "Git repository identity changed before dispatch")
                    yield current
                finally:
                    self._repository_lock_depth = 0
                    self._repository_lock_owner = None
                    if acquired:
                        try:
                            if os.name == "nt":
                                import msvcrt

                                os.lseek(descriptor, 0, os.SEEK_SET)
                                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                            else:
                                import fcntl

                                fcntl.flock(descriptor, fcntl.LOCK_UN)
                        except OSError:
                            pass
        finally:
            self._thread_lock.release()

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        read_only = operation in _READ_OPERATIONS or bool(context.get("read_only"))
        remote = operation in _REMOTE_OPERATIONS or context.get("remote") is not None
        result_data = result if isinstance(result, dict) else {}
        receipt = {
            key: result_data[key]
            for key in (
                "repository_id",
                "worktree_id",
                "before_state_token",
                "after_state_token",
                "created_oid",
                "base_oid",
                "head_oid",
                "merged_oid",
                "remote",
                "remote_ref",
                "remote_old_oid",
                "remote_new_oid",
                "patch_sha256",
            )
            if result_data.get(key) is not None
        }
        if read_only:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"provider": "git", "remote": remote},
            )
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=remote,
            metadata={
                "provider": "git",
                "remote": remote,
                "provider_receipt": receipt,
            },
        )

    def reconcile_external_effect(self, effect: Any) -> dict[str, Any]:
        """Query repository identity/state for an ambiguous effect; never replay it."""

        metadata = getattr(effect, "provider_metadata", {})
        context = metadata.get("context") if isinstance(metadata, dict) else None
        observation = context if isinstance(context, dict) else {}
        worktree_id = str(observation.get("worktree_id") or "main")
        if worktree_id == "main":
            worktree: Path | None = None
        elif _MANAGED_WORKTREE_ID_RE.fullmatch(worktree_id):
            worktree = self.managed_worktree_root / worktree_id
        else:
            worktree = None
            worktree_id = "main"
        state = self.repository_state(worktree=worktree)
        receipt: dict[str, Any] = {
            "reconciliation": "query_only",
            "repository_id": state.layout.repository_id,
            "worktree_id": state.layout.worktree_id,
            "head_ref": state.head_ref,
            "head_oid": state.head_oid,
            "index_sha256": state.index_sha256,
            "refs_sha256": state.refs_sha256,
            "worktrees_sha256": state.worktrees_sha256,
            "pull_requests_sha256": state.pull_requests_sha256,
            "worktree_sha256": state.worktree_sha256,
        }
        remote = observation.get("remote")
        if isinstance(remote, str) and _REMOTE_NAME_RE.fullmatch(remote):
            try:
                fingerprint = self.remote_fingerprint(
                    remote,
                    worktree=worktree,
                )
                receipt["remote_fingerprint"] = {
                    key: fingerprint[key]
                    for key in (
                        "remote",
                        "fetch_url_sha256",
                        "push_url_sha256",
                        "config_sha256",
                        "refs_sha256",
                        "fingerprint",
                    )
                }
            except GitError:
                receipt["remote_fingerprint"] = None
        return {"state": "unknown", "provider_receipt": receipt}


__all__ = ["LocalGitProvider"]
