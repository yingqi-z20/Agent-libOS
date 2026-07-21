from __future__ import annotations

import hashlib
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
from agent_libos.models.exceptions import GitError
from agent_libos.substrate.base import (
    CommandMetrics,
    GitCommandResult,
    GitRepositoryLayout,
    GitRepositoryState,
    ProviderEffectNotStarted,
    executable_content_sha256,
)

_VERSION_RE = re.compile(rb"(?:git version\s+)?(\d+)\.(\d+)(?:\.(\d+))?")
_REMOTE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MANAGED_WORKTREE_ID_RE = re.compile(r"wt_[A-Za-z0-9_-]{1,96}\Z")
_PULL_REQUEST_ID_RE = re.compile(r"pr_[A-Za-z0-9_-]{1,96}\Z")
_SCP_REMOTE_RE = re.compile(
    r"(?:(?P<user>[A-Za-z0-9._-]+)@)?(?P<host>[A-Za-z0-9.-]+):(?P<path>[^\x00\r\n]+)\Z"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ATTRIBUTE_DRIVER_RE = re.compile(
    rb"(?:^|[ \t])(?P<kind>filter|diff|merge)=(?P<name>[A-Za-z0-9._-]+)(?=$|[ \t\r])",
    re.MULTILINE,
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
_FILTER_DRIVER_SUFFIXES = frozenset({"clean", "smudge", "process"})
_DIFF_DRIVER_SUFFIXES = frozenset({"command"})
_MERGE_DRIVER_SUFFIXES = frozenset({"driver"})
_SAFE_REF_SUFFIX_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,500}\Z")


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
    if not stat.S_ISDIR(value.st_mode):
        raise NotADirectoryError(path)
    return int(value.st_dev), int(value.st_ino)


def _digest_fields(*fields: str) -> str:
    digest = hashlib.sha256()
    for field in fields:
        encoded = field.encode("utf-8", errors="surrogateescape")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class LocalGitProvider:
    """Pinned, byte-preserving system-Git provider.

    This provider deliberately has no public arbitrary-argv compatibility
    surface.  ``run`` is the narrow Host boundary used only by GitPrimitive,
    which constructs every accepted argument.  Repository discovery is also
    avoided: the workspace's lexical ``.git`` entry is validated first and is
    then supplied to Git explicitly.
    """

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
        self._lock_state = threading.local()
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
        if timeout <= 0 or timeout > self.config.timeout_hard_limit_s:
            raise self._error(GitErrorCode.TIMEOUT, "invalid Git timeout", operation=operation)
        if max_output_bytes <= 0 or max_output_bytes > self.config.output_hard_limit_bytes:
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "requested Git output limit exceeds the configured hard limit",
                operation=operation,
            )
        executable, identity_before, content_before = self._resolve_executable()
        full_argv = [str(executable), *map(str, argv)]
        if any("\x00" in item for item in full_argv):
            raise self._error(GitErrorCode.COMMAND_FAILED, "Git argv contains a NUL byte")
        started = time.monotonic()
        peak_memory = 0
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                environment = self._safe_env(read_only=read_only)
                environment.update(dict(env_overrides or {}))
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
                raise self._error(GitErrorCode.GIT_UNAVAILABLE, "system Git is unavailable") from exc
            except OSError as exc:
                raise self._error(GitErrorCode.COMMAND_FAILED, "system Git could not start") from exc
            if stdin is not None and process.stdin is not None:
                try:
                    process.stdin.write(stdin)
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            deadline = started + timeout
            oversized = False
            timed_out = False
            while process.poll() is None:
                now = time.monotonic()
                try:
                    usage = psutil.Process(process.pid).memory_info().rss
                    peak_memory = max(peak_memory, int(usage))
                except (psutil.Error, OSError):
                    pass
                if stdout_file.tell() > max_output_bytes or stderr_file.tell() > max_output_bytes:
                    oversized = True
                    self._kill_process_tree(process)
                    break
                if now >= deadline:
                    timed_out = True
                    self._kill_process_tree(process)
                    break
                time.sleep(0.01)
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._kill_process_tree(process)
                process.wait()
            elapsed = max(0.0, time.monotonic() - started)
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(max_output_bytes + 1)
            stderr = stderr_file.read(max_output_bytes + 1)
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
        metrics = CommandMetrics(
            wall_seconds=elapsed,
            peak_memory_bytes=peak_memory,
            killed=timed_out or oversized,
            limit_kind="wall" if timed_out else ("output" if oversized else None),
        )
        if timed_out:
            raise self._error(
                GitErrorCode.TIMEOUT,
                "Git operation timed out; its effect may be unknown",
                operation=operation,
                retryable=True,
                details={"effect": "unknown"},
            )
        if oversized or len(stdout) > max_output_bytes or len(stderr) > max_output_bytes:
            raise self._error(
                GitErrorCode.OUTPUT_TOO_LARGE,
                "Git output exceeded the configured hard limit",
                operation=operation,
                details={"effect": "unknown" if not read_only else "none"},
            )
        return GitCommandResult(
            argv=tuple(full_argv),
            returncode=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            stderr_sha256=hashlib.sha256(stderr).hexdigest(),
            metrics=metrics,
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
        resolved = selected.resolve(strict=False)
        if resolved != self.workspace_root and not _is_within(resolved, self.managed_worktree_root):
            raise self._error(
                GitErrorCode.INVALID_PATH,
                "worktree is outside the Runtime-managed worktree root",
            )
        return resolved

    @staticmethod
    def _read_small_file(path: Path, *, limit: int = 65536) -> bytes:
        with path.open("rb") as handle:
            data = handle.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f"metadata file exceeds {limit} bytes")
        return data

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
        if stat.S_ISLNK(entry_state.st_mode):
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
        try:
            git_dir = candidate.resolve(strict=True)
            target_state = candidate.lstat()
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "worktree gitdir is unavailable") from exc
        if stat.S_ISLNK(target_state.st_mode) or not git_dir.is_dir():
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "worktree gitdir is not trusted")
        primary = (self.workspace_root / ".git").resolve(strict=False)
        if not any(_is_within(git_dir, root) for root in self._trusted_metadata_roots(primary)):
            raise self._error(
                GitErrorCode.UNSAFE_REPOSITORY,
                "linked worktree metadata is outside trusted metadata roots",
            )
        return git_dir, True

    def _common_dir(self, git_dir: Path) -> Path:
        marker = git_dir / "commondir"
        if not marker.exists():
            return git_dir
        try:
            marker_state = marker.lstat()
            if stat.S_ISLNK(marker_state.st_mode) or not stat.S_ISREG(marker_state.st_mode):
                raise ValueError("invalid commondir marker")
            raw = self._read_small_file(marker, limit=8192).strip()
            if not raw or b"\x00" in raw or b"\n" in raw or b"\r" in raw:
                raise ValueError("invalid commondir marker")
            selected = Path(os.fsdecode(raw))
            candidate = selected if selected.is_absolute() else git_dir / selected
            resolved = candidate.resolve(strict=True)
            target = candidate.lstat()
            if stat.S_ISLNK(target.st_mode) or not resolved.is_dir():
                raise ValueError("invalid common directory")
            return resolved
        except (OSError, ValueError) as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "invalid Git common directory") from exc

    def _repo_prefix(self, layout: GitRepositoryLayout | tuple[Path, Path, Path]) -> list[str]:
        if isinstance(layout, GitRepositoryLayout):
            worktree, git_dir = layout.root, layout.git_dir
        else:
            worktree, git_dir, _common = layout
        protocol_file = "always" if self.config.allow_file_remotes else "never"
        return [
            "--no-pager",
            "--literal-pathspecs",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
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
            if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
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
        if stat.S_ISLNK(selected_state.st_mode) or not stat.S_ISDIR(selected_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git worktree root is not a trusted directory")
        git_dir, linked = self._git_dir_from_entry(selected)
        common_dir = self._common_dir(git_dir)
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
        key: str,
        value: str,
        remote: str | None,
    ) -> None:
        if remote is None:
            return
        remote_key = f"remote.{remote.casefold()}."
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

    def _validate_repository_config(
        self,
        layout: GitRepositoryLayout,
        *,
        remote: str | None,
        operation: str,
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        entries = self._config_entries(layout)
        normalized: list[str] = []
        helper_identities: list[tuple[str, str]] = []
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
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "repository Git config is not a trusted regular file",
                )
        active_drivers: dict[str, set[str]] | None = None

        def driver_is_active(kind: str, key: str) -> bool:
            nonlocal active_drivers
            if active_drivers is None:
                active_drivers = self._active_attribute_drivers(layout, entries)
            parts = key.split(".")
            return len(parts) >= 3 and parts[1] in active_drivers[kind]
        for scope, origin, key, value in entries:
            if scope == "command":
                continue
            normalized.append(f"{scope}\0{origin}\0{key}\0{value}")
            origin_path: Path | None = None
            if origin.startswith("file:"):
                origin_path = Path(origin[5:]).expanduser().resolve(strict=False)
                if scope in {"local", "worktree"} and origin_path not in allowed_local_origins:
                    raise self._error(
                        GitErrorCode.UNSAFE_CONFIG,
                        "repository Git config includes are not allowed",
                    )
                if _is_within(origin_path, self.workspace_root) and origin_path not in allowed_local_origins:
                    raise self._error(GitErrorCode.UNSAFE_CONFIG, "workspace-controlled Git config includes are not allowed")
            if self._config_selects_executable_extension(
                key=key,
                value=value,
                remote=remote,
                operation=operation,
                driver_is_active=driver_is_active,
            ):
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "repository config contains an executable Git extension")
            self._validate_remote_config_entry(
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
            helper_identity = self._credential_helper_identity(
                scope=scope,
                origin_path=origin_path,
                key=key,
                value=value,
                remote=remote,
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
    ) -> dict[str, set[str]]:
        """Inspect attribute declarations as data, without invoking Git drivers."""

        files: list[Path] = []
        seen: set[Path] = set()

        def include(candidate: Path) -> None:
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)

        include(layout.common_dir / "info" / "attributes")
        for scope, _origin, key, value in entries:
            if scope != "command" and key == "core.attributesfile" and value:
                selected = Path(value)
                if not selected.is_absolute():
                    home = os.environ.get("HOME")
                    if not home:
                        raise self._error(
                            GitErrorCode.UNSAFE_CONFIG,
                            "relative Host attributes file cannot be resolved safely",
                        )
                    selected = Path(home) / selected
                include(selected)
        visited = 0
        for directory, directory_names, file_names in os.walk(layout.root, followlinks=False):
            visited += len(directory_names) + len(file_names)
            if visited > 100_000:
                raise self._error(
                    GitErrorCode.UNSAFE_CONFIG,
                    "attribute discovery exceeded its safety bound",
                )
            current = Path(directory)
            directory_names[:] = [
                name
                for name in directory_names
                if name.casefold() != ".git"
                and (current / name).resolve(strict=False) != self.managed_worktree_root
            ]
            if ".gitattributes" in file_names:
                include(current / ".gitattributes")
        drivers = {"filter": set(), "diff": set(), "merge": set()}
        total = 0
        for selected in files:
            try:
                metadata = selected.lstat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes could not be inspected") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes source is not a regular file")
            try:
                raw = self._read_small_file(selected, limit=1_048_576)
            except (OSError, ValueError) as exc:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes exceed their safety bound") from exc
            total += len(raw)
            if total > 1_048_576:
                raise self._error(GitErrorCode.UNSAFE_CONFIG, "Git attributes exceed their aggregate safety bound")
            for match in _ATTRIBUTE_DRIVER_RE.finditer(raw):
                kind = match.group("kind").decode("ascii")
                name = match.group("name").decode("ascii")
                drivers[kind].add(name.casefold())
        return drivers

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
        if scheme not in set(self.config.allowed_remote_schemes):
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
            remote=None,
            operation="list_remotes",
        )
        fetch_url, push_url = self._remote_urls(layout, remote)
        after = self.repository_layout(worktree=layout.root)
        if not self._same_layout(layout, after):
            raise self._error(GitErrorCode.STALE_STATE, "Git repository identity changed during remote inspection")
        return fetch_url, push_url, {
            "config_sha256": config_sha256,
            "fetch_url_sha256": hashlib.sha256(fetch_url.encode("utf-8")).hexdigest(),
            "push_url_sha256": hashlib.sha256(push_url.encode("utf-8")).hexdigest(),
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
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "managed worktree root is not a trusted directory")
        target = self.managed_worktree_root / worktree_id
        if target.exists() or target.is_symlink():
            raise self._error(GitErrorCode.ALREADY_EXISTS, "managed worktree target already exists")
        return target

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
        if stat.S_ISLNK(info_state.st_mode) or not stat.S_ISDIR(info_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git info directory is not trusted")

        exclude = info / "exclude"
        try:
            exclude_state = exclude.lstat()
        except FileNotFoundError:
            current = b""
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "Git exclude file could not be inspected") from exc
        else:
            if stat.S_ISLNK(exclude_state.st_mode) or not stat.S_ISREG(exclude_state.st_mode):
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
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
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
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata is not a trusted regular file")
        try:
            data = self._read_small_file(selected, limit=self.config.output_hard_limit_bytes)
        except (OSError, ValueError) as exc:
            raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata exceeds its hard limit") from exc
        return data, hashlib.sha256(data).hexdigest()

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
        if stat.S_ISLNK(directory_state.st_mode) or not stat.S_ISDIR(directory_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory is not trusted")
        rows: list[tuple[str, bytes, str]] = []
        try:
            names = sorted(directory.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be listed") from exc
        if len(names) > limit:
            names = names[:limit]
        total = 0
        for selected in names:
            if selected.suffix != ".json" or not _PULL_REQUEST_ID_RE.fullmatch(selected.stem):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unexpected pull request metadata entry")
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
        if stat.S_ISLNK(directory_state.st_mode) or not stat.S_ISDIR(directory_state.st_mode):
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata directory is not trusted")
        digest = hashlib.sha256()
        total = 0
        try:
            names = sorted(directory.iterdir(), key=lambda item: item.name)
        except OSError as exc:
            raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "pull request metadata could not be inspected") from exc
        if len(names) > self.config.status_entry_hard_limit:
            raise self._error(GitErrorCode.OUTPUT_TOO_LARGE, "pull request metadata count exceeds its hard limit")
        for selected in names:
            if selected.suffix != ".json" or not _PULL_REQUEST_ID_RE.fullmatch(selected.stem):
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "unexpected pull request metadata entry")
            try:
                metadata = selected.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
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
        if stat.S_ISLNK(state.st_mode) or not stat.S_ISREG(state.st_mode):
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
        self._validate_repository_config(before, remote=remote, operation=operation)
        remote_env: dict[str, str] = {}
        if remote is not None:
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
            remote_env, _ssh_identities = self._remote_dispatch_environment(fetch_url, push_url)
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
        if verify_after:
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
                    GitErrorCode.UNKNOWN_EFFECT if not read_only else GitErrorCode.UNSAFE_REPOSITORY,
                    "Git repository identity changed during operation",
                    operation=operation,
                    details={"effect": "unknown" if not read_only else "none"},
                )
        return result

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
        with self._thread_lock:
            depth = int(getattr(self._lock_state, "depth", 0))
            if depth:
                self._lock_state.depth = depth + 1
                try:
                    yield self.repository_layout(worktree=worktree)
                finally:
                    self._lock_state.depth -= 1
                return
            layout = self.repository_layout(worktree=worktree)
            lock_directory = layout.common_dir / "agent-libos"
            try:
                lock_directory.mkdir(mode=0o700, exist_ok=True)
                if lock_directory.is_symlink() or not lock_directory.is_dir():
                    raise OSError("unsafe lock directory")
                lock_path = lock_directory / "repository.lock"
                flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(lock_path, flags, 0o600)
            except OSError as exc:
                raise self._error(GitErrorCode.UNSAFE_REPOSITORY, "repository lock could not be opened") from exc
            acquired = False
            deadline = time.monotonic() + selected_timeout
            try:
                while not acquired:
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
                        time.sleep(0.02)
                self._lock_state.depth = 1
                current = self.repository_layout(worktree=worktree)
                if not self._same_layout(layout, current):
                    raise self._error(GitErrorCode.STALE_STATE, "Git repository identity changed before dispatch")
                yield current
            finally:
                self._lock_state.depth = 0
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
                os.close(descriptor)

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
