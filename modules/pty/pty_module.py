from __future__ import annotations

import errno
import contextlib
import math
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, TYPE_CHECKING

import psutil
from pydantic import BaseModel, Field

from agent_libos.memory.data_labels import propagate_object_labels
from agent_libos.models import (
    AgentImage,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityRight,
    DataFlowContext,
    DataIntegrity,
    DataLabels,
    DataSensitivity,
    DataSink,
    DataTrustLevel,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectMetadata,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    ResourceUsage,
    ViewMode,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.primitives.git_command_policy import trusted_git_read_operation
from agent_libos.primitives.shell import ShellExecutionPolicy, ShellPolicyDecision
from agent_libos.substrate import (
    ExecutableSnapshot,
    LocalGitProvider,
    ProviderEffectNotStarted,
    resolve_runtime_python_alias,
    SubprocessLimits,
)
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProviderPhase,
)
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.ids import new_id, utc_now

if TYPE_CHECKING:
    from agent_libos.modules.context import ModuleHost

_PTY_ADAPTER_ATTR = "_agent_libos_pty_adapter"
_SAFE_PTY_ENV_KEYS = {
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


class PtySession(Protocol):
    backend: str
    pid: int | None

    def read(self, *, timeout_s: float = 0.0) -> str: ...

    def write(self, text: str) -> int: ...

    def resize(self, cols: int, rows: int) -> None: ...

    def is_alive(self) -> bool: ...

    def exit_code(self) -> int | None: ...

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None: ...


class PtyProvider(Protocol):
    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]: ...

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> PtySession: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


class _PtySpawnContainmentError(RuntimeError):
    """A provider-created session whose immediate containment did not finish."""

    def __init__(self, message: str, *, orphan_session: PtySession) -> None:
        super().__init__(message)
        self.orphan_session = orphan_session


@dataclass(frozen=True)
class PtyModuleSettings:
    max_sessions_global: int = 16
    max_sessions_per_process: int = 4
    buffer_max_chars: int = 200_000
    startup_output_max_chars: int = 4_000
    read_max_chars: int = 32_000
    read_hard_limit_chars: int = 200_000
    input_max_chars: int = 32_768
    input_hard_limit_chars: int = 131_072
    default_cols: int = 80
    default_rows: int = 24
    max_cols: int = 512
    max_rows: int = 200
    startup_timeout_s: float = 0.2
    startup_timeout_hard_limit_s: float = 5.0
    read_timeout_s: float = 0.0
    read_timeout_hard_limit_s: float = 30.0
    close_timeout_s: float = 2.0
    close_timeout_hard_limit_s: float = 10.0
    resource_sample_interval_s: float = 0.05
    session_name_prefix: str = "pty_session"


@dataclass(frozen=True)
class PtyModuleConfig:
    pty: PtyModuleSettings = field(default_factory=PtyModuleSettings)


def _coerce_pty_settings(value: Any) -> PtyModuleSettings:
    if value is None:
        return PtyModuleSettings()
    if isinstance(value, PtyModuleSettings):
        settings = value
    elif isinstance(value, dict):
        settings = PtyModuleSettings(**value)
    else:
        settings = PtyModuleSettings(
            **{field_name: getattr(value, field_name) for field_name in PtyModuleSettings.__dataclass_fields__ if hasattr(value, field_name)}
        )
    _validate_pty_settings(settings)
    return settings


def _validate_pty_settings(settings: PtyModuleSettings) -> None:
    _validate_numeric_settings(
        settings,
        (
            "max_sessions_global",
            "max_sessions_per_process",
            "buffer_max_chars",
            "startup_output_max_chars",
            "read_max_chars",
            "read_hard_limit_chars",
            "input_max_chars",
            "input_hard_limit_chars",
            "default_cols",
            "default_rows",
            "max_cols",
            "max_rows",
            "resource_sample_interval_s",
        ),
        allow_zero=False,
    )
    _validate_numeric_settings(
        settings,
        (
            "startup_timeout_s",
            "startup_timeout_hard_limit_s",
            "read_timeout_s",
            "read_timeout_hard_limit_s",
            "close_timeout_s",
            "close_timeout_hard_limit_s",
        ),
        allow_zero=True,
    )
    if not settings.session_name_prefix.strip():
        raise ValidationError("pty module setting session_name_prefix must be non-empty")
    _validate_setting_bounds(settings)


def _validate_numeric_settings(
    settings: PtyModuleSettings,
    names: tuple[str, ...],
    *,
    allow_zero: bool,
) -> None:
    for name in names:
        value = getattr(settings, name)
        valid_number = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        )
        if not valid_number or value < 0 or (not allow_zero and value == 0):
            qualifier = ">= 0" if allow_zero else "> 0"
            raise ValidationError(f"pty module setting {name} must be {qualifier}")


def _validate_setting_bounds(settings: PtyModuleSettings) -> None:
    bounds = (
        ("max_sessions_global", "max_sessions_per_process"),
        ("read_hard_limit_chars", "read_max_chars"),
        ("read_hard_limit_chars", "startup_output_max_chars"),
        ("input_hard_limit_chars", "input_max_chars"),
        ("max_cols", "default_cols"),
        ("max_rows", "default_rows"),
        ("startup_timeout_hard_limit_s", "startup_timeout_s"),
        ("read_timeout_hard_limit_s", "read_timeout_s"),
        ("close_timeout_hard_limit_s", "close_timeout_s"),
    )
    for ceiling_name, default_name in bounds:
        if getattr(settings, ceiling_name) < getattr(settings, default_name):
            raise ValidationError(
                f"pty module {ceiling_name} must be >= {default_name}"
            )


def initialize_pty(runtime: "ModuleHost") -> None:
    if runtime.get_runtime_attribute(_PTY_ADAPTER_ATTR) is not None:
        return
    settings = _coerce_pty_settings(getattr(runtime.substrate, "pty_settings", None))
    provider = getattr(runtime.substrate, "pty", None) or LocalPtyProvider(
        runtime.workspace_root,
        git_config=runtime.config.git,
    )
    adapter = PtyAdapter(
        runtime,
        runtime.shell,
        runtime.human,
        runtime.audit,
        runtime.events,
        provider=provider,
        config=PtyModuleConfig(settings),
        resources=runtime.resources,
    )
    adapter.release_stale_session_objects()
    runtime.set_runtime_attribute(_PTY_ADAPTER_ATTR, adapter)
    runtime.bind_object_release_finalizer(_object_release_finalizer(adapter))
    runtime.bind_shutdown_finalizer(adapter.shutdown)
    runtime.bind_recovery_cleanup(adapter.release_recovery_diagnostics)


def _object_release_finalizer(adapter: "PtyAdapter"):
    def finalize(obj: Any, actor: str, reason: str) -> None:
        if getattr(obj, "type", None) == ObjectType.EXTERNAL_REF and isinstance(getattr(obj, "payload", None), dict):
            if obj.payload.get("kind") == "pty_session":
                adapter.close_for_object_release(obj.oid, actor=actor, reason=reason)

    return finalize


class LocalPtyProvider:
    """Subprocess-backed PTY provider scoped to a configured workspace."""

    supports_subprocess_limits = os.name != "nt"
    supports_executable_snapshots = True

    def __init__(self, cwd: str | Path, *, git_config: Any | None = None):
        self.cwd = Path(cwd).resolve()
        self._git_config = git_config
        self._git_read_guard: LocalGitProvider | None = None

    def resolve_argv(self, argv: list[str], *, cwd: str | None = None) -> list[str]:
        return self._resolve_argv0(argv, self._resolve_cwd(cwd))

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> PtySession:
        if limits is not None and not self.supports_subprocess_limits:
            raise ValidationError("PTY provider cannot enforce SubprocessLimits on this platform")
        selected_cwd = self._resolve_cwd(cwd)
        safe_path = self._safe_path()
        requested_argv0 = argv[0] if argv else None
        resolved_argv = self._resolve_argv0(argv, selected_cwd)
        git_operation = trusted_git_read_operation(argv, hardened_only=True)
        if git_operation is not None:
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
        if executable_snapshot is not None:
            executable_snapshot.verify()
            if executable_snapshot.source_path != Path(resolved_argv[0]).resolve(
                strict=False
            ):
                raise ValidationError(
                    "PTY executable snapshot does not match resolved argv[0]"
                )
        if os.name == "nt":
            return _WinPtySession.spawn(
                resolved_argv,
                cwd=selected_cwd,
                home=self.cwd,
                path=safe_path,
                cols=cols,
                rows=rows,
                executable_snapshot=executable_snapshot,
            )
        dispatch_argv = resolved_argv
        if executable_snapshot is not None and requested_argv0 is not None:
            # POSIX executable selection pins the launched bytes independently
            # from argv[0]. Preserve the requested launcher spelling so
            # virtualenv and other argv[0]-sensitive runtimes keep their normal
            # semantics.
            dispatch_argv = [requested_argv0, *resolved_argv[1:]]
        return _PosixPtySession.spawn(
            dispatch_argv,
            cwd=selected_cwd,
            home=self.cwd,
            path=safe_path,
            cols=cols,
            rows=rows,
            executable_snapshot=executable_snapshot,
        )

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
        if operation in {"spawn", "write", "close"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                state_mutation=True,
                information_flow=True,
                metadata={
                    "operation": operation,
                    "argv": context.get("argv") if operation == "spawn" else None,
                    "cwd": context.get("cwd") if operation == "spawn" else None,
                },
            )
        if operation in {"read", "ingest"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                state_mutation=False,
                information_flow=True,
                metadata={"operation": operation},
            )
        if operation == "resize":
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.ROLLBACKABLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_APPLIED,
                state_mutation=True,
                information_flow=True,
                metadata={
                    "operation": operation,
                    "previous_cols": context.get("previous_cols"),
                    "previous_rows": context.get("previous_rows"),
                },
            )
        raise ValueError(f"unsupported pty external effect operation: {operation}")

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None or cwd in {"", "."}:
            return self.cwd
        raw = Path(cwd)
        target = raw.resolve() if raw.is_absolute() else (self.cwd / raw).resolve()
        if self.cwd not in target.parents and target != self.cwd:
            raise CapabilityDenied(f"pty working directory escapes workspace root: {cwd}")
        return target

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
            raise FileNotFoundError(f"pty executable not found on safe PATH: {argv[0]}")
        target = Path(resolved).resolve()
        if self.cwd in target.parents or target == self.cwd or selected_cwd in target.parents or target == selected_cwd:
            raise CapabilityDenied(f"bare pty executable resolves inside workspace: {argv[0]}")
        return [str(target), *argv[1:]]

    def _safe_path(self) -> str:
        entries: list[str] = []
        for item in os.environ.get("PATH", "").split(os.pathsep):
            if not item:
                continue
            raw = Path(item).expanduser()
            if not raw.is_absolute():
                continue
            resolved = raw.resolve()
            if self.cwd in resolved.parents or resolved == self.cwd:
                continue
            entries.append(str(resolved))
        return os.pathsep.join(entries)

    def _argv0_has_path(self, value: str) -> bool:
        return "/" in value or "\\" in value or Path(value).is_absolute()


class _PosixPtySession:
    backend = "posix-pty"

    def __init__(
        self,
        master_fd: int,
        proc: subprocess.Popen[bytes],
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> None:
        self.master_fd = master_fd
        self.proc = proc
        self.executable_snapshot = executable_snapshot
        self.pid = proc.pid
        self._closed = False

    @classmethod
    def spawn(
        cls,
        argv: list[str],
        *,
        cwd: Path,
        home: Path,
        path: str,
        cols: int,
        rows: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> "_PosixPtySession":
        if sys.platform == "win32":
            raise RuntimeError("POSIX PTY backend is unavailable on Windows")
        import fcntl
        import pty
        import struct
        import termios

        try:
            master_fd, slave_fd = pty.openpty()
        except Exception as exc:
            raise ProviderEffectNotStarted(f"PTY allocation failed before process spawn: {exc}") from exc
        proc: subprocess.Popen[bytes] | None = None
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            proc = subprocess.Popen(
                argv,
                executable=(
                    str(executable_snapshot.executable_path)
                    if executable_snapshot is not None
                    else None
                ),
                cwd=cwd,
                env=_safe_subprocess_env(path=path, home=home),
                shell=False,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
                start_new_session=True,
            )
            try:
                os.set_blocking(master_fd, False)
            except AttributeError:
                pass
            return cls(master_fd, proc, executable_snapshot)
        except Exception as exc:
            if proc is None:
                with contextlib.suppress(OSError):
                    os.close(master_fd)
                raise ProviderEffectNotStarted(f"PTY process failed before spawn completed: {exc}") from exc
            # A process existed, so the outcome remains externally ambiguous.
            # Contain it before surfacing the original exception.
            cleanup = cls(master_fd, proc)
            try:
                cleanup.close(force=True, timeout_s=1.0)
            except Exception as cleanup_exc:
                raise _PtySpawnContainmentError(
                    "PTY post-spawn initialization and containment failed",
                    orphan_session=cleanup,
                ) from cleanup_exc
            raise
        finally:
            with contextlib.suppress(OSError):
                os.close(slave_fd)

    def read(self, *, timeout_s: float = 0.0) -> str:
        if self._closed:
            return ""
        import select

        chunks: list[bytes] = []
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            wait_s = max(0.0, deadline - time.monotonic()) if not chunks else 0.0
            ready, _, _ = select.select([self.master_fd], [], [], wait_s)
            if not ready:
                break
            try:
                data = os.read(self.master_fd, 8192)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not data:
                break
            chunks.append(data)
            if timeout_s == 0:
                continue
            if time.monotonic() >= deadline:
                break
        return b"".join(chunks).decode("utf-8", errors="replace")

    def write(self, text: str) -> int:
        if self._closed:
            return 0
        data = text.encode("utf-8")
        return os.write(self.master_fd, data)

    def resize(self, cols: int, rows: int) -> None:
        if self._closed:
            return
        import fcntl
        import struct
        import termios

        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def exit_code(self) -> int | None:
        return self.proc.poll()

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None:
        if self._closed:
            return self.exit_code()
        deadline = time.monotonic() + max(0.0, timeout_s)
        if force:
            self._signal_process_group(signal.SIGTERM)
        if self.proc.poll() is None:
            try:
                graceful_timeout = max(0.0, deadline - time.monotonic())
                if force:
                    graceful_timeout *= 0.5
                self.proc.wait(timeout=graceful_timeout)
            except subprocess.TimeoutExpired:
                self._signal_process_group(signal.SIGKILL)
                try:
                    self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        self.proc.kill()
                    try:
                        self.proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired as exc:
                        raise TimeoutError(
                            "PTY process did not exit within the close deadline"
                        ) from exc
        if self.proc.poll() is None:
            raise TimeoutError("PTY process did not exit within the close deadline")
        self._closed = True
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        if self.executable_snapshot is not None:
            self.executable_snapshot.close()
            self.executable_snapshot = None
        return self.exit_code()

    def _signal_process_group(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            # Falling back to only the direct child would leave descendants
            # running outside the PTY lifecycle.  psutil gives us a second,
            # explicit tree-containment path; if that is also denied, surface
            # the cleanup failure instead of reporting the session closed.
            self._signal_process_tree(sig)

    def _signal_process_tree(self, sig: int) -> None:
        try:
            root = psutil.Process(self.proc.pid)
        except psutil.NoSuchProcess:
            return
        descendants = root.children(recursive=True)
        for process in reversed(descendants):
            try:
                process.send_signal(sig)
            except psutil.NoSuchProcess:
                continue
        try:
            root.send_signal(sig)
        except psutil.NoSuchProcess:
            return


class _WinPtySession:
    backend = "winpty"

    def __init__(
        self,
        proc: Any,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> None:
        self.proc = proc
        self.pid = getattr(proc, "pid", None)
        self.executable_snapshot = executable_snapshot

    @classmethod
    def spawn(
        cls,
        argv: list[str],
        *,
        cwd: Path,
        home: Path,
        path: str,
        cols: int,
        rows: int,
        executable_snapshot: ExecutableSnapshot | None = None,
    ) -> "_WinPtySession":
        try:
            from winpty import PtyProcess
        except ImportError as exc:
            raise RuntimeError("Windows PTY backend requires the pywinpty package") from exc

        dispatch_argv = (
            [str(executable_snapshot.executable_path), *argv[1:]]
            if executable_snapshot is not None
            else argv
        )
        command_line = subprocess.list2cmdline(dispatch_argv)
        spawn = PtyProcess.spawn
        kwargs = {
            "cwd": str(cwd),
            "env": _safe_subprocess_env(path=path, home=home),
            "dimensions": (rows, cols),
        }
        try:
            proc = spawn(dispatch_argv, **kwargs)
        except TypeError:
            try:
                proc = spawn(command_line, **kwargs)
            except TypeError:
                proc = spawn(command_line, cwd=str(cwd), dimensions=(rows, cols))
        return cls(proc, executable_snapshot)

    def read(self, *, timeout_s: float = 0.0) -> str:
        read = getattr(self.proc, "read")
        try:
            return str(read(timeout=timeout_s))
        except TypeError:
            try:
                return str(read())
            except EOFError:
                return ""
        except EOFError:
            return ""

    def write(self, text: str) -> int:
        write = getattr(self.proc, "write")
        result = write(text)
        return len(text) if result is None else int(result)

    def resize(self, cols: int, rows: int) -> None:
        resize = getattr(self.proc, "setwinsize", None)
        if callable(resize):
            resize(rows, cols)

    def is_alive(self) -> bool:
        isalive = getattr(self.proc, "isalive", None) or getattr(self.proc, "is_alive", None)
        if callable(isalive):
            return bool(isalive())
        return self.exit_code() is None

    def exit_code(self) -> int | None:
        for name in ("exitstatus", "returncode"):
            value = getattr(self.proc, name, None)
            if value is not None:
                return int(value)
        return None

    def close(self, *, force: bool = True, timeout_s: float = 2.0) -> int | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        terminate = getattr(self.proc, "terminate", None)
        close = getattr(self.proc, "close", None)
        if self.is_alive():
            if callable(terminate):
                try:
                    terminate(force=force)
                except TypeError:
                    terminate()
                except (OSError, PermissionError):
                    if callable(close):
                        close()
            elif callable(close):
                close()
        wait = getattr(self.proc, "wait", None)
        if callable(wait):
            try:
                wait(timeout=max(0.0, deadline - time.monotonic()))
            except TypeError:
                while self.is_alive() and time.monotonic() < deadline:
                    time.sleep(
                        min(0.02, max(0.0, deadline - time.monotonic()))
                    )
        else:
            while self.is_alive() and time.monotonic() < deadline:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        if self.is_alive():
            raise TimeoutError("PTY process did not exit within the close deadline")
        if self.executable_snapshot is not None:
            self.executable_snapshot.close()
            self.executable_snapshot = None
        return self.exit_code()


def _safe_subprocess_env(*, path: str, home: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key.upper() in _SAFE_PTY_ENV_KEYS}
    env["PATH"] = path
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
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


@dataclass
class PtyCreateResult:
    session_oid: str
    namespace: str
    name: str
    type: str
    alive: bool
    output: str
    output_truncated: bool
    dropped_chars: int


@dataclass
class PtyReadResult:
    session_oid: str
    output: str
    output_truncated: bool
    alive: bool
    exit_code: int | None
    dropped_chars: int


@dataclass
class PtyWriteResult:
    session_oid: str
    bytes_written: int
    alive: bool


@dataclass
class PtyResizeResult:
    session_oid: str
    cols: int
    rows: int
    alive: bool


@dataclass
class PtyCloseResult:
    session_oid: str
    closed: bool
    exit_code: int | None


@dataclass
class PtySessionListEntry:
    session_oid: str
    name: str
    namespace: str
    argv: list[str]
    cwd: str
    backend: str
    alive: bool
    exit_code: int | None
    cols: int
    rows: int
    dropped_chars: int


@dataclass
class _PtyRuntimeSession:
    session_oid: str
    session_id: str
    owner_pid: str
    argv: list[str]
    cwd: str
    backend: str
    handle: PtySession
    cols: int
    rows: int
    started_at: str
    started_monotonic: float
    buffer_max_chars: int
    data_sink: DataSink
    data_flow_context: DataFlowContext
    buffer: deque[str] = field(default_factory=deque)
    buffer_chars: int = 0
    dropped_chars: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    control_lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    close_complete: threading.Event = field(default_factory=threading.Event)
    reader_thread: threading.Thread | None = None
    monitor_thread: threading.Thread | None = None
    closing: bool = False
    closed: bool = False
    close_outcome_unknown: bool = False
    host_only_orphan: bool = False
    public_session_oid: str | None = None
    recovery_abandoning: bool = False
    exit_code: int | None = None
    last_wall_seconds: float = 0.0
    last_cpu_seconds: float = 0.0
    last_peak_memory_bytes: int = 0
    cpu_seconds_by_process: dict[tuple[int, float], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.close_complete.set()


@dataclass(frozen=True)
class _PtyResourceObservation:
    wall_seconds: float
    cpu_seconds_by_process: dict[tuple[int, float], float]
    memory_bytes: int
    sampling_error: Exception | None


@dataclass(frozen=True)
class _PtyResourceDelta:
    wall_seconds: float
    cpu_seconds: float
    memory_bytes: int
    peak_changed: bool

    @property
    def changed(self) -> bool:
        return self.wall_seconds > 0 or self.cpu_seconds > 0 or self.peak_changed


@dataclass(frozen=True)
class _PtyClosePlan:
    session_oid: str
    session: _PtyRuntimeSession
    actor: str
    reason: str
    force: bool
    timeout_s: float
    authority_decision: CapabilityDecision | None
    data_sink: DataSink | None
    data_flow_context: DataFlowContext | None
    data_flow_payload: dict[str, Any] | None
    effect_context: dict[str, Any]


@dataclass
class _PtyCloseMutationState:
    attempted: bool = False


@dataclass
class _PtyCreatePlan:
    pid: str
    argv: list[str]
    dispatch_argv: list[str]
    cwd: str
    cols: int
    rows: int
    startup_timeout_s: float
    output_chars: int
    name: str | None
    resource: str
    decision: ShellPolicyDecision
    capability_decision: CapabilityDecision
    spawn_sink: DataSink
    executable_identity: str
    flow_context: DataFlowContext
    data_flow_payload: dict[str, Any]
    limits: SubprocessLimits | None
    intent_record: Any
    session_id: str
    effect_context: dict[str, Any]


@dataclass
class _PtySpawnGuards:
    provider_argv: list[str]
    failure_phase: str = "provider_resolve"
    dispatch_pinned: bool = False
    cleanup_attempted: bool = False
    cleanup_succeeded: bool = False
    cleanup_retained: bool = False

    def cleanup_evidence(self) -> dict[str, bool]:
        return {
            "attempted": self.cleanup_attempted,
            "succeeded": self.cleanup_succeeded,
            "retained": self.cleanup_retained,
        }


@dataclass
class _PtySpawnResources:
    executable_snapshot: ExecutableSnapshot | None = None
    handle: PtySession | None = None
    session_oid: str | None = None


@dataclass
class _PtyStartedSession:
    plan: _PtyCreatePlan
    protected_cm: Any
    protected: Any
    session: _PtyRuntimeSession
    namespace: str
    object_name: str


class PtyAdapter:
    """Object-bound PTY primitive."""

    def __init__(
        self,
        host: "ModuleHost",
        shell_policy: ShellExecutionPolicy,
        human: Any,
        audit: AuditPort,
        events: EventPort,
        provider: PtyProvider,
        *,
        config: PtyModuleConfig | None = None,
        resources: Any | None = None,
    ) -> None:
        self.host = host
        self.shell_policy = shell_policy
        self.human = human
        self.audit = audit
        self.events = events
        self.provider = provider
        self.config = config or PtyModuleConfig()
        self.resources = resources
        self._sessions: dict[str, _PtyRuntimeSession] = {}
        self._pending_session_creates = 0
        self._pending_session_creates_by_process: dict[str, int] = {}
        self._lock = threading.RLock()
        self._worker_condition = threading.Condition(self._lock)
        self._active_worker_threads: set[threading.Thread] = set()

    def create(
        self,
        pid: str,
        argv: list[str],
        *,
        cwd: str,
        cols: int | None = None,
        rows: int | None = None,
        startup_timeout_s: float | None = None,
        max_output_chars: int | None = None,
        name: str | None = None,
        source_oids: Iterable[str] | None = None,
    ) -> PtyCreateResult:
        plan = self._prepare_create_plan(
            pid,
            argv,
            cwd=cwd,
            cols=cols,
            rows=rows,
            startup_timeout_s=startup_timeout_s,
            max_output_chars=max_output_chars,
            name=name,
            source_oids=source_oids,
        )
        started = self._spawn_prepared_session(plan)
        return self._complete_started_session(started)

    def _prepare_create_plan(
        self,
        pid: str,
        argv: list[str],
        *,
        cwd: str,
        cols: int | None,
        rows: int | None,
        startup_timeout_s: float | None,
        max_output_chars: int | None,
        name: str | None,
        source_oids: Iterable[str] | None,
    ) -> _PtyCreatePlan:
        (
            checked,
            selected_cols,
            selected_rows,
            selected_timeout,
            selected_output_chars,
        ) = self._validate_create_request(
            argv,
            cols=cols,
            rows=rows,
            startup_timeout_s=startup_timeout_s,
            max_output_chars=max_output_chars,
        )
        resource, dispatch_argv = self._prepare_spawn_argv(checked, cwd=cwd)
        decision = self._authorize_pty_spawn(
            pid,
            checked,
            resource,
            timeout_s=selected_timeout,
            cwd=cwd,
        )
        if not decision.allowed and not decision.ask_human:
            raise CapabilityDenied(
                f"{pid} denied pty spawn on {resource}: {decision.reason}"
            )
        spawn_sink = self.shell_policy.executable_data_sink(
            "pty:spawn",
            checked[0],
            cwd=cwd,
        )
        executable_identity = spawn_sink.identity.split("pty:spawn:", 1)[1]
        flow_context = self.host.data_flow.context_from_source_oids(
            pid,
            source_oids,
        )
        data_flow_payload = {
            "argv": [executable_identity, *dispatch_argv[1:]],
            "cwd": cwd,
        }
        self.host.data_flow.authorize_egress(
            pid=pid,
            sink=spawn_sink,
            context=flow_context,
            payload=data_flow_payload,
            operation="pty.spawn",
        )
        if decision.ask_human:
            self._request_human_approval(
                pid,
                checked,
                resource,
                decision,
                timeout=selected_timeout,
                cwd=cwd,
                source_oids=source_oids,
            )
        if not decision.allowed:
            raise CapabilityDenied(
                f"{pid} denied pty spawn on {resource}: {decision.reason}"
            )
        limits = self._pty_subprocess_limits(pid)
        intent_record = self._record_spawn_intent(
            pid,
            resource,
            checked,
            decision,
            cwd=cwd,
            cols=selected_cols,
            rows=selected_rows,
        )
        capability_decision = decision.authority_decision
        if capability_decision is None or not capability_decision.allowed:
            raise CapabilityDenied(
                "allowed PTY policy decision is missing its capability authority"
            )
        session_id = new_id("pty")
        effect_context = {
            "argv": list(checked),
            "provider_argv": list(data_flow_payload["argv"]),
            "resource": resource,
            "cwd": cwd,
            "cols": selected_cols,
            "rows": selected_rows,
            "session_id": session_id,
        }
        return _PtyCreatePlan(
            pid=pid,
            argv=checked,
            dispatch_argv=dispatch_argv,
            cwd=cwd,
            cols=selected_cols,
            rows=selected_rows,
            startup_timeout_s=selected_timeout,
            output_chars=selected_output_chars,
            name=name,
            resource=resource,
            decision=decision,
            capability_decision=capability_decision,
            spawn_sink=spawn_sink,
            executable_identity=executable_identity,
            flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            limits=limits,
            intent_record=intent_record,
            session_id=session_id,
            effect_context=effect_context,
        )

    def _prepare_spawn_argv(
        self,
        checked: list[str],
        *,
        cwd: str,
    ) -> tuple[str, list[str]]:
        resource = self.shell_policy.resource_for(checked)
        dispatch_argv = self.shell_policy.harden_read_only_git_argv(checked)
        self.shell_policy.enforce_workspace_argv_scope(checked, cwd=cwd)
        return resource, dispatch_argv

    def _validate_create_request(
        self,
        argv: list[str],
        *,
        cols: int | None,
        rows: int | None,
        startup_timeout_s: float | None,
        max_output_chars: int | None,
    ) -> tuple[list[str], int, int, float, int]:
        checked = self.shell_policy.validate_argv(argv)
        selected_cols, selected_rows = self._validate_size(cols, rows)
        selected_timeout = self._validate_timeout(
            startup_timeout_s,
            default=self.config.pty.startup_timeout_s,
            hard_limit=self.config.pty.startup_timeout_hard_limit_s,
            label="pty startup timeout",
        )
        selected_output_chars = self._validate_char_limit(
            max_output_chars,
            default=self.config.pty.startup_output_max_chars,
            hard_limit=self.config.pty.read_hard_limit_chars,
            label="pty max_output_chars",
        )
        return (
            checked,
            selected_cols,
            selected_rows,
            selected_timeout,
            selected_output_chars,
        )

    def _pty_subprocess_limits(self, pid: str) -> SubprocessLimits | None:
        limits = self.shell_policy.subprocess_limits(pid)
        supports_limits = bool(
            getattr(self.provider, "supports_subprocess_limits", False)
        )
        if limits is not None and not supports_limits:
            raise ValidationError(
                "PTY provider must explicitly support SubprocessLimits before "
                "budgeted execution"
            )
        return limits

    def _authorize_pty_spawn(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        *,
        timeout_s: float,
        cwd: str,
    ) -> ShellPolicyDecision:
        return self.shell_policy.authorize_operation(
            pid,
            argv,
            resource,
            timeout=timeout_s,
            cwd=cwd,
            adapter="pty",
            primitive="runtime.pty.spawn",
            operation="pty.spawn",
            authority_operation="pty.spawn",
            include_timeout_in_authority=False,
            continuous_session=True,
            extra_context={"startup_timeout_s": timeout_s},
        )

    def _spawn_invocation(
        self,
        plan: _PtyCreatePlan,
        guards: _PtySpawnGuards,
    ) -> ProtectedOperationInvocation:
        def revalidate_spawn_authority() -> tuple[CapabilityDecision, ...]:
            current = self._authorize_pty_spawn(
                plan.pid,
                plan.argv,
                plan.resource,
                timeout_s=plan.startup_timeout_s,
                cwd=plan.cwd,
            )
            authority = current.authority_decision
            if not current.allowed or current.ask_human or authority is None:
                raise CapabilityDenied(
                    f"{plan.pid} PTY authority changed before provider dispatch: "
                    f"{current.reason}"
                )
            return (authority,)

        def revalidate_spawn_sink() -> DataSink:
            if guards.dispatch_pinned:
                return plan.spawn_sink
            return self.shell_policy.executable_data_sink(
                "pty:spawn",
                guards.provider_argv[0],
                cwd=plan.cwd,
            )

        canonical_args = self.shell_policy.operation_context(
            plan.pid,
            plan.argv,
            plan.resource,
            timeout=plan.startup_timeout_s,
            cwd=plan.cwd,
            profile=plan.decision.sandbox_profile,
            adapter="pty",
            primitive="runtime.pty.spawn",
            operation="pty.spawn",
            authority_operation="pty.spawn",
            include_timeout=False,
            continuous_session=True,
            extra={"startup_timeout_s": plan.startup_timeout_s},
        )
        return ProtectedOperationInvocation(
            pid=plan.pid,
            actor=plan.pid,
            target=f"pty:{plan.session_id}",
            decisions=(plan.capability_decision,),
            canonical_args=canonical_args,
            observation=plan.effect_context,
            data_sink=plan.spawn_sink,
            data_sink_revalidator=revalidate_spawn_sink,
            data_flow_context=plan.flow_context,
            data_flow_ingress_context=self._pty_ingress_context(plan.flow_context),
            data_flow_payload=plan.data_flow_payload,
            data_flow_operation="pty.spawn",
            authority_revalidator=revalidate_spawn_authority,
            failure_evidence=lambda error, phase: self._spawn_failure_evidence(
                plan,
                guards,
                error,
                phase,
            ),
        )

    def _spawn_failure_evidence(
        self,
        plan: _PtyCreatePlan,
        guards: _PtySpawnGuards,
        error: BaseException,
        phase: str,
    ) -> ProtectedOperationEvidence:
        selected_phase = (
            guards.failure_phase
            if phase == "caller_failed_after_provider"
            else phase
        )
        evidence = self._protected_pty_failure_evidence(
            plan.pid,
            plan.session_id,
            "spawn",
            error,
            selected_phase,
            correlation_id=plan.intent_record.record_id,
            parent_record_id=plan.intent_record.record_id,
        )
        return replace(
            evidence,
            effect_metadata={
                **dict(evidence.effect_metadata),
                "cleanup": guards.cleanup_evidence(),
            },
        )

    def _spawn_prepared_session(
        self,
        plan: _PtyCreatePlan,
    ) -> _PtyStartedSession:
        guards = _PtySpawnGuards(provider_argv=list(plan.dispatch_argv))
        resources = _PtySpawnResources()
        protected_cm: Any | None = None
        protected: Any | None = None
        reserved_capacity = False
        self._reserve_session_capacity(plan.pid)
        reserved_capacity = True
        try:
            invocation = self._spawn_invocation(plan, guards)
            protected_cm = self._protected().start(
                "primitive.pty.spawn",
                invocation,
                provider=self.provider,
            )
            protected = protected_cm.__enter__()
            object_name, namespace = self._dispatch_prepared_spawn(
                plan,
                guards,
                resources,
                protected,
            )
            if resources.handle is None or resources.session_oid is None:
                raise ValidationError("PTY provider spawn did not produce a session")
            session = self._runtime_session_from_spawn(plan, resources)
            with self._lock:
                self._sessions[session.session_oid] = session
                self._release_session_capacity_locked(plan.pid)
                reserved_capacity = False
            return _PtyStartedSession(
                plan=plan,
                protected_cm=protected_cm,
                protected=protected,
                session=session,
                namespace=namespace,
                object_name=object_name,
            )
        except BaseException as error:
            error_info = sys.exc_info()
            orphan_session = getattr(error, "orphan_session", None)
            if resources.handle is None and orphan_session is not None:
                resources.handle = orphan_session
            self._cleanup_failed_spawn(plan, guards, resources, protected)
            if reserved_capacity:
                self._release_session_capacity(plan.pid)
            if protected_cm is not None and protected is not None:
                protected_cm.__exit__(*error_info)
            raise

    def _dispatch_prepared_spawn(
        self,
        plan: _PtyCreatePlan,
        guards: _PtySpawnGuards,
        resources: _PtySpawnResources,
        protected: Any,
    ) -> tuple[str, str]:
        resolver = getattr(self.provider, "resolve_argv", None)
        if callable(resolver):
            guards.provider_argv = protected.call(
                ProviderPhase("resolve_argv", information_flow=True),
                self.shell_policy.resolve_provider_argv,
                guards.provider_argv,
                cwd=plan.cwd,
                provider=self.provider,
            )
        guards.failure_phase = "provider_identity_validation"
        self.shell_policy.require_provider_executable_identity(
            guards.provider_argv[0],
            expected=plan.executable_identity,
            cwd=plan.cwd,
        )
        plan.effect_context["provider_argv"] = list(guards.provider_argv)
        guards.failure_phase = "provider_spawn"
        resources.executable_snapshot = (
            self.shell_policy.snapshot_executable_for_dispatch(
                pid=plan.pid,
                provider=self.provider,
                requested_argv0=plan.argv[0],
                provider_argv0=guards.provider_argv[0],
                cwd=plan.cwd,
                expected_sink=plan.spawn_sink,
                expected_executable_identity=plan.executable_identity,
                flow_context=plan.flow_context,
                data_flow_payload=plan.data_flow_payload,
            )
        )
        spawn_kwargs: dict[str, Any] = {
            "cwd": plan.cwd,
            "cols": plan.cols,
            "rows": plan.rows,
            "limits": plan.limits,
        }
        if resources.executable_snapshot is not None:
            spawn_kwargs["executable_snapshot"] = resources.executable_snapshot
        resources.handle = protected.call(
            ProviderPhase("spawn", state_mutation=True, information_flow=True),
            self.provider.spawn,
            guards.provider_argv,
            **spawn_kwargs,
        )
        guards.dispatch_pinned = True
        # The returned session owns the executable snapshot after spawn.
        resources.executable_snapshot = None
        guards.failure_phase = "session_object_creation"
        session_oid, object_name, namespace = self._create_session_object(
            plan.pid,
            session_id=plan.session_id,
            argv=plan.argv,
            cwd=plan.cwd,
            backend=resources.handle.backend,
            cols=plan.cols,
            rows=plan.rows,
            name=plan.name,
            data_flow_context=plan.flow_context,
        )
        resources.session_oid = session_oid
        return object_name, namespace

    def _runtime_session_from_spawn(
        self,
        plan: _PtyCreatePlan,
        resources: _PtySpawnResources,
    ) -> _PtyRuntimeSession:
        assert resources.handle is not None
        assert resources.session_oid is not None
        return _PtyRuntimeSession(
            session_oid=resources.session_oid,
            session_id=plan.session_id,
            owner_pid=plan.pid,
            argv=list(plan.argv),
            cwd=plan.cwd,
            backend=resources.handle.backend,
            handle=resources.handle,
            cols=plan.cols,
            rows=plan.rows,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
            buffer_max_chars=self.config.pty.buffer_max_chars,
            data_sink=plan.spawn_sink,
            data_flow_context=plan.flow_context,
        )

    def _cleanup_failed_spawn(
        self,
        plan: _PtyCreatePlan,
        guards: _PtySpawnGuards,
        resources: _PtySpawnResources,
        protected: Any | None,
    ) -> None:
        public_session_oid = resources.session_oid
        if resources.executable_snapshot is not None:
            resources.executable_snapshot.close()
        if resources.handle is not None:
            try:
                guards.cleanup_attempted = True
                # Predeclare the fallback owner so protected failure evidence
                # produced inside cleanup_close records the final containment
                # strategy. A successful close clears it below.
                guards.cleanup_retained = True
                if protected is None or protected.terminal:
                    raise RuntimeError(
                        "failed PTY spawn cleanup requires a new lifecycle owner"
                    )
                protected.call(
                    ProviderPhase("cleanup_close", state_mutation=True),
                    resources.handle.close,
                    force=True,
                    timeout_s=self.config.pty.close_timeout_s,
                )
                alive = protected.call(
                    ProviderPhase("cleanup_status", information_flow=True),
                    resources.handle.is_alive,
                )
                if alive:
                    raise TimeoutError(
                        "failed PTY spawn cleanup returned while the session remained alive"
                    )
                guards.cleanup_succeeded = True
                guards.cleanup_retained = False
            except BaseException:
                self._retain_failed_spawn(plan, resources)
        if public_session_oid is not None:
            try:
                self.host.memory.delete_object_trusted(
                    plan.pid,
                    public_session_oid,
                    reason="pty_create_pre_registration_failure",
                )
            except Exception:
                pass

    def _retain_failed_spawn(
        self,
        plan: _PtyCreatePlan,
        resources: _PtySpawnResources,
    ) -> None:
        assert resources.handle is not None
        public_session_oid = resources.session_oid
        orphan_oid = f"orphan:{plan.session_id}"
        resources.session_oid = orphan_oid
        session = _PtyRuntimeSession(
            session_oid=orphan_oid,
            session_id=plan.session_id,
            owner_pid=plan.pid,
            argv=list(plan.argv),
            cwd=plan.cwd,
            backend=resources.handle.backend,
            handle=resources.handle,
            cols=plan.cols,
            rows=plan.rows,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
            buffer_max_chars=self.config.pty.buffer_max_chars,
            data_sink=plan.spawn_sink,
            data_flow_context=plan.flow_context,
            host_only_orphan=True,
            public_session_oid=public_session_oid,
        )
        session.stop_event.set()
        with self._lock:
            self._sessions[orphan_oid] = session

    def _complete_started_session(
        self,
        started: _PtyStartedSession,
    ) -> PtyCreateResult:
        plan = started.plan
        session = started.session
        try:
            parent_operation_id = self.host.operations.current_id()
            self._start_reader(
                session,
                resource=plan.resource,
                parent_operation_id=parent_operation_id,
            )
            self._start_monitor(session, resource=plan.resource)
            if plan.startup_timeout_s > 0:
                time.sleep(plan.startup_timeout_s)
            output, output_truncated = self._take_output(
                session,
                plan.output_chars,
            )
            result = PtyCreateResult(
                session_oid=session.session_oid,
                namespace=started.namespace,
                name=started.object_name,
                type=ObjectType.EXTERNAL_REF.value,
                alive=started.protected.call(
                    ProviderPhase("spawn_status", information_flow=True),
                    session.handle.is_alive,
                ),
                output=output,
                output_truncated=output_truncated,
                dropped_chars=session.dropped_chars,
            )
            result_payload = {
                "session_oid": session.session_oid,
                "backend": session.backend,
            }
            started.protected.complete(
                result,
                self._protected_pty_evidence(
                    plan.pid,
                    session.session_oid,
                    "spawn",
                    {
                        "argv": plan.argv,
                        "cwd": plan.cwd,
                        "resource": plan.resource,
                        "backend": session.backend,
                        "policy_level": plan.decision.policy_level,
                        "policy_reason": plan.decision.reason,
                        "risk": plan.decision.risk.value,
                        "rule_id": plan.decision.rule_id,
                        "cols": plan.cols,
                        "rows": plan.rows,
                    },
                    output_refs=(session.session_oid,),
                    correlation_id=plan.intent_record.record_id,
                    parent_record_id=plan.intent_record.record_id,
                ),
                classification_context={
                    **plan.effect_context,
                    "backend": session.backend,
                    "session_oid": session.session_oid,
                },
                classification_result=result_payload,
            )
            started.protected_cm.__exit__(None, None, None)
            return result
        except BaseException:
            error_info = sys.exc_info()
            self._cleanup_failed_started_session(
                session,
                actor=plan.pid,
                reason="pty_create_post_spawn_failure",
            )
            started.protected_cm.__exit__(*error_info)
            raise

    def read(self, pid: str, session_oid: str, *, timeout_s: float | None = None, max_chars: int | None = None) -> PtyReadResult:
        selected_timeout = self._validate_timeout(
            timeout_s,
            default=self.config.pty.read_timeout_s,
            hard_limit=self.config.pty.read_timeout_hard_limit_s,
            label="pty read timeout",
        )
        selected_max_chars = self._validate_char_limit(
            max_chars,
            default=self.config.pty.read_max_chars,
            hard_limit=self.config.pty.read_hard_limit_chars,
            label="pty max_chars",
        )
        authority = self._require_object_right(
            pid, session_oid, ObjectRight.READ.value, consume=False
        )
        session = self._require_session(session_oid)
        effect_context = {
            "session_oid": session_oid,
            "backend": session.backend,
            "timeout_s": selected_timeout,
            "max_chars": selected_max_chars,
        }
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=f"pty:{session_oid}",
            decisions=(authority,),
            canonical_args=effect_context,
            observation=effect_context,
            data_flow_ingress_context=self._session_ingress_context(session),
            failure_evidence=lambda error, phase: self._protected_pty_failure_evidence(
                pid, session_oid, "read", error, phase
            ),
        )
        with self._protected().start("primitive.pty.read", invocation, provider=self.provider) as protected:

            def read_buffer() -> tuple[str, bool, bool, int | None, int]:
                if selected_timeout > 0:
                    deadline = time.monotonic() + selected_timeout
                    while (
                        time.monotonic() < deadline
                        and self._buffer_is_empty(session)
                        and self._session_alive(session)
                    ):
                        time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
                # A control operation may have produced output before its
                # success/ambiguous settlement publishes a higher session
                # label. Serialize only final extraction (not the potentially
                # blocking wait above) with that publication boundary.
                with session.control_lock:
                    self._require_session_open(session)
                    self.host.data_flow.observe_ingress(
                        self._session_ingress_context(session)
                    )
                    output, truncated = self._take_output(session, selected_max_chars)
                    with session.lock:
                        dropped_chars = session.dropped_chars
                    return (
                        output,
                        truncated,
                        self._session_alive(session),
                        self._session_exit_code(session),
                        dropped_chars,
                    )

            output, truncated, alive, exit_code, dropped_chars = protected.call(
                ProviderPhase("read", information_flow=True), read_buffer
            )
            result = PtyReadResult(
                session_oid=session_oid,
                output=output,
                output_truncated=truncated,
                alive=alive,
                exit_code=exit_code,
                dropped_chars=dropped_chars,
            )
            result_payload = {
                "chars": len(output),
                "truncated": truncated,
                "alive": alive,
                "exit_code": exit_code,
            }
            completed = protected.complete(
                result,
                self._protected_pty_evidence(
                    pid, session_oid, "read", result_payload, input_refs=(session_oid,)
                ),
                classification_context=effect_context,
                classification_result=result_payload,
            )
            return completed

    def write(self, pid: str, session_oid: str, text: str) -> PtyWriteResult:
        if "\x00" in text:
            raise ValidationError("pty input cannot contain NUL bytes")
        if len(text) > self.config.pty.input_max_chars:
            raise ValidationError(f"pty input exceeds configured limit {self.config.pty.input_max_chars} chars")
        if len(text) > self.config.pty.input_hard_limit_chars:
            raise ValidationError(f"pty input exceeds hard limit {self.config.pty.input_hard_limit_chars} chars")
        session = self._require_session(session_oid)
        if session.owner_pid != pid:
            raise CapabilityDenied(f"{pid} cannot write to PTY session owned by {session.owner_pid}")
        authority = self._require_object_right(
            pid,
            session_oid,
            ObjectRight.WRITE.value,
            consume=False,
        )
        with session.control_lock:
            return self._write_with_session_control(
                pid,
                session_oid,
                text,
                session=session,
                authority=authority,
            )

    def _write_with_session_control(
        self,
        pid: str,
        session_oid: str,
        text: str,
        *,
        session: _PtyRuntimeSession,
        authority: CapabilityDecision,
    ) -> PtyWriteResult:
        self._require_session_open(session)
        effect_context = {
            "session_oid": session_oid,
            "backend": session.backend,
            "chars": len(text),
            "cwd": session.cwd,
        }
        flow_context = self._session_egress_context(session)
        session_sink = self._session_sink(session)
        mutation_attempted = False

        def write_provider() -> int:
            nonlocal mutation_attempted
            mutation_attempted = True
            return session.handle.write(text)

        def settle_ambiguous_write(error: BaseException, _phase: str) -> None:
            if not mutation_attempted or isinstance(error, ProviderEffectNotStarted):
                return
            self._raise_session_flow(session, flow_context)

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=f"pty:{session_oid}",
            decisions=(authority,),
            canonical_args={"session_oid": session_oid, "text": text},
            observation=effect_context,
            data_sink=session_sink,
            data_flow_context=flow_context,
            data_flow_payload={"session_id": session.session_id, "text": text},
            data_flow_operation="pty.write",
            failure_evidence=lambda error, phase: self._protected_pty_failure_evidence(
                pid, session_oid, "write", error, phase
            ),
            failure_settlement=settle_ambiguous_write,
        )
        with self._protected().start("primitive.pty.write", invocation, provider=self.provider) as protected:
            bytes_written = protected.call(
                ProviderPhase("write", state_mutation=True, information_flow=True),
                write_provider,
            )
            alive = protected.call(
                ProviderPhase("status", information_flow=True),
                self._session_alive,
                session,
            )
            result = PtyWriteResult(
                session_oid=session_oid, bytes_written=bytes_written, alive=alive
            )
            result_payload = {
                "chars": len(text),
                "bytes_written": bytes_written,
                "alive": alive,
            }
            return protected.complete(
                result,
                self._protected_pty_evidence(
                    pid, session_oid, "write", result_payload, input_refs=(session_oid,)
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                settle_success=lambda: self._raise_session_flow(
                    session,
                    flow_context,
                ),
            )

    def resize(self, pid: str, session_oid: str, *, cols: int, rows: int) -> PtyResizeResult:
        selected_cols, selected_rows = self._validate_size(cols, rows)
        authority = self._require_object_right(
            pid,
            session_oid,
            ObjectRight.WRITE.value,
            consume=False,
        )
        session = self._require_session(session_oid)
        with session.control_lock:
            return self._resize_with_session_control(
                pid,
                session_oid,
                selected_cols=selected_cols,
                selected_rows=selected_rows,
                session=session,
                authority=authority,
            )

    def _resize_with_session_control(
        self,
        pid: str,
        session_oid: str,
        *,
        selected_cols: int,
        selected_rows: int,
        session: _PtyRuntimeSession,
        authority: CapabilityDecision,
    ) -> PtyResizeResult:
        self._require_session_open(session)
        with session.lock:
            previous_cols = session.cols
            previous_rows = session.rows
        effect_context = {
            "session_oid": session_oid,
            "backend": session.backend,
            "previous_cols": previous_cols,
            "previous_rows": previous_rows,
            "cols": selected_cols,
            "rows": selected_rows,
        }
        flow_context = self._session_egress_context(session)
        session_sink = self._session_sink(session)
        data_flow_payload = {
            "session_id": session.session_id,
            "cols": selected_cols,
            "rows": selected_rows,
        }
        mutation_attempted = False

        def resize_provider() -> None:
            nonlocal mutation_attempted
            mutation_attempted = True
            session.handle.resize(selected_cols, selected_rows)

        def settle_ambiguous_resize(error: BaseException, _phase: str) -> None:
            if not mutation_attempted or isinstance(error, ProviderEffectNotStarted):
                return
            self._raise_session_flow(session, flow_context)

        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=f"pty:{session_oid}",
            decisions=(authority,),
            canonical_args=effect_context,
            observation=effect_context,
            data_sink=session_sink,
            data_flow_context=flow_context,
            data_flow_payload=data_flow_payload,
            data_flow_operation="pty.resize",
            failure_evidence=lambda error, phase: self._protected_pty_failure_evidence(
                pid, session_oid, "resize", error, phase
            ),
            failure_settlement=settle_ambiguous_resize,
        )
        with self._protected().start("primitive.pty.resize", invocation, provider=self.provider) as protected:
            protected.call(
                ProviderPhase("resize", state_mutation=True, information_flow=True),
                resize_provider,
            )

            def settle_success() -> None:
                with session.lock:
                    session.cols = selected_cols
                    session.rows = selected_rows
                self._raise_session_flow(session, flow_context)

            alive = protected.call(
                ProviderPhase("status", information_flow=True),
                self._session_alive,
                session,
            )
            result = PtyResizeResult(
                session_oid=session_oid, cols=selected_cols, rows=selected_rows, alive=alive
            )
            result_payload = {"cols": selected_cols, "rows": selected_rows, "alive": alive}
            return protected.complete(
                result,
                self._protected_pty_evidence(
                    pid, session_oid, "resize", result_payload, input_refs=(session_oid,)
                ),
                classification_context=effect_context,
                classification_result=result_payload,
                settle_success=settle_success,
            )

    def close(
        self,
        pid: str,
        session_oid: str,
        *,
        force: bool = True,
        timeout_s: float | None = None,
    ) -> PtyCloseResult:
        selected_timeout = self._validate_timeout(
            timeout_s,
            default=self.config.pty.close_timeout_s,
            hard_limit=self.config.pty.close_timeout_hard_limit_s,
            label="pty close timeout",
        )
        authority = self._require_object_right(
            pid,
            session_oid,
            ObjectRight.DELETE.value,
            consume=False,
        )
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            return self._close_with_session_control(
                pid,
                session_oid,
                force=force,
                selected_timeout=selected_timeout,
                authority=authority,
                session=None,
            )
        with session.control_lock:
            return self._close_with_session_control(
                pid,
                session_oid,
                force=force,
                selected_timeout=selected_timeout,
                authority=authority,
                session=session,
            )

    def _close_with_session_control(
        self,
        pid: str,
        session_oid: str,
        *,
        force: bool,
        selected_timeout: float,
        authority: CapabilityDecision,
        session: _PtyRuntimeSession | None,
    ) -> PtyCloseResult:
        session_missing_at_entry = session is None
        if session is not None:
            with self._lock:
                if self._sessions.get(session_oid) is not session:
                    session = None
        data_sink: DataSink | None = None
        data_flow_context: DataFlowContext | None = None
        data_flow_payload: dict[str, Any] | None = None
        if session is not None:
            data_sink = self._session_sink(session)
            data_flow_context = self._session_egress_context(session)
            data_flow_payload = {
                "session_id": session.session_id,
                "force": force,
                "timeout_s": selected_timeout,
            }
            # Public close sets the reader stop flag before the protected SDK
            # enters, so reject impossible egress before mutating session state.
            self.host.data_flow.authorize_egress(
                pid=pid,
                sink=data_sink,
                context=data_flow_context,
                payload=data_flow_payload,
                operation="pty.close",
            )
        exit_code = self._close_session(
            session_oid,
            actor=pid,
            reason="pty_close",
            force=force,
            timeout_s=selected_timeout,
            wait_if_closing=True,
            # A provider session that was already absent at public-close
            # entry is the natural-exit path and still has to settle finite
            # DELETE authority.  If a session disappeared while this caller
            # waited on its control lock, another public closer already
            # crossed and settled the provider boundary; replaying its stale
            # pre-lock decision would spuriously fail an idempotent close.
            authority_decision=(
                authority if session is not None or session_missing_at_entry else None
            ),
            data_sink=data_sink,
            data_flow_context=data_flow_context,
            data_flow_payload=data_flow_payload,
        )
        self.host.memory.delete_object_trusted(pid, session_oid, reason="pty_close")
        return PtyCloseResult(session_oid=session_oid, closed=True, exit_code=exit_code)

    def list(self, pid: str) -> list[PtySessionListEntry]:
        entries: list[PtySessionListEntry] = []
        returned_contexts: list[DataFlowContext] = []
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            obj = self.host.store.get_object(session.session_oid)
            if obj is None:
                continue
            try:
                self._require_object_right(pid, session.session_oid, ObjectRight.READ.value)
            except (CapabilityDenied, NotFound):
                continue
            with session.lock:
                alive = not session.closed and session.exit_code is None
                returned_contexts.append(session.data_flow_context)
                entries.append(
                    PtySessionListEntry(
                        session_oid=session.session_oid,
                        name=obj.name,
                        namespace=obj.namespace,
                        argv=list(session.argv),
                        cwd=session.cwd,
                        backend=session.backend,
                        alive=alive,
                        exit_code=session.exit_code,
                        cols=session.cols,
                        rows=session.rows,
                        dropped_chars=session.dropped_chars,
                    )
                )
        if returned_contexts:
            self.host.data_flow.observe_ingress(
                DataFlowContext.aggregate(returned_contexts)
            )
        self.audit.record(actor=pid, action="primitive.pty.list", target="pty:*", decision={"count": len(entries)})
        return entries

    def close_for_object_release(self, oid: str, *, actor: str, reason: str) -> None:
        if oid not in self._sessions:
            return
        self._close_session(
            oid,
            actor=actor,
            reason=f"object_release:{reason}",
            force=True,
            timeout_s=self.config.pty.close_timeout_s,
            wait_if_closing=True,
        )

    def _cleanup_failed_started_session(self, session: _PtyRuntimeSession, *, actor: str, reason: str) -> None:
        public_session_oid = self._fence_failed_started_session(session)
        try:
            self._close_session(
                session.session_oid,
                actor=actor,
                reason=reason,
                force=True,
                timeout_s=self.config.pty.close_timeout_s,
                wait_if_closing=True,
            )
        except Exception:
            # The close phase already crossed a protected provider boundary.
            # An ambiguous result must not be retried blindly; keep the session
            # tracked so shutdown/reconciliation evidence is not replaced by a
            # false local-cleanup claim.
            pass
        try:
            self.host.memory.delete_object_trusted(
                "runtime.pty",
                public_session_oid,
                reason=reason,
            )
        except Exception:
            pass

    def _fence_failed_started_session(
        self,
        session: _PtyRuntimeSession,
    ) -> str:
        """Remove a failed create from every caller-addressable session key."""

        with session.control_lock:
            public_session_oid = session.session_oid
            orphan_oid = f"orphan:{session.session_id}"
            with self._lock:
                if self._sessions.get(public_session_oid) is session:
                    self._sessions.pop(public_session_oid, None)
                with session.lock:
                    session.public_session_oid = public_session_oid
                    session.session_oid = orphan_oid
                    session.host_only_orphan = True
                self._sessions[orphan_oid] = session
            session.stop_event.set()
        return public_session_oid

    def release_stale_session_objects(self) -> list[str]:
        released: list[str] = []
        for obj in list(self.host.store.list_objects()):
            if obj.type != ObjectType.EXTERNAL_REF:
                continue
            if not isinstance(obj.payload, dict) or obj.payload.get("kind") != "pty_session":
                continue
            if obj.oid in self._sessions:
                continue
            if self.host.memory.delete_object_trusted("runtime.pty", obj.oid, reason="stale_pty_session"):
                released.append(obj.oid)
        if released:
            self.audit.record(
                actor="runtime.pty",
                action="primitive.pty.release_stale_objects",
                target="pty:*",
                input_refs=released,
                decision={"released": released},
            )
        return released

    def shutdown(self) -> bool:
        ok = True
        with self._lock:
            session_oids = list(self._sessions)
        for oid in session_oids:
            try:
                with self._lock:
                    session = self._sessions.get(oid)
                self._close_session(
                    oid,
                    actor="runtime",
                    reason="runtime.shutdown",
                    force=True,
                    timeout_s=self.config.pty.close_timeout_s,
                    wait_if_closing=True,
                    retry_host_only_orphan=bool(
                        session is not None and session.host_only_orphan
                    ),
                )
            except Exception as exc:
                ok = False
                self.audit.record(
                    actor="runtime.pty",
                    action="primitive.pty.shutdown_close_failed",
                    target=f"pty:{oid}",
                    decision={"error_type": type(exc).__name__, "error": str(exc)},
                )
        return self._wait_for_worker_threads(self.config.pty.close_timeout_s) and ok

    def release_recovery_diagnostics(self) -> bool:
        """Abandon process-local PTY ownership without durable publication.

        RuntimeLifecycle invokes this callback only after a recovery fence and
        admission drain.  It deliberately bypasses the protected close path:
        the durable PTY Object and its evidence are diagnostics for the next
        Runtime's stale-session recovery, while the current Runtime must stop
        its reader/monitor threads and release the live host handle before the
        store can be closed.  A partial failure keeps the session registered so
        the same callback can retry it.
        """

        return self.abandon_transient_sessions()

    def abandon_transient_sessions(self) -> bool:
        """Idempotently release every live PTY handle and worker in memory."""

        with self._lock:
            sessions = tuple(self._sessions.values())
        ok = True
        for session in sessions:
            if not self._abandon_transient_session(session):
                ok = False
        workers_stopped = self._wait_for_worker_threads(
            self.config.pty.close_timeout_s
        )
        return ok and workers_stopped

    def _abandon_transient_session(self, session: _PtyRuntimeSession) -> bool:
        # This is the only intentionally evidence-free provider phase in the
        # PTY module. The opaque lifecycle lease makes the raw close reachable
        # only from Runtime's explicit recovery-diagnostics handoff callback;
        # ordinary/public calls fail before any session or provider mutation.
        self.host.require_recovery_cleanup_lease()
        with session.control_lock:
            with self._lock:
                if self._sessions.get(session.session_oid) is not session:
                    return True
            with session.lock:
                session.recovery_abandoning = True
                session.stop_event.set()

            close_error: BaseException | None = None
            exit_code: int | None = None
            try:
                exit_code = session.handle.close(
                    force=True,
                    timeout_s=self.config.pty.close_timeout_s,
                )
            except BaseException as exc:
                close_error = exc
            finally:
                # Provider close is expected to unblock a pending read. Always
                # join after the attempt, including ambiguous/interrupt paths.
                session.stop_event.set()
                self._join_session_workers(
                    session,
                    timeout_s=self.config.pty.close_timeout_s,
                )

            workers_stopped = self._session_workers_stopped(session)
            if close_error is not None:
                if isinstance(close_error, Exception):
                    return False
                raise close_error.with_traceback(close_error.__traceback__)
            if not workers_stopped:
                return False

            with session.lock:
                session.closed = True
                session.closing = False
                session.exit_code = exit_code
                session.close_complete.set()
            with self._lock:
                if self._sessions.get(session.session_oid) is session:
                    self._sessions.pop(session.session_oid, None)
            return True

    def _create_session_object(
        self,
        pid: str,
        *,
        session_id: str,
        argv: list[str],
        cwd: str,
        backend: str,
        cols: int,
        rows: int,
        name: str | None,
        data_flow_context: DataFlowContext,
    ) -> tuple[str, str, str]:
        object_name = name or f"{self.config.pty.session_name_prefix}:{session_id.rsplit('_', 1)[-1]}"
        payload = {
            "kind": "pty_session",
            "session_id": session_id,
            "argv": list(argv),
            "cwd": cwd,
            "backend": backend,
            "cols": cols,
            "rows": rows,
            "created_at": utc_now(),
        }
        handle = self.host.memory.create_object(
            pid=pid,
            object_type=ObjectType.EXTERNAL_REF,
            payload=payload,
            metadata=ObjectMetadata(
                title="PTY session",
                tags=["pty", "external_ref"],
                **data_flow_context.labels.to_dict(),
            ),
            immutable=False,
            name=object_name,
        )
        obj = self.host.memory.get_object(pid, handle)
        self.host.add_handle_to_process_view(pid, handle)
        return handle.oid, obj.name, obj.namespace

    @staticmethod
    def _pty_ingress_context(request_context: DataFlowContext) -> DataFlowContext:
        external = DataFlowContext(
            labels=DataLabels(
                sensitivity=DataSensitivity.NORMAL,
                trust_level=DataTrustLevel.UNTRUSTED,
                integrity=DataIntegrity.UNTRUSTED,
                origin="external:pty",
            )
        )
        return DataFlowContext.aggregate((request_context, external))

    def _raise_session_flow(
        self,
        session: _PtyRuntimeSession,
        context: DataFlowContext,
    ) -> None:
        with session.lock:
            combined = DataFlowContext.aggregate(
                (session.data_flow_context, context)
            )
            session.data_flow_context = combined

        current = self.host.store.get_object(session.session_oid)
        if current is None:
            raise NotFound(f"PTY session Object not found: {session.session_oid}")
        metadata = propagate_object_labels(
            current.metadata,
            [ObjectMetadata(**combined.labels.to_dict())],
        )
        if metadata == current.metadata:
            return
        handle = self.host.memory.handle_for_oid(
            session.owner_pid,
            session.session_oid,
            required_rights={ObjectRight.WRITE.value},
            issued_by="runtime.pty.data_flow",
        )
        self.host.memory.update_object(
            session.owner_pid,
            handle,
            ObjectPatch(metadata=metadata),
            expected_version=current.version,
            _trusted_label_propagation=True,
        )

    def _session_egress_context(
        self,
        session: _PtyRuntimeSession,
    ) -> DataFlowContext:
        with session.lock:
            session_context = session.data_flow_context
        return DataFlowContext.aggregate(
            (self.host.data_flow.current_context(), session_context)
        )

    def _session_ingress_context(
        self,
        session: _PtyRuntimeSession,
    ) -> DataFlowContext:
        with session.lock:
            session_context = session.data_flow_context
        return self._pty_ingress_context(session_context)

    @staticmethod
    def _session_sink(session: _PtyRuntimeSession) -> DataSink:
        return DataSink(
            identity=f"pty:session:{session.session_id}",
            trust_identity=session.data_sink.identity,
            trust_identity_sha256=session.data_sink.identity_sha256,
        )

    def _start_reader(
        self,
        session: _PtyRuntimeSession,
        *,
        resource: str,
        parent_operation_id: str | None,
    ) -> None:
        # The reader drains continuously so interactive children cannot block on
        # a full PTY output buffer while the model is between tool calls.
        thread = threading.Thread(
            target=self._reader_loop,
            args=(session, resource, parent_operation_id),
            name=f"agent-libos-pty-reader-{session.session_id}",
            daemon=True,
        )
        session.reader_thread = thread
        with self._worker_condition:
            self._active_worker_threads.add(thread)
            try:
                thread.start()
            except Exception:
                self._active_worker_threads.discard(thread)
                self._worker_condition.notify_all()
                session.reader_thread = None
                raise

    def _start_monitor(self, session: _PtyRuntimeSession, *, resource: str) -> None:
        if self.resources is None or session.handle.pid is None:
            return
        thread = threading.Thread(
            target=self._monitor_loop,
            args=(session, resource),
            name=f"agent-libos-pty-monitor-{session.session_id}",
            daemon=True,
        )
        session.monitor_thread = thread
        with self._worker_condition:
            self._active_worker_threads.add(thread)
            try:
                thread.start()
            except Exception:
                self._active_worker_threads.discard(thread)
                self._worker_condition.notify_all()
                session.monitor_thread = None
                raise

    def _reader_loop(
        self,
        session: _PtyRuntimeSession,
        resource: str,
        parent_operation_id: str | None,
    ) -> None:
        try:
            try:
                operation_context = (
                    self.host.operations.attach(parent_operation_id)
                    if parent_operation_id is not None
                    else contextlib.nullcontext()
                )
                with operation_context:
                    effect_context = {
                        "session_oid": session.session_oid,
                        "backend": session.backend,
                        "resource": resource,
                        "mode": "continuous_drain",
                    }
                    invocation = ProtectedOperationInvocation(
                        pid=session.owner_pid,
                        actor="runtime.pty",
                        target=f"pty:{session.session_oid}",
                        canonical_args=effect_context,
                        observation=effect_context,
                        data_flow_ingress_context=self._session_ingress_context(session),
                        failure_evidence=lambda error, phase: self._protected_pty_failure_evidence(
                            "runtime.pty", session.session_oid, "ingest", error, phase
                        ),
                    )
                    with self._protected().start(
                        "primitive.pty.ingest", invocation, provider=self.provider
                    ) as protected:
                        result = protected.call(
                            ProviderPhase("ingest", information_flow=True),
                            self._reader_provider_phase,
                            session,
                        )
                        protected.complete(
                            result,
                            self._protected_pty_evidence(
                                "runtime.pty",
                                session.session_oid,
                                "ingest",
                                result,
                                input_refs=(session.session_oid,),
                            ),
                            classification_context=effect_context,
                            classification_result=result,
                        )
                if result["exited"]:
                    self._mark_session_exited(session, resource=resource)
            except BaseException:
                # Recovery handoff intentionally fences the continuous ingest
                # settlement. Its commit guard rolls back every attempted
                # store/evidence write; suppress that expected worker failure
                # only after the explicit transient-abandon flag is visible.
                with session.lock:
                    recovery_abandoning = session.recovery_abandoning
                if not recovery_abandoning:
                    raise
        finally:
            self._worker_finished(threading.current_thread())

    def _reader_provider_phase(self, session: _PtyRuntimeSession) -> dict[str, Any]:
        chars = 0
        exited = False
        while not session.stop_event.is_set():
            try:
                chunk = session.handle.read(timeout_s=0.05)
                if session.stop_event.is_set():
                    break
                if chunk:
                    self._append_output(session, chunk)
                    chars += len(chunk)
                if session.stop_event.is_set():
                    break
                if not session.handle.is_alive() and not chunk:
                    exited = True
                    break
            except Exception:
                if session.stop_event.is_set() or self._session_is_closing_or_closed(session):
                    break
                raise
        return {
            "chars": chars,
            "exited": exited,
            "stopped": session.stop_event.is_set(),
        }

    def _monitor_loop(self, session: _PtyRuntimeSession, resource: str) -> None:
        try:
            while not session.stop_event.is_set():
                try:
                    self._sample_and_charge(session, resource)
                    if session.stop_event.is_set():
                        return
                except Exception as exc:
                    if session.stop_event.is_set() or self._session_is_closing_or_closed(session):
                        return
                    self._fail_closed_resource_monitor(session, resource=resource, error=exc)
                    return
                session.stop_event.wait(self.config.pty.resource_sample_interval_s)
        finally:
            self._worker_finished(threading.current_thread())

    def _append_output(self, session: _PtyRuntimeSession, output: str) -> None:
        with session.lock:
            session.buffer.append(output)
            session.buffer_chars += len(output)
            while session.buffer_chars > session.buffer_max_chars and session.buffer:
                removed = session.buffer.popleft()
                session.buffer_chars -= len(removed)
                session.dropped_chars += len(removed)

    def _take_output(self, session: _PtyRuntimeSession, max_chars: int) -> tuple[str, bool]:
        with session.lock:
            chunks = list(session.buffer)
            session.buffer.clear()
            session.buffer_chars = 0
        output = "".join(chunks)
        if len(output) <= max_chars:
            return output, False
        remainder = output[max_chars:]
        if remainder:
            self._append_output(session, remainder)
        return output[:max_chars], True

    def _buffer_is_empty(self, session: _PtyRuntimeSession) -> bool:
        with session.lock:
            return session.buffer_chars == 0

    def _sample_and_charge(self, session: _PtyRuntimeSession, resource: str) -> None:
        if self.resources is None or session.handle.pid is None:
            return
        with session.lock:
            if session.stop_event.is_set() or session.closing or session.closed:
                return
        observation = self._observe_session_resources(session)
        delta = self._update_session_resource_totals(session, observation)
        if delta is None:
            return
        wall_only_monitoring = self._wall_only_monitoring(
            session,
            observation.sampling_error,
        )
        if not delta.changed:
            if observation.sampling_error is not None and not wall_only_monitoring:
                raise observation.sampling_error
            return
        if self._charge_session_resources(session, resource, delta):
            return
        if observation.sampling_error is not None and not wall_only_monitoring:
            raise observation.sampling_error

    def _observe_session_resources(
        self,
        session: _PtyRuntimeSession,
    ) -> _PtyResourceObservation:
        wall_seconds = max(0.0, time.monotonic() - session.started_monotonic)
        processes, sampling_error = self._process_tree(session.handle.pid)
        cpu_seconds, memory_bytes, sampling_error = self._process_metrics(
            processes,
            sampling_error,
        )
        return _PtyResourceObservation(
            wall_seconds=wall_seconds,
            cpu_seconds_by_process=cpu_seconds,
            memory_bytes=memory_bytes,
            sampling_error=sampling_error,
        )

    def _process_tree(
        self,
        pid: int,
    ) -> tuple[list[Any], Exception | None]:
        processes: list[Any] = []
        sampling_error: Exception | None = None
        try:
            proc = psutil.Process(pid)
            processes = [proc]
            try:
                processes.extend(proc.children(recursive=True))
            except (psutil.AccessDenied, PermissionError) as exc:
                sampling_error = exc
                processes = []
            except psutil.Error:
                pass
        except psutil.NoSuchProcess:
            # The host process may exit between the provider liveness check and
            # this sample. Wall time still needs a final cumulative charge.
            processes = []
        except (psutil.AccessDenied, PermissionError) as exc:
            # Provider metrics may be inaccessible even though monotonic wall
            # time is available. Preserve that independent observation so a
            # wall-time overage cannot be masked by a secondary sampler error;
            # if the wall charge is still within budget, the error is raised
            # after charging and the monitor retains its fail-closed behavior.
            sampling_error = exc
            processes = []
        return processes, sampling_error

    def _process_metrics(
        self,
        processes: list[Any],
        sampling_error: Exception | None,
    ) -> tuple[dict[tuple[int, float], float], int, Exception | None]:
        observed_cpu: dict[tuple[int, float], float] = {}
        current_memory = 0
        for item in processes:
            try:
                identity = (int(item.pid), float(item.create_time()))
                times = item.cpu_times()
                observed_cpu[identity] = max(0.0, float(times.user) + float(times.system))
                current_memory += max(0, int(item.memory_info().rss))
            except (psutil.AccessDenied, PermissionError) as exc:
                sampling_error = exc
                observed_cpu.clear()
                current_memory = 0
                break
            except psutil.Error:
                continue
        return observed_cpu, current_memory, sampling_error

    def _update_session_resource_totals(
        self,
        session: _PtyRuntimeSession,
        observation: _PtyResourceObservation,
    ) -> _PtyResourceDelta | None:
        with session.lock:
            if session.stop_event.is_set() or session.closing or session.closed:
                return None
            for identity, total in observation.cpu_seconds_by_process.items():
                previous = session.cpu_seconds_by_process.get(identity, 0.0)
                session.cpu_seconds_by_process[identity] = max(previous, total)
            cpu_seconds = sum(session.cpu_seconds_by_process.values())
            wall_delta = max(
                0.0,
                observation.wall_seconds - session.last_wall_seconds,
            )
            cpu_delta = max(0.0, cpu_seconds - session.last_cpu_seconds)
            peak_delta_changed = (
                observation.memory_bytes > session.last_peak_memory_bytes
            )
            session.last_wall_seconds = observation.wall_seconds
            session.last_cpu_seconds = cpu_seconds
            session.last_peak_memory_bytes = max(
                session.last_peak_memory_bytes,
                observation.memory_bytes,
            )
        return _PtyResourceDelta(
            wall_seconds=wall_delta,
            cpu_seconds=cpu_delta,
            memory_bytes=observation.memory_bytes,
            peak_changed=peak_delta_changed,
        )

    def _wall_only_monitoring(
        self,
        session: _PtyRuntimeSession,
        sampling_error: Exception | None,
    ) -> bool:
        if sampling_error is None:
            return False
        remaining = self.resources.remaining_budget(session.owner_pid)
        return (
            remaining.max_subprocess_wall_seconds is not None
            and remaining.max_subprocess_cpu_seconds is None
            and remaining.max_subprocess_memory_bytes is None
        )

    def _charge_session_resources(
        self,
        session: _PtyRuntimeSession,
        resource: str,
        delta: _PtyResourceDelta,
    ) -> bool:
        try:
            self.resources.charge(
                session.owner_pid,
                ResourceUsage(
                    subprocess_wall_seconds=delta.wall_seconds,
                    subprocess_cpu_seconds=delta.cpu_seconds,
                    subprocess_peak_memory_bytes=delta.memory_bytes,
                ),
                source="primitive.pty.spawn",
                context={"resource": resource, "session_oid": session.session_oid},
                allow_overage=True,
                kill_on_exceed=True,
            )
        except ResourceLimitExceeded as exc:
            try:
                self._close_session(
                    session.session_oid,
                    actor="runtime.pty",
                    reason="resource_limit_exceeded",
                    force=True,
                    timeout_s=self.config.pty.close_timeout_s,
                    wait_if_closing=True,
                )
            finally:
                self.audit.record(
                    actor="runtime.pty",
                    action="primitive.pty.resource_limit_exceeded",
                    target=f"pty:{session.session_oid}",
                    decision={"reason": str(exc)},
                )
            return True
        return False

    def _fail_closed_resource_monitor(
        self,
        session: _PtyRuntimeSession,
        *,
        resource: str,
        error: Exception,
    ) -> None:
        try:
            self._close_session(
                session.session_oid,
                actor="runtime.pty",
                reason="resource_monitor_access_denied",
                force=True,
                timeout_s=self.config.pty.close_timeout_s,
                wait_if_closing=True,
            )
            self.host.memory.delete_object_trusted(
                "runtime.pty",
                session.session_oid,
                reason="resource_monitor_access_denied",
            )
        finally:
            self.audit.record(
                actor="runtime.pty",
                action="primitive.pty.resource_monitor_denied",
                target=f"pty:{session.session_oid}",
                decision={
                    "resource": resource,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fail_closed": True,
                },
            )

    def _close_session(
        self,
        session_oid: str,
        *,
        actor: str,
        reason: str,
        force: bool,
        timeout_s: float,
        wait_if_closing: bool = False,
        authority_decision: CapabilityDecision | None = None,
        data_sink: DataSink | None = None,
        data_flow_context: DataFlowContext | None = None,
        data_flow_payload: dict[str, Any] | None = None,
        retry_host_only_orphan: bool = False,
    ) -> int | None:
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            if authority_decision is not None:
                self.host.capability.claim_decision_use(
                    authority_decision,
                    used_by=actor,
                    reason=(
                        "PTY Object deletion completed after natural-exit "
                        "cleanup removed the provider session"
                    ),
                )
            return None
        self._validate_close_descriptors(
            authority_decision,
            data_sink,
            data_flow_context,
            data_flow_payload,
        )
        remove_only, exit_code = self._begin_session_close(
            session,
            timeout_s=timeout_s,
            wait_if_closing=wait_if_closing,
            retry_host_only_orphan=retry_host_only_orphan,
        )
        if remove_only:
            self._settle_remove_only_close(
                session,
                actor=actor,
                authority_decision=authority_decision,
            )
            return exit_code
        plan = _PtyClosePlan(
            session_oid=session_oid,
            session=session,
            actor=actor,
            reason=reason,
            force=force,
            timeout_s=timeout_s,
            authority_decision=authority_decision,
            data_sink=data_sink,
            data_flow_context=data_flow_context,
            data_flow_payload=data_flow_payload,
            effect_context=self._close_effect_context(
                session,
                reason=reason,
                force=force,
                timeout_s=timeout_s,
            ),
        )
        try:
            return self._perform_protected_close(plan)
        except BaseException:
            self._mark_close_failed(plan)
            raise

    def _validate_close_descriptors(
        self,
        authority_decision: CapabilityDecision | None,
        data_sink: DataSink | None,
        data_flow_context: DataFlowContext | None,
        data_flow_payload: dict[str, Any] | None,
    ) -> None:
        if authority_decision is None:
            return
        if data_sink is None or data_flow_context is None or data_flow_payload is None:
            raise ValidationError(
                "public PTY close requires complete data-flow egress descriptors"
            )

    def _begin_session_close(
        self,
        session: _PtyRuntimeSession,
        *,
        timeout_s: float,
        wait_if_closing: bool,
        retry_host_only_orphan: bool,
    ) -> tuple[bool, int | None]:
        wait_for_close: threading.Event | None = None
        exit_code: int | None = None
        with session.lock:
            if session.close_outcome_unknown:
                if not (retry_host_only_orphan and session.host_only_orphan):
                    raise ValidationError(
                        "PTY session has an unresolved prior close outcome and "
                        f"cannot be retried: {session.session_oid}"
                    )
                # A failed create has already been removed from every public
                # session key and its reader is stopped. Repeating force-close
                # is therefore a Host lifecycle convergence attempt, not a
                # caller-visible replay. The earlier unknown effect remains
                # durable; the retry records its own protected effect.
                session.close_outcome_unknown = False
            if session.closing:
                if not wait_if_closing:
                    raise ValidationError(
                        "PTY session close is already in progress: "
                        f"{session.session_oid}"
                    )
                wait_for_close = session.close_complete
            if session.closed:
                exit_code = session.exit_code
                remove_only = True
            elif wait_for_close is None:
                session.closing = True
                session.close_complete.clear()
                remove_only = False
            else:
                remove_only = True
                exit_code = session.exit_code
        if wait_for_close is not None:
            if not wait_for_close.wait(timeout=max(0.0, timeout_s)):
                raise ValidationError(
                    f"timed out waiting for PTY session close: {session.session_oid}"
                )
            with session.lock:
                if not session.closed:
                    raise ValidationError(
                        f"PTY session close did not complete: {session.session_oid}"
                    )
                exit_code = session.exit_code
        return remove_only, exit_code

    def _settle_remove_only_close(
        self,
        session: _PtyRuntimeSession,
        *,
        actor: str,
        authority_decision: CapabilityDecision | None,
    ) -> None:
        # Another closer (or natural-exit cleanup) crossed the provider
        # boundary. This caller may consume DELETE authority, but must not
        # fabricate another provider effect or intent.
        if authority_decision is not None:
            self.host.capability.claim_decision_use(
                authority_decision,
                used_by=actor,
                reason=(
                    "PTY close completed after another closer crossed the "
                    "provider boundary"
                ),
            )
        with self._lock:
            self._sessions.pop(session.session_oid, None)

    def _close_effect_context(
        self,
        session: _PtyRuntimeSession,
        *,
        reason: str,
        force: bool,
        timeout_s: float,
    ) -> dict[str, Any]:
        return {
            "session_oid": session.session_oid,
            "backend": session.backend,
            "reason": reason,
            "force": force,
            "timeout_s": timeout_s,
            "provider_close_performed": True,
        }

    def _protected_close_operation(
        self,
        plan: _PtyClosePlan,
        state: _PtyCloseMutationState,
    ) -> Any:

        def restore_not_started() -> None:
            with plan.session.lock:
                plan.session.closing = False
                plan.session.close_complete.set()

        def failure_evidence(
            error: BaseException,
            phase: str,
        ) -> ProtectedOperationEvidence:
            return self._protected_pty_failure_evidence(
                plan.actor,
                plan.session_oid,
                "close",
                error,
                phase,
            )

        def settle_ambiguous_close(error: BaseException, _phase: str) -> None:
            if (
                plan.data_flow_context is not None
                and state.attempted
                and not isinstance(error, ProviderEffectNotStarted)
            ):
                self._raise_session_flow(plan.session, plan.data_flow_context)

        if plan.authority_decision is None:
            invocation = ProtectedOperationInvocation(
                pid=plan.session.owner_pid,
                actor=plan.actor,
                target=f"pty:{plan.session_oid}",
                canonical_args=plan.effect_context,
                observation=plan.effect_context,
                restore_not_started=restore_not_started,
                failure_evidence=failure_evidence,
            )
            operation = "primitive.pty.close.internal"
        else:
            assert plan.data_sink is not None
            assert plan.data_flow_context is not None
            assert plan.data_flow_payload is not None
            invocation = ProtectedOperationInvocation(
                pid=plan.session.owner_pid,
                actor=plan.actor,
                target=f"pty:{plan.session_oid}",
                decisions=(plan.authority_decision,),
                canonical_args=plan.effect_context,
                observation=plan.effect_context,
                data_sink=plan.data_sink,
                data_flow_context=plan.data_flow_context,
                data_flow_payload=plan.data_flow_payload,
                data_flow_operation="pty.close",
                restore_not_started=restore_not_started,
                failure_evidence=failure_evidence,
                failure_settlement=settle_ambiguous_close,
            )
            operation = "primitive.pty.close"
        return self._protected().start(
            operation,
            invocation,
            provider=self.provider,
        )

    def _close_provider(
        self,
        plan: _PtyClosePlan,
        state: _PtyCloseMutationState,
    ) -> int | None:
        state.attempted = True
        try:
            exit_code = plan.session.handle.close(
                force=plan.force,
                timeout_s=plan.timeout_s,
            )
            if plan.session.handle.is_alive():
                raise TimeoutError(
                    "PTY provider close returned while the session remained alive"
                )
        except ProviderEffectNotStarted:
            raise
        except BaseException:
            plan.session.stop_event.set()
            raise
        plan.session.stop_event.set()
        return exit_code

    def _perform_protected_close(self, plan: _PtyClosePlan) -> int | None:
        state = _PtyCloseMutationState()
        protected_operation = self._protected_close_operation(plan, state)
        with protected_operation as protected:
            if plan.reason == "process_exit":
                protected.call(
                    ProviderPhase("exit_code", information_flow=True),
                    plan.session.handle.exit_code,
                )
            exit_code = protected.call(
                ProviderPhase(
                    "close",
                    state_mutation=True,
                    information_flow=True,
                ),
                self._close_provider,
                plan,
                state,
            )
            plan.session.stop_event.set()
            self._join_session_workers(
                plan.session,
                timeout_s=min(plan.timeout_s, 1.0),
            )

            def settle_success() -> None:
                with plan.session.lock:
                    plan.session.closed = True
                    plan.session.closing = False
                    plan.session.exit_code = exit_code
                    plan.session.close_complete.set()
                with self._lock:
                    self._sessions.pop(plan.session_oid, None)

            result_payload = {
                "reason": plan.reason,
                "force": plan.force,
                "exit_code": exit_code,
                "closed": True,
                "provider_close_performed": True,
            }
            evidence_operation = (
                "exit" if plan.reason == "process_exit" else "close"
            )
            protected.complete(
                exit_code,
                self._protected_pty_evidence(
                    plan.actor,
                    plan.session_oid,
                    evidence_operation,
                    result_payload,
                    input_refs=(plan.session_oid,),
                ),
                classification_context=plan.effect_context,
                classification_result=result_payload,
                settle_success=settle_success,
            )
        return exit_code

    def _mark_close_failed(self, plan: _PtyClosePlan) -> None:
        uncertain_close = any(
            effect.operation == "close"
            and effect.target == f"pty:{plan.session_oid}"
            and (
                effect.effect_state == "pending"
                or effect.transaction_state == "unknown"
            )
            for effect in self.host.store.list_external_effects(
                pid=plan.session.owner_pid
            )
        )
        with plan.session.lock:
            plan.session.closing = False
            plan.session.close_outcome_unknown = uncertain_close
            plan.session.close_complete.set()

    def _session_is_closing_or_closed(self, session: _PtyRuntimeSession) -> bool:
        with session.lock:
            return session.closing or session.closed

    def _join_session_workers(self, session: _PtyRuntimeSession, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        current = threading.current_thread()
        for thread in (session.reader_thread, session.monitor_thread):
            if thread is None or thread is current or not thread.is_alive():
                continue
            thread.join(timeout=max(0.0, deadline - time.monotonic()))

    def _worker_finished(self, thread: threading.Thread) -> None:
        with self._worker_condition:
            self._active_worker_threads.discard(thread)
            self._worker_condition.notify_all()

    def _wait_for_worker_threads(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        current = threading.current_thread()
        with self._worker_condition:
            while True:
                self._active_worker_threads = {
                    thread for thread in self._active_worker_threads if thread.is_alive()
                }
                waiting = [thread for thread in self._active_worker_threads if thread is not current]
                if not waiting:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._worker_condition.wait(timeout=min(remaining, 0.05))

    @staticmethod
    def _session_workers_stopped(session: _PtyRuntimeSession) -> bool:
        return all(
            thread is None or not thread.is_alive()
            for thread in (session.reader_thread, session.monitor_thread)
        )

    def _require_session(self, session_oid: str) -> _PtyRuntimeSession:
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            raise NotFound(f"PTY session is not active: {session_oid}")
        with session.lock:
            if session.host_only_orphan:
                raise NotFound(f"PTY session is not active: {session_oid}")
        return session

    def _protected(self) -> Any:
        sdk = self.host.protected_operations
        if sdk is None:
            raise ValidationError("PTY protected-operation SDK is not attached")
        return sdk

    def _protected_pty_evidence(
        self,
        actor: str,
        session_oid: str,
        operation: str,
        result: dict[str, Any],
        *,
        input_refs: tuple[str, ...] = (),
        output_refs: tuple[str, ...] = (),
        correlation_id: str | None = None,
        parent_record_id: str | None = None,
    ) -> ProtectedOperationEvidence:
        return ProtectedOperationEvidence(
            event_type=(
                EventType.EXTERNAL_READ
                if operation in {"read", "ingest"}
                else EventType.EXTERNAL_WRITE
            ),
            event_source=actor,
            event_target=f"pty:{session_oid}",
            event_payload={"operation": operation, **result},
            audit_action=f"primitive.pty.{operation}",
            audit_actor=actor,
            audit_target=f"pty:{session_oid}",
            audit_decision=result,
            input_refs=input_refs,
            output_refs=output_refs,
            correlation_id=correlation_id,
            parent_record_id=parent_record_id,
            effect_metadata={"session_oid": session_oid, **result},
        )

    def _protected_pty_failure_evidence(
        self,
        actor: str,
        session_oid: str,
        operation: str,
        error: BaseException,
        phase: str,
        *,
        correlation_id: str | None = None,
        parent_record_id: str | None = None,
    ) -> ProtectedOperationEvidence:
        result = {
            "outcome": "unknown",
            "phase": phase,
            "failure_phase": phase,
            "error_type": type(error).__name__,
        }
        evidence = self._protected_pty_evidence(
            actor,
            session_oid,
            operation,
            result,
            correlation_id=correlation_id,
            parent_record_id=parent_record_id,
        )
        return replace(
            evidence,
            audit_action=f"primitive.pty.{operation}.failed",
        )

    def _require_session_open(self, session: _PtyRuntimeSession) -> None:
        with session.lock:
            if session.closed or session.host_only_orphan:
                raise NotFound(f"PTY session is not active: {session.session_oid}")

    def _session_alive(self, session: _PtyRuntimeSession) -> bool:
        with session.lock:
            if session.closed or session.host_only_orphan:
                return False
        return session.handle.is_alive()

    def _session_exit_code(self, session: _PtyRuntimeSession) -> int | None:
        with session.lock:
            if session.closed:
                return session.exit_code
            if session.host_only_orphan:
                return None
        return session.handle.exit_code()

    def _mark_session_exited(self, session: _PtyRuntimeSession, *, resource: str) -> None:
        with session.lock:
            if session.closed or session.closing:
                return
        try:
            self._close_session(
                session.session_oid,
                actor="runtime.pty",
                reason="process_exit",
                force=True,
                timeout_s=0.0,
                wait_if_closing=False,
            )
        except Exception as error:
            pending = any(
                effect.operation == "close"
                and effect.target == f"pty:{session.session_oid}"
                and effect.effect_state == "pending"
                for effect in self.host.store.list_external_effects(pid=session.owner_pid)
            )
            if pending:
                raise
            self.audit.record(
                actor="runtime.pty",
                action="primitive.pty.exit_cleanup_failed",
                target=f"pty:{session.session_oid}",
                decision={
                    "resource": resource,
                    "error_type": type(error).__name__,
                    "fail_closed": True,
                },
            )

    def _require_object_right(
        self,
        pid: str,
        oid: str,
        right: str,
        *,
        consume: bool = True,
    ) -> CapabilityDecision:
        if self.host.store.get_object(oid) is None:
            raise NotFound(f"object not found: {oid}")
        return self.host.capability.require(
            pid,
            f"object:{oid}",
            right,
            consume=consume,
            used_by="pty",
            reason="one-time PTY object permission consumed",
        )

    def _reserve_session_capacity(self, pid: str) -> None:
        with self._lock:
            active_sessions = [session for session in self._sessions.values() if not session.closed]
            global_count = len(active_sessions) + self._pending_session_creates
            process_count = (
                sum(1 for session in active_sessions if session.owner_pid == pid)
                + self._pending_session_creates_by_process.get(pid, 0)
            )
            if global_count >= self.config.pty.max_sessions_global:
                raise ValidationError("PTY session global limit reached")
            if process_count >= self.config.pty.max_sessions_per_process:
                raise ValidationError("PTY session per-process limit reached")
            self._pending_session_creates += 1
            self._pending_session_creates_by_process[pid] = self._pending_session_creates_by_process.get(pid, 0) + 1

    def _release_session_capacity(self, pid: str) -> None:
        with self._lock:
            self._release_session_capacity_locked(pid)

    def _release_session_capacity_locked(self, pid: str) -> None:
        if self._pending_session_creates <= 0:
            return
        self._pending_session_creates -= 1
        process_pending = self._pending_session_creates_by_process.get(pid, 0)
        if process_pending <= 1:
            self._pending_session_creates_by_process.pop(pid, None)
        else:
            self._pending_session_creates_by_process[pid] = process_pending - 1

    def _validate_size(self, cols: int | None, rows: int | None) -> tuple[int, int]:
        selected_cols = self.config.pty.default_cols if cols is None else int(cols)
        selected_rows = self.config.pty.default_rows if rows is None else int(rows)
        if selected_cols < 1 or selected_cols > self.config.pty.max_cols:
            raise ValidationError(f"pty cols must be between 1 and {self.config.pty.max_cols}")
        if selected_rows < 1 or selected_rows > self.config.pty.max_rows:
            raise ValidationError(f"pty rows must be between 1 and {self.config.pty.max_rows}")
        return selected_cols, selected_rows

    def _validate_timeout(self, value: float | None, *, default: float, hard_limit: float, label: str) -> float:
        selected = default if value is None else float(value)
        if not math.isfinite(selected) or selected < 0:
            raise ValidationError(f"{label} must be a non-negative finite number")
        if selected > hard_limit:
            raise ValidationError(f"{label} exceeds hard limit {hard_limit}s")
        return selected

    def _validate_char_limit(self, value: int | None, *, default: int, hard_limit: int, label: str) -> int:
        selected = default if value is None else int(value)
        if selected < 1:
            raise ValidationError(f"{label} must be >= 1")
        if selected > hard_limit:
            raise ValidationError(f"{label} exceeds hard limit {hard_limit}")
        return selected

    def _record_spawn_intent(
        self,
        pid: str,
        resource: str,
        argv: list[str],
        decision: ShellPolicyDecision,
        *,
        cwd: str,
        cols: int,
        rows: int,
    ) -> Any:
        return self.audit.record(
            actor=pid,
            action="primitive.pty.intent",
            target=resource,
            decision={
                "argv": argv,
                "cwd": cwd,
                "cols": cols,
                "rows": rows,
                "policy_level": decision.policy_level,
                "policy_reason": decision.reason,
                "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                "high_risk": decision.high_risk,
                "risk": decision.risk.value,
                "rule_id": decision.rule_id,
                "sandbox_profile": self.shell_policy.profile_json(decision.sandbox_profile),
                "continuous_session": True,
            },
        )

    def _request_human_approval(
        self,
        pid: str,
        argv: list[str],
        resource: str,
        decision: ShellPolicyDecision,
        *,
        timeout: float,
        cwd: str,
        source_oids: Iterable[str] | None = None,
    ) -> None:
        if self.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for pty spawn on {resource}")
        request_id = self.human.query(
            pid=pid,
            human=self.host.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to open an interactive PTY for {argv[0]!r}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.EXECUTE.value],
                    "constraints": self.shell_policy.approval_constraints(
                        argv,
                        decision,
                        timeout=timeout,
                        cwd=cwd,
                        operation="pty.spawn",
                        include_timeout=False,
                        extra_conditions={"continuous_session": True},
                        description="one-shot human approval for exact PTY spawn",
                    ),
                },
                "context": {
                    "adapter": "pty",
                    "primitive": "runtime.pty.spawn",
                    "operation": "pty.spawn",
                    "continuous_session": True,
                    "pid": pid,
                    "workspace_root": str(getattr(self.provider, "cwd", "")),
                    "working_directory": cwd,
                    "argv": list(argv),
                    "command": argv[0],
                    "resource": resource,
                    "right": CapabilityRight.EXECUTE.value,
                    "grant_scope": "one_time",
                    "policy_level": decision.policy_level,
                    "policy_reason": decision.reason,
                    "matched_rule": list(decision.matched_rule) if decision.matched_rule else None,
                    "high_risk": decision.high_risk,
                    "risk": decision.risk.value,
                    "rule_id": decision.rule_id,
                    "rule_effect": decision.rule_effect.value,
                    "sandbox_profile": self.shell_policy.profile_json(decision.sandbox_profile),
                },
            },
            blocking=True,
            source_oids=source_oids,
        )
        raise HumanApprovalRequired(
            request_id=request_id,
            message=f"{pid} is waiting for per-use human approval to open PTY for {resource}",
        )


class PtyCreateArgs(BaseModel):
    argv: list[str] = Field(min_length=1, description="Command argv array used to start the interactive PTY.")
    cwd: str | None = Field(default=None, description="Workspace-relative working directory. Defaults to process cwd.")
    cols: int | None = Field(default=None, description="Terminal columns. Defaults to the PTY module settings.")
    rows: int | None = Field(default=None, description="Terminal rows. Defaults to the PTY module settings.")
    startup_timeout_s: float | None = Field(default=None, ge=0, description="Seconds to wait for initial output.")
    max_output_chars: int | None = Field(default=None, ge=1, description="Maximum initial output chars returned.")
    name: str | None = Field(default=None, description="Optional Object Memory name for the PTY session object.")


class PtyCreateOutput(BaseModel):
    session_oid: str
    namespace: str
    name: str
    type: str
    alive: bool
    output: str
    output_truncated: bool
    dropped_chars: int


class PtyReadArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by pty_create.")
    timeout_s: float | None = Field(default=None, ge=0, description="Seconds to wait for new output.")
    max_chars: int | None = Field(default=None, ge=1, description="Maximum output chars returned.")


class PtyReadOutput(BaseModel):
    session_oid: str
    output: str
    output_truncated: bool
    alive: bool
    exit_code: int | None
    dropped_chars: int


class PtyWriteArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by pty_create.")
    text: str = Field(description="Text to write to the PTY.")


class PtyWriteOutput(BaseModel):
    session_oid: str
    bytes_written: int
    alive: bool


class PtyResizeArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by pty_create.")
    cols: int = Field(ge=1, description="Terminal columns.")
    rows: int = Field(ge=1, description="Terminal rows.")


class PtyResizeOutput(BaseModel):
    session_oid: str
    cols: int
    rows: int
    alive: bool


class PtyCloseArgs(BaseModel):
    session_oid: str = Field(description="Object oid returned by pty_create.")
    force: bool = Field(default=True, description="Terminate the PTY process if it is still alive.")
    timeout_s: float | None = Field(default=None, ge=0, description="Seconds to wait for process termination.")


class PtyCloseOutput(BaseModel):
    session_oid: str
    closed: bool
    exit_code: int | None


class PtyListArgs(BaseModel):
    pass


class PtyListEntry(BaseModel):
    session_oid: str
    name: str
    namespace: str
    argv: list[str]
    cwd: str
    backend: str
    alive: bool
    exit_code: int | None
    cols: int
    rows: int
    dropped_chars: int


class PtyListOutput(BaseModel):
    sessions: list[PtyListEntry]


class PtyCreateTool(SyncAgentTool[PtyCreateArgs]):
    name = "pty_create"
    description = "Create an interactive PTY session and return an Object Memory EXTERNAL_REF handle for it."
    args_schema = PtyCreateArgs
    output_schema = PtyCreateOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"filesystem.read", "object.write", "shell.execute"},
        timeout_s=None,
    )
    tags = ["pty", "external", "side_effect"]

    def run(self, args: PtyCreateArgs, ctx: ToolContext) -> PtyCreateOutput:
        runtime = _runtime(ctx)
        cwd = (
            runtime.process.working_directory(ctx.pid)
            if args.cwd is None
            else runtime.resolve_process_working_directory(ctx.pid, args.cwd)
        )
        result = _pty_adapter(runtime).create(
            ctx.pid,
            args.argv,
            cwd=cwd,
            cols=args.cols,
            rows=args.rows,
            startup_timeout_s=args.startup_timeout_s,
            max_output_chars=args.max_output_chars,
            name=args.name,
        )
        return PtyCreateOutput(**asdict(result))


class PtyReadTool(SyncAgentTool[PtyReadArgs]):
    name = "pty_read"
    description = "Read buffered output from an active Object-bound PTY session."
    args_schema = PtyReadArgs
    output_schema = PtyReadOutput
    policy = ToolPolicy(side_effects=False, idempotent=False, declared_permissions={"object.read"}, timeout_s=None)
    tags = ["pty", "read"]

    def run(self, args: PtyReadArgs, ctx: ToolContext) -> PtyReadOutput:
        result = _pty_adapter(_runtime(ctx)).read(
            ctx.pid,
            args.session_oid,
            timeout_s=args.timeout_s,
            max_chars=args.max_chars,
        )
        return PtyReadOutput(**asdict(result))


class PtyWriteTool(SyncAgentTool[PtyWriteArgs]):
    name = "pty_write"
    description = "Write input to an active Object-bound PTY session."
    args_schema = PtyWriteArgs
    output_schema = PtyWriteOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "shell.execute"},
        timeout_s=None,
    )
    tags = ["pty", "write", "external", "side_effect"]

    def run(self, args: PtyWriteArgs, ctx: ToolContext) -> PtyWriteOutput:
        result = _pty_adapter(_runtime(ctx)).write(ctx.pid, args.session_oid, args.text)
        return PtyWriteOutput(**asdict(result))


class PtyResizeTool(SyncAgentTool[PtyResizeArgs]):
    name = "pty_resize"
    description = "Resize an active Object-bound PTY session."
    args_schema = PtyResizeArgs
    output_schema = PtyResizeOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.write", "shell.execute"},
        timeout_s=None,
    )
    tags = ["pty", "resize", "external", "side_effect"]

    def run(self, args: PtyResizeArgs, ctx: ToolContext) -> PtyResizeOutput:
        result = _pty_adapter(_runtime(ctx)).resize(ctx.pid, args.session_oid, cols=args.cols, rows=args.rows)
        return PtyResizeOutput(**asdict(result))


class PtyCloseTool(SyncAgentTool[PtyCloseArgs]):
    name = "pty_close"
    description = "Close an active Object-bound PTY session and release its Object Memory handle."
    args_schema = PtyCloseArgs
    output_schema = PtyCloseOutput
    policy = ToolPolicy(
        side_effects=True,
        idempotent=False,
        declared_permissions={"object.delete", "shell.execute"},
        timeout_s=None,
    )
    tags = ["pty", "close", "external", "side_effect"]

    def run(self, args: PtyCloseArgs, ctx: ToolContext) -> PtyCloseOutput:
        result = _pty_adapter(_runtime(ctx)).close(
            ctx.pid,
            args.session_oid,
            force=args.force,
            timeout_s=args.timeout_s,
        )
        return PtyCloseOutput(**asdict(result))


class PtyListTool(SyncAgentTool[PtyListArgs]):
    name = "pty_list"
    description = "List active PTY sessions whose Object handles are readable by this process."
    args_schema = PtyListArgs
    output_schema = PtyListOutput
    policy = ToolPolicy(side_effects=False, idempotent=True, declared_permissions={"object.read"}, timeout_s=None)
    tags = ["pty", "list"]

    def run(self, args: PtyListArgs, ctx: ToolContext) -> PtyListOutput:
        entries = [PtyListEntry(**asdict(entry)) for entry in _pty_adapter(_runtime(ctx)).list(ctx.pid)]
        return PtyListOutput(sessions=entries)


def register_module(ctx: Any) -> None:
    for tool in [
        PtyCreateTool(),
        PtyReadTool(),
        PtyWriteTool(),
        PtyResizeTool(),
        PtyCloseTool(),
        PtyListTool(),
    ]:
        ctx.register_tool(tool)

    shell = ctx.runtime.config.shell
    ctx.register_image(
        AgentImage(
            image_id="pty-agent:v0",
            name="pty-agent",
            default_tools=[
                "process_exit",
                "pty_close",
                "pty_create",
                "pty_list",
                "pty_read",
                "pty_resize",
                "pty_write",
            ],
            required_capabilities=[
                {
                    "resource": shell.policy_resource,
                    "rights": ["execute"],
                    "constraints": {shell.policy_capability_key: shell.default_policy_level},
                }
            ],
            metadata={"module": "agent-libos-pty:v0"},
        )
    )
    ctx.add_startup_hook(initialize_pty)


def _runtime(ctx: ToolContext) -> Any:
    if ctx.runtime is None:
        raise ToolExecutionError("Runtime is unavailable.", code=ToolErrorCode.EXECUTION_ERROR)
    return ctx.runtime


def _pty_adapter(runtime: Any) -> PtyAdapter:
    adapter = runtime.module_state.get(_PTY_ADAPTER_ATTR)
    if adapter is None:
        raise ToolExecutionError("PTY module has not initialized.", code=ToolErrorCode.EXECUTION_ERROR)
    return adapter
