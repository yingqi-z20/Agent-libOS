from __future__ import annotations

import errno
import contextlib
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

import psutil
from pydantic import BaseModel, Field

from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    ObjectMetadata,
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
from agent_libos.primitives.shell import ShellAdapter, ShellPolicyDecision
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.event_bus import EventBus
from agent_libos.runtime.external_effects import (
    classify_external_effect,
    record_external_effect,
    require_external_effect_classifier,
)
from agent_libos.substrate import SubprocessLimits
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolErrorCode, ToolExecutionError, ToolPolicy
from agent_libos.utils.ids import new_id, utc_now

if TYPE_CHECKING:
    from agent_libos.runtime.runtime import Runtime

_PTY_ADAPTER_ATTR = "_agent_libos_pty_adapter"
_SAFE_PTY_ENV_KEYS = {
    "COMSPEC",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PATHEXT",
    "PYTHONIOENCODING",
    "PYTHONUTF8",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "USERPROFILE",
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
    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> PtySession: ...

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification: ...


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
    positive_fields = (
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
    )
    for name in positive_fields:
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValidationError(f"pty module setting {name} must be > 0")
    timeout_fields = (
        "startup_timeout_s",
        "startup_timeout_hard_limit_s",
        "read_timeout_s",
        "read_timeout_hard_limit_s",
        "close_timeout_s",
        "close_timeout_hard_limit_s",
    )
    for name in timeout_fields:
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValidationError(f"pty module setting {name} must be >= 0")
    if not settings.session_name_prefix.strip():
        raise ValidationError("pty module setting session_name_prefix must be non-empty")
    if settings.max_sessions_global < settings.max_sessions_per_process:
        raise ValidationError("pty module max_sessions_global must be >= max_sessions_per_process")
    if settings.read_hard_limit_chars < settings.read_max_chars:
        raise ValidationError("pty module read_hard_limit_chars must be >= read_max_chars")
    if settings.read_hard_limit_chars < settings.startup_output_max_chars:
        raise ValidationError("pty module read_hard_limit_chars must be >= startup_output_max_chars")
    if settings.input_hard_limit_chars < settings.input_max_chars:
        raise ValidationError("pty module input_hard_limit_chars must be >= input_max_chars")
    if settings.max_cols < settings.default_cols or settings.max_rows < settings.default_rows:
        raise ValidationError("pty module max terminal dimensions must cover defaults")
    if settings.startup_timeout_hard_limit_s < settings.startup_timeout_s:
        raise ValidationError("pty module startup timeout hard limit must cover default")
    if settings.read_timeout_hard_limit_s < settings.read_timeout_s:
        raise ValidationError("pty module read timeout hard limit must cover default")
    if settings.close_timeout_hard_limit_s < settings.close_timeout_s:
        raise ValidationError("pty module close timeout hard limit must cover default")


def initialize_pty(runtime: "Runtime") -> None:
    if getattr(runtime, _PTY_ADAPTER_ATTR, None) is not None:
        return
    settings = _coerce_pty_settings(getattr(runtime.substrate, "pty_settings", None))
    provider = getattr(runtime.substrate, "pty", None) or LocalPtyProvider(runtime.workspace_root)
    adapter = PtyAdapter(
        runtime,
        runtime.shell,
        runtime.audit,
        runtime.events,
        provider=provider,
        config=PtyModuleConfig(settings),
        resources=runtime.resources,
    )
    adapter.release_stale_session_objects()
    setattr(runtime, _PTY_ADAPTER_ATTR, adapter)
    runtime.memory.bind_object_release_finalizer(_object_release_finalizer(adapter))
    bind_shutdown = getattr(runtime, "bind_shutdown_finalizer", None)
    if callable(bind_shutdown):
        bind_shutdown(adapter.shutdown)


def _object_release_finalizer(adapter: "PtyAdapter"):
    def finalize(obj: Any, actor: str, reason: str) -> None:
        if getattr(obj, "type", None) == ObjectType.EXTERNAL_REF and isinstance(getattr(obj, "payload", None), dict):
            if obj.payload.get("kind") == "pty_session":
                adapter.close_for_object_release(obj.oid, actor=actor, reason=reason)

    return finalize


class LocalPtyProvider:
    """Subprocess-backed PTY provider scoped to a configured workspace."""

    supports_subprocess_limits = True

    def __init__(self, cwd: str | Path):
        self.cwd = Path(cwd).resolve()

    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        cols: int = 80,
        rows: int = 24,
        limits: SubprocessLimits | None = None,
    ) -> PtySession:
        selected_cwd = self._resolve_cwd(cwd)
        if os.name == "nt":
            return _WinPtySession.spawn(argv, cwd=selected_cwd, cols=cols, rows=rows)
        return _PosixPtySession.spawn(argv, cwd=selected_cwd, cols=cols, rows=rows)

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation != "spawn":
            raise ValueError(f"unsupported pty external effect operation: {operation}")
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
            raise CapabilityDenied(f"pty working directory escapes workspace root: {cwd}")
        return target


class _PosixPtySession:
    backend = "posix-pty"

    def __init__(self, master_fd: int, proc: subprocess.Popen[bytes]) -> None:
        self.master_fd = master_fd
        self.proc = proc
        self.pid = proc.pid
        self._closed = False

    @classmethod
    def spawn(cls, argv: list[str], *, cwd: Path, cols: int, rows: int) -> "_PosixPtySession":
        if sys.platform == "win32":
            raise RuntimeError("POSIX PTY backend is unavailable on Windows")
        import fcntl
        import pty
        import struct
        import termios

        master_fd, slave_fd = pty.openpty()
        try:
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            proc = subprocess.Popen(
                argv,
                cwd=cwd,
                env=_safe_subprocess_env(),
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
            return cls(master_fd, proc)
        except Exception:
            os.close(master_fd)
            raise
        finally:
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
        if force:
            self._signal_process_group(signal.SIGTERM)
        if self.proc.poll() is None:
            try:
                self.proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                self._signal_process_group(signal.SIGKILL)
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        self.proc.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        self.proc.wait(timeout=1.0)
        self._closed = True
        try:
            os.close(self.master_fd)
        except OSError:
            pass
        return self.exit_code()

    def _signal_process_group(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            if sig == signal.SIGKILL:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.kill()
            else:
                with contextlib.suppress(ProcessLookupError):
                    self.proc.terminate()


class _WinPtySession:
    backend = "winpty"

    def __init__(self, proc: Any) -> None:
        self.proc = proc
        self.pid = getattr(proc, "pid", None)

    @classmethod
    def spawn(cls, argv: list[str], *, cwd: Path, cols: int, rows: int) -> "_WinPtySession":
        try:
            from winpty import PtyProcess
        except ImportError as exc:
            raise RuntimeError("Windows PTY backend requires the pywinpty package") from exc

        command_line = subprocess.list2cmdline(argv)
        spawn = PtyProcess.spawn
        kwargs = {
            "cwd": str(cwd),
            "env": _safe_subprocess_env(),
            "dimensions": (rows, cols),
        }
        try:
            proc = spawn(argv, **kwargs)
        except TypeError:
            try:
                proc = spawn(command_line, **kwargs)
            except TypeError:
                proc = spawn(command_line, cwd=str(cwd), dimensions=(rows, cols))
        return cls(proc)

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
                wait(timeout=timeout_s)
            except TypeError:
                wait()
        return self.exit_code()


def _safe_subprocess_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.upper() in _SAFE_PTY_ENV_KEYS}


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
    buffer: deque[str] = field(default_factory=deque)
    buffer_chars: int = 0
    dropped_chars: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    reader_thread: threading.Thread | None = None
    closing: bool = False
    closed: bool = False
    exit_code: int | None = None
    last_wall_seconds: float = 0.0
    last_cpu_seconds: float = 0.0
    last_peak_memory_bytes: int = 0


class PtyAdapter:
    """Object-bound PTY primitive."""

    def __init__(
        self,
        runtime: "Runtime",
        shell: ShellAdapter,
        audit: AuditManager,
        events: EventBus,
        provider: PtyProvider,
        *,
        config: PtyModuleConfig | None = None,
        resources: Any | None = None,
    ) -> None:
        self.runtime = runtime
        self.shell = shell
        self.audit = audit
        self.events = events
        self.provider = provider
        self.config = config or PtyModuleConfig()
        self.resources = resources
        self._sessions: dict[str, _PtyRuntimeSession] = {}
        self._pending_session_creates = 0
        self._pending_session_creates_by_process: dict[str, int] = {}
        self._lock = threading.RLock()

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
    ) -> PtyCreateResult:
        checked = self.shell._validate_argv(argv)
        selected_cols, selected_rows = self._validate_size(cols, rows)
        selected_startup_timeout = self._validate_timeout(
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
        resource = self.shell.resource_for(checked)
        self.shell._enforce_workspace_argv_scope(checked, cwd=cwd)
        # PTY creation launches an interactive host process. It must reuse the
        # same shell authority path as shell.run before any provider side effect.
        decision = self.shell._authorize_operation(
            pid,
            checked,
            resource,
            timeout=selected_startup_timeout,
            cwd=cwd,
            adapter="pty",
            primitive="runtime.pty.spawn",
            operation="pty.spawn",
            authority_operation="pty.spawn",
            include_timeout_in_authority=False,
            continuous_session=True,
            extra_context={"startup_timeout_s": selected_startup_timeout},
        )
        if decision.ask_human:
            self._request_human_approval(pid, checked, resource, decision, timeout=selected_startup_timeout, cwd=cwd)
        if not decision.allowed:
            raise CapabilityDenied(f"{pid} denied pty spawn on {resource}: {decision.reason}")
        limits = self.shell._subprocess_limits(pid)
        require_external_effect_classifier(self.provider, "spawn")
        intent_record = self._record_spawn_intent(
            pid,
            resource,
            checked,
            decision,
            cwd=cwd,
            cols=selected_cols,
            rows=selected_rows,
        )
        self._reserve_session_capacity(pid)
        reserved_capacity = True
        session_id = new_id("pty")
        handle: PtySession | None = None
        try:
            if decision.consume_once and decision.consume_capability_id is not None:
                self.shell.capabilities.consume_use(
                    decision.consume_capability_id,
                    used_by="pty",
                    reason="one-time pty spawn permission consumed",
                )
            handle = self.provider.spawn(
                checked,
                cwd=cwd,
                cols=selected_cols,
                rows=selected_rows,
                limits=limits,
            )
            session_oid, object_name, namespace = self._create_session_object(
                pid,
                session_id=session_id,
                argv=checked,
                cwd=cwd,
                backend=handle.backend,
                cols=selected_cols,
                rows=selected_rows,
                name=name,
            )
        except Exception as exc:
            if handle is not None:
                try:
                    handle.close(force=True, timeout_s=self.config.pty.close_timeout_s)
                except Exception:
                    pass
            if reserved_capacity:
                self._release_session_capacity(pid)
            self.audit.record(
                actor=pid,
                action="primitive.pty.failed",
                target=resource,
                decision={"argv": checked, "cwd": cwd, "error_type": type(exc).__name__, "error": str(exc)},
                correlation_id=intent_record.record_id,
                parent_record_id=intent_record.record_id,
            )
            raise

        session = _PtyRuntimeSession(
            session_oid=session_oid,
            session_id=session_id,
            owner_pid=pid,
            argv=list(checked),
            cwd=cwd,
            backend=handle.backend,
            handle=handle,
            cols=selected_cols,
            rows=selected_rows,
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
            buffer_max_chars=self.config.pty.buffer_max_chars,
        )
        with self._lock:
            self._sessions[session_oid] = session
            self._release_session_capacity_locked(pid)
            reserved_capacity = False
        self._start_reader(session, resource=resource)
        if selected_startup_timeout > 0:
            time.sleep(selected_startup_timeout)
        output, output_truncated = self._take_output(session, selected_output_chars)
        event = self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=f"pty:{session_oid}",
            payload={"operation": "spawn", "argv": checked, "cwd": cwd, "backend": session.backend},
            correlation_id=intent_record.record_id,
        )
        audit_record = self.audit.record(
            actor=pid,
            action="primitive.pty.spawn",
            target=f"pty:{session_oid}",
            output_refs=[session_oid],
            decision={
                "argv": checked,
                "cwd": cwd,
                "resource": resource,
                "backend": session.backend,
                "policy_level": decision.policy_level,
                "policy_reason": decision.reason,
                "risk": decision.risk.value,
                "rule_id": decision.rule_id,
                "cols": selected_cols,
                "rows": selected_rows,
            },
            correlation_id=intent_record.record_id,
            parent_record_id=intent_record.record_id,
        )
        classification = classify_external_effect(
            self.provider,
            "spawn",
            {
                "argv": checked,
                "resource": resource,
                "cwd": cwd,
                "backend": session.backend,
                "session_oid": session_oid,
            },
            {"session_oid": session_oid, "backend": session.backend},
        )
        record_external_effect(
            self.runtime.store,
            pid=pid,
            provider="pty",
            operation="spawn",
            target=f"pty:{session_oid}",
            classification=classification,
            audit_record=audit_record,
            event=event,
            metadata={"session_oid": session_oid, "resource": resource},
        )
        return PtyCreateResult(
            session_oid=session_oid,
            namespace=namespace,
            name=object_name,
            type=ObjectType.EXTERNAL_REF.value,
            alive=session.handle.is_alive(),
            output=output,
            output_truncated=output_truncated,
            dropped_chars=session.dropped_chars,
        )

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
        self._require_object_right(pid, session_oid, ObjectRight.READ.value)
        session = self._require_session(session_oid)
        if selected_timeout > 0:
            deadline = time.monotonic() + selected_timeout
            while time.monotonic() < deadline and self._buffer_is_empty(session) and self._session_alive(session):
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
        output, truncated = self._take_output(session, selected_max_chars)
        alive = self._session_alive(session)
        exit_code = self._session_exit_code(session)
        self.events.emit(
            EventType.EXTERNAL_READ,
            source=pid,
            target=f"pty:{session_oid}",
            payload={"operation": "read", "chars": len(output), "truncated": truncated, "alive": alive},
        )
        self.audit.record(
            actor=pid,
            action="primitive.pty.read",
            target=f"pty:{session_oid}",
            input_refs=[session_oid],
            decision={"chars": len(output), "truncated": truncated, "alive": alive, "exit_code": exit_code},
        )
        return PtyReadResult(
            session_oid=session_oid,
            output=output,
            output_truncated=truncated,
            alive=alive,
            exit_code=exit_code,
            dropped_chars=session.dropped_chars,
        )

    def write(self, pid: str, session_oid: str, text: str) -> PtyWriteResult:
        if "\x00" in text:
            raise ValidationError("pty input cannot contain NUL bytes")
        if len(text) > self.config.pty.input_max_chars:
            raise ValidationError(f"pty input exceeds configured limit {self.config.pty.input_max_chars} chars")
        if len(text) > self.config.pty.input_hard_limit_chars:
            raise ValidationError(f"pty input exceeds hard limit {self.config.pty.input_hard_limit_chars} chars")
        self._require_object_right(pid, session_oid, ObjectRight.WRITE.value)
        session = self._require_session(session_oid)
        self._require_session_open(session)
        bytes_written = session.handle.write(text)
        alive = self._session_alive(session)
        self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=pid,
            target=f"pty:{session_oid}",
            payload={"operation": "write", "chars": len(text), "bytes_written": bytes_written, "alive": alive},
        )
        self.audit.record(
            actor=pid,
            action="primitive.pty.write",
            target=f"pty:{session_oid}",
            input_refs=[session_oid],
            decision={"chars": len(text), "bytes_written": bytes_written, "alive": alive},
        )
        return PtyWriteResult(session_oid=session_oid, bytes_written=bytes_written, alive=alive)

    def resize(self, pid: str, session_oid: str, *, cols: int, rows: int) -> PtyResizeResult:
        selected_cols, selected_rows = self._validate_size(cols, rows)
        self._require_object_right(pid, session_oid, ObjectRight.WRITE.value)
        session = self._require_session(session_oid)
        self._require_session_open(session)
        session.handle.resize(selected_cols, selected_rows)
        with session.lock:
            session.cols = selected_cols
            session.rows = selected_rows
        alive = self._session_alive(session)
        self.audit.record(
            actor=pid,
            action="primitive.pty.resize",
            target=f"pty:{session_oid}",
            input_refs=[session_oid],
            decision={"cols": selected_cols, "rows": selected_rows, "alive": alive},
        )
        return PtyResizeResult(session_oid=session_oid, cols=selected_cols, rows=selected_rows, alive=alive)

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
        self._require_object_right(pid, session_oid, ObjectRight.DELETE.value)
        exit_code = self._close_session(session_oid, actor=pid, reason="pty_close", force=force, timeout_s=selected_timeout)
        self.runtime.memory.delete_object_trusted(pid, session_oid, reason="pty_close")
        return PtyCloseResult(session_oid=session_oid, closed=True, exit_code=exit_code)

    def list(self, pid: str) -> list[PtySessionListEntry]:
        entries: list[PtySessionListEntry] = []
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            obj = self.runtime.store.get_object(session.session_oid)
            if obj is None or not self.runtime.capability.check(pid, f"object:{session.session_oid}", ObjectRight.READ):
                continue
            with session.lock:
                entries.append(
                    PtySessionListEntry(
                        session_oid=session.session_oid,
                        name=obj.name,
                        namespace=obj.namespace,
                        argv=list(session.argv),
                        cwd=session.cwd,
                        backend=session.backend,
                        alive=self._session_alive(session),
                        exit_code=self._session_exit_code(session),
                        cols=session.cols,
                        rows=session.rows,
                        dropped_chars=session.dropped_chars,
                    )
                )
        self.audit.record(actor=pid, action="primitive.pty.list", target="pty:*", decision={"count": len(entries)})
        return entries

    def close_for_object_release(self, oid: str, *, actor: str, reason: str) -> None:
        if oid not in self._sessions:
            return
        self._close_session(oid, actor=actor, reason=f"object_release:{reason}", force=True, timeout_s=self.config.pty.close_timeout_s)

    def release_stale_session_objects(self) -> list[str]:
        released: list[str] = []
        for obj in list(self.runtime.store.list_objects()):
            if obj.type != ObjectType.EXTERNAL_REF:
                continue
            if not isinstance(obj.payload, dict) or obj.payload.get("kind") != "pty_session":
                continue
            if obj.oid in self._sessions:
                continue
            if self.runtime.memory.delete_object_trusted("runtime.pty", obj.oid, reason="stale_pty_session"):
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
        with self._lock:
            session_oids = list(self._sessions)
        for oid in session_oids:
            self._close_session(
                oid,
                actor="runtime",
                reason="runtime.shutdown",
                force=True,
                timeout_s=self.config.pty.close_timeout_s,
            )
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
        handle = self.runtime.memory.create_object(
            pid=pid,
            object_type=ObjectType.EXTERNAL_REF,
            payload=payload,
            metadata=ObjectMetadata(title="PTY session", tags=["pty", "external_ref"]),
            immutable=False,
            name=object_name,
        )
        obj = self.runtime.memory.get_object(pid, handle)
        with self.runtime.store._lock:
            process = self.runtime.process.get(pid)
            if process.memory_view is None:
                process.memory_view = self.runtime.memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
            elif all(existing.oid != handle.oid for existing in process.memory_view.roots):
                process.memory_view.roots.append(handle)
            self.runtime.store.update_process(process)
        return handle.oid, obj.name, obj.namespace

    def _start_reader(self, session: _PtyRuntimeSession, *, resource: str) -> None:
        # The reader drains continuously so interactive children cannot block on
        # a full PTY output buffer while the model is between tool calls.
        thread = threading.Thread(
            target=self._reader_loop,
            args=(session, resource),
            name=f"agent-libos-pty-reader-{session.session_id}",
            daemon=True,
        )
        session.reader_thread = thread
        thread.start()

    def _reader_loop(self, session: _PtyRuntimeSession, resource: str) -> None:
        exited = False
        while not session.stop_event.is_set():
            try:
                chunk = session.handle.read(timeout_s=0.05)
                if chunk:
                    self._append_output(session, chunk)
                self._sample_and_charge(session, resource)
                if not session.handle.is_alive() and not chunk:
                    exited = True
                    break
            except Exception as exc:
                self.audit.record(
                    actor="runtime.pty",
                    action="primitive.pty.reader_failed",
                    target=f"pty:{session.session_oid}",
                    decision={"error_type": type(exc).__name__, "error": str(exc)},
                )
                return
        if session.stop_event.is_set():
            session.exit_code = session.handle.exit_code()
            return
        if exited:
            self._mark_session_exited(session, resource=resource)

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
            if session.closed:
                return
        try:
            proc = psutil.Process(session.handle.pid)
            wall_seconds = max(0.0, time.monotonic() - session.started_monotonic)
            cpu_seconds = 0.0
            peak_memory = 0
            processes = [proc]
            try:
                processes.extend(proc.children(recursive=True))
            except psutil.Error:
                pass
            for item in processes:
                try:
                    times = item.cpu_times()
                    cpu_seconds += float(times.user) + float(times.system)
                    peak_memory += int(item.memory_info().rss)
                except psutil.Error:
                    continue
        except psutil.Error:
            return
        wall_delta = max(0.0, wall_seconds - session.last_wall_seconds)
        cpu_delta = max(0.0, cpu_seconds - session.last_cpu_seconds)
        peak_delta_changed = peak_memory > session.last_peak_memory_bytes
        session.last_wall_seconds = wall_seconds
        session.last_cpu_seconds = cpu_seconds
        session.last_peak_memory_bytes = max(session.last_peak_memory_bytes, peak_memory)
        if wall_delta == 0 and cpu_delta == 0 and not peak_delta_changed:
            return
        try:
            self.resources.charge(
                session.owner_pid,
                ResourceUsage(
                    subprocess_wall_seconds=wall_delta,
                    subprocess_cpu_seconds=cpu_delta,
                    subprocess_peak_memory_bytes=peak_memory,
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
                )
            finally:
                self.audit.record(
                    actor="runtime.pty",
                    action="primitive.pty.resource_limit_exceeded",
                    target=f"pty:{session.session_oid}",
                    decision={"reason": str(exc)},
                )

    def _close_session(
        self,
        session_oid: str,
        *,
        actor: str,
        reason: str,
        force: bool,
        timeout_s: float,
    ) -> int | None:
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            return None
        with session.lock:
            if session.closing:
                raise ValidationError(f"PTY session close is already in progress: {session_oid}")
            if session.closed:
                exit_code = session.exit_code
                remove_only = True
            else:
                session.closing = True
                remove_only = False
        if not remove_only:
            try:
                exit_code = session.handle.close(force=force, timeout_s=timeout_s)
            except Exception:
                with session.lock:
                    session.closing = False
                raise
            session.stop_event.set()
            if (
                session.reader_thread is not None
                and session.reader_thread.is_alive()
                and session.reader_thread is not threading.current_thread()
            ):
                session.reader_thread.join(timeout=min(timeout_s, 1.0))
            with session.lock:
                session.closed = True
                session.closing = False
                session.exit_code = exit_code
        with self._lock:
            self._sessions.pop(session_oid, None)
        self.events.emit(
            EventType.EXTERNAL_WRITE,
            source=actor,
            target=f"pty:{session_oid}",
            payload={"operation": "close", "reason": reason, "exit_code": exit_code},
        )
        self.audit.record(
            actor=actor,
            action="primitive.pty.close",
            target=f"pty:{session_oid}",
            input_refs=[session_oid],
            decision={"reason": reason, "force": force, "exit_code": exit_code},
        )
        return exit_code

    def _require_session(self, session_oid: str) -> _PtyRuntimeSession:
        with self._lock:
            session = self._sessions.get(session_oid)
        if session is None:
            raise NotFound(f"PTY session is not active: {session_oid}")
        return session

    def _require_session_open(self, session: _PtyRuntimeSession) -> None:
        with session.lock:
            if session.closed:
                raise NotFound(f"PTY session is not active: {session.session_oid}")

    def _session_alive(self, session: _PtyRuntimeSession) -> bool:
        with session.lock:
            if session.closed:
                return False
        return session.handle.is_alive()

    def _session_exit_code(self, session: _PtyRuntimeSession) -> int | None:
        with session.lock:
            if session.closed:
                return session.exit_code
        return session.handle.exit_code()

    def _mark_session_exited(self, session: _PtyRuntimeSession, *, resource: str) -> None:
        with session.lock:
            if session.closed or session.closing:
                return
            session.closing = True
        exit_code = session.handle.exit_code()
        try:
            exit_code = session.handle.close(force=False, timeout_s=0.0)
        except Exception as exc:
            with session.lock:
                session.closing = False
            self.audit.record(
                actor="runtime.pty",
                action="primitive.pty.exit_cleanup_failed",
                target=f"pty:{session.session_oid}",
                decision={"error_type": type(exc).__name__, "error": str(exc)},
            )
            return
        session.stop_event.set()
        with session.lock:
            session.closed = True
            session.closing = False
            session.exit_code = exit_code
        self.events.emit(
            EventType.EXTERNAL_WRITE,
            source="runtime.pty",
            target=f"pty:{session.session_oid}",
            payload={"operation": "exit", "reason": "process_exit", "exit_code": exit_code},
        )
        self.audit.record(
            actor="runtime.pty",
            action="primitive.pty.exit",
            target=f"pty:{session.session_oid}",
            input_refs=[session.session_oid],
            decision={"resource": resource, "exit_code": exit_code},
        )

    def _require_object_right(self, pid: str, oid: str, right: str) -> None:
        if self.runtime.store.get_object(oid) is None:
            raise NotFound(f"object not found: {oid}")
        decision = self.runtime.capability.require(pid, f"object:{oid}", right)
        if decision.consume_capability_id is not None:
            self.runtime.capability.consume_use(
                decision.consume_capability_id,
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
                "sandbox_profile": self.shell._profile_json(decision.sandbox_profile),
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
    ) -> None:
        if self.shell.human is None:
            raise CapabilityDenied(f"{pid} requires human approval for pty spawn on {resource}")
        request_id = self.shell.human.query(
            pid=pid,
            human=self.runtime.config.runtime.default_human,
            request={
                "type": "external_operation_approval",
                "question": f"Allow this process to open an interactive PTY for {argv[0]!r}?",
                "requested_once_capability": {
                    "subject": pid,
                    "resource": resource,
                    "rights": [CapabilityRight.EXECUTE.value],
                    "constraints": self.shell._approval_constraints(
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
                    "sandbox_profile": self.shell._profile_json(decision.sandbox_profile),
                },
            },
            blocking=True,
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
        declared_permissions={"object.write", "shell.execute"},
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
    adapter = getattr(runtime, _PTY_ADAPTER_ATTR, None)
    if adapter is None:
        raise ToolExecutionError("PTY module has not initialized.", code=ToolErrorCode.EXECUTION_ERROR)
    return adapter
