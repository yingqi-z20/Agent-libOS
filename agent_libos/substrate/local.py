from __future__ import annotations

import asyncio
import contextlib
import http.client
import heapq
import os
import signal
import shutil
import socket
import ssl
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
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
)
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.substrate.base import (
    CommandMetrics,
    CommandResult,
    DirectoryEntrySnapshot,
    PathState,
    ResolvedPath,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
)

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_SHELL_DEFAULTS = DEFAULT_CONFIG.shell
_SAFE_SHELL_ENV_KEYS = {
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


class LocalFilesystemProvider:
    """Local-workspace implementation of the filesystem substrate."""

    def __init__(self, root: str | Path, namespace: str = _RUNTIME_DEFAULTS.workspace_namespace):
        self.root = Path(root).resolve()
        self.namespace = namespace
        self.root_display = str(self.root)

    def resolve(self, path: Any) -> ResolvedPath:
        raw = Path(path)
        target = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
        if self.root not in target.parents and target != self.root:
            raise CapabilityDenied(f"path escapes filesystem adapter root: {path}")
        relative = target.relative_to(self.root).as_posix()
        return ResolvedPath(relative=relative, display=str(target), is_root=target == self.root)

    def state(self, path: ResolvedPath) -> PathState:
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
        if max_bytes is None:
            return self._target(path).read_bytes()
        with self._target(path).open("rb") as handle:
            return handle.read(max(0, max_bytes))

    def write_text(self, path: ResolvedPath, text: str, encoding: str, newline: str | None = "\n") -> None:
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target = self._target(path)
        target.write_text(text, encoding=encoding, newline=newline)

    def make_directory(self, path: ResolvedPath, *, parents: bool, exist_ok: bool) -> None:
        self._target(path).mkdir(parents=parents, exist_ok=exist_ok)

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None) -> list[DirectoryEntrySnapshot]:
        target = self._target(path)
        if limit is not None and limit > 0:
            children = heapq.nsmallest(limit, target.iterdir(), key=lambda item: item.name)
        else:
            children = sorted(target.iterdir(), key=lambda item: item.name)
        return [self._directory_entry(child) for child in children]

    def delete_file(self, path: ResolvedPath) -> None:
        self._target(path).unlink()

    def delete_directory(self, path: ResolvedPath, *, recursive: bool) -> None:
        target = self._target(path)
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        if operation in {"write_text", "make_directory", "delete_file", "delete_directory"}:
            return ExternalEffectClassification(
                rollback_class=ExternalEffectRollbackClass.ROLLBACKABLE,
                rollback_status=ExternalEffectRollbackStatus.NOT_APPLIED,
                state_mutation=True,
                information_flow=False,
                metadata={"namespace": self.namespace, "path": context.get("path")},
            )
        if operation in {"read_bytes", "list_directory"}:
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

    def _directory_entry(self, target: Path) -> DirectoryEntrySnapshot:
        stat_result = target.lstat()
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
        return DirectoryEntrySnapshot(
            name=target.name,
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

    supports_subprocess_limits = True

    def __init__(self, cwd: str | Path):
        self.cwd = Path(cwd).resolve()

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = _TOOL_DEFAULTS.shell_timeout_s,
        cwd: str | None = None,
        limits: SubprocessLimits | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        selected_cwd = self._resolve_cwd(cwd)
        stdout_limit = _SHELL_DEFAULTS.stdout_hard_limit_chars if stdout_limit_chars is None else max(0, int(stdout_limit_chars))
        stderr_limit = _SHELL_DEFAULTS.stderr_hard_limit_chars if stderr_limit_chars is None else max(0, int(stderr_limit_chars))
        checked_argv = self._resolve_argv0(argv, selected_cwd)
        started_at = time.monotonic()
        with tempfile.TemporaryFile("w+b") as stdout_file, tempfile.TemporaryFile("w+b") as stderr_file:
            proc = subprocess.Popen(
                checked_argv,
                cwd=selected_cwd,
                env=self._safe_env(),
                shell=False,
                stdout=stdout_file,
                stderr=stderr_file,
                **self._process_group_kwargs(),
            )
            ps_proc = psutil.Process(proc.pid)
            peak_memory = 0
            cpu_seconds = 0.0
            limit_kind: str | None = None
            timed_out = False
            try:
                while True:
                    wall_seconds = time.monotonic() - started_at
                    cpu_seconds, peak_memory = self._sample_process_tree(ps_proc, peak_memory)
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
            stdout, stdout_truncated = self._read_limited_output(stdout_file, stdout_limit)
            stderr, stderr_truncated = self._read_limited_output(stderr_file, stderr_limit)
            wall_seconds = time.monotonic() - started_at
            final_cpu_seconds, peak_memory = self._sample_process_tree(ps_proc, peak_memory)
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
        return env

    def _resolve_argv0(self, argv: list[str], selected_cwd: Path) -> list[str]:
        if not argv or self._argv0_has_path(argv[0]):
            return argv
        resolved = shutil.which(argv[0], path=self._safe_path())
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

    def _sample_process_tree(self, proc: psutil.Process, peak_memory: int) -> tuple[float, int]:
        cpu_seconds = 0.0
        memory_bytes = 0
        processes = [proc]
        try:
            processes.extend(proc.children(recursive=True))
        except psutil.Error:
            pass
        for item in processes:
            try:
                times = item.cpu_times()
                cpu_seconds += float(times.user) + float(times.system)
                memory_bytes += int(item.memory_info().rss)
            except psutil.Error:
                continue
        return cpu_seconds, max(peak_memory, memory_bytes)

    def _process_group_kwargs(self) -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    def _kill_process_tree(self, ps_proc: psutil.Process, proc: subprocess.Popen[str]) -> None:
        # The direct child may exit after spawning background work, at which
        # point psutil no longer sees those processes as descendants. A process
        # group gives the provider one cleanup handle for the whole shell run.
        self._terminate_process_group(proc)
        processes: list[psutil.Process] = []
        try:
            processes.extend(ps_proc.children(recursive=True))
            processes.append(ps_proc)
        except psutil.Error:
            pass
        for item in processes:
            try:
                item.terminate()
            except psutil.Error:
                continue
        alive = psutil.wait_procs(processes, timeout=1.0)[1] if processes else []
        for item in alive:
            try:
                item.kill()
            except psutil.Error:
                continue
        if proc.poll() is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def _terminate_process_group(self, proc: subprocess.Popen[str]) -> None:
        if os.name == "nt":
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.05)
        with contextlib.suppress(ProcessLookupError, PermissionError):
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


class LocalResourceProviderSubstrate:
    """Default Resource Provider Substrate backed by the host OS."""

    def __init__(self, workspace_root: str | Path, namespace: str = _RUNTIME_DEFAULTS.workspace_namespace):
        self.workspace_root = Path(workspace_root).resolve()
        self.workspace_display = str(self.workspace_root)
        self.filesystem = LocalFilesystemProvider(self.workspace_root, namespace=namespace)
        self.clock = LocalClockProvider()
        self.shell = LocalShellProvider(self.workspace_root)
        self.human = LocalHumanProvider()
        self.jsonrpc = HttpJsonRpcProvider()
