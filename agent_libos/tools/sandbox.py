from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import psutil

from agent_libos.capability.profiles import SandboxProfileBuilder
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ProviderHostError, SandboxError
from agent_libos.models import ValidationResult
from agent_libos.substrate import (
    CommandMetrics,
    SubprocessLimitExceeded,
    SubprocessLimits,
    SubprocessTimeoutExpired,
    WindowsJobObject,
)
from agent_libos.tools.observability import ensure_json_size, sanitize_for_observability
from agent_libos.utils.public_errors import (
    public_error_envelope,
    provider_error_envelope_from_mapping,
)
from agent_libos.utils.serde import to_jsonable

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_RUNTIME_CODE_GENERATION_NAMES = {"eval", "Function", "AsyncFunction", "GeneratorFunction", "AsyncGeneratorFunction"}
_RUNTIME_GLOBAL_OBJECT_NAMES = {"globalThis", "window"}
_TYPESCRIPT_DEPENDENCY_DIRECTIVE_RE = re.compile(
    r"(?im)^\s*///\s*<\s*(?:reference|amd-dependency)\b"
)
_DENO_TYPES_DIRECTIVE_RE = re.compile(r"(?im)^\s*//\s*@deno-types\s*=")
_PROCESS_CLEANUP_TIMEOUT_S = 1.0
_PROCESS_CLEANUP_POLL_INTERVAL_S = 0.02
_VALIDATION_SUMMARY_MAX_DEPTH = 64
_SANDBOX_FAILURE_METRICS_ATTR = "_agent_libos_command_metrics"
_POSIX_HOST_PATH_RE = re.compile(
    r"(?<![:/A-Za-z0-9._-])(?:file://)?/"
    r"(?:[^\s/\\\"'<>|,;()[\]{}]+/)+"
    r"[^\s/\\\"'<>|,;()[\]{}]+"
)
_WINDOWS_HOST_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9._-])[A-Z]:\\"
    r"(?:[^\s/\\\"'<>|,;()[\]{}]+\\)+"
    r"[^\s/\\\"'<>|,;()[\]{}]+"
)
_STACK_LINE_RE = re.compile(
    r"^\s*(?:"
    r"File\s+[\"']|"
    r"at\s+(?:async\s+)?(?:.*[/\\].*:\d+(?::\d+)?\)?|.*\([^)]*:\d+(?::\d+)?\))"
    r").*$"
)
_PYTHON_EXCEPTION_LINE_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Warning)?\s*:"
)
_VALIDATION_CREDENTIAL_PATTERNS = (
    re.compile(
        r"(?is)-----BEGIN [^-]*PRIVATE KEY-----.*?"
        r"-----END [^-]*PRIVATE KEY-----"
    ),
    re.compile(
        r"(?i)\b[A-Za-z0-9_-]*(?:"
        r"api[_-]?key|authorization|auth[_-]?token|credential|password|passwd|"
        r"private[_-]?key|secret|session[_-]?token|token"
        r")[A-Za-z0-9_-]*\b\s*[:=]\s*"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;\"'}]+)"
    ),
    re.compile(r"(?i)\b(?:sk|gh[pousr])[-_][A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\bhttps?://[^/\s:@]+:[^/\s@]+@"),
)
_SANDBOX_SOURCE_LOCATION_RE = re.compile(
    r"(?P<name>candidate\.ts|runner\.ts|deno\.json)"
    r"(?P<location>:\d+(?::\d+)?)?"
)

SyscallHandler = Callable[[str, dict[str, Any]], Any | Awaitable[Any]]


def compact_validation_diagnostic(
    value: object,
    *,
    max_chars: int,
    head_tail: bool = True,
) -> str:
    """Return a credential/path-safe, stack-free validation diagnostic.

    Validation text is model-facing data. It must stay actionable, but must
    not become an accidental carrier for provider credentials, host paths, or
    embedded tracebacks. The digest is over the original text so a bounded
    diagnostic can still be correlated with durable validation evidence.
    """

    text = str(value)
    safe = _redact_validation_text(text)
    safe = _strip_validation_stack(safe)
    safe = _redact_host_paths(safe)
    return _bound_validation_text(
        safe,
        original=text,
        max_chars=max(1, max_chars),
        head_tail=head_tail,
    )


def summarize_validation_value(
    value: Any,
    *,
    preview_chars: int,
    head_tail: bool = False,
) -> dict[str, Any]:
    """Build a deterministic, bounded summary for one candidate test value."""

    canonical, lossy = _canonical_validation_value(value)
    serialized = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded = serialized.encode("utf-8")
    safe_serialized = _redact_validation_text(serialized)
    safe_serialized = _redact_host_paths(safe_serialized)
    preview = _bound_validation_text(
        safe_serialized,
        original=serialized,
        max_chars=max(1, preview_chars),
        head_tail=head_tail,
    )
    return {
        "type": _validation_value_type(value),
        "serialized_size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "preview": preview,
        "preview_truncated": len(safe_serialized) > max(1, preview_chars),
        "lossy": lossy,
    }


def _exact_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON types exactly while treating all finite numbers alike."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is bool and type(expected) is bool and actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return bool(actual == expected)
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(
            _exact_json_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(
            _exact_json_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _canonical_validation_value(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> tuple[Any, bool]:
    if depth >= _VALIDATION_SUMMARY_MAX_DEPTH:
        return {"$non_json_type": _qualified_type_name(value), "$depth_limited": True}, True
    if value is None or isinstance(value, (str, bool, int)):
        return value, False
    if isinstance(value, float):
        if math.isfinite(value):
            return value, False
        return {"$non_finite_number": str(value)}, True
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "$binary_type": type(value).__name__,
            "serialized_size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }, True

    selected_seen = seen if seen is not None else set()
    identity = id(value)
    if identity in selected_seen:
        return {"$non_json_type": _qualified_type_name(value), "$cycle": True}, True

    if isinstance(value, Mapping):
        selected_seen.add(identity)
        try:
            result: dict[str, Any] = {}
            lossy = False
            for key, item in value.items():
                key_text, key_lossy = _canonical_validation_key(key)
                lossy = lossy or key_lossy
                converted, item_lossy = _canonical_validation_value(
                    item,
                    depth=depth + 1,
                    seen=selected_seen,
                )
                if key_text in result:
                    key_text = f"{key_text}#{len(result)}"
                    lossy = True
                result[key_text] = converted
                lossy = lossy or item_lossy
            return result, lossy
        finally:
            selected_seen.remove(identity)

    if isinstance(value, (list, tuple, set, frozenset)):
        selected_seen.add(identity)
        try:
            converted_items: list[Any] = []
            lossy = not isinstance(value, list)
            for item in value:
                converted, item_lossy = _canonical_validation_value(
                    item,
                    depth=depth + 1,
                    seen=selected_seen,
                )
                converted_items.append(converted)
                lossy = lossy or item_lossy
            if isinstance(value, (set, frozenset)):
                converted_items.sort(
                    key=lambda item: json.dumps(
                        item,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            return converted_items, lossy
        finally:
            selected_seen.remove(identity)

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        selected_seen.add(identity)
        try:
            try:
                dumped = model_dump(mode="json")
            except TypeError:
                dumped = model_dump()
            converted, _lossy = _canonical_validation_value(
                dumped,
                depth=depth + 1,
                seen=selected_seen,
            )
            return converted, True
        finally:
            selected_seen.remove(identity)

    return {"$non_json_type": _qualified_type_name(value)}, True


def _canonical_validation_key(key: Any) -> tuple[str, bool]:
    if isinstance(key, str):
        return key, False
    if key is None or isinstance(key, (bool, int, float)):
        return json.dumps(key, ensure_ascii=False, sort_keys=True), True
    return f"<{_qualified_type_name(key)}>", True


def _validation_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (list, tuple, set, frozenset)):
        return "array"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "binary"
    return f"non-json:{_qualified_type_name(value)}"


def _qualified_type_name(value: Any) -> str:
    value_type = type(value)
    module = value_type.__module__
    qualified = value_type.__qualname__
    return qualified if module == "builtins" else f"{module}.{qualified}"


def _redact_validation_text(value: str) -> str:
    locally_redacted = _redact_additional_credentials(value)
    encoded = json.dumps(locally_redacted, ensure_ascii=True)
    envelope = sanitize_for_observability(
        locally_redacted,
        preview_chars=len(encoded) + _TOOL_DEFAULTS.tool_observability_preview_chars,
    )
    preview = envelope.get("preview")
    if isinstance(preview, str) and not envelope.get("truncated", False):
        try:
            decoded = json.loads(preview)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, str):
            return decoded
    fallback = (
        "[validation diagnostic redacted "
        f"sha256={hashlib.sha256(value.encode('utf-8', errors='replace')).hexdigest()}]"
    )
    return _redact_additional_credentials(fallback)


def _redact_additional_credentials(value: str) -> str:
    redacted = value
    for pattern in _VALIDATION_CREDENTIAL_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    return redacted


def _strip_validation_stack(value: str) -> str:
    kept: list[str] = []
    stack_omitted = False
    python_traceback = False
    for line in value.splitlines():
        if line.lstrip().startswith("Traceback ("):
            if not stack_omitted:
                kept.append("[stack omitted]")
                stack_omitted = True
            python_traceback = True
            continue
        if python_traceback:
            if _PYTHON_EXCEPTION_LINE_RE.match(line):
                python_traceback = False
                kept.append(line)
            continue
        if _STACK_LINE_RE.match(line):
            if not stack_omitted:
                kept.append("[stack omitted]")
                stack_omitted = True
            continue
        kept.append(line)
    return "\n".join(kept)


def _redact_host_paths(value: str) -> str:
    return _WINDOWS_HOST_PATH_RE.sub(
        _host_path_replacement,
        _POSIX_HOST_PATH_RE.sub(_host_path_replacement, value),
    )


def _host_path_replacement(match: re.Match[str]) -> str:
    source_location = _SANDBOX_SOURCE_LOCATION_RE.search(match.group(0))
    if source_location is not None:
        return (
            source_location.group("name")
            + (source_location.group("location") or "")
        )
    return "<host-path>"


def _bound_validation_text(
    value: str,
    *,
    original: str,
    max_chars: int,
    head_tail: bool,
) -> str:
    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(original.encode("utf-8", errors="replace")).hexdigest()
    marker = (
        f"\n[truncated chars={len(original)} sha256={digest}]\n"
        if head_tail
        else f"... [truncated chars={len(original)} sha256={digest}]"
    )
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return marker.strip()
    if not head_tail:
        return value[:remaining] + marker
    head_chars = (remaining + 1) // 2
    tail_chars = remaining - head_chars
    return value[:head_chars] + marker + (value[-tail_chars:] if tail_chars else "")


@dataclass(frozen=True)
class SandboxExecutionResult:
    value: Any
    metrics: CommandMetrics | None = None


def sandbox_failure_metrics(error: BaseException) -> CommandMetrics | None:
    """Return metrics retained on an ordinary sandbox execution failure."""

    metrics = getattr(error, _SANDBOX_FAILURE_METRICS_ATTR, None)
    return metrics if isinstance(metrics, CommandMetrics) else None


def _attach_sandbox_failure_metrics(
    error: BaseException,
    metrics: CommandMetrics | None,
) -> None:
    if metrics is None:
        return
    try:
        setattr(error, _SANDBOX_FAILURE_METRICS_ATTR, metrics)
    except (AttributeError, TypeError):
        # All built-in Agent libOS sandbox exceptions currently support
        # attributes. Keep the original failure type if a foreign one does not.
        return


@dataclass(frozen=True)
class _SupervisedProcessHandle:
    process: asyncio.subprocess.Process
    death_write_fd: int | None
    windows_job: WindowsJobObject | None


@dataclass(frozen=True)
class _DenoCheckTasks:
    communicate: asyncio.Task[tuple[bytes, bytes]]
    monitor: asyncio.Task[CommandMetrics]

    def all(self) -> tuple[asyncio.Task[Any], ...]:
        return (self.communicate, self.monitor)


class SandboxBackend:
    language = "typescript"

    def static_check(self, source_code: str) -> ValidationResult:
        raise NotImplementedError

    async def arun_source(
        self,
        source_code: str,
        args: dict[str, Any],
        *,
        pid: str | None = None,
        syscall_handler: SyscallHandler | None = None,
        timeout: float | None = None,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
        cached_only: bool = True,
    ) -> Any:
        raise NotImplementedError

    def run_source(
        self,
        source_code: str,
        args: dict[str, Any],
        *,
        pid: str | None = None,
        syscall_handler: SyscallHandler | None = None,
        timeout: float | None = None,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
        cached_only: bool = True,
    ) -> Any:
        kwargs = self._arun_source_kwargs(
            pid=pid,
            syscall_handler=syscall_handler,
            timeout=timeout,
            limits=limits,
            return_metrics=return_metrics,
            cached_only=cached_only,
        )
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun_source(source_code, args, **kwargs))
        raise RuntimeError("Cannot call run_source() inside a running event loop. Use await arun_source(...).")

    def _arun_source_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Pass new optional sandbox controls only to backends that support them."""
        signature = inspect.signature(self.arun_source)
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in signature.parameters}

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        raise NotImplementedError

    def metadata_for_source(self, source_code: str) -> dict[str, Any]:
        return {"language": self.language}


class DenoTypescriptSandbox(SandboxBackend):
    """Deno/TypeScript sandbox for Agent-authored tools.

    Candidate tools run as Deno userland programs without host permissions.
    Their only libOS access path is the NDJSON syscall protocol handled by the
    libOS runtime broker over stdin/stdout.
    """

    _RUN_EXPORT_RE = re.compile(r"export\s+(?:async\s+)?function\s+run\s*\(\s*args\b[\s\S]*?,\s*libos\b")

    def __init__(
        self,
        *,
        deno_executable: str = _TOOL_DEFAULTS.deno_executable,
        default_timeout_s: float = _TOOL_DEFAULTS.deno_timeout_s,
        max_rpc_calls: int = _TOOL_DEFAULTS.deno_max_rpc_calls,
        max_stdout_bytes: int = _TOOL_DEFAULTS.deno_max_stdout_bytes,
        max_stderr_bytes: int = _TOOL_DEFAULTS.deno_max_stderr_bytes,
        jsr_allowlist: tuple[str, ...] = _TOOL_DEFAULTS.deno_jsr_allowlist,
        max_source_chars: int = _TOOL_DEFAULTS.jit_source_max_chars,
        max_tests: int = _TOOL_DEFAULTS.jit_tests_max_count,
        max_test_case_bytes: int = _TOOL_DEFAULTS.jit_test_case_max_bytes,
        max_validation_log_chars: int = _TOOL_DEFAULTS.jit_validation_log_max_chars,
        forbidden_executable_roots: Iterable[str | Path] = (),
    ) -> None:
        self.deno_executable = deno_executable
        self.default_timeout_s = default_timeout_s
        self.max_rpc_calls = max_rpc_calls
        self.max_stdout_bytes = max_stdout_bytes
        self.max_stderr_bytes = max_stderr_bytes
        self.jsr_allowlist = tuple(jsr_allowlist)
        self.max_source_chars = max_source_chars
        self.max_tests = max_tests
        self.max_test_case_bytes = max_test_case_bytes
        self.max_validation_log_chars = max_validation_log_chars
        self.validation_result_preview_chars = min(
            max_validation_log_chars,
            _TOOL_DEFAULTS.tool_observability_preview_chars,
        )
        self.forbidden_executable_roots = tuple(Path(root).resolve() for root in forbidden_executable_roots)
        self.profile_builder = SandboxProfileBuilder()

    def static_check(self, source_code: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if len(source_code) > self.max_source_chars:
            errors.append(f"TypeScript tool source exceeds max chars: {self.max_source_chars}")
        if "\ufeff" in source_code:
            errors.append("Unicode byte-order marks are not allowed in JIT tool source")
        if not self._RUN_EXPORT_RE.search(source_code):
            errors.append("TypeScript tool source must export function run(args, libos)")
        if self._contains_dynamic_import(source_code):
            errors.append("dynamic import() is not allowed")
        if self._contains_runtime_code_generation(source_code):
            errors.append("runtime code generation is not allowed")
        if (
            _TYPESCRIPT_DEPENDENCY_DIRECTIVE_RE.search(source_code)
            or _DENO_TYPES_DIRECTIVE_RE.search(source_code)
        ):
            errors.append("TypeScript dependency directives are not allowed")
        for specifier in self._extract_imports(source_code):
            errors.append(f"imports are not allowed in JIT tool source: {specifier}")
        return ValidationResult(ok=not errors, errors=errors, warnings=warnings)

    async def arun_source(
        self,
        source_code: str,
        args: dict[str, Any],
        *,
        pid: str | None = None,
        syscall_handler: SyscallHandler | None = None,
        timeout: float | None = None,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
        cached_only: bool = True,
    ) -> Any:
        validation = self.static_check(source_code)
        if not validation.ok:
            raise SandboxError("; ".join(validation.errors))
        deno = self._resolve_deno()
        selected_timeout = self.default_timeout_s if timeout is None else timeout
        with tempfile.TemporaryDirectory(prefix="agent_libos_deno_tool_") as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "candidate.ts").write_text(source_code, encoding="utf-8")
            (tmp_path / "runner.ts").write_text(self._runner_source(), encoding="utf-8")
            command = [
                deno,
                "run",
                "--no-prompt",
                "--no-code-cache",
                "--no-remote",
                "--no-npm",
                "--deny-import",
            ]
            if cached_only:
                command.append("--cached-only")
            command.append("runner.ts")
            (
                launch_command,
                launch_kwargs,
                death_read_fd,
                death_write_fd,
                windows_job,
                windows_gate,
            ) = self._prepare_supervised_launch(command, tmp_path)
            launch_kwargs["env"] = self._isolated_deno_environment(tmp_path)
            proc: asyncio.subprocess.Process | None = None
            monitor_task: asyncio.Task[CommandMetrics] | None = None
            serve_task: asyncio.Task[Any] | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *launch_command,
                    cwd=tmp,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    limit=max(self.max_stdout_bytes, self.max_stderr_bytes) + 1,
                    **launch_kwargs,
                )
                if death_read_fd is not None:
                    os.close(death_read_fd)
                    death_read_fd = None
                self._attach_windows_supervisor(proc, windows_job, windows_gate)
                monitor_task = asyncio.create_task(self._monitor_process(proc, limits), name="deno-resource-monitor")
                serve_task = asyncio.create_task(
                    self._serve_process(proc, args, syscall_handler),
                    name="deno-syscall-server",
                )
                done, pending = await asyncio.wait(
                    {serve_task, monitor_task},
                    timeout=selected_timeout,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                if not done:
                    await self._raise_deno_execution_timeout(
                        proc,
                        monitor_task,
                        pending,
                        selected_timeout,
                    )
                self._raise_completed_task_failure(done)
                if serve_task not in done:
                    done_all, pending = await asyncio.wait(
                        {serve_task, monitor_task},
                        timeout=1.0,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                    if serve_task not in done_all:
                        await self._kill_process(proc)
                        for task in pending:
                            task.cancel()
                        raise SandboxError("Deno JIT tool exited before result")
                value = serve_task.result()
                metrics = await self._settle_monitor_metrics(monitor_task)
                wrapped = SandboxExecutionResult(value=value, metrics=metrics)
                return wrapped if return_metrics else value
            except (SubprocessLimitExceeded, SubprocessTimeoutExpired):
                await self._kill_process(proc)
                raise
            except TimeoutError as exc:
                await self._kill_process(proc)
                raise TimeoutError(f"Deno JIT tool timed out after {selected_timeout}s") from exc
            except Exception as exc:
                await self._retain_failed_execution_metrics(
                    exc,
                    proc,
                    monitor_task,
                )
                raise
            finally:
                try:
                    if proc is not None:
                        await asyncio.shield(self._kill_process(proc))
                    tasks = [task for task in (serve_task, monitor_task) if task is not None]
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    if death_read_fd is not None:
                        os.close(death_read_fd)
                    if death_write_fd is not None:
                        os.close(death_write_fd)
                    if windows_job is not None:
                        windows_job.close()

    async def _raise_deno_execution_timeout(
        self,
        proc: asyncio.subprocess.Process,
        monitor_task: asyncio.Task[CommandMetrics],
        pending: set[asyncio.Task[Any]],
        selected_timeout: float,
    ) -> NoReturn:
        await self._kill_process(proc)
        observed = await self._settle_monitor_metrics(monitor_task)
        metrics = observed or CommandMetrics()
        self._cancel_tasks(task for task in pending if task is not monitor_task)
        raise SubprocessTimeoutExpired(
            f"Deno JIT tool timed out after {selected_timeout}s",
            metrics=CommandMetrics(
                wall_seconds=max(float(selected_timeout), metrics.wall_seconds),
                cpu_seconds=metrics.cpu_seconds,
                peak_memory_bytes=metrics.peak_memory_bytes,
                killed=True,
                limit_kind="subprocess_timeout",
            ),
        )

    @staticmethod
    def _raise_completed_task_failure(done: set[asyncio.Task[Any]]) -> None:
        for task in done:
            error = task.exception()
            if error is not None:
                raise error

    async def _retain_failed_execution_metrics(
        self,
        error: Exception,
        proc: asyncio.subprocess.Process | None,
        monitor_task: asyncio.Task[CommandMetrics] | None,
    ) -> None:
        if proc is not None:
            await self._kill_process(proc)
        if monitor_task is None:
            return
        if (
            monitor_task.done()
            and not monitor_task.cancelled()
            and monitor_task.exception() is error
        ):
            return
        metrics = await self._settle_monitor_metrics(monitor_task)
        _attach_sandbox_failure_metrics(error, metrics)

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        validation = self.static_check(source_code)
        if not validation.ok:
            return validation
        size_errors = self._test_size_errors(tests)
        if size_errors:
            return ValidationResult(ok=False, errors=size_errors)
        try:
            version = self.deno_version()
        except SandboxError as exc:
            return ValidationResult(
                ok=False,
                errors=[
                    public_error_envelope(
                        exc,
                        code="deno_version_error",
                    )["message"]
                ],
            )
        errors: list[str] = []
        logs: list[str] = [f"language=typescript", f"deno={version}"]
        metrics: list[CommandMetrics] = []
        if not tests:
            try:
                check_metrics = self._check_source_with_deno(
                    source_code,
                    timeout=timeout,
                    limits=limits,
                )
                metrics.append(check_metrics)
                logs.append("deno_check=ok")
            except (SubprocessLimitExceeded, SubprocessTimeoutExpired):
                raise
            except Exception as exc:
                failure_metrics = sandbox_failure_metrics(exc)
                if failure_metrics is not None:
                    metrics.append(failure_metrics)
                errors.append(
                    self._validation_failure(
                        status="failed",
                        reason="Deno type-check failed",
                        error=exc,
                    )
                )
        for index, test in enumerate(tests, start=1):
            syscall_handler, assert_syscalls_consumed = self._test_syscall_handler(test, index)
            try:
                result = self.run_source(
                    source_code,
                    test.get("args", {}),
                    syscall_handler=syscall_handler,
                    timeout=timeout,
                    limits=limits,
                    return_metrics=True,
                    cached_only=False,
                )
                if isinstance(result, SandboxExecutionResult):
                    metrics.append(result.metrics or CommandMetrics())
                    result_value = result.value
                else:
                    result_value = result
                assert_syscalls_consumed()
            except (SubprocessLimitExceeded, SubprocessTimeoutExpired):
                raise
            except Exception as exc:
                failure_metrics = sandbox_failure_metrics(exc)
                if failure_metrics is not None:
                    metrics.append(failure_metrics)
                errors.append(
                    self._validation_failure(
                        status="failed",
                        reason="execution_error",
                        test=index,
                        error=exc,
                    )
                )
                continue
            matches = _exact_json_equal(result_value, test["expected"])
            result_summary = summarize_validation_value(
                result_value,
                preview_chars=self.validation_result_preview_chars,
                head_tail=not matches,
            )
            logs.append(
                self._validation_record(
                    {
                        "result": result_summary,
                        "status": "passed" if matches else "failed",
                        "test": index,
                    }
                )
            )
            if not matches:
                errors.append(
                    self._validation_record(
                        {
                            "actual": result_summary,
                            "expected": summarize_validation_value(
                                test["expected"],
                                preview_chars=self.validation_result_preview_chars,
                                head_tail=True,
                            ),
                            "reason": "result_mismatch",
                            "status": "failed",
                            "test": index,
                        }
                    )
                )
        metadata = {"metrics": self._aggregate_metrics(metrics)} if return_metrics else {}
        return ValidationResult(ok=not errors, errors=errors, logs=self._bounded_logs(logs), metadata=metadata)

    def _check_source_with_deno(
        self,
        source_code: str,
        *,
        timeout: float | None,
        limits: SubprocessLimits | None,
    ) -> CommandMetrics:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._acheck_source_with_deno(
                    source_code,
                    timeout=timeout,
                    limits=limits,
                )
            )
        raise RuntimeError(
            "Cannot type-check JIT source inside a running event loop. "
            "Run validation in a worker."
        )

    async def _acheck_source_with_deno(
        self,
        source_code: str,
        *,
        timeout: float | None,
        limits: SubprocessLimits | None,
    ) -> CommandMetrics:
        """Type-check an otherwise untested candidate without executing it."""

        deno = self._resolve_deno()
        selected_timeout = self.default_timeout_s if timeout is None else timeout
        with tempfile.TemporaryDirectory(prefix="agent_libos_deno_check_") as tmp:
            tmp_path = Path(tmp)
            self._write_deno_check_files(tmp_path, source_code)
            handle = await self._start_deno_check_process(
                self._deno_check_command(deno),
                tmp_path,
            )
            tasks: _DenoCheckTasks | None = None
            try:
                tasks = self._start_deno_check_tasks(handle.process, limits)
                stdout, stderr, metrics = await self._wait_for_deno_check(
                    tasks,
                    selected_timeout,
                )
                try:
                    self._validate_deno_check_output(
                        handle.process,
                        stdout,
                        stderr,
                    )
                except Exception as exc:
                    _attach_sandbox_failure_metrics(exc, metrics)
                    raise
                return metrics
            finally:
                await self._cleanup_deno_check(handle, tasks)

    @staticmethod
    def _write_deno_check_files(tmp_path: Path, source_code: str) -> None:
        (tmp_path / "candidate.ts").write_text(source_code, encoding="utf-8")
        (tmp_path / "deno.json").write_text(
            json.dumps({"compilerOptions": {"strict": False}}),
            encoding="utf-8",
        )

    @staticmethod
    def _deno_check_command(deno: str) -> list[str]:
        return [
            deno,
            "check",
            "--quiet",
            "--no-code-cache",
            "--config",
            "deno.json",
            "--no-lock",
            "--no-remote",
            "--no-npm",
            "candidate.ts",
        ]

    async def _start_deno_check_process(
        self,
        command: list[str],
        tmp_path: Path,
    ) -> _SupervisedProcessHandle:
        (
            launch_command,
            launch_kwargs,
            death_read_fd,
            death_write_fd,
            windows_job,
            windows_gate,
        ) = self._prepare_supervised_launch(command, tmp_path)
        launch_kwargs["env"] = self._isolated_deno_environment(tmp_path)
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *launch_command,
                cwd=str(tmp_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **launch_kwargs,
            )
            if death_read_fd is not None:
                os.close(death_read_fd)
                death_read_fd = None
            self._attach_windows_supervisor(process, windows_job, windows_gate)
            return _SupervisedProcessHandle(
                process=process,
                death_write_fd=death_write_fd,
                windows_job=windows_job,
            )
        except BaseException:
            try:
                if process is not None and process.returncode is None:
                    await self._kill_process(process)
            finally:
                self._close_supervisor_resources(
                    death_read_fd=death_read_fd,
                    death_write_fd=death_write_fd,
                    windows_job=windows_job,
                )
            raise

    @staticmethod
    def _attach_windows_supervisor(
        process: asyncio.subprocess.Process,
        windows_job: WindowsJobObject | None,
        windows_gate: Path | None,
    ) -> None:
        if windows_job is None:
            return
        try:
            windows_job.assign_pid(process.pid)
            assert windows_gate is not None
            windows_gate.write_text("contained\n", encoding="utf-8")
        except Exception as exc:
            raise SandboxError(
                "failed to attach Deno supervisor to Windows Job Object"
            ) from exc

    def _start_deno_check_tasks(
        self,
        process: asyncio.subprocess.Process,
        limits: SubprocessLimits | None,
    ) -> _DenoCheckTasks:
        return _DenoCheckTasks(
            communicate=asyncio.create_task(
                process.communicate(),
                name="deno-check-communicate",
            ),
            monitor=asyncio.create_task(
                self._monitor_process(process, limits),
                name="deno-check-resource-monitor",
            ),
        )

    async def _wait_for_deno_check(
        self,
        tasks: _DenoCheckTasks,
        selected_timeout: float,
    ) -> tuple[bytes, bytes, CommandMetrics]:
        all_tasks = tasks.all()
        done, pending = await asyncio.wait(
            all_tasks,
            timeout=selected_timeout,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        if not done:
            self._cancel_tasks(pending)
            raise SubprocessTimeoutExpired(
                f"Deno JIT type-check timed out after {selected_timeout}s",
                metrics=CommandMetrics(
                    wall_seconds=float(selected_timeout),
                    killed=True,
                    limit_kind="subprocess_timeout",
                ),
            )
        for task in done:
            exc = task.exception()
            if exc is not None:
                self._cancel_tasks(pending)
                raise exc
        if tasks.communicate not in done or tasks.monitor not in done:
            self._cancel_tasks(pending)
            raise SandboxError("Deno JIT type-check did not settle")
        stdout, stderr = tasks.communicate.result()
        return stdout, stderr, tasks.monitor.result()

    def _validate_deno_check_output(
        self,
        process: asyncio.subprocess.Process,
        stdout: bytes,
        stderr: bytes,
    ) -> None:
        ready_line, separator, child_stdout = stdout.partition(b"\n")
        try:
            ready_frame = json.loads(ready_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxError(
                f"Deno supervisor produced invalid readiness frame: {ready_line[:200]!r}"
            ) from exc
        if not separator or ready_frame != {
            "type": "supervisor_ready",
            "version": 1,
        }:
            raise SandboxError(
                f"unexpected Deno supervisor readiness frame: {ready_frame!r}"
            )
        if len(child_stdout) > self.max_stdout_bytes:
            raise SandboxError("Deno JIT type-check stdout exceeded max bytes")
        if len(stderr) > self.max_stderr_bytes:
            raise SandboxError("Deno JIT type-check stderr exceeded max bytes")
        if process.returncode != 0:
            detail = (
                stderr.decode("utf-8", errors="replace").strip()
                or child_stdout.decode("utf-8", errors="replace").strip()
                or f"deno check exited {process.returncode}"
            )
            raise SandboxError(detail)

    async def _cleanup_deno_check(
        self,
        handle: _SupervisedProcessHandle,
        tasks: _DenoCheckTasks | None,
    ) -> None:
        all_tasks = tasks.all() if tasks is not None else ()
        try:
            if handle.process.returncode is None:
                await asyncio.shield(self._kill_process(handle.process))
        finally:
            try:
                self._cancel_tasks(all_tasks)
                if all_tasks:
                    await asyncio.gather(*all_tasks, return_exceptions=True)
            finally:
                self._close_supervisor_resources(
                    death_write_fd=handle.death_write_fd,
                    windows_job=handle.windows_job,
                )

    @staticmethod
    def _cancel_tasks(tasks: Iterable[asyncio.Task[Any]]) -> None:
        for task in tasks:
            if not task.done():
                task.cancel()

    @staticmethod
    def _close_supervisor_resources(
        *,
        death_read_fd: int | None = None,
        death_write_fd: int | None = None,
        windows_job: WindowsJobObject | None = None,
    ) -> None:
        if death_read_fd is not None:
            os.close(death_read_fd)
        if death_write_fd is not None:
            os.close(death_write_fd)
        if windows_job is not None:
            windows_job.close()

    def metadata_for_source(self, source_code: str) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "language": "typescript",
            "imports": self._extract_imports(source_code),
            "jsr_allowlist": list(self.jsr_allowlist),
            "sandbox_profile": self._profile_json(self.profile_builder.deno_jit()),
        }
        try:
            metadata["deno_version"] = self.deno_version()
        except SandboxError as exc:
            metadata["deno_version_error"] = public_error_envelope(
                exc,
                code="deno_version_error",
            )["message"]
        return metadata

    def deno_version(self) -> str:
        deno = self._resolve_deno()
        try:
            proc = subprocess.run(
                [deno, "--version"],
                text=True,
                capture_output=True,
                timeout=min(self.default_timeout_s, 5.0),
            )
        except Exception as exc:
            raise SandboxError(f"failed to run Deno executable {deno!r}: {exc}") from exc
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip() or f"deno exited {proc.returncode}"
            raise SandboxError(message)
        return proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "deno"

    async def _serve_process(
        self,
        proc: asyncio.subprocess.Process,
        args: dict[str, Any],
        syscall_handler: SyscallHandler | None,
    ) -> Any:
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise SandboxError("Deno process was not created with stdio pipes")
        stderr_task = asyncio.create_task(proc.stderr.read(self.max_stderr_bytes + 1))
        ready_line = await proc.stdout.readline()
        if not ready_line:
            stderr, _stderr_truncated = await self._finish_stderr(stderr_task)
            code = await proc.wait()
            raise SandboxError(stderr.strip() or f"Deno supervisor exited before containment readiness: {code}")
        try:
            ready_frame = json.loads(ready_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxError(f"Deno supervisor produced invalid readiness frame: {ready_line[:200]!r}") from exc
        if ready_frame != {"type": "supervisor_ready", "version": 1}:
            raise SandboxError(f"unexpected Deno supervisor readiness frame: {ready_frame!r}")
        provider_error_proof = secrets.token_urlsafe(32)
        await self._write_frame(
            proc,
            {
                "type": "run",
                "args": to_jsonable(args),
                "provider_error_proof": provider_error_proof,
            },
        )
        stdout_bytes = 0
        rpc_calls = 0
        while True:
            line = await proc.stdout.readline()
            if not line:
                stderr, _stderr_truncated = await self._finish_stderr(stderr_task)
                code = await proc.wait()
                raise SandboxError(stderr.strip() or f"Deno JIT tool exited before result: {code}")
            stdout_bytes += len(line)
            if stdout_bytes > self.max_stdout_bytes:
                raise SandboxError("Deno JIT stdout exceeded max bytes")
            try:
                frame = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise SandboxError(f"Deno JIT produced non-protocol stdout: {line[:200]!r}") from exc
            frame_type = frame.get("type")
            if frame_type == "syscall":
                rpc_calls += 1
                if rpc_calls > self.max_rpc_calls:
                    await self._write_frame(
                        proc,
                        {
                            "type": "syscall_result",
                            "id": frame.get("id"),
                            "ok": False,
                            "error": f"Deno JIT exceeded max_rpc_calls={self.max_rpc_calls}",
                        },
                    )
                    continue
                await self._handle_syscall_frame(proc, frame, syscall_handler)
                continue
            if frame_type == "result":
                # The result frame is the protocol boundary for a successful
                # tool call. Any remaining Deno event-loop handles belong to
                # this transient tool process and must not delay lifecycle
                # syscalls or scheduler progress.
                value = frame.get("value")
                await self._kill_process(proc)
                stderr, stderr_truncated = await self._finish_stderr(stderr_task)
                if stderr_truncated:
                    raise SandboxError("Deno JIT stderr exceeded max bytes")
                return value
            if frame_type == "error":
                await self._kill_process(proc)
                stderr, _stderr_truncated = await self._finish_stderr(stderr_task)
                self._raise_runner_error(
                    frame,
                    stderr,
                    provider_error_proof=provider_error_proof,
                )
            raise SandboxError(f"unknown Deno JIT protocol frame: {frame_type!r}")

    @staticmethod
    def _raise_runner_error(
        frame: dict[str, Any],
        stderr: str,
        *,
        provider_error_proof: str,
    ) -> NoReturn:
        frame_proof = frame.get("provider_error_proof")
        proof_matches = isinstance(frame_proof, str) and secrets.compare_digest(
            frame_proof,
            provider_error_proof,
        )
        public_error = (
            provider_error_envelope_from_mapping(frame)
            if proof_matches
            else None
        )
        if public_error is not None:
            raise ProviderHostError(
                code=public_error["code"],
                error_type=public_error["error_type"],
                correlation_id=public_error["correlation_id"],
            )
        message = str(frame.get("message") or stderr or "Deno JIT tool failed")
        raise SandboxError(message)

    async def _monitor_process(
        self,
        proc: asyncio.subprocess.Process,
        limits: SubprocessLimits | None,
    ) -> CommandMetrics:
        started_at = time.monotonic()
        peak_memory = 0
        cpu_seconds = 0.0
        cpu_high_water_by_pid: dict[int, float] = {}
        try:
            ps_proc = psutil.Process(proc.pid)
        except (psutil.Error, OSError) as exc:
            if limits is not None:
                await self._kill_process(proc)
                raise SandboxError("Deno resource monitor cannot inspect a budgeted subprocess") from exc
            return CommandMetrics(wall_seconds=max(0.0, time.monotonic() - started_at))
        while proc.returncode is None:
            wall_seconds = time.monotonic() - started_at
            try:
                cpu_seconds, peak_memory = self._sample_process_tree(
                    ps_proc,
                    peak_memory,
                    cpu_high_water_by_pid,
                )
            except OSError as exc:
                if limits is not None:
                    await self._kill_process(proc)
                    raise SandboxError("Deno resource monitor cannot enforce subprocess limits") from exc
                # The outer wall timeout and process-group containment remain
                # active even where the host denies process-tree inspection.
            limit_kind = self._limit_kind(
                wall_seconds=wall_seconds,
                cpu_seconds=cpu_seconds,
                peak_memory=peak_memory,
                limits=limits,
            )
            if limit_kind is not None:
                await self._kill_process(proc)
                metrics = CommandMetrics(
                    wall_seconds=wall_seconds,
                    cpu_seconds=cpu_seconds,
                    peak_memory_bytes=peak_memory,
                    killed=True,
                    limit_kind=limit_kind,
                )
                raise SubprocessLimitExceeded(
                    f"Deno JIT subprocess exceeded {limit_kind}",
                    metrics=metrics,
                )
            await asyncio.sleep(0.02)
        wall_seconds = time.monotonic() - started_at
        try:
            final_cpu_seconds, peak_memory = self._sample_process_tree(
                ps_proc,
                peak_memory,
                cpu_high_water_by_pid,
            )
        except OSError as exc:
            if limits is not None:
                raise SandboxError("Deno resource monitor could not verify subprocess limits") from exc
            final_cpu_seconds = cpu_seconds
        return CommandMetrics(
            wall_seconds=wall_seconds,
            cpu_seconds=max(cpu_seconds, final_cpu_seconds),
            peak_memory_bytes=peak_memory,
            killed=False,
            limit_kind=None,
        )

    @staticmethod
    async def _settle_monitor_metrics(
        monitor_task: asyncio.Task[CommandMetrics],
    ) -> CommandMetrics | None:
        if not monitor_task.done():
            done, _pending = await asyncio.wait(
                {monitor_task},
                timeout=_PROCESS_CLEANUP_TIMEOUT_S,
                return_when=asyncio.ALL_COMPLETED,
            )
            if not done:
                monitor_task.cancel()
                return None
        if monitor_task.cancelled():
            return None
        return monitor_task.result()

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
        cpu_high_water_by_pid: dict[int, float] | None = None,
    ) -> tuple[float, int]:
        cpu_by_pid = (
            cpu_high_water_by_pid
            if cpu_high_water_by_pid is not None
            else {}
        )
        memory_bytes = 0
        processes = [proc]
        try:
            processes.extend(proc.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            pass
        except (psutil.Error, OSError) as exc:
            raise OSError("cannot enumerate Deno subprocess tree") from exc
        for item in processes:
            try:
                times = item.cpu_times()
                sampled_cpu = float(times.user) + float(times.system)
                memory_bytes += int(item.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
            except (psutil.Error, OSError) as exc:
                raise OSError("cannot inspect Deno subprocess resource usage") from exc
            cpu_by_pid[item.pid] = max(cpu_by_pid.get(item.pid, 0.0), sampled_cpu)
        return sum(cpu_by_pid.values()), max(peak_memory, memory_bytes)

    async def _handle_syscall_frame(
        self,
        proc: asyncio.subprocess.Process,
        frame: dict[str, Any],
        syscall_handler: SyscallHandler | None,
    ) -> None:
        frame_id = frame.get("id")
        if syscall_handler is None:
            await self._write_frame(
                proc,
                {"type": "syscall_result", "id": frame_id, "ok": False, "error": "libOS syscall handler is unavailable"},
            )
            return
        name = str(frame.get("name") or "")
        args = frame.get("args")
        if not isinstance(args, dict):
            args = {}
        try:
            result = syscall_handler(name, args)
            if inspect.isawaitable(result):
                result = await result
            await self._write_frame(
                proc,
                {"type": "syscall_result", "id": frame_id, "ok": True, "payload": to_jsonable(result)},
            )
        except Exception as exc:
            public_error = public_error_envelope(exc, code="syscall_error")
            await self._write_frame(
                proc,
                {
                    "type": "syscall_result",
                    "id": frame_id,
                    "ok": False,
                    "error": public_error["message"],
                    "error_type": public_error["error_type"],
                    "code": public_error["code"],
                    "correlation_id": public_error["correlation_id"],
                },
            )

    async def _write_frame(self, proc: asyncio.subprocess.Process, frame: dict[str, Any]) -> None:
        if proc.stdin is None:
            raise SandboxError("Deno process stdin is closed")
        proc.stdin.write((json.dumps(frame, ensure_ascii=True, default=str) + "\n").encode("utf-8"))
        await proc.stdin.drain()

    async def _finish_stderr(self, stderr_task: asyncio.Task[bytes]) -> tuple[str, bool]:
        try:
            data = await stderr_task
        except Exception:
            return "", False
        return data[: self.max_stderr_bytes].decode("utf-8", errors="replace"), len(data) > self.max_stderr_bytes

    def _prepare_supervised_launch(
        self,
        command: list[str],
        tmp_path: Path,
    ) -> tuple[
        list[str],
        dict[str, Any],
        int | None,
        int | None,
        WindowsJobObject | None,
        Path | None,
    ]:
        supervisor = Path(__file__).with_name("_process_supervisor.py")
        if not supervisor.is_file():
            raise SandboxError(f"Deno subprocess supervisor is unavailable: {supervisor}")
        launch_command = [
            sys.executable,
            "-I",
            str(supervisor),
            "--parent-pid",
            str(os.getpid()),
        ]
        launch_kwargs = self._subprocess_group_kwargs()
        if os.name == "posix":
            try:
                death_read_fd, death_write_fd = os.pipe()
            except OSError as exc:
                raise SandboxError("failed to create Deno supervisor death pipe") from exc
            launch_command.extend(["--death-fd", str(death_read_fd), "--", *command])
            launch_kwargs["pass_fds"] = (death_read_fd,)
            return launch_command, launch_kwargs, death_read_fd, death_write_fd, None, None
        if os.name == "nt":
            try:
                job = WindowsJobObject.create()
            except OSError as exc:
                raise SandboxError("failed to create Deno KILL_ON_JOB_CLOSE Job Object") from exc
            gate = tmp_path / ".deno-supervisor-contained"
            launch_command.extend(["--gate-file", str(gate), "--", *command])
            return launch_command, launch_kwargs, None, None, job, gate
        raise SandboxError(f"Deno subprocess containment is unavailable on platform: {os.name}")

    def _subprocess_group_kwargs(self) -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return {"start_new_session": True}

    @staticmethod
    def _isolated_deno_environment(tmp_path: Path) -> dict[str, str]:
        """Keep Deno's permission-exempt caches and storage inside one run."""

        roots = {
            "DENO_DIR": tmp_path / ".deno",
            "HOME": tmp_path / ".home",
            "TMPDIR": tmp_path / ".tmp",
            "TMP": tmp_path / ".tmp",
            "TEMP": tmp_path / ".tmp",
            "XDG_CACHE_HOME": tmp_path / ".cache",
            "XDG_CONFIG_HOME": tmp_path / ".config",
            "XDG_DATA_HOME": tmp_path / ".data",
        }
        for root in set(roots.values()):
            root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update({key: str(value) for key, value in roots.items()})
        environment.update(
            {
                "DENO_NO_PROMPT": "1",
                "DENO_NO_UPDATE_CHECK": "1",
                "NO_COLOR": "1",
            }
        )
        if os.name == "nt":
            environment.update(
                {
                    "APPDATA": str(tmp_path / ".data"),
                    "LOCALAPPDATA": str(tmp_path / ".data"),
                    "USERPROFILE": str(tmp_path / ".home"),
                }
            )
        return environment

    async def _kill_process(self, proc: asyncio.subprocess.Process) -> None:
        # Cleanup is intentionally idempotent. Re-signalling a process-group ID
        # after asyncio has reaped its leader risks targeting a recycled ID.
        if proc.returncode is not None:
            return
        descendants: list[psutil.Process] = []
        tree_available = True
        try:
            descendants = psutil.Process(proc.pid).children(recursive=True)
        except (psutil.Error, OSError):
            tree_available = False
        group_permission_error: PermissionError | None = None
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                group_permission_error = exc
        for child in reversed(descendants):
            try:
                child.kill()
            except (psutil.Error, OSError):
                continue
        if proc.returncode is None:
            try:
                proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
        root_settled = await self._wait_for_deno_process(proc)
        if group_permission_error is None:
            if not root_settled:
                raise SandboxError(f"Deno supervisor {proc.pid} did not terminate during cleanup")
            return
        descendants_settled = await self._wait_for_deno_descendants(descendants)
        if tree_available and root_settled and descendants_settled:
            return
        raise SandboxError(
            f"failed to terminate Deno process group {proc.pid} and could not verify process-tree fallback"
        ) from group_permission_error

    @staticmethod
    async def _wait_for_deno_process(proc: asyncio.subprocess.Process) -> bool:
        if proc.returncode is not None:
            return True
        try:
            await asyncio.wait_for(proc.wait(), timeout=_PROCESS_CLEANUP_TIMEOUT_S)
        except (TimeoutError, ProcessLookupError, PermissionError):
            return proc.returncode is not None
        return proc.returncode is not None

    @staticmethod
    async def _wait_for_deno_descendants(descendants: list[psutil.Process]) -> bool:
        if not descendants:
            return True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _PROCESS_CLEANUP_TIMEOUT_S
        alive = descendants
        while alive:
            try:
                _gone, alive = psutil.wait_procs(alive, timeout=0.0)
            except (psutil.Error, OSError):
                return False
            if not alive:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(_PROCESS_CLEANUP_POLL_INTERVAL_S, remaining))
        return True

    def _test_syscall_handler(self, test: dict[str, Any], index: int) -> tuple[SyscallHandler, Callable[[], None]]:
        expected = list(test.get("syscalls", []))
        total_expected = len(expected)

        async def handler(name: str, args: dict[str, Any]) -> Any:
            if not expected:
                raise SandboxError(f"test {index} did not expect syscall {name}")
            spec = expected.pop(0)
            expected_name = spec.get("name")
            if expected_name != name:
                raise SandboxError(f"test {index} expected syscall {expected_name}, got {name}")
            if "args" in spec and spec["args"] != args:
                raise SandboxError(f"test {index} syscall {name} expected args {spec['args']!r}, got {args!r}")
            if spec.get("ok", True) is False:
                # Candidate self-tests must exercise the same text-free syscall
                # boundary as a live JIT call while remaining deterministic.
                # The spec-authored diagnostic is intentionally not exposed to
                # candidate code or included in equality-based expectations.
                ordinal = total_expected - len(expected)
                raise ProviderHostError(
                    code="syscall_error",
                    error_type="SandboxError",
                    correlation_id=f"corr_jit_test_{index}_{ordinal}",
                )
            return spec.get("result", spec.get("payload"))

        def assert_consumed() -> None:
            if expected:
                missing = [str(spec.get("name", "<unnamed>")) for spec in expected]
                raise SandboxError(f"test {index} expected syscall(s) not performed: {missing}")

        return handler, assert_consumed

    def _test_size_errors(self, tests: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        if len(tests) > self.max_tests:
            errors.append(f"JIT tests exceed max count: {self.max_tests}")
        for index, test in enumerate(tests, start=1):
            if not isinstance(test, dict):
                errors.append(f"JIT test {index} must be a JSON object")
                continue
            try:
                ensure_json_size(test, self.max_test_case_bytes, f"JIT test {index}")
            except Exception as exc:
                errors.append(str(exc))
            if "expected" not in test:
                errors.append(f"JIT test {index} must include expected")
        return errors

    def _aggregate_metrics(self, metrics: list[CommandMetrics]) -> dict[str, Any]:
        if not metrics:
            return {
                "wall_seconds": 0.0,
                "cpu_seconds": 0.0,
                "peak_memory_bytes": 0,
                "killed": False,
                "limit_kind": None,
            }
        return {
            "wall_seconds": sum(item.wall_seconds for item in metrics),
            "cpu_seconds": sum(item.cpu_seconds for item in metrics),
            "peak_memory_bytes": max(item.peak_memory_bytes for item in metrics),
            "killed": any(item.killed for item in metrics),
            "limit_kind": next((item.limit_kind for item in metrics if item.limit_kind), None),
        }

    def _validation_record(self, value: Mapping[str, Any]) -> str:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return compact_validation_diagnostic(
            serialized,
            max_chars=self.max_validation_log_chars,
            head_tail=True,
        )

    def _validation_failure(
        self,
        *,
        status: str,
        reason: str,
        error: BaseException,
        test: int | None = None,
    ) -> str:
        record: dict[str, Any] = {
            "error": {
                "message": compact_validation_diagnostic(
                    error,
                    max_chars=self.validation_result_preview_chars,
                    head_tail=True,
                ),
                "type": type(error).__name__,
            },
            "reason": reason,
            "status": status,
        }
        if test is not None:
            record["test"] = test
        return self._validation_record(record)

    def _bounded_logs(self, logs: list[str]) -> str:
        text = "\n".join(logs)
        return compact_validation_diagnostic(
            text,
            max_chars=self.max_validation_log_chars,
            head_tail=True,
        )

    def _contains_dynamic_import(self, source_code: str) -> bool:
        tokens = self._typescript_tokens(source_code, preserve_line_breaks=True)
        for index, token in enumerate(tokens):
            if token != ("identifier", "import"):
                continue
            previous_index = self._previous_significant_token_index(tokens, index)
            previous_token = tokens[previous_index] if previous_index is not None else None
            if previous_token == ("punct", "."):
                continue
            next_index = self._next_significant_token_index(tokens, index)
            next_token = tokens[next_index] if next_index is not None else None
            if next_token == ("punct", "("):
                if self._looks_like_method_definition(tokens, index):
                    continue
                return True
        return False

    def _contains_runtime_code_generation(self, source_code: str) -> bool:
        tokens = self._typescript_tokens(source_code)
        for index, token in enumerate(tokens):
            if token[0] == "identifier" and token[1] in _RUNTIME_GLOBAL_OBJECT_NAMES:
                if self._token_starts_global_runtime_code_generation_property(tokens, index):
                    return True
            if token[0] == "identifier" and token[1] in _RUNTIME_CODE_GENERATION_NAMES:
                if self._looks_like_method_definition(tokens, index):
                    continue
                previous_token = tokens[index - 1] if index > 0 else None
                if previous_token == ("punct", "."):
                    continue
                if previous_token == ("punct", "["):
                    continue
                if self._looks_like_property_key(tokens, index):
                    continue
                if self._looks_like_declaration_name(tokens, index):
                    continue
                if self._looks_like_member_name(tokens, index):
                    continue
                return True
            if token[0] == "string" and token[1] in _RUNTIME_CODE_GENERATION_NAMES:
                if self._token_is_bracketed_global_property(tokens, index):
                    return True
            if token == ("identifier", "constructor"):
                previous_token = tokens[index - 1] if index > 0 else None
                if previous_token == ("punct", "."):
                    return True
            if token == ("string", "constructor") and self._token_is_bracketed_property(tokens, index):
                return True
        return False

    def _token_is_bracketed_global_property(self, tokens: list[tuple[str, str]], index: int) -> bool:
        property_name, open_index, _ = self._constant_bracket_property_name_at(tokens, index)
        if property_name not in _RUNTIME_CODE_GENERATION_NAMES:
            return False
        object_index = open_index - 1
        if object_index < 0:
            return False
        if tokens[object_index] == ("punct", ".") and object_index > 0:
            object_index -= 1
        if tokens[object_index] == ("punct", "?") and object_index > 0:
            object_index -= 1
        return tokens[object_index][0] == "identifier" and tokens[object_index][1] in _RUNTIME_GLOBAL_OBJECT_NAMES

    def _token_is_bracketed_property(self, tokens: list[tuple[str, str]], index: int) -> bool:
        property_name, open_index, _ = self._constant_bracket_property_name_at(tokens, index)
        return property_name == "constructor" and open_index > 0

    def _token_starts_global_runtime_code_generation_property(self, tokens: list[tuple[str, str]], index: int) -> bool:
        cursor = index + 1
        if cursor < len(tokens) and tokens[cursor] == ("punct", "?"):
            cursor += 1
        if cursor < len(tokens) and tokens[cursor] == ("punct", "."):
            cursor += 1
        if cursor >= len(tokens):
            return False
        token = tokens[cursor]
        if token[0] == "identifier":
            return token[1] in _RUNTIME_CODE_GENERATION_NAMES
        if token == ("punct", "["):
            property_name, _ = self._constant_bracket_property_name_from_open(tokens, cursor)
            return property_name in _RUNTIME_CODE_GENERATION_NAMES
        return False

    def _constant_bracket_property_name_at(
        self, tokens: list[tuple[str, str]], index: int
    ) -> tuple[str | None, int, int]:
        if index < 1 or tokens[index - 1] != ("punct", "["):
            return None, -1, -1
        property_name, close_index = self._constant_bracket_property_name_from_open(tokens, index - 1)
        return property_name, index - 1, close_index

    def _constant_bracket_property_name_from_open(
        self, tokens: list[tuple[str, str]], open_index: int
    ) -> tuple[str | None, int]:
        parts: list[str] = []
        cursor = open_index + 1
        expecting_string = True
        while cursor < len(tokens):
            token = tokens[cursor]
            if token == ("punct", "]"):
                return ("".join(parts), cursor) if parts and not expecting_string else (None, cursor)
            if expecting_string and token[0] == "string":
                parts.append(token[1])
                expecting_string = False
                cursor += 1
                continue
            if not expecting_string and token == ("punct", "+"):
                expecting_string = True
                cursor += 1
                continue
            return None, cursor
        return None, cursor

    def _looks_like_property_key(self, tokens: list[tuple[str, str]], index: int) -> bool:
        previous_token = tokens[index - 1] if index > 0 else None
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        return previous_token in {("punct", "{"), ("punct", ",")} and next_token == ("punct", ":")

    def _looks_like_declaration_name(self, tokens: list[tuple[str, str]], index: int) -> bool:
        previous_token = tokens[index - 1] if index > 0 else None
        return previous_token in {
            ("identifier", "const"),
            ("identifier", "let"),
            ("identifier", "var"),
            ("identifier", "function"),
            ("identifier", "class"),
            ("identifier", "interface"),
            ("identifier", "type"),
        }

    def _looks_like_member_name(self, tokens: list[tuple[str, str]], index: int) -> bool:
        previous_token = tokens[index - 1] if index > 0 else None
        return previous_token == ("punct", ".")

    def _extract_imports(self, source_code: str) -> list[str]:
        tokens = self._typescript_tokens(source_code)
        imports: set[str] = set()
        for index, token in enumerate(tokens):
            if token == ("identifier", "import"):
                imports.update(self._module_specifier_after_import(tokens, index))
            elif token == ("identifier", "export"):
                imports.update(self._module_specifier_after_export(tokens, index))
        return sorted(imports)

    def _module_specifier_after_import(self, tokens: list[tuple[str, str]], index: int) -> list[str]:
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if next_token is None or next_token == ("punct", "("):
            return []
        if next_token[0] == "string":
            return [next_token[1]]
        specifier = self._string_after_from(tokens, index + 1)
        if specifier is None:
            specifier = self._string_after_import_require(tokens, index + 1)
        return [specifier] if specifier is not None else []

    def _string_after_import_require(
        self,
        tokens: list[tuple[str, str]],
        index: int,
    ) -> str | None:
        for cursor in range(index, len(tokens)):
            token = tokens[cursor]
            if token == ("punct", ";"):
                return None
            if token != ("identifier", "require"):
                continue
            open_token = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
            value_token = tokens[cursor + 2] if cursor + 2 < len(tokens) else None
            if (
                open_token == ("punct", "(")
                and value_token is not None
                and value_token[0] == "string"
            ):
                return value_token[1]
        return None

    def _module_specifier_after_export(self, tokens: list[tuple[str, str]], index: int) -> list[str]:
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        if next_token is not None and next_token[0] == "identifier" and next_token[1] in {
            "abstract",
            "async",
            "class",
            "const",
            "declare",
            "default",
            "enum",
            "function",
            "interface",
            "let",
            "namespace",
            "var",
        }:
            return []
        specifier = self._string_after_from(tokens, index + 1)
        return [specifier] if specifier is not None else []

    def _looks_like_method_definition(self, tokens: list[tuple[str, str]], index: int) -> bool:
        previous_index = self._previous_significant_token_index(tokens, index)
        previous_token = tokens[previous_index] if previous_index is not None else None
        if previous_token not in {("punct", "{"), ("punct", ",")}:
            return False
        depth = 0
        for cursor in range(index + 1, len(tokens)):
            token = tokens[cursor]
            if token == ("punct", "("):
                depth += 1
                continue
            if token == ("punct", ")"):
                depth -= 1
                if depth == 0:
                    next_index = self._next_significant_token_index(tokens, cursor)
                    if next_index is None:
                        return False
                    if any(
                        token_kind == "linebreak"
                        for token_kind, _token_value in tokens[cursor + 1 : next_index]
                    ):
                        return False
                    return tokens[next_index] == ("punct", "{")
        return False

    @staticmethod
    def _previous_significant_token_index(
        tokens: list[tuple[str, str]],
        index: int,
    ) -> int | None:
        cursor = index - 1
        while cursor >= 0 and tokens[cursor][0] == "linebreak":
            cursor -= 1
        return cursor if cursor >= 0 else None

    @staticmethod
    def _next_significant_token_index(
        tokens: list[tuple[str, str]],
        index: int,
    ) -> int | None:
        cursor = index + 1
        while cursor < len(tokens) and tokens[cursor][0] == "linebreak":
            cursor += 1
        return cursor if cursor < len(tokens) else None

    def _string_after_from(self, tokens: list[tuple[str, str]], index: int) -> str | None:
        for cursor in range(index, len(tokens)):
            kind, value = tokens[cursor]
            if (kind, value) == ("punct", ";"):
                return None
            if (kind, value) != ("identifier", "from"):
                continue
            next_token = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
            return next_token[1] if next_token is not None and next_token[0] == "string" else None
        return None

    def _typescript_tokens(
        self,
        source_code: str,
        *,
        preserve_line_breaks: bool = False,
    ) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        index = 0
        length = len(source_code)
        while index < length:
            char = source_code[index]
            trivia_end = self._typescript_trivia_end(source_code, index)
            if trivia_end is not None:
                self._append_linebreak_token(
                    tokens,
                    source_code[index:trivia_end],
                    preserve_line_breaks=preserve_line_breaks,
                )
                index = trivia_end
                continue
            if char in {"'", '"'}:
                value, index = self._read_string_literal(source_code, index)
                tokens.append(("string", value))
                continue
            if char == "`":
                template_tokens, index = self._read_template_literal_tokens(
                    source_code,
                    index + 1,
                    preserve_line_breaks=preserve_line_breaks,
                )
                tokens.extend(template_tokens)
                continue
            if self._is_identifier_start(char):
                start = index
                index += 1
                while index < length and self._is_identifier_part(source_code[index]):
                    index += 1
                tokens.append(("identifier", source_code[start:index]))
                continue
            if char in "(){}[];,.?+:":
                tokens.append(("punct", char))
            index += 1
        return tokens

    def _typescript_trivia_end(
        self,
        source_code: str,
        index: int,
    ) -> int | None:
        char = source_code[index]
        if char.isspace():
            cursor = index + 1
            while cursor < len(source_code) and source_code[cursor].isspace():
                cursor += 1
            return cursor
        if char != "/" or index + 1 >= len(source_code):
            return None
        if source_code[index + 1] == "/":
            return self._skip_line_comment(source_code, index + 2)
        if source_code[index + 1] == "*":
            return self._skip_block_comment(source_code, index + 2)
        return None

    @staticmethod
    def _append_linebreak_token(
        tokens: list[tuple[str, str]],
        trivia: str,
        *,
        preserve_line_breaks: bool,
    ) -> None:
        if preserve_line_breaks and ("\n" in trivia or "\r" in trivia):
            tokens.append(("linebreak", "\n"))

    def _skip_line_comment(self, source_code: str, index: int) -> int:
        newline = source_code.find("\n", index)
        return len(source_code) if newline == -1 else newline + 1

    def _skip_block_comment(self, source_code: str, index: int) -> int:
        end = source_code.find("*/", index)
        return len(source_code) if end == -1 else end + 2

    def _read_string_literal(self, source_code: str, index: int) -> tuple[str, int]:
        quote = source_code[index]
        chars: list[str] = []
        index += 1
        while index < len(source_code):
            char = source_code[index]
            if char == "\\":
                if index + 1 < len(source_code):
                    chars.append(source_code[index + 1])
                    index += 2
                    continue
                index += 1
                break
            if char == quote:
                return "".join(chars), index + 1
            chars.append(char)
            index += 1
        return "".join(chars), index

    def _read_template_literal_tokens(
        self,
        source_code: str,
        index: int,
        *,
        preserve_line_breaks: bool = False,
    ) -> tuple[list[tuple[str, str]], int]:
        tokens: list[tuple[str, str]] = []
        while index < len(source_code):
            char = source_code[index]
            if char == "\\":
                index += 2
                continue
            if char == "`":
                return tokens, index + 1
            if char == "$" and index + 1 < len(source_code) and source_code[index + 1] == "{":
                expression_tokens, index = self._read_template_expression_tokens(
                    source_code,
                    index + 2,
                    preserve_line_breaks=preserve_line_breaks,
                )
                tokens.extend(expression_tokens)
                continue
            index += 1
        return tokens, index

    def _read_template_expression_tokens(
        self,
        source_code: str,
        index: int,
        *,
        preserve_line_breaks: bool = False,
    ) -> tuple[list[tuple[str, str]], int]:
        tokens: list[tuple[str, str]] = []
        depth = 1
        length = len(source_code)
        while index < length:
            char = source_code[index]
            trivia_end = self._typescript_trivia_end(source_code, index)
            if trivia_end is not None:
                self._append_linebreak_token(
                    tokens,
                    source_code[index:trivia_end],
                    preserve_line_breaks=preserve_line_breaks,
                )
                index = trivia_end
                continue
            if char in {"'", '"'}:
                value, index = self._read_string_literal(source_code, index)
                tokens.append(("string", value))
                continue
            if char == "`":
                template_tokens, index = self._read_template_literal_tokens(
                    source_code,
                    index + 1,
                    preserve_line_breaks=preserve_line_breaks,
                )
                tokens.extend(template_tokens)
                continue
            if self._is_identifier_start(char):
                start = index
                index += 1
                while index < length and self._is_identifier_part(source_code[index]):
                    index += 1
                tokens.append(("identifier", source_code[start:index]))
                continue
            if char == "{":
                depth += 1
                tokens.append(("punct", char))
                index += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return tokens, index + 1
                tokens.append(("punct", char))
                index += 1
                continue
            if char in "()[];,.?+:":
                tokens.append(("punct", char))
            index += 1
        return tokens, index

    def _is_identifier_start(self, char: str) -> bool:
        return char == "_" or char == "$" or char.isalpha()

    def _is_identifier_part(self, char: str) -> bool:
        return self._is_identifier_start(char) or char.isdigit()

    def _resolve_deno(self) -> str:
        candidate = self.deno_executable
        if os.path.sep in candidate or (os.path.altsep and os.path.altsep in candidate):
            path = Path(candidate).expanduser().resolve()
            if path.exists():
                return str(self._require_allowed_executable(path))
        resolved = shutil.which(candidate, path=self._safe_executable_search_path())
        if resolved is None:
            unsafe_resolved = shutil.which(candidate)
            if unsafe_resolved is not None and self._path_is_forbidden(Path(unsafe_resolved).resolve()):
                raise SandboxError(f"Deno executable resolves inside a forbidden root: {Path(unsafe_resolved).resolve()}")
            raise SandboxError(
                f"Deno executable not found: {candidate!r}. Install Deno or configure tools.deno_executable."
            )
        return str(self._require_allowed_executable(Path(resolved).resolve()))

    def _safe_executable_search_path(self) -> str:
        entries: list[str] = []
        for item in os.environ.get("PATH", "").split(os.pathsep):
            if not item:
                continue
            raw = Path(item).expanduser()
            if not raw.is_absolute():
                continue
            resolved = raw.resolve()
            if self._path_is_forbidden(resolved):
                continue
            entries.append(str(resolved))
        return os.pathsep.join(entries)

    def _require_allowed_executable(self, path: Path) -> Path:
        if self._path_is_forbidden(path):
            raise SandboxError(f"Deno executable resolves inside a forbidden root: {path}")
        return path

    def _path_is_forbidden(self, path: Path) -> bool:
        return any(path == root or root in path.parents for root in self.forbidden_executable_roots)

    def _profile_json(self, profile: Any) -> dict[str, Any]:
        return {
            "operation": profile.operation,
            "resource": profile.resource,
            "effect": profile.effect.value,
            "risk": profile.risk.value,
            "rule_id": profile.rule_id,
            "restrictions": dict(profile.restrictions),
        }

    def _runner_source(self) -> str:
        return textwrap.dedent(
            """
            const decoder = new TextDecoder(), encoder = new TextEncoder();
            const stdout = Deno.stdout.writable.getWriter(), stdin = Deno.stdin.readable.getReader();
            const NativeError = Error, nativeString = String, apply = Reflect.apply;
            const jsonParse = JSON.parse, jsonStringify = JSON.stringify;
            const objectCreate = Object.create, objectKeys = Object.keys;
            const hasOwnProperty = Object.prototype.hasOwnProperty;
            const stringIndexOf = String.prototype.indexOf, stringSlice = String.prototype.slice;
            const stringTrim = String.prototype.trim;
            const weakMapGet = WeakMap.prototype.get, weakMapSet = WeakMap.prototype.set;
            const stdinRead = stdin.read, stdoutWrite = stdout.write;
            const decode = decoder.decode, encode = encoder.encode, exit = Deno.exit;
            const hostSyscallErrors = new WeakMap<object, Record<string, unknown>>();
            let buffer = "";

            console.log = (...args: unknown[]) => console.error(...args);

            function rememberHostSyscallError(error: Error, frame: Record<string, unknown>): void {
              apply(weakMapSet, hostSyscallErrors, [error, {
                code: ownFrameValue(frame, "code"),
                error_type: ownFrameValue(frame, "error_type"),
                correlation_id: ownFrameValue(frame, "correlation_id"),
              }]);
            }

            function ownFrameValue(frame: Record<string, unknown>, key: string): unknown {
              return apply(hasOwnProperty, frame, [key]) ? frame[key] : undefined;
            }

            function hostSyscallErrorDetails(error: unknown): Record<string, unknown> | undefined {
              if (
                error === null
                || (typeof error !== "object" && typeof error !== "function")
              ) {
                return undefined;
              }
              return apply(weakMapGet, hostSyscallErrors, [error]) as
                Record<string, unknown> | undefined;
            }

            async function readFrame(): Promise<Record<string, unknown>> {
              while (true) {
                const newline = apply(stringIndexOf, buffer, ["\\n"]);
                if (newline >= 0) {
                  const line = apply(stringSlice, buffer, [0, newline]);
                  buffer = apply(stringSlice, buffer, [newline + 1]);
                  if (apply(stringTrim, line, []).length === 0) continue;
                  return jsonParse(line);
                }
                const chunk = await apply(stdinRead, stdin, []);
                if (chunk.done) throw new NativeError("stdin closed before protocol frame");
                buffer += apply(decode, decoder, [chunk.value, { stream: true }]);
              }
            }

            async function writeFrame(frame: Record<string, unknown>): Promise<void> {
              const protocolFrame = objectCreate(null) as Record<string, unknown>;
              const keys = objectKeys(frame);
              for (let index = 0; index < keys.length; index += 1) {
                const key = keys[index];
                protocolFrame[key] = frame[key];
              }
              const serialized = jsonStringify(protocolFrame) + "\\n";
              await apply(stdoutWrite, stdout, [apply(encode, encoder, [serialized])]);
            }

            const libos = {
              async syscall(name: string, args: Record<string, unknown> = {}): Promise<unknown> {
                const id = crypto.randomUUID();
                await writeFrame({ type: "syscall", id, name, args });
                while (true) {
                  const frame = await readFrame();
                  if (frame.type !== "syscall_result" || frame.id !== id) continue;
                  if (frame.ok) return frame.payload;
                  const error = new NativeError(nativeString(frame.error ?? "libOS syscall failed"));
                  rememberHostSyscallError(error, frame);
                  throw error;
                }
              },
            };

            let providerErrorProof: string | undefined;
            try {
              const frame = await readFrame();
              if (frame.type !== "run") throw new NativeError("first protocol frame must be run");
              providerErrorProof = typeof frame.provider_error_proof === "string"
                ? frame.provider_error_proof
                : undefined;
              const candidate = await import("./candidate.ts");
              if (typeof candidate.run !== "function") {
                throw new NativeError("candidate module must export a run function");
              }
              const run = candidate.run as (
                args: unknown,
                libos: typeof libos,
              ) => unknown | Promise<unknown>;
              const value = await run(frame.args ?? {}, libos);
              await writeFrame({ type: "result", value });
            } catch (error) {
              const details = hostSyscallErrorDetails(error);
              await writeFrame({
                type: "error",
                message: error instanceof NativeError ? error.message : nativeString(error),
                stack: error instanceof NativeError ? error.stack : undefined,
                code: details?.code,
                error_type: details?.error_type,
                correlation_id: details?.correlation_id,
                provider_error_proof: details === undefined ? undefined : providerErrorProof,
              });
              apply(exit, Deno, [1]);
            } finally {
              stdout.releaseLock();
              stdin.releaseLock();
            }
            """
        ).strip()
