from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
import time
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import psutil

from agent_libos.capability.profiles import SandboxProfileBuilder
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import SandboxError
from agent_libos.models import ValidationResult
from agent_libos.substrate import CommandMetrics, SubprocessLimitExceeded, SubprocessLimits, SubprocessTimeoutExpired
from agent_libos.tools.observability import ensure_json_size
from agent_libos.utils.serde import to_jsonable

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_EXACT_JSR_VERSION_RE = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)

SyscallHandler = Callable[[str, dict[str, Any]], Any | Awaitable[Any]]


@dataclass(frozen=True)
class SandboxExecutionResult:
    value: Any
    metrics: CommandMetrics | None = None


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
    ) -> Any:
        kwargs = self._arun_source_kwargs(
            pid=pid,
            syscall_handler=syscall_handler,
            timeout=timeout,
            limits=limits,
            return_metrics=return_metrics,
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
        self.profile_builder = SandboxProfileBuilder()

    def static_check(self, source_code: str) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        if len(source_code) > self.max_source_chars:
            errors.append(f"TypeScript tool source exceeds max chars: {self.max_source_chars}")
        if not self._RUN_EXPORT_RE.search(source_code):
            errors.append("TypeScript tool source must export function run(args, libos)")
        if self._contains_dynamic_import(source_code):
            errors.append("dynamic import() is not allowed")
        if self._contains_runtime_code_generation(source_code):
            errors.append("runtime code generation is not allowed")
        for specifier in self._extract_imports(source_code):
            parsed = self._jsr_package_and_version(specifier)
            package = parsed[0] if parsed is not None else None
            if package is None:
                errors.append(f"import is not allowed: {specifier}")
                continue
            if package not in self.jsr_allowlist:
                errors.append(f"JSR package is not in allowlist: {package}")
                continue
            if parsed[1] is None:
                errors.append(f"JSR import must pin a package version: {specifier}")
                continue
            if not self._is_exact_jsr_version(parsed[1]):
                errors.append(f"JSR import must use an exact semantic version: {specifier}")
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
            proc = await asyncio.create_subprocess_exec(
                deno,
                "run",
                "--no-prompt",
                "runner.ts",
                cwd=tmp,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            monitor_task = asyncio.create_task(self._monitor_process(proc, limits), name="deno-resource-monitor")
            serve_task = asyncio.create_task(
                self._serve_process(proc, args, syscall_handler),
                name="deno-syscall-server",
            )
            try:
                done, pending = await asyncio.wait(
                    {serve_task, monitor_task},
                    timeout=selected_timeout,
                    return_when=asyncio.FIRST_EXCEPTION,
                )
                if not done:
                    await self._kill_process(proc)
                    for task in pending:
                        task.cancel()
                    raise SubprocessTimeoutExpired(
                        f"Deno JIT tool timed out after {selected_timeout}s",
                        metrics=CommandMetrics(
                            wall_seconds=float(selected_timeout),
                            killed=True,
                            limit_kind="subprocess_timeout",
                        ),
                    )
                for task in done:
                    exc = task.exception()
                    if exc is not None:
                        await self._kill_process(proc)
                        for pending_task in pending:
                            pending_task.cancel()
                        raise exc
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
                if not monitor_task.done():
                    await asyncio.wait({monitor_task}, timeout=1.0, return_when=asyncio.ALL_COMPLETED)
                metrics = monitor_task.result() if monitor_task.done() and not monitor_task.cancelled() else None
                wrapped = SandboxExecutionResult(value=value, metrics=metrics)
                return wrapped if return_metrics else value
            except SubprocessTimeoutExpired:
                await self._kill_process(proc)
                raise
            except TimeoutError as exc:
                await self._kill_process(proc)
                raise TimeoutError(f"Deno JIT tool timed out after {selected_timeout}s") from exc
            except Exception:
                await self._kill_process(proc)
                raise
            finally:
                for task in (serve_task, monitor_task):
                    if not task.done():
                        task.cancel()

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
            return ValidationResult(ok=False, errors=[str(exc)])
        errors: list[str] = []
        logs: list[str] = [f"language=typescript", f"deno={version}"]
        metrics: list[CommandMetrics] = []
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
                errors.append(f"test {index} failed to run: {self._bounded_result_repr(exc)}")
                continue
            logs.append(f"test {index} result: {self._bounded_result_repr(result_value)}")
            if "expected" in test and result_value != test["expected"]:
                errors.append(
                    "test "
                    f"{index} expected {self._bounded_result_repr(test['expected'])}, "
                    f"got {self._bounded_result_repr(result_value)}"
                )
        metadata = {"metrics": self._aggregate_metrics(metrics)} if return_metrics else {}
        return ValidationResult(ok=not errors, errors=errors, logs=self._bounded_logs(logs), metadata=metadata)

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
            metadata["deno_version_error"] = str(exc)
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
        await self._write_frame(proc, {"type": "run", "args": to_jsonable(args)})
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
                stderr, _stderr_truncated = await self._finish_stderr(stderr_task)
                message = str(frame.get("message") or stderr or "Deno JIT tool failed")
                raise SandboxError(message)
            raise SandboxError(f"unknown Deno JIT protocol frame: {frame_type!r}")

    async def _monitor_process(
        self,
        proc: asyncio.subprocess.Process,
        limits: SubprocessLimits | None,
    ) -> CommandMetrics:
        started_at = time.monotonic()
        peak_memory = 0
        cpu_seconds = 0.0
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.Error:
            return CommandMetrics(wall_seconds=max(0.0, time.monotonic() - started_at))
        while proc.returncode is None:
            wall_seconds = time.monotonic() - started_at
            cpu_seconds, peak_memory = self._sample_process_tree(ps_proc, peak_memory)
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
        final_cpu_seconds, peak_memory = self._sample_process_tree(ps_proc, peak_memory)
        return CommandMetrics(
            wall_seconds=wall_seconds,
            cpu_seconds=max(cpu_seconds, final_cpu_seconds),
            peak_memory_bytes=peak_memory,
            killed=False,
            limit_kind=None,
        )

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
            await self._write_frame(
                proc,
                {
                    "type": "syscall_result",
                    "id": frame_id,
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
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

    async def _kill_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        try:
            proc.kill()
        except ProcessLookupError:
            return
        await proc.wait()

    def _test_syscall_handler(self, test: dict[str, Any], index: int) -> tuple[SyscallHandler, Callable[[], None]]:
        expected = list(test.get("syscalls", []))

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
                raise SandboxError(str(spec.get("error", "mock syscall failed")))
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
            try:
                ensure_json_size(test, self.max_test_case_bytes, f"JIT test {index}")
            except Exception as exc:
                errors.append(str(exc))
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

    def _bounded_result_repr(self, value: Any) -> str:
        text = repr(value)
        if len(text) <= self.max_validation_log_chars:
            return text
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        return (
            text[: self.max_validation_log_chars]
            + f"... [truncated validation result repr chars={len(text)} sha256={digest}]"
        )

    def _bounded_logs(self, logs: list[str]) -> str:
        text = "\n".join(logs)
        if len(text) <= self.max_validation_log_chars:
            return text
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
        return text[: self.max_validation_log_chars] + f"\n[validation logs truncated chars={len(text)} sha256={digest}]"

    def _contains_dynamic_import(self, source_code: str) -> bool:
        tokens = self._typescript_tokens(source_code)
        for index, token in enumerate(tokens):
            if token != ("identifier", "import"):
                continue
            previous_token = tokens[index - 1] if index > 0 else None
            if previous_token == ("punct", "."):
                continue
            next_token = tokens[index + 1] if index + 1 < len(tokens) else None
            if next_token == ("punct", "("):
                if self._looks_like_method_definition(tokens, index):
                    continue
                return True
        return False

    def _contains_runtime_code_generation(self, source_code: str) -> bool:
        tokens = self._typescript_tokens(source_code)
        for index, token in enumerate(tokens):
            if token == ("identifier", "eval"):
                if self._looks_like_method_definition(tokens, index):
                    continue
                previous_token = tokens[index - 1] if index > 0 else None
                previous_previous_token = tokens[index - 2] if index > 1 else None
                if previous_token == ("punct", ".") and previous_previous_token in {
                    ("identifier", "globalThis"),
                    ("identifier", "window"),
                }:
                    return True
                if previous_token != ("punct", ".") and self._token_is_call(tokens, index):
                    return True
            if token == ("identifier", "Function"):
                if self._looks_like_method_definition(tokens, index):
                    continue
                previous_token = tokens[index - 1] if index > 0 else None
                previous_previous_token = tokens[index - 2] if index > 1 else None
                if previous_token == ("punct", ".") and previous_previous_token in {
                    ("identifier", "globalThis"),
                    ("identifier", "window"),
                }:
                    return True
                if previous_token != ("punct", ".") and self._token_is_call(tokens, index):
                    return True
            if token[0] == "string" and token[1] in {"eval", "Function"}:
                if self._token_is_bracketed_global_property(tokens, index):
                    return True
        return False

    def _token_is_call(self, tokens: list[tuple[str, str]], index: int) -> bool:
        next_token = tokens[index + 1] if index + 1 < len(tokens) else None
        return next_token == ("punct", "(")

    def _token_is_bracketed_global_property(self, tokens: list[tuple[str, str]], index: int) -> bool:
        if index < 2 or index + 1 >= len(tokens):
            return False
        if tokens[index - 1] != ("punct", "["):
            return False
        if tokens[index + 1] != ("punct", "]"):
            return False
        object_index = index - 2
        if tokens[object_index] == ("punct", ".") and object_index > 0:
            object_index -= 1
        return tokens[object_index] in {("identifier", "globalThis"), ("identifier", "window")}

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
        return [specifier] if specifier is not None else []

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
        previous_token = tokens[index - 1] if index > 0 else None
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
                    next_token = tokens[cursor + 1] if cursor + 1 < len(tokens) else None
                    return next_token == ("punct", "{")
        return False

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

    def _typescript_tokens(self, source_code: str) -> list[tuple[str, str]]:
        tokens: list[tuple[str, str]] = []
        index = 0
        length = len(source_code)
        while index < length:
            char = source_code[index]
            if char.isspace():
                index += 1
                continue
            if char == "/" and index + 1 < length and source_code[index + 1] == "/":
                index = self._skip_line_comment(source_code, index + 2)
                continue
            if char == "/" and index + 1 < length and source_code[index + 1] == "*":
                index = self._skip_block_comment(source_code, index + 2)
                continue
            if char in {"'", '"'}:
                value, index = self._read_string_literal(source_code, index)
                tokens.append(("string", value))
                continue
            if char == "`":
                template_tokens, index = self._read_template_literal_tokens(source_code, index + 1)
                tokens.extend(template_tokens)
                continue
            if self._is_identifier_start(char):
                start = index
                index += 1
                while index < length and self._is_identifier_part(source_code[index]):
                    index += 1
                tokens.append(("identifier", source_code[start:index]))
                continue
            if char in "(){}[];,.":
                tokens.append(("punct", char))
            index += 1
        return tokens

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

    def _read_template_literal_tokens(self, source_code: str, index: int) -> tuple[list[tuple[str, str]], int]:
        tokens: list[tuple[str, str]] = []
        while index < len(source_code):
            char = source_code[index]
            if char == "\\":
                index += 2
                continue
            if char == "`":
                return tokens, index + 1
            if char == "$" and index + 1 < len(source_code) and source_code[index + 1] == "{":
                expression_tokens, index = self._read_template_expression_tokens(source_code, index + 2)
                tokens.extend(expression_tokens)
                continue
            index += 1
        return tokens, index

    def _read_template_expression_tokens(self, source_code: str, index: int) -> tuple[list[tuple[str, str]], int]:
        tokens: list[tuple[str, str]] = []
        depth = 1
        length = len(source_code)
        while index < length:
            char = source_code[index]
            if char.isspace():
                index += 1
                continue
            if char == "/" and index + 1 < length and source_code[index + 1] == "/":
                index = self._skip_line_comment(source_code, index + 2)
                continue
            if char == "/" and index + 1 < length and source_code[index + 1] == "*":
                index = self._skip_block_comment(source_code, index + 2)
                continue
            if char in {"'", '"'}:
                value, index = self._read_string_literal(source_code, index)
                tokens.append(("string", value))
                continue
            if char == "`":
                template_tokens, index = self._read_template_literal_tokens(source_code, index + 1)
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
            if char in "()[];,.":
                tokens.append(("punct", char))
            index += 1
        return tokens, index

    def _is_identifier_start(self, char: str) -> bool:
        return char == "_" or char == "$" or char.isalpha()

    def _is_identifier_part(self, char: str) -> bool:
        return self._is_identifier_start(char) or char.isdigit()

    def _jsr_package_and_version(self, specifier: str) -> tuple[str, str | None] | None:
        if not specifier.startswith("jsr:"):
            return None
        body = specifier[4:]
        if not body.startswith("@"):
            return None
        parts = body.split("/")
        if len(parts) < 2:
            return None
        scope = parts[0]
        name_part = parts[1]
        name, version = name_part.split("@", 1) if "@" in name_part else (name_part, None)
        if not scope or not name:
            return None
        return f"{scope}/{name}", version or None

    def _is_exact_jsr_version(self, version: str) -> bool:
        return _EXACT_JSR_VERSION_RE.fullmatch(version) is not None

    def _resolve_deno(self) -> str:
        candidate = self.deno_executable
        if os.path.sep in candidate or (os.path.altsep and os.path.altsep in candidate):
            path = Path(candidate)
            if path.exists():
                return str(path)
        resolved = shutil.which(candidate)
        if resolved is None:
            raise SandboxError(
                f"Deno executable not found: {candidate!r}. Install Deno or configure tools.deno_executable."
            )
        return resolved

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
            import { run } from "./candidate.ts";

            const decoder = new TextDecoder();
            const encoder = new TextEncoder();
            const stdout = Deno.stdout.writable.getWriter();
            const stdin = Deno.stdin.readable.getReader();
            let buffer = "";

            console.log = (...args: unknown[]) => console.error(...args);

            async function readFrame(): Promise<Record<string, unknown>> {
              while (true) {
                const newline = buffer.indexOf("\\n");
                if (newline >= 0) {
                  const line = buffer.slice(0, newline);
                  buffer = buffer.slice(newline + 1);
                  if (line.trim().length === 0) continue;
                  return JSON.parse(line);
                }
                const chunk = await stdin.read();
                if (chunk.done) throw new Error("stdin closed before protocol frame");
                buffer += decoder.decode(chunk.value, { stream: true });
              }
            }

            async function writeFrame(frame: Record<string, unknown>): Promise<void> {
              await stdout.write(encoder.encode(JSON.stringify(frame) + "\\n"));
            }

            const libos = {
              async syscall(name: string, args: Record<string, unknown> = {}): Promise<unknown> {
                const id = crypto.randomUUID();
                await writeFrame({ type: "syscall", id, name, args });
                while (true) {
                  const frame = await readFrame();
                  if (frame.type !== "syscall_result" || frame.id !== id) continue;
                  if (frame.ok) return frame.payload;
                  const error = new Error(String(frame.error ?? "libOS syscall failed"));
                  (error as Error & { details?: unknown }).details = frame;
                  throw error;
                }
              },
            };

            try {
              const frame = await readFrame();
              if (frame.type !== "run") throw new Error("first protocol frame must be run");
              const value = await run(frame.args ?? {}, libos);
              await writeFrame({ type: "result", value });
            } catch (error) {
              await writeFrame({
                type: "error",
                message: error instanceof Error ? error.message : String(error),
                stack: error instanceof Error ? error.stack : undefined,
              });
              Deno.exit(1);
            } finally {
              stdout.releaseLock();
              stdin.releaseLock();
            }
            """
        ).strip()
