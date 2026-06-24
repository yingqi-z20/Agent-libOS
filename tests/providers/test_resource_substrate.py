from __future__ import annotations
import contextlib
import os
import pytest
import psutil
import sys
import tempfile
import time
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any
from agent_libos import Runtime
from agent_libos.primitives.shell import ShellAdapter
from agent_libos.models import CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import CommandResult, LocalClockProvider, LocalFilesystemProvider, LocalHumanProvider, LocalResourceProviderSubstrate, LocalShellProvider, ResolvedPath, SubprocessLimitExceeded, SubprocessLimits, SubprocessTimeoutExpired
from modules.pty.pty_module import LocalPtyProvider

class TestResourceProviderSubstrate:

    def test_runtime_filesystem_primitive_uses_injected_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            substrate = RecordingSubstrate(temp_dir)
            runtime = Runtime.open('local', substrate=substrate)
            try:
                path = 'agent_outputs/substrate_write.txt'
                pid = runtime.process.spawn(image='review-agent:v0', goal='write through substrate')
                runtime.filesystem.grant_path(pid, path, [CapabilityRight.WRITE], issued_by='test')
                result = runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'via provider'})
                assert result.ok, result.error
                assert 'resolve' in substrate.filesystem.calls
                assert 'write_text' in substrate.filesystem.calls
                assert (Path(temp_dir) / path).read_text(encoding='utf-8') == 'via provider'
            finally:
                runtime.close()

    def test_filesystem_directory_limit_is_pushed_to_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index in range(5):
                (root / f"item-{index}.txt").write_text("x", encoding="utf-8")
            substrate = RecordingSubstrate(temp_dir)
            runtime = Runtime.open("local", substrate=substrate)
            try:
                pid = runtime.process.spawn(image="base-agent:v0", goal="bounded directory listing")
                runtime.filesystem.grant_directory(pid, ".", [CapabilityRight.READ], issued_by="test")
                result = runtime.filesystem.read_directory(pid, ".", limit=2)
                assert result.count == 2
                assert result.truncated
                assert substrate.filesystem.list_limits == [3]
            finally:
                runtime.close()

    def test_runtime_clock_primitive_uses_injected_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_clock = FakeClockProvider()
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.clock = fake_clock
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='use fake clock')
                now = runtime.tools.call(pid, 'get_current_time', {'timezone': 'UTC'})
                slept = runtime.tools.call(pid, 'sleep', {'seconds': 0.1})
                assert now.ok, now.error
                assert now.payload['iso8601'] == '2040-01-02T03:04:05+00:00'
                assert slept.ok, slept.error
                assert slept.payload['elapsed_seconds'] == 0.25
                assert fake_clock.sleeps == [('async', 0.1)]
            finally:
                runtime.close()

    def test_shell_adapter_uses_injected_provider(self) -> None:
        runtime = Runtime.open('local')
        provider = FakeShellProvider()
        shell = ShellAdapter(runtime.capability, runtime.audit, provider=provider)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='run shell through substrate')
            runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            result = shell.run(pid, ['git', 'status', '--short'], timeout=2.0)
            assert result.stdout == 'ok\n'
            assert provider.calls == [(['git', 'status', '--short'], 2.0)]
        finally:
            runtime.close()

    def test_local_shell_provider_enforces_subprocess_wall_limit_with_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            with pytest.raises(SubprocessLimitExceeded) as exc_info:
                provider.run(
                    [sys.executable, "-c", "import time; time.sleep(0.2)"],
                    timeout=5.0,
                    limits=SubprocessLimits(wall_seconds=0.05),
                )
            assert exc_info.value.metrics.killed
            assert exc_info.value.metrics.limit_kind == "subprocess_wall_seconds"
            assert exc_info.value.metrics.wall_seconds > 0

    def test_local_shell_provider_timeout_returns_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            with pytest.raises(SubprocessTimeoutExpired) as exc_info:
                provider.run(
                    [sys.executable, "-c", "import time; time.sleep(0.2)"],
                    timeout=0.05,
                )
            assert exc_info.value.metrics.killed
            assert exc_info.value.metrics.limit_kind == "subprocess_timeout"
            assert exc_info.value.metrics.wall_seconds > 0

    def test_local_shell_provider_drains_large_stdout_while_monitoring(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            result = provider.run(
                [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"],
                timeout=5.0,
            )
            assert result.returncode == 0
            assert len(result.stdout) == 200000
            assert result.metrics is not None

    def test_local_shell_provider_terminates_background_child_process_tree(self) -> None:
        if sys.platform == "win32":
            pytest.skip("POSIX process-group cleanup does not apply to Windows shell provider")
        marker = f"SHELL_PROVIDER_TREE_{time.monotonic_ns()}"
        child_script = "import time; time.sleep(60)"
        parent_script = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child_script!r}, {marker!r}]); "
            "time.sleep(0.2)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            try:
                result = provider.run([sys.executable, "-c", parent_script, marker], timeout=5.0)
                assert result.returncode == 0

                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and _processes_with_marker(marker):
                    time.sleep(0.05)
                assert _processes_with_marker(marker) == []
            finally:
                for proc in _processes_with_marker(marker):
                    with contextlib.suppress(psutil.Error):
                        proc.kill()

    def test_local_pty_provider_smoke(self) -> None:
        if sys.platform == "win32":
            pytest.importorskip("winpty")
            argv = ["cmd.exe", "/c", "echo", "PTY_PROVIDER_SMOKE"]
        else:
            argv = [sys.executable, "-c", "print('PTY_PROVIDER_SMOKE')"]
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalPtyProvider(temp_dir)
            session = provider.spawn(argv, cols=80, rows=24)
            try:
                output = ""
                deadline = time.monotonic() + 5.0
                while "PTY_PROVIDER_SMOKE" not in output and time.monotonic() < deadline:
                    output += session.read(timeout_s=0.1)
                    if not session.is_alive() and "PTY_PROVIDER_SMOKE" not in output:
                        time.sleep(0.05)
                assert "PTY_PROVIDER_SMOKE" in output
            finally:
                session.close(force=True, timeout_s=1.0)

    def test_local_pty_provider_close_terminates_child_process_tree(self) -> None:
        if sys.platform == "win32":
            pytest.skip("POSIX process-group cleanup does not apply to pywinpty")
        marker = f"PTY_PROVIDER_TREE_{time.monotonic_ns()}"
        child_script = "import time; time.sleep(60)"
        parent_script = (
            "import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', {child_script!r}, {marker!r}]); "
            "time.sleep(60)"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalPtyProvider(temp_dir)
            session = provider.spawn([sys.executable, "-c", parent_script, marker], cols=80, rows=24)
            try:
                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and len(_processes_with_marker(marker)) < 2:
                    time.sleep(0.05)
                assert len(_processes_with_marker(marker)) >= 2

                session.close(force=True, timeout_s=0.5)

                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and _processes_with_marker(marker):
                    time.sleep(0.05)
                assert _processes_with_marker(marker) == []
            finally:
                session.close(force=True, timeout_s=0.5)
                for proc in _processes_with_marker(marker):
                    with contextlib.suppress(psutil.Error):
                        proc.kill()

    def test_effectful_provider_without_classification_fails_closed(self) -> None:
        runtime = Runtime.open('local')
        provider = NoClassificationShellProvider()
        shell = ShellAdapter(runtime.capability, runtime.audit, provider=provider)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='missing effect classifier')
            runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(ValidationError):
                shell.run(pid, ['git', 'status', '--short'], timeout=2.0)
            assert provider.calls == []
            assert runtime.store.list_external_effects() == []
        finally:
            runtime.close()

    def test_runtime_human_primitive_uses_injected_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            human = RecordingHumanProvider(answers=['blue'])
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.human = human
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='use fake human')
                output = runtime.tools.call(pid, 'human_output', {'message': 'hello'})
                question_id = runtime.human.ask(pid, 'Favorite color?', blocking=True)
                processed = runtime.human.drain_terminal_queue()
                assert output.ok, output.error
                assert human.outputs[0] == 'hello'
                assert human.prompts == ['Favorite color? ']
                assert processed[0].request_id == question_id
                assert processed[0].decision['answer'] == 'blue'
            finally:
                runtime.close()

class RecordingSubstrate:

    def __init__(self, root: str):
        self.workspace_root = Path(root).resolve()
        self.workspace_display = str(self.workspace_root)
        self.filesystem = RecordingFilesystemProvider(root)
        self.clock = LocalClockProvider()
        self.shell = LocalShellProvider(root)
        self.human = LocalHumanProvider()

class RecordingFilesystemProvider:

    def __init__(self, root: str):
        self.inner = LocalFilesystemProvider(root)
        self.namespace = self.inner.namespace
        self.root_display = self.inner.root_display
        self.calls: list[str] = []
        self.list_limits: list[int | None] = []

    def resolve(self, path: Any) -> ResolvedPath:
        self.calls.append('resolve')
        return self.inner.resolve(path)

    def state(self, path: ResolvedPath):
        self.calls.append('state')
        return self.inner.state(path)

    def write_text(self, path: ResolvedPath, text: str, encoding: str, newline: str | None='\n') -> None:
        self.calls.append('write_text')
        self.inner.write_text(path, text, encoding, newline)

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None):
        self.calls.append('list_directory')
        self.list_limits.append(limit)
        return self.inner.list_directory(path, limit=limit)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def _processes_with_marker(marker: str) -> list[psutil.Process]:
    matches: list[psutil.Process] = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.pid == current_pid:
            continue
        try:
            cmdline = proc.info.get("cmdline") or []
        except psutil.Error:
            continue
        if marker in " ".join(str(part) for part in cmdline):
            matches.append(proc)
    return matches

class FakeClockProvider:

    def __init__(self):
        self.sleeps: list[tuple[str, float]] = []
        self._monotonic = [100.0, 100.25]

    def now(self, timezone_: tzinfo) -> datetime:
        return datetime(2040, 1, 2, 3, 4, 5, tzinfo=timezone_)

    def monotonic(self) -> float:
        return self._monotonic.pop(0)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(('sync', seconds))

    async def asleep(self, seconds: float) -> None:
        self.sleeps.append(('async', seconds))

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        return ExternalEffectClassification(rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED, rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED, state_mutation=False, information_flow=operation == 'now', metadata={'operation': operation})

class FakeShellProvider:

    def __init__(self):
        self.calls: list[tuple[list[str], float]] = []

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        limits: object | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        self.calls.append((list(argv), timeout))
        return CommandResult(argv=list(argv), returncode=0, stdout='ok\n', stderr='')

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        return ExternalEffectClassification(rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE, rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED, state_mutation=True, information_flow=True, metadata={'operation': operation})

class NoClassificationShellProvider:

    def __init__(self):
        self.calls: list[tuple[list[str], float]] = []

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        limits: object | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        self.calls.append((list(argv), timeout))
        return CommandResult(argv=list(argv), returncode=0, stdout='ok\n', stderr='')

class RecordingHumanProvider:

    def __init__(self, answers: list[str] | None=None):
        self.outputs: list[str] = []
        self.prompts: list[str] = []
        self.answers = list(answers or [])

    def write(self, message: str) -> None:
        self.outputs.append(message)

    def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else ''

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        return ExternalEffectClassification(rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED, rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED, state_mutation=False, information_flow=True, metadata={'operation': operation})
