from __future__ import annotations
import contextlib
import hashlib
import json
import os
import pytest
import psutil
import signal
import sys
import tempfile
import time
import venv
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any
from agent_libos import Runtime
from agent_libos.primitives.shell import ShellAdapter
import agent_libos.substrate.base as substrate_base
from agent_libos.models import CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.substrate import CommandResult, FilesystemCompareAndSwapProvider, FilesystemContentConflict, LocalClockProvider, LocalFilesystemProvider, LocalHumanProvider, LocalResourceProviderSubstrate, LocalShellProvider, ResolvedPath, resolve_runtime_python_alias, snapshot_executable, SubprocessLimitExceeded, SubprocessLimits, SubprocessTimeoutExpired
from modules.pty.pty_module import LocalPtyProvider, _PosixPtySession

class TestResourceProviderSubstrate:

    def test_local_filesystem_provider_write_text_compare_and_swap(self, tmp_path: Path) -> None:
        provider = LocalFilesystemProvider(tmp_path)
        target = provider.resolve("document.txt")

        assert isinstance(provider, FilesystemCompareAndSwapProvider)
        provider.write_text_compare_and_swap(
            target,
            "first",
            "utf-8",
            expected_content_sha256="missing",
        )
        target = provider.resolve("document.txt")
        first_digest = hashlib.sha256(b"first").hexdigest()
        provider.write_text_compare_and_swap(
            target,
            "second",
            "utf-8",
            expected_content_sha256=first_digest,
        )

        assert (tmp_path / "document.txt").read_text(encoding="utf-8") == "second"
        with pytest.raises(FilesystemContentConflict):
            provider.write_text_compare_and_swap(
                target,
                "stale",
                "utf-8",
                expected_content_sha256=first_digest,
            )
        with pytest.raises(FilesystemContentConflict):
            provider.write_text_compare_and_swap(
                target,
                "duplicate-create",
                "utf-8",
                expected_content_sha256="missing",
            )
        assert (tmp_path / "document.txt").read_text(encoding="utf-8") == "second"

    @pytest.mark.parametrize(
        "invalid",
        ["", "0" * 63, "A" * 64, "not-a-digest"],
    )
    def test_local_filesystem_provider_rejects_invalid_cas_token(
        self,
        tmp_path: Path,
        invalid: str,
    ) -> None:
        provider = LocalFilesystemProvider(tmp_path)

        with pytest.raises(ValidationError, match="expected_content_sha256"):
            provider.write_text_compare_and_swap(
                provider.resolve("document.txt"),
                "content",
                "utf-8",
                expected_content_sha256=invalid,
            )

        assert not (tmp_path / "document.txt").exists()

    def test_local_filesystem_provider_directory_limit_avoids_eager_listdir(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        provider = LocalFilesystemProvider(tmp_path)
        for index in range(20):
            (tmp_path / f"entry-{index:02d}").write_text("x", encoding="utf-8")

        monkeypatch.setattr(
            os,
            "listdir",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("bounded listing must not materialize os.listdir")
            ),
        )
        entries = provider.list_directory(provider.resolve("."), limit=4)

        assert len(entries) == 4

    def test_local_filesystem_provider_unlimited_listing_fails_on_hard_overflow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        import agent_libos.substrate.local as local_substrate

        provider = LocalFilesystemProvider(tmp_path)
        for index in range(4):
            (tmp_path / f"entry-{index}").write_text("x", encoding="utf-8")
        monkeypatch.setattr(
            local_substrate,
            "_TOOL_DEFAULTS",
            type("SmallDirectoryDefaults", (), {"directory_entry_hard_limit": 3})(),
        )

        with pytest.raises(ValidationError, match="bounded provider listing limit"):
            provider.list_directory(provider.resolve("."))

    def test_windows_executable_hash_uses_consistent_path_and_descriptor_identity(
        self,
    ) -> None:
        if os.name != "nt":
            pytest.skip("Windows stat identity regression")
        executable = Path(sys.executable)
        expected = hashlib.sha256(executable.read_bytes()).hexdigest()

        assert substrate_base.executable_content_sha256(executable) == expected

    def test_executable_hash_stops_when_deadline_expires_during_read(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        executable = tmp_path / "tool"
        executable.write_bytes(b"x" * (3 * 1024 * 1024))
        actual_read = os.read
        observed_fds: list[int] = []

        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

        clock = Clock()

        class OsProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def read(self, descriptor: int, size: int) -> bytes:
                observed_fds.append(descriptor)
                chunk = actual_read(descriptor, size)
                clock.now = 2.0
                return chunk

        monkeypatch.setattr(substrate_base, "time", clock)
        monkeypatch.setattr(substrate_base, "os", OsProxy())

        with pytest.raises(TimeoutError, match="fingerprint read"):
            substrate_base.executable_content_sha256(executable, deadline=1.0)

        assert len(observed_fds) == 1
        with pytest.raises(OSError):
            os.fstat(observed_fds[0])

    def test_executable_hash_fifo_swap_uses_nonblocking_open_and_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        if os.name == "nt" or not hasattr(os, "mkfifo"):
            pytest.skip("FIFO executable swap regression is POSIX-only")
        executable = tmp_path / "tool"
        executable.write_bytes(b"regular-before-race")
        actual_open = os.open
        actual_stat = os.stat
        swapped = False
        observed_flags: list[int] = []

        class OsProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def stat(self, path: Any, *, follow_symlinks: bool = True) -> Any:
                nonlocal swapped
                result = actual_stat(path, follow_symlinks=follow_symlinks)
                if not swapped and Path(path) == executable:
                    executable.unlink()
                    os.mkfifo(executable)
                    swapped = True
                return result

            def open(self, path: Any, flags: int, *args: Any) -> int:
                observed_flags.append(flags)
                assert flags & os.O_NONBLOCK
                return actual_open(path, flags, *args)

        monkeypatch.setattr(substrate_base, "os", OsProxy())

        with pytest.raises(ValidationError, match="regular file"):
            substrate_base.executable_content_sha256(
                executable,
                deadline=time.monotonic() + 1.0,
            )

        assert swapped is True
        assert len(observed_flags) == 1

    def test_expired_snapshot_deadline_does_not_create_private_directory(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        executable = tmp_path / "tool"
        executable.write_bytes(b"snapshot-content")

        class ExpiredClock:
            @staticmethod
            def monotonic() -> float:
                return 2.0

        class ForbiddenTempfile:
            @staticmethod
            def mkdtemp(*_args: Any, **_kwargs: Any) -> str:
                raise AssertionError("expired snapshot must not create a directory")

        monkeypatch.setattr(substrate_base, "time", ExpiredClock())
        monkeypatch.setattr(substrate_base, "tempfile", ForbiddenTempfile())

        with pytest.raises(TimeoutError, match="snapshot setup"):
            snapshot_executable(executable, deadline=1.0)

    @pytest.mark.parametrize("expire_during", ["read", "write"])
    def test_executable_snapshot_copy_checks_deadline_after_each_io(
        self,
        expire_during: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / "source"
        destination = tmp_path / "destination"
        source.write_bytes(b"snapshot-content")
        source_fd = os.open(source, os.O_RDONLY)
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        actual_read = os.read
        actual_write = os.write
        calls = {"read": 0, "write": 0}

        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

        clock = Clock()

        class OsProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def read(self, descriptor: int, size: int) -> bytes:
                calls["read"] += 1
                chunk = actual_read(descriptor, size)
                if expire_during == "read":
                    clock.now = 2.0
                return chunk

            def write(self, descriptor: int, value: bytes) -> int:
                calls["write"] += 1
                written = actual_write(descriptor, value)
                if expire_during == "write":
                    clock.now = 2.0
                return written

        monkeypatch.setattr(substrate_base, "time", clock)
        monkeypatch.setattr(substrate_base, "os", OsProxy())
        try:
            with pytest.raises(TimeoutError, match=f"snapshot {expire_during}"):
                substrate_base._copy_executable_snapshot_bytes(
                    source_fd,
                    destination_fd,
                    deadline=1.0,
                )
        finally:
            os.close(destination_fd)
            os.close(source_fd)

        assert calls["read"] == 1
        assert calls["write"] == (0 if expire_during == "read" else 1)

    def test_snapshot_deadline_failure_closes_fds_and_removes_private_copy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        executable = tmp_path / "tool"
        executable.write_bytes(b"snapshot-content")
        actual_open = os.open
        actual_read = os.read
        actual_mkdtemp = tempfile.mkdtemp
        opened_fds: list[int] = []
        snapshot_directories: list[Path] = []

        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

        clock = Clock()

        class OsProxy:
            def __getattr__(self, name: str) -> Any:
                return getattr(os, name)

            def open(self, path: Any, flags: int, *args: Any) -> int:
                descriptor = actual_open(path, flags, *args)
                opened_fds.append(descriptor)
                return descriptor

            def read(self, descriptor: int, size: int) -> bytes:
                chunk = actual_read(descriptor, size)
                clock.now = 2.0
                return chunk

        class TempfileProxy:
            def mkdtemp(self, *args: Any, **kwargs: Any) -> str:
                selected = actual_mkdtemp(*args, **kwargs)
                snapshot_directories.append(Path(selected))
                return selected

        monkeypatch.setattr(substrate_base, "time", clock)
        monkeypatch.setattr(substrate_base, "os", OsProxy())
        monkeypatch.setattr(substrate_base, "tempfile", TempfileProxy())

        with pytest.raises(TimeoutError, match="snapshot read"):
            snapshot_executable(
                executable,
                sibling_policy="none",
                deadline=1.0,
            )

        assert len(opened_fds) == 2
        for descriptor in opened_fds:
            with pytest.raises(OSError):
                os.fstat(descriptor)
        assert len(snapshot_directories) == 1
        assert not snapshot_directories[0].exists()

    def test_snapshot_sibling_mirroring_stops_between_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        if os.name == "nt":
            pytest.skip("symlink iteration regression requires POSIX symlinks")
        source_directory = tmp_path / "source"
        snapshot_directory = tmp_path / "snapshot"
        source_directory.mkdir()
        snapshot_directory.mkdir()
        executable = source_directory / "tool"
        executable.write_bytes(b"tool")
        (source_directory / "first").write_bytes(b"first")
        (source_directory / "second").write_bytes(b"second")
        actual_symlink_to = Path.symlink_to

        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

        clock = Clock()

        def expiring_symlink(
            path: Path,
            target: str | Path,
            target_is_directory: bool = False,
        ) -> None:
            actual_symlink_to(
                path,
                target,
                target_is_directory=target_is_directory,
            )
            clock.now = 2.0

        monkeypatch.setattr(substrate_base, "time", clock)
        monkeypatch.setattr(Path, "symlink_to", expiring_symlink)

        with pytest.raises(TimeoutError, match="snapshot sibling mirror"):
            substrate_base._mirror_executable_siblings(
                executable,
                snapshot_directory,
                sibling_limit=4,
                deadline=1.0,
            )

        assert len(list(snapshot_directory.iterdir())) == 1

    def test_snapshot_deadline_is_optional_and_preserved_by_verify(
        self,
        tmp_path: Path,
    ) -> None:
        executable = tmp_path / "tool"
        executable.write_bytes(b"snapshot-content")
        expected = hashlib.sha256(executable.read_bytes()).hexdigest()

        assert substrate_base.executable_content_sha256(
            executable,
            deadline=time.monotonic() + 30.0,
        ) == expected
        with snapshot_executable(
            executable,
            sibling_policy="none",
            deadline=time.monotonic() + 30.0,
        ) as snapshot:
            snapshot.verify(deadline=time.monotonic() + 30.0)
            assert snapshot.content_sha256 == expected
            with pytest.raises(TimeoutError, match="fingerprint setup"):
                snapshot.verify(deadline=time.monotonic() - 1.0)

    def test_runtime_python_alias_requires_an_external_host_interpreter(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        workspace = tmp_path / "workspace"
        external = tmp_path / "host-runtime" / "python3"
        workspace.mkdir()
        monkeypatch.setattr(substrate_base.sys, "_base_executable", str(external))
        monkeypatch.setattr(substrate_base.sys, "executable", str(external))

        assert resolve_runtime_python_alias(
            "python",
            workspace_root=workspace,
        ) == str(external.resolve(strict=False))
        assert resolve_runtime_python_alias(
            "python3",
            workspace_root=workspace,
        ) == str(external.resolve(strict=False))
        assert resolve_runtime_python_alias(
            "pip",
            workspace_root=workspace,
        ) is None
        assert resolve_runtime_python_alias(
            "bin/python",
            workspace_root=workspace,
        ) is None

        workspace_interpreter = workspace / ".venv" / "bin" / "python"
        monkeypatch.setattr(
            substrate_base.sys,
            "_base_executable",
            str(workspace_interpreter),
        )
        monkeypatch.setattr(
            substrate_base.sys,
            "executable",
            str(workspace_interpreter),
        )
        assert resolve_runtime_python_alias(
            "python",
            workspace_root=workspace,
        ) is None

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
                runtime.capability.grant(pid, 'clock:*', [CapabilityRight.READ], issued_by='test')
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
        shell = ShellAdapter(
            runtime.capability,
            runtime.audit,
            protected_operations=runtime.protected_operations,
            provider=provider,
        )
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='run shell through substrate')
            runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
            result = shell.run(pid, ['git', 'status', '--short'], timeout=2.0)
            assert result.stdout == 'ok\n'
            assert provider.calls == [
                (
                    [
                        'git',
                        '--no-pager',
                        '--no-optional-locks',
                        '-c',
                        'core.fsmonitor=false',
                        '-c',
                        'core.untrackedCache=false',
                        '-c',
                        'maintenance.auto=false',
                        '-c',
                        'submodule.recurse=false',
                        '-c',
                        'diff.external=',
                        '-c',
                        'color.ui=false',
                        'status',
                        '--short',
                    ],
                    2.0,
                )
            ]
            assert result.argv == ['git', 'status', '--short']
        finally:
            runtime.close()

    def test_local_shell_provider_enforces_subprocess_wall_limit_with_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            if sys.platform == "win32":
                with pytest.raises(ValidationError, match="SubprocessLimits"):
                    provider.run(
                        [sys.executable, "-c", "import time; time.sleep(0.2)"],
                        timeout=5.0,
                        limits=SubprocessLimits(wall_seconds=0.05),
                    )
                return
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

    def test_local_shell_provider_enforces_character_limit_for_multibyte_utf8(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            result = provider.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.buffer.write(bytes.fromhex('e99baa') * 5)",
                ],
                timeout=5.0,
                stdout_limit_chars=5,
            )

            assert result.stdout == "雪" * 5
            assert not result.stdout_truncated

            with pytest.raises(SubprocessLimitExceeded) as caught:
                provider.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.buffer.write(bytes.fromhex('e99baa') * 6)",
                    ],
                    timeout=5.0,
                    stdout_limit_chars=5,
                )

            assert caught.value.metrics.limit_kind == "subprocess_stdout_chars"
            assert caught.value.result is not None
            assert caught.value.result.stdout == "雪" * 5
            assert caught.value.result.stdout_truncated

    def test_local_shell_provider_does_not_use_unbounded_temporary_output_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            tempfile,
            "TemporaryFile",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("shell output capture must use bounded pipes")
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            result = provider.run(
                [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'bounded\\n')"],
                timeout=5.0,
            )

        assert result.stdout == "bounded\n"

    def test_local_shell_snapshot_preserves_explicit_virtualenv_argv0(
        self,
        tmp_path: Path,
    ) -> None:
        if os.name == "nt":
            pytest.skip("POSIX executable/argv[0] separation is not available on Windows")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        environment = workspace / ".venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        executable = environment / "bin" / "python"
        provider = LocalShellProvider(workspace)
        argv = [
            ".venv/bin/python",
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({'prefix': sys.prefix, 'executable': sys.executable}))"
            ),
        ]
        resolved = provider.resolve_argv(argv)

        with snapshot_executable(resolved[0]) as snapshot:
            result = provider.run(argv, executable_snapshot=snapshot)

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert Path(payload["prefix"]) == environment
        assert Path(payload["executable"]) == executable

    @pytest.mark.parametrize("mode", ["success", "wall", "timeout"])
    def test_local_shell_provider_without_cpu_memory_limits_tolerates_tree_enumeration_denial(
        self,
        mode: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        instances: list[Any] = []

        class TreeEnumerationDenied:
            def __init__(self, pid: int) -> None:
                self.pid = pid
                self.children_calls = 0
                instances.append(self)

            def children(self, *, recursive: bool) -> list[Any]:
                assert recursive
                self.children_calls += 1
                raise PermissionError("process tree enumeration denied")

            def cpu_times(self) -> Any:
                return type("CpuTimes", (), {"user": 0.0, "system": 0.0})()

            def memory_info(self) -> Any:
                return type("MemoryInfo", (), {"rss": 0})()

            def terminate(self) -> None:
                raise psutil.NoSuchProcess(self.pid)

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                raise psutil.NoSuchProcess(self.pid)

        monkeypatch.setattr("agent_libos.substrate.local.psutil.Process", TreeEnumerationDenied)

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            if mode == "wall" and not provider.supports_subprocess_limits:
                pytest.skip("wall-time SubprocessLimits are unavailable on Windows")
            if mode == "success":
                result = provider.run([sys.executable, "-c", "print('ok')"], timeout=2.0)
                assert result.returncode == 0
                assert result.stdout.strip() == "ok"
            elif mode == "wall":
                with pytest.raises(SubprocessLimitExceeded) as exc_info:
                    provider.run(
                        [sys.executable, "-c", "import time; time.sleep(0.2)"],
                        timeout=2.0,
                        limits=SubprocessLimits(wall_seconds=0.02),
                    )
                assert exc_info.value.metrics.limit_kind == "subprocess_wall_seconds"
            else:
                with pytest.raises(SubprocessTimeoutExpired) as exc_info:
                    provider.run(
                        [sys.executable, "-c", "import time; time.sleep(0.2)"],
                        timeout=0.02,
                    )
                assert exc_info.value.metrics.limit_kind == "subprocess_timeout"

        assert len(instances) == 1
        assert instances[0].children_calls == (0 if mode == "success" else 1)

    @pytest.mark.parametrize(
        "limits",
        [
            SubprocessLimits(cpu_seconds=1.0),
            SubprocessLimits(memory_bytes=512_000_000),
        ],
        ids=["cpu", "memory"],
    )
    def test_local_shell_provider_cpu_memory_limits_fail_closed_when_tree_metrics_are_denied(
        self,
        limits: SubprocessLimits,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class TreeEnumerationDenied:
            def __init__(self, pid: int) -> None:
                self.pid = pid

            def children(self, *, recursive: bool) -> list[Any]:
                assert recursive
                raise PermissionError("process tree enumeration denied")

            def terminate(self) -> None:
                raise psutil.NoSuchProcess(self.pid)

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                raise psutil.NoSuchProcess(self.pid)

        monkeypatch.setattr("agent_libos.substrate.local.psutil.Process", TreeEnumerationDenied)

        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalShellProvider(temp_dir)
            expected = (
                "complete process metrics are unavailable"
                if provider.supports_subprocess_limits
                else "cannot enforce SubprocessLimits"
            )
            with pytest.raises(ValidationError, match=expected):
                provider.run(
                    [sys.executable, "-c", "import time; time.sleep(0.2)"],
                    timeout=2.0,
                    limits=limits,
                )

    def test_local_shell_provider_kill_falls_back_to_direct_child_when_group_and_tree_access_are_denied(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not hasattr(os, "killpg"):
            pytest.skip("POSIX process-group fallback is unavailable")
        class TreeAccessDenied:
            def children(self, *, recursive: bool) -> list[Any]:
                assert recursive
                raise PermissionError("process tree enumeration denied")

            def terminate(self) -> None:
                raise PermissionError("process terminate denied")

            def kill(self) -> None:
                raise PermissionError("process kill denied")

        class DirectChild:
            pid = 424242

            def __init__(self) -> None:
                self.kill_calls = 0

            def poll(self) -> int | None:
                return None

            def kill(self) -> None:
                self.kill_calls += 1

        def deny_group_signal(_pid: int, _signal: int) -> None:
            raise PermissionError("process group signal denied")

        def deny_wait(_processes: list[Any], *, timeout: float) -> tuple[list[Any], list[Any]]:
            raise PermissionError("process wait denied")

        monkeypatch.setattr(os, "killpg", deny_group_signal)
        monkeypatch.setattr("agent_libos.substrate.local.psutil.wait_procs", deny_wait)
        child = DirectChild()

        LocalShellProvider(".")._kill_process_tree(TreeAccessDenied(), child)  # type: ignore[arg-type]

        assert child.kill_calls == 1

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

    def test_local_shell_provider_cleanup_kills_background_sentinel_writer(self) -> None:
        marker = f"SHELL_PROVIDER_SENTINEL_{time.monotonic_ns()}"
        child_script = (
            "import pathlib, sys, time; "
            "time.sleep(0.5); "
            "pathlib.Path(sys.argv[1]).write_text(sys.argv[2], encoding='utf-8')"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            sentinel = Path(temp_dir) / "sentinel.txt"
            if sys.platform == "win32":
                child_kwargs = "dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)"
            else:
                child_kwargs = "{}"
            parent_script = (
                "import subprocess, sys; "
                f"kwargs = {child_kwargs}; "
                "subprocess.Popen("
                f"[sys.executable, '-c', {child_script!r}, {str(sentinel)!r}, {marker!r}], "
                "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs)"
            )
            provider = LocalShellProvider(temp_dir)
            try:
                result = provider.run([sys.executable, "-c", parent_script], timeout=5.0)
                assert result.returncode == 0

                deadline = time.monotonic() + 1.5
                while time.monotonic() < deadline and not sentinel.exists():
                    time.sleep(0.05)
                assert not sentinel.exists()
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

    def test_local_pty_snapshot_preserves_explicit_virtualenv_argv0(
        self,
        tmp_path: Path,
    ) -> None:
        if os.name == "nt":
            pytest.skip("POSIX executable/argv[0] separation is not available on Windows")
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        environment = workspace / ".venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        executable = environment / "bin" / "python"
        provider = LocalPtyProvider(workspace)
        argv = [
            ".venv/bin/python",
            "-c",
            (
                "import json, sys; "
                "print(json.dumps({'prefix': sys.prefix, 'executable': sys.executable}), flush=True)"
            ),
        ]
        resolved = provider.resolve_argv(argv)
        snapshot = snapshot_executable(resolved[0])
        session = provider.spawn(argv, executable_snapshot=snapshot)
        output = ""
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                output += session.read(timeout_s=0.1)
                if not session.is_alive():
                    output += session.read(timeout_s=0.0)
                    break
        finally:
            session.close(force=True, timeout_s=1.0)

        payload = json.loads(output.replace("\r", "").strip())
        assert Path(payload["prefix"]) == environment
        assert Path(payload["executable"]) == executable

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

    def test_local_pty_provider_contains_process_when_post_spawn_initialization_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if sys.platform == "win32":
            pytest.skip("POSIX PTY initialization containment is platform-specific")
        marker = f"PTY_PROVIDER_INIT_FAILURE_{time.monotonic_ns()}"
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = LocalPtyProvider(temp_dir)

            def fail_set_blocking(_fd: int, _blocking: bool) -> None:
                raise OSError("simulated post-spawn PTY initialization failure")

            monkeypatch.setattr(os, "set_blocking", fail_set_blocking)
            try:
                with pytest.raises(OSError, match="post-spawn PTY initialization failure"):
                    provider.spawn([sys.executable, "-c", "import time; time.sleep(60)", marker])

                deadline = time.monotonic() + 3.0
                while time.monotonic() < deadline and _processes_with_marker(marker):
                    time.sleep(0.05)
                assert _processes_with_marker(marker) == []
            finally:
                for proc in _processes_with_marker(marker):
                    with contextlib.suppress(psutil.Error):
                        proc.kill()

    def test_local_pty_provider_permission_denied_group_signal_uses_tree_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if sys.platform == "win32":
            pytest.skip("POSIX process-group cleanup does not apply to pywinpty")

        class SignalNode:
            def __init__(self, pid: int, children: list["SignalNode"] | None = None) -> None:
                self.pid = pid
                self._children = list(children or [])
                self.signals: list[int] = []

            def children(self, *, recursive: bool) -> list["SignalNode"]:
                assert recursive
                return list(self._children)

            def send_signal(self, selected_signal: int) -> None:
                self.signals.append(selected_signal)

        child = SignalNode(102)
        root = SignalNode(101, [child])
        process = type("FakePopen", (), {"pid": root.pid})()
        session = _PosixPtySession(master_fd=-1, proc=process)

        def deny_group_signal(_pid: int, _signal: int) -> None:
            raise PermissionError("simulated process-group denial")

        monkeypatch.setattr(os, "killpg", deny_group_signal)
        monkeypatch.setattr("modules.pty.pty_module.psutil.Process", lambda pid: root)

        session._signal_process_group(signal.SIGTERM)

        assert child.signals == [signal.SIGTERM]
        assert root.signals == [signal.SIGTERM]

    def test_effectful_provider_without_classification_fails_closed(self) -> None:
        runtime = Runtime.open('local')
        provider = NoClassificationShellProvider()
        shell = ShellAdapter(
            runtime.capability,
            runtime.audit,
            protected_operations=runtime.protected_operations,
            provider=provider,
        )
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
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='use fake human',
                    authority_manifest={
                        'authorized_capabilities': [
                            {
                                'resource': 'human:owner',
                                'rights': [CapabilityRight.WRITE.value],
                            }
                        ]
                    },
                )
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

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None='\n',
        *,
        overwrite: bool = True,
    ) -> None:
        self.calls.append('write_text')
        self.inner.write_text(
            path,
            text,
            encoding,
            newline,
            overwrite=overwrite,
        )

    def write_text_compare_and_swap(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None='\n',
        *,
        overwrite: bool = True,
        expected_content_sha256: str,
    ) -> None:
        self.calls.append('write_text_compare_and_swap')
        self.inner.write_text_compare_and_swap(
            path,
            text,
            encoding,
            newline,
            overwrite=overwrite,
            expected_content_sha256=expected_content_sha256,
        )

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
