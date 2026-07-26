from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ResourceBudget
from agent_libos.models.exceptions import ResourceLimitExceeded, ValidationError
from agent_libos.substrate import (
    DirectoryEntrySnapshot,
    LocalFilesystemProvider,
    LocalResourceProviderSubstrate,
    ResolvedPath,
)
from tests.security.test_shell_primitive import FakeShellProvider, RecordingShellSubstrate


class TestResourceConstraints:
    def test_capability_grant_does_not_expand_tool_call_budget(self) -> None:
        provider = FakeShellProvider()
        runtime = Runtime.open(
            "local",
            substrate=RecordingShellSubstrate(".", provider),
        )
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="budget cannot be bypassed",
                resource_budget=ResourceBudget(max_tool_calls=0),
            )
            runtime.tools.configure_process_tools(pid, ["run_shell_command"], assigned_by="test")
            runtime.shell.grant_policy(pid, "always_allow", issued_by="test")
            runtime.capability.grant(pid, "shell:git", [CapabilityRight.EXECUTE], issued_by="test")

            result = runtime.tools.call(pid, "run_shell_command", {"argv": ["git", "status"]})

            assert not result.ok
            assert result.error == (
                "Tool call resource budget was exceeded before execution."
            )
            assert provider.calls == []
        finally:
            runtime.close()

    def test_exec_process_does_not_reset_resource_usage_or_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="exec resource",
                resource_budget=ResourceBudget(max_tool_calls=2),
            )
            runtime.tools.configure_process_tools(pid, ["get_working_directory"], assigned_by="test")
            assert runtime.tools.call(pid, "get_working_directory", {}).ok

            runtime.exec_process(pid, "base-agent:v0", goal="after exec")
            process = runtime.process.get(pid)

            assert process.resource_budget.max_tool_calls == 2
            assert process.resource_usage.tool_calls == 1
        finally:
            runtime.close()

    def test_always_allow_shell_policy_does_not_bypass_subprocess_wall_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(
                    image="base-agent:v0",
                    goal="subprocess limit",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=0.05),
                )
                runtime.tools.configure_process_tools(pid, ["run_shell_command"], assigned_by="test")
                runtime.shell.grant_policy(pid, "always_allow", issued_by="test")

                result = runtime.tools.call(
                    pid,
                    "run_shell_command",
                    {"argv": ["python", "-c", "import time; time.sleep(0.2)"], "timeout_s": 5.0},
                )

                assert not result.ok
                if sys.platform == "win32":
                    assert (result.error or "").startswith(
                        "validation_error: ValidationError"
                    )
                    assert runtime.process.get(pid).status.value == "runnable"
                else:
                    assert runtime.process.get(pid).status.value == "killed"
                    assert any(record.action == "resource.limit_exceeded" for record in runtime.audit.trace())
            finally:
                runtime.close()

    def test_shell_provider_without_limits_support_fails_closed_when_budgeted(self) -> None:
        provider = NoLimitsShellProvider()
        runtime = Runtime.open(
            "local",
            substrate=RecordingShellSubstrate(".", provider),
        )
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="provider limits required",
                resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
            )
            runtime.tools.configure_process_tools(pid, ["run_shell_command"], assigned_by="test")
            runtime.shell.grant_policy(pid, "always_allow", issued_by="test")

            result = runtime.tools.call(pid, "run_shell_command", {"argv": ["git", "status"]})

            assert not result.ok
            assert (result.error or "").startswith(
                "validation_error: ValidationError"
            )
            assert provider.calls == []
        finally:
            runtime.close()

    def test_shell_timeout_charges_metrics_without_killing_when_budget_remains(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open("local", substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(
                    image="base-agent:v0",
                    goal="timeout charge",
                    resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
                )
                runtime.shell.grant_policy(pid, "always_allow", issued_by="test")

                if sys.platform == "win32":
                    with pytest.raises(ValidationError, match="SubprocessLimits"):
                        runtime.shell.run(
                            pid,
                            ["python", "-c", "import time; time.sleep(0.2)"],
                            timeout=0.05,
                        )
                    assert runtime.process.get(pid).status.value == "runnable"
                    return

                with pytest.raises(TimeoutError):
                    runtime.shell.run(
                        pid,
                        ["python", "-c", "import time; time.sleep(0.2)"],
                        timeout=0.05,
                    )

                process = runtime.process.get(pid)
                assert process.status.value == "runnable"
                assert process.resource_usage.subprocess_wall_seconds > 0
            finally:
                runtime.close()

    def test_terminal_process_cannot_call_visible_tools_directly(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="terminal")
            runtime.tools.configure_process_tools(pid, ["get_working_directory"], assigned_by="test")
            runtime.process.exit(pid, failed=False, message="done")

            result = runtime.tools.call(pid, "get_working_directory", {})

            assert not result.ok
            assert "terminal process" in (result.error or "")
        finally:
            runtime.close()

    def test_filesystem_read_uses_provider_limited_read_and_charges_actual_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = RecordingLimitedFilesystemProvider(temp_dir)
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.filesystem = provider
            Path(temp_dir, "large.txt").write_text("x" * 100, encoding="utf-8")
            runtime = Runtime.open("local", substrate=substrate)
            try:
                pid = runtime.process.spawn(
                    image="base-agent:v0",
                    goal="limited read",
                    resource_budget=ResourceBudget(max_external_read_bytes=10),
                )
                runtime.filesystem.grant_path(pid, "large.txt", [CapabilityRight.READ], issued_by="test")

                result = runtime.filesystem.read_bytes(pid, "large.txt", max_bytes=10)

                assert result.bytes_read == 10
                assert result.truncated
                assert provider.read_limits == [10]
                assert runtime.process.get(pid).resource_usage.external_read_bytes == 10
            finally:
                runtime.close()

    def test_filesystem_read_rejects_provider_limit_contract_violations(
        self,
    ) -> None:
        for provider_type in (OversizedFilesystemProvider, NonBytesFilesystemProvider):
            with tempfile.TemporaryDirectory() as temp_dir:
                provider = provider_type(temp_dir)
                substrate = LocalResourceProviderSubstrate(temp_dir)
                substrate.filesystem = provider
                Path(temp_dir, "payload.bin").write_bytes(b"trusted")
                runtime = Runtime.open("local", substrate=substrate)
                try:
                    pid = runtime.process.spawn(image="base-agent:v0", goal="reject dishonest read")
                    runtime.filesystem.grant_path(
                        pid,
                        "payload.bin",
                        [CapabilityRight.READ],
                        issued_by="test",
                    )

                    with pytest.raises(ValidationError, match="filesystem provider"):
                        runtime.filesystem.read_bytes(pid, "payload.bin", max_bytes=4)
                finally:
                    runtime.close()

    def test_directory_listing_consumes_only_sentinel_and_closes_provider_iterator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = InfiniteListFilesystemProvider(temp_dir)
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.filesystem = provider
            Path(temp_dir, "items").mkdir()
            runtime = Runtime.open("local", substrate=substrate)
            try:
                pid = runtime.process.spawn(image="base-agent:v0", goal="bounded directory list")
                runtime.filesystem.grant_directory(
                    pid,
                    "items",
                    [CapabilityRight.READ],
                    issued_by="test",
                )

                result = runtime.filesystem.read_directory(pid, "items", limit=2)

                assert result.count == 2
                assert result.truncated
                assert provider.iterator.yield_count == 3
                assert provider.iterator.closed
            finally:
                runtime.close()

    def test_directory_listing_preflights_metadata_budget_before_provider_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            provider = RecordingListFilesystemProvider(temp_dir)
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.filesystem = provider
            Path(temp_dir, "item.txt").write_text("content", encoding="utf-8")
            runtime = Runtime.open("local", substrate=substrate)
            try:
                pid = runtime.process.spawn(
                    image="base-agent:v0",
                    goal="directory budget",
                    resource_budget=ResourceBudget(max_external_read_bytes=1),
                )
                runtime.filesystem.grant_directory(pid, ".", [CapabilityRight.READ], issued_by="test")

                with pytest.raises(ResourceLimitExceeded):
                    runtime.filesystem.read_directory(pid, ".")

                assert provider.list_calls == 0
                assert runtime.process.get(pid).status.value == "runnable"
                assert runtime.process.get(pid).resource_usage.external_read_bytes == 0
            finally:
                runtime.close()


class NoLimitsShellProvider(FakeShellProvider):
    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ):
        self.calls.append((list(argv), timeout))
        return super().run(
            argv,
            timeout=timeout,
            cwd=cwd,
            stdout_limit_chars=stdout_limit_chars,
            stderr_limit_chars=stderr_limit_chars,
        )


class RecordingLimitedFilesystemProvider(LocalFilesystemProvider):
    def __init__(self, root: str):
        super().__init__(root)
        self.read_limits: list[int | None] = []

    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes:
        self.read_limits.append(max_bytes)
        return super().read_bytes(path, max_bytes=max_bytes)


class OversizedFilesystemProvider(LocalFilesystemProvider):
    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes:
        assert max_bytes is not None
        return b"x" * (max_bytes + 1)


class NonBytesFilesystemProvider(LocalFilesystemProvider):
    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes:
        del path, max_bytes
        return "not-bytes"  # type: ignore[return-value]


class _InfiniteDirectoryIterator:
    def __init__(self) -> None:
        self.yield_count = 0
        self.closed = False

    def __iter__(self) -> _InfiniteDirectoryIterator:
        return self

    def __next__(self) -> DirectoryEntrySnapshot:
        self.yield_count += 1
        if self.yield_count > 4:
            raise AssertionError("directory iterator was consumed without a bound")
        index = self.yield_count
        return DirectoryEntrySnapshot(
            name=f"item-{index}",
            path=f"items/item-{index}",
            kind="file",
            size_bytes=index,
            modified_at="2026-01-01T00:00:00Z",
        )

    def close(self) -> None:
        self.closed = True


class InfiniteListFilesystemProvider(LocalFilesystemProvider):
    def __init__(self, root: str | Path):
        super().__init__(root)
        self.iterator = _InfiniteDirectoryIterator()

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None):
        del path, limit
        return self.iterator


class RecordingListFilesystemProvider(LocalFilesystemProvider):
    def __init__(self, root: str | Path):
        super().__init__(root)
        self.list_calls = 0

    def list_directory(self, path: ResolvedPath, *, limit: int | None = None):
        self.list_calls += 1
        return super().list_directory(path, limit=limit)
