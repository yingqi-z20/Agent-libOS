from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import ValidationError
from agent_libos.primitives.filesystem import FileReadResult
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import (
    FilesystemCompareAndSwapProvider,
    FilesystemContentConflict,
    LocalFilesystemProvider,
    LocalResourceProviderSubstrate,
    ResolvedPath,
)
from tests.support.runtime import workspace_runtime


class _LegacyFilesystemProvider:
    """Pre-CAS custom provider whose write method rejects unknown keywords."""

    def __init__(self, root: Path) -> None:
        self.inner = LocalFilesystemProvider(root)
        self.namespace = self.inner.namespace
        self.root_display = self.inner.root_display
        self.write_calls = 0

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = "\n",
        *,
        overwrite: bool = True,
    ) -> None:
        self.write_calls += 1
        self.inner.write_text(
            path,
            text,
            encoding,
            newline,
            overwrite=overwrite,
        )

    def __getattr__(self, name: str) -> Any:
        if name == "write_text_compare_and_swap":
            raise AttributeError(name)
        return getattr(self.inner, name)


def test_file_read_result_legacy_four_argument_constructor_remains_valid() -> None:
    result = FileReadResult("legacy.txt", "content", 7, False)

    assert result.content_sha256 is None


def test_read_text_returns_full_raw_digest_and_omits_it_for_prefixes() -> None:
    with workspace_runtime() as (runtime, root):
        path = "digest.txt"
        content = "snow: 雪\n"
        (root / path).write_bytes(content.encode("utf-8"))
        pid = runtime.process.spawn(goal="read filesystem edit token")
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.READ],
            issued_by="test.host",
        )

        complete = runtime.filesystem.read_text(pid, path)
        prefix = runtime.filesystem.read_text(pid, path, max_bytes=4)

        assert complete.content_sha256 == hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
        assert complete.truncated is False
        assert prefix.truncated is True
        assert prefix.content_sha256 is None


def test_compare_and_swap_success_and_stale_conflict_are_atomic() -> None:
    with workspace_runtime() as (runtime, root):
        path = "edit.txt"
        target = root / path
        target.write_text("old", encoding="utf-8")
        pid = runtime.process.spawn(goal="atomic filesystem edit")
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        stale_digest = runtime.filesystem.read_text(pid, path).content_sha256
        assert stale_digest is not None
        first_write = runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        result = runtime.filesystem.write_text(
            pid,
            path,
            "new",
            expected_content_sha256=stale_digest,
        )

        assert result.created is False
        assert target.read_text(encoding="utf-8") == "new"
        assert runtime.store.get_capability(first_write.cap_id).uses_remaining == 0
        binding_after_success = runtime.store.get_file_label_binding(path)
        effects_after_success = tuple(runtime.store.list_external_effects(pid=pid))
        second_write = runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        with pytest.raises(FilesystemContentConflict):
            runtime.filesystem.write_text(
                pid,
                path,
                "must not replace newer bytes",
                expected_content_sha256=stale_digest,
            )

        assert target.read_text(encoding="utf-8") == "new"
        assert runtime.store.get_capability(second_write.cap_id).uses_remaining == 1
        assert runtime.store.get_file_label_binding(path) == binding_after_success
        assert tuple(runtime.store.list_external_effects(pid=pid)) == effects_after_success


def test_compare_and_swap_missing_precondition_is_atomic() -> None:
    with workspace_runtime() as (runtime, root):
        pid = runtime.process.spawn(goal="atomic filesystem creation")
        created_path = "created.txt"
        runtime.filesystem.grant_path(
            pid,
            created_path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        created = runtime.filesystem.write_text(
            pid,
            created_path,
            "created",
            expected_content_sha256="missing",
        )

        assert created.created is True
        assert (root / created_path).read_text(encoding="utf-8") == "created"

        existing_path = "existing.txt"
        existing = root / existing_path
        existing.write_text("preserve", encoding="utf-8")
        finite_write = runtime.capability.grant_once(
            pid,
            runtime.filesystem.resource_for_path(existing_path),
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        effects_before = tuple(runtime.store.list_external_effects(pid=pid))

        with pytest.raises(FilesystemContentConflict):
            runtime.filesystem.write_text(
                pid,
                existing_path,
                "must not overwrite",
                expected_content_sha256="missing",
            )

        assert existing.read_text(encoding="utf-8") == "preserve"
        assert runtime.store.get_capability(finite_write.cap_id).uses_remaining == 1
        assert runtime.store.get_file_label_binding(existing_path) is None
        assert tuple(runtime.store.list_external_effects(pid=pid)) == effects_before


def test_legacy_provider_supports_plain_writes_and_rejects_cas_before_mutation(
    tmp_path: Path,
) -> None:
    provider = _LegacyFilesystemProvider(tmp_path)
    assert not isinstance(provider, FilesystemCompareAndSwapProvider)
    substrate = LocalResourceProviderSubstrate(tmp_path)
    substrate.filesystem = provider
    runtime = Runtime.open("local", substrate=substrate)
    try:
        plain_pid = runtime.process.spawn(goal="legacy provider plain write")
        path = "legacy.txt"
        runtime.filesystem.grant_path(
            plain_pid,
            path,
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )

        runtime.filesystem.write_text(plain_pid, path, "compatible")

        assert provider.write_calls == 1
        assert (tmp_path / path).read_text(encoding="utf-8") == "compatible"

        cas_pid = runtime.process.spawn(goal="legacy provider rejected CAS")
        finite_write = runtime.capability.grant_once(
            cas_pid,
            runtime.filesystem.resource_for_path(path),
            [CapabilityRight.WRITE],
            issued_by="test.host",
        )
        effects_before = tuple(runtime.store.list_external_effects(pid=cas_pid))
        binding_before = runtime.store.get_file_label_binding(path)

        with pytest.raises(
            ValidationError,
            match="provider does not support content compare-and-swap",
        ):
            runtime.filesystem.write_text(
                cas_pid,
                path,
                "must not dispatch",
                expected_content_sha256="0" * 64,
            )

        assert provider.write_calls == 1
        assert (tmp_path / path).read_text(encoding="utf-8") == "compatible"
        assert runtime.store.get_capability(finite_write.cap_id).uses_remaining == 1
        assert runtime.store.get_file_label_binding(path) == binding_before
        assert tuple(runtime.store.list_external_effects(pid=cas_pid)) == effects_before
    finally:
        runtime.close()


def test_tool_and_syscall_forward_filesystem_edit_tokens() -> None:
    with workspace_runtime() as (runtime, root):
        path = "forwarded.txt"
        (root / path).write_text("initial", encoding="utf-8")
        pid = runtime.process.spawn(image="review-agent:v0", goal="forward edit token")
        runtime.filesystem.grant_path(
            pid,
            path,
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="test.host",
        )

        tool_read = runtime.tools.call(pid, "read_text_file", {"path": path})
        assert tool_read.ok, tool_read.error
        tool_write = runtime.tools.call(
            pid,
            "write_text_file",
            {
                "path": path,
                "content": "from tool",
                "expected_content_sha256": tool_read.payload["content_sha256"],
            },
        )
        assert tool_write.ok, tool_write.error

        async def exercise_syscall() -> None:
            session = LibOSSyscallSession(runtime, pid)
            syscall_read = await session.handle("filesystem.read_text", {"path": path})
            await session.handle(
                "filesystem.write_text",
                {
                    "path": path,
                    "text": "from syscall",
                    "expected_content_sha256": syscall_read["content_sha256"],
                },
            )

        asyncio.run(exercise_syscall())
        assert (root / path).read_text(encoding="utf-8") == "from syscall"
