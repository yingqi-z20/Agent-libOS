from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib
import inspect
import os
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, GitDefaults
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    EventType,
    GitErrorCode,
    GitPullRequestStatus,
    ObjectMetadata,
    ObjectType,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    GitError,
    HumanApprovalRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.primitives.git import GitPrimitive
from agent_libos.substrate import (
    GitCommandResult,
    GitLimitedRunProvider,
    GitProvider,
    GitProviderEffectNotStarted,
    GitSubprocessScopeProvider,
    LocalGitProvider,
    LocalResourceProviderSubstrate,
    LocalShellProvider,
    SubprocessLimitExceeded,
    SubprocessLimits,
)
from agent_libos.utils.secure_host_files import (
    SecureDirectoryGuard,
    SecureFileDescriptor,
    StablePathSnapshot,
)


_T = TypeVar("_T")

# Git for Windows performs the same repository-identity and lineage checks with
# substantially higher process and filesystem latency. Keep the platform's
# already-established 300-second ceiling consistent across this module; the
# provider lane retains its independent hard deadline.
pytestmark = pytest.mark.timeout(300 if os.name == "nt" else 120)


class _LegacyGitProvider:
    """1.0.0-shape adapter with no subprocess supervision extensions."""

    _EXTENSION_ATTRIBUTES = frozenset(
        {
            "run_with_limits",
            "subprocess_scope",
            "supports_subprocess_limits",
        }
    )

    def __init__(self, delegate: LocalGitProvider) -> None:
        self._delegate = delegate
        self.run_calls = 0

    def __getattr__(self, name: str) -> Any:
        if name in self._EXTENSION_ATTRIBUTES:
            raise AttributeError(name)
        return getattr(self._delegate, name)

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
        self.run_calls += 1
        return self._delegate.run(
            args,
            worktree=worktree,
            timeout=timeout,
            stdin=stdin,
            max_output_bytes=max_output_bytes,
            read_only=read_only,
            remote=remote,
            expected_remote_fingerprint=expected_remote_fingerprint,
            verify_after=verify_after,
        )


@pytest.fixture(autouse=True)
def _isolate_host_git_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    git_home = tmp_path / "git-home"
    git_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(git_home))


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_TRACE": "0",
            "GIT_TRACE2": "0",
            "GIT_TRACE2_EVENT": "0",
            "GIT_TRACE2_PERF": "0",
        }
    )
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        env=environment,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )
    return result.stdout


def _init_repository(root: Path) -> None:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root, "config", "user.name", "Agent libOS Test")
    _git(root, "config", "user.email", "agent-libos@example.test")
    _git(root, "config", "core.autocrlf", "false")
    (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(root, "add", "--", "tracked.txt")
    _git(root, "commit", "-q", "-m", "initial")


def _runtime_config(*, git: GitDefaults | None = None) -> AgentLibOSConfig:
    return replace(
        DEFAULT_CONFIG,
        git=git or DEFAULT_CONFIG.git,
        modules=replace(
            DEFAULT_CONFIG.modules,
            manifest_paths=(),
            trusted_modules=(),
            trusted_sha256=(),
        ),
    )


def _open_runtime(root: Path, *, git: GitDefaults | None = None) -> Runtime:
    selected_git = git or DEFAULT_CONFIG.git
    return Runtime.open(
        ":memory:",
        config=_runtime_config(git=selected_git),
        substrate=LocalResourceProviderSubstrate(root, git_config=selected_git),
        module_manifests=(),
    )


def test_git_provider_preserves_0_3_4_run_contract() -> None:
    expected_parameters = [
        "self",
        "args",
        "worktree",
        "timeout",
        "stdin",
        "max_output_bytes",
        "read_only",
        "remote",
        "expected_remote_fingerprint",
        "verify_after",
    ]

    assert list(inspect.signature(GitProvider.run).parameters) == expected_parameters
    assert inspect.signature(LocalGitProvider.run) == inspect.signature(GitProvider.run)
    assert inspect.signature(_LegacyGitProvider.run) == inspect.signature(GitProvider.run)
    assert "supports_subprocess_limits" not in GitProvider.__annotations__
    assert "subprocess_scope" not in vars(GitProvider)
    substrate_exports = importlib.import_module("agent_libos.substrate")
    assert {
        "GitLimitedRunProvider",
        "GitSubprocessScopeProvider",
    } <= set(substrate_exports.__all__)


def test_runtime_accepts_legacy_git_provider_without_optional_extensions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root)
    legacy = _LegacyGitProvider(substrate.git)
    substrate.git = legacy
    runtime = Runtime.open(
        ":memory:",
        config=_runtime_config(),
        substrate=substrate,
        module_manifests=(),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="use legacy Git provider")
        _grant_git_authority(runtime, pid)

        assert runtime.git.diff(pid).sha256
        assert legacy.run_calls > 0
        assert not isinstance(legacy, GitSubprocessScopeProvider)
        assert not isinstance(legacy, GitLimitedRunProvider)
    finally:
        runtime.close()


def test_budgeted_git_execution_rejects_legacy_provider_before_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root)
    legacy = _LegacyGitProvider(substrate.git)
    substrate.git = legacy
    runtime = Runtime.open(
        ":memory:",
        config=_runtime_config(),
        substrate=substrate,
        module_manifests=(),
    )
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject unsupervised legacy Git",
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
        )
        _grant_git_authority(runtime, pid)

        with pytest.raises(
            ValidationError,
            match="must support SubprocessLimits",
        ):
            runtime.git.status(pid)
        assert legacy.run_calls == 0
    finally:
        runtime.close()


def _grant_git_authority(runtime: Runtime, pid: str, *, remote: str | None = None) -> None:
    _grant_git_repository_authority(runtime, pid, remote=remote)
    runtime.filesystem.grant_directory(
        pid,
        ".",
        [CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE],
        issued_by="git-provider-test",
    )


def _grant_git_repository_authority(
    runtime: Runtime,
    pid: str,
    *,
    remote: str | None = None,
) -> None:
    runtime.capability.issue_trusted(
        pid,
        "git:workspace",
        [
            CapabilityRight.READ,
            CapabilityRight.DIFF,
            CapabilityRight.WRITE,
            CapabilityRight.DELETE,
            CapabilityRight.ADMIN,
        ],
        issued_by="git-provider-test",
    )
    runtime.capability.issue_trusted(
        pid,
        "git_pr:workspace:*",
        [
            CapabilityRight.READ,
            CapabilityRight.WRITE,
            CapabilityRight.APPROVE,
            CapabilityRight.DELETE,
        ],
        issued_by="git-provider-test",
    )
    if remote is not None:
        runtime.capability.issue_trusted(
            pid,
            f"git_remote:workspace:{remote}",
            [
                CapabilityRight.READ,
                CapabilityRight.WRITE,
                CapabilityRight.DELETE,
                CapabilityRight.ADMIN,
            ],
            issued_by="git-provider-test",
        )


def _with_auto_approvals(runtime: Runtime, callback: Callable[[], _T]) -> _T:
    for _attempt in range(8):
        try:
            return callback()
        except HumanApprovalRequired:
            assert runtime.human.drain_terminal_queue(auto_approve=True)
    raise AssertionError("Git operation requested too many sequential approvals")


def test_git_unavailable_is_lazy_and_does_not_block_runtime_startup(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    git_config = replace(DEFAULT_CONFIG.git, executable="definitely-missing-agent-libos-git")
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect unavailable Git")
        _grant_git_authority(runtime, pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.repository_info(pid)
        assert exc_info.value.code == GitErrorCode.GIT_UNAVAILABLE.value
    finally:
        runtime.close()


def test_provider_defers_hook_isolation_until_a_git_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    git_provider_module = importlib.import_module("agent_libos.substrate.git")

    def unavailable_temp_directory(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("injected unavailable temporary directory")

    monkeypatch.setattr(
        git_provider_module.tempfile,
        "TemporaryDirectory",
        unavailable_temp_directory,
    )
    provider = LocalGitProvider(root)
    assert provider._hooks_tempdir is None

    with pytest.raises(GitError) as exc_info:
        provider.repository_layout()
    assert exc_info.value.code == GitErrorCode.COMMAND_FAILED.value


def test_provider_repository_lock_thread_waiter_honors_timeout(
    tmp_path: Path,
) -> None:
    sync_timeout_s = 30.0
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    holder_entered = threading.Event()
    holder_release = threading.Event()
    holder_done = threading.Event()
    waiter_done = threading.Event()
    holder_failures: list[BaseException] = []
    waiter_results: list[str] = []

    def hold_repository_lock() -> None:
        try:
            with provider.repository_lock(timeout=DEFAULT_CONFIG.git.lock_timeout_s):
                holder_entered.set()
                if not holder_release.wait(timeout=sync_timeout_s):
                    raise AssertionError("repository lock holder was not released")
        except BaseException as exc:  # pragma: no cover - surfaced below
            holder_failures.append(exc)
        finally:
            holder_done.set()

    def wait_for_repository_lock() -> None:
        try:
            with provider.repository_lock(timeout=0.05):
                waiter_results.append("acquired")
        except GitError as exc:
            waiter_results.append(exc.code)
        except BaseException as exc:  # pragma: no cover - surfaced below
            waiter_results.append(f"unexpected:{exc!r}")
        finally:
            waiter_done.set()

    holder = threading.Thread(target=hold_repository_lock)
    waiter = threading.Thread(target=wait_for_repository_lock)
    holder.start()
    assert holder_entered.wait(timeout=sync_timeout_s)
    waiter.start()

    waiter_completed_while_held = waiter_done.wait(timeout=sync_timeout_s)
    holder_was_still_holding = not holder_done.is_set()
    holder_release.set()
    holder.join(timeout=sync_timeout_s)
    waiter.join(timeout=sync_timeout_s)

    assert not holder.is_alive()
    assert not waiter.is_alive()
    assert holder_failures == []
    assert waiter_completed_while_held
    assert holder_was_still_holding
    assert waiter_results == [GitErrorCode.REPOSITORY_BUSY.value]

    with provider.repository_lock(timeout=DEFAULT_CONFIG.git.lock_timeout_s):
        pass


def test_provider_repository_lock_remains_reentrant_in_same_thread(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)

    with provider.repository_lock(
        timeout=DEFAULT_CONFIG.git.lock_timeout_s
    ) as outer:
        with provider.repository_lock(timeout=0.05) as inner:
            assert outer.repository_id == inner.repository_id
            assert outer.worktree_id == inner.worktree_id


def test_provider_repository_lock_rejects_mocked_reparse_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    lock_directory = root / ".git" / "agent-libos"
    lock_directory.mkdir()
    original_open_child = SecureDirectoryGuard.open_child_directory

    def reject_reparse_child(
        directory: SecureDirectoryGuard,
        name: str,
    ) -> SecureDirectoryGuard:
        if directory.path == root / ".git" and name == "agent-libos":
            raise OSError("mocked reparse directory")
        return original_open_child(directory, name)

    monkeypatch.setattr(
        SecureDirectoryGuard,
        "open_child_directory",
        reject_reparse_child,
    )

    with pytest.raises(GitError) as exc_info:
        with provider.repository_lock():
            raise AssertionError("unsafe lock directory must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert not (lock_directory / "repository.lock").exists()


def test_provider_repository_lock_rejects_mocked_reparse_file_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    lock_directory = root / ".git" / "agent-libos"
    lock_directory.mkdir()
    lock_path = lock_directory / "repository.lock"
    lock_path.write_bytes(b"")
    git_provider_module = importlib.import_module("agent_libos.substrate.git")
    original_open = git_provider_module.open_secure_readwrite_child

    def mocked_reparse_file(*args: Any, **kwargs: Any) -> SecureFileDescriptor:
        secure_file = original_open(*args, **kwargs)
        original_snapshot = secure_file.snapshot

        def reparse_snapshot() -> StablePathSnapshot:
            return replace(original_snapshot(), is_reparse_point=True)

        secure_file.snapshot = reparse_snapshot  # type: ignore[method-assign]
        return secure_file

    monkeypatch.setattr(
        git_provider_module,
        "open_secure_readwrite_child",
        mocked_reparse_file,
    )

    with pytest.raises(GitError) as exc_info:
        with provider.repository_lock():
            raise AssertionError("unsafe lock file must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert lock_path.read_bytes() == b""


def test_provider_repository_lock_rejects_opened_path_identity_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    original_linked_snapshot = SecureFileDescriptor.linked_snapshot

    def mismatched_linked_snapshot(
        secure_file: SecureFileDescriptor,
    ) -> StablePathSnapshot:
        observed = original_linked_snapshot(secure_file)
        return replace(observed, inode=observed.inode + 1)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            SecureFileDescriptor,
            "linked_snapshot",
            mismatched_linked_snapshot,
        )
        with pytest.raises(GitError) as exc_info:
            with provider.repository_lock():
                raise AssertionError("mismatched lock identity must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    with provider.repository_lock():
        pass


def test_provider_repository_lock_requires_identity_or_held_replacement_lock(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    unavailable = StablePathSnapshot(
        device=0,
        inode=0,
        mode=stat.S_IFREG,
        links=1,
        size=0,
        modified_ns=0,
        changed_ns=0,
    )

    with pytest.raises(OSError, match="stable regular file"):
        provider._validate_repository_lock_snapshot(unavailable)

    held = replace(unavailable, replacement_locked=True)
    assert provider._validate_repository_lock_snapshot(held) == held
    provider._require_same_repository_lock_identity(held, held)


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX dir_fd semantics")
def test_provider_repository_lock_ancestor_swap_cannot_create_external_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    lock_directory = root / ".git" / "agent-libos"
    lock_directory.mkdir()
    moved_directory = root / ".git" / "agent-libos-moved"
    outside = tmp_path / "outside-lock-directory"
    outside.mkdir()
    original_open = os.open
    swapped = False

    def swap_ancestor_before_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "repository.lock" and dir_fd is not None and not swapped:
            lock_directory.rename(moved_directory)
            lock_directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap_ancestor_before_open)

    with pytest.raises(GitError) as exc_info:
        with provider.repository_lock():
            raise AssertionError("replaced lock ancestor must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert swapped
    assert (moved_directory / "repository.lock").is_file()
    assert not (outside / "repository.lock").exists()


def test_provider_repository_lock_rejects_symlink_file_without_touching_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    lock_directory = root / ".git" / "agent-libos"
    lock_directory.mkdir()
    lock_path = lock_directory / "repository.lock"
    outside = tmp_path / "outside-lock-target"
    outside.write_bytes(b"outside")
    try:
        lock_path.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(GitError) as exc_info:
        with provider.repository_lock():
            raise AssertionError("symlinked lock file must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert outside.read_bytes() == b"outside"


def test_provider_repository_lock_rejects_hard_linked_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    lock_directory = root / ".git" / "agent-libos"
    lock_directory.mkdir()
    lock_path = lock_directory / "repository.lock"
    external = tmp_path / "external-lock-target"
    external.write_bytes(b"")
    try:
        os.link(external, lock_path)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable: {exc}")

    with pytest.raises(GitError) as exc_info:
        with provider.repository_lock():
            raise AssertionError("hard-linked lock file must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert external.read_bytes() == b""


def test_prepare_managed_worktree_rejects_mocked_reparse_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    target = Path(provider.managed_worktree_root) / "wt_reparse"
    target.mkdir(parents=True)
    original_lstat = Path.lstat

    def mocked_lstat(path: Path) -> Any:
        observed = original_lstat(path)
        if path == target:
            return type(
                "ReparseTargetMetadata",
                (),
                {
                    "st_mode": observed.st_mode,
                    "st_file_attributes": 0x400,
                },
            )()
        return observed

    monkeypatch.setattr(Path, "lstat", mocked_lstat)

    with pytest.raises(GitError) as exc_info:
        provider.prepare_managed_worktree("wt_reparse")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


@pytest.mark.skipif(os.name != "nt", reason="requires Windows junction semantics")
def test_provider_repository_lock_rejects_windows_junction_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    outside = tmp_path / "outside-lock"
    _init_repository(root)
    outside.mkdir()
    lock_directory = root / ".git" / "agent-libos"
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(lock_directory), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(
            "junction creation is unavailable: "
            + (result.stderr or result.stdout).strip()
        )
    metadata = lock_directory.lstat()
    reparse_attribute = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    assert int(getattr(metadata, "st_file_attributes", 0)) & reparse_attribute

    with pytest.raises(GitError) as exc_info:
        with LocalGitProvider(root).repository_lock():
            raise AssertionError("junction lock directory must not be entered")

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    assert not (outside / "repository.lock").exists()


def test_builder_does_not_eagerly_construct_a_fallback_git_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root, git_config=DEFAULT_CONFIG.git)
    builder_module = importlib.import_module("agent_libos.runtime.builder")

    def forbidden_fallback(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("fallback Git provider was constructed eagerly")

    monkeypatch.setattr(builder_module, "LocalGitProvider", forbidden_fallback)
    runtime = Runtime.open(
        ":memory:",
        config=_runtime_config(),
        substrate=substrate,
        module_manifests=(),
    )
    runtime.close()


def test_builder_binds_runtime_git_config_to_supplied_local_substrate(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    selected_git = replace(
        DEFAULT_CONFIG.git,
        enabled=False,
        executable="runtime-selected-git",
        inherit_credential_helpers=False,
        inherit_ssh_agent=False,
    )
    config = replace(
        _runtime_config(git=selected_git),
        shell=replace(DEFAULT_CONFIG.shell, default_policy_level="always_allow"),
    )
    substrate = LocalResourceProviderSubstrate(root)
    runtime = Runtime.open(
        ":memory:",
        config=config,
        substrate=substrate,
        module_manifests=(),
    )
    try:
        assert runtime.git.provider is substrate.git
        assert runtime.git.provider.config is selected_git
        assert runtime.shell.provider is substrate.shell
        assert substrate.shell._git_config is selected_git

        pid = runtime.process.spawn(image="base-agent:v0", goal="verify disabled Git")
        _grant_git_authority(runtime, pid)
        runtime.capability.issue_trusted(
            pid,
            "shell:git",
            [CapabilityRight.EXECUTE],
            issued_by="git-provider-test",
        )
        with pytest.raises(GitError) as typed_error:
            runtime.git.repository_info(pid)
        assert typed_error.value.code == GitErrorCode.GIT_UNAVAILABLE.value
        with pytest.raises(GitError) as shell_error:
            runtime.shell.run(pid, ["git", "status"])
        assert shell_error.value.code == GitErrorCode.GIT_UNAVAILABLE.value
    finally:
        runtime.close()


def test_builder_rejects_conflicting_explicit_local_substrate_git_config(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(
        root,
        git_config=DEFAULT_CONFIG.git,
    )
    config = _runtime_config(
        git=replace(DEFAULT_CONFIG.git, inherit_credential_helpers=False),
    )

    with pytest.raises(
        ValidationError,
        match="substrate Git configuration does not match",
    ):
        Runtime.open(
            ":memory:",
            config=config,
            substrate=substrate,
            module_manifests=(),
        )


@pytest.mark.parametrize("provider_kind", ("git", "shell"))
def test_builder_rejects_local_provider_subclass_with_mismatched_git_config(
    tmp_path: Path,
    provider_kind: str,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root)

    if provider_kind == "git":
        class DerivedLocalGitProvider(LocalGitProvider):
            pass

        original_provider: Any = DerivedLocalGitProvider(root)
        substrate.git = original_provider
    else:
        class DerivedLocalShellProvider(LocalShellProvider):
            pass

        original_provider = DerivedLocalShellProvider(root)
        substrate.shell = original_provider

    config = _runtime_config(git=replace(DEFAULT_CONFIG.git, enabled=False))
    with pytest.raises(
        ValidationError,
        match="provider subclass.*does not match",
    ):
        Runtime.open(
            ":memory:",
            config=config,
            substrate=substrate,
            module_manifests=(),
        )

    assert getattr(substrate, provider_kind) is original_provider
    if provider_kind == "git":
        assert original_provider.config == DEFAULT_CONFIG.git
    else:
        assert original_provider._git_config is None


def test_local_substrate_runtime_git_binding_is_idempotent_and_atomic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root)
    selected_git = replace(DEFAULT_CONFIG.git, inherit_credential_helpers=False)
    conflicting_git = replace(selected_git, enabled=False)

    substrate.bind_runtime_git_config(selected_git)
    bound_git_provider = substrate.git
    bound_shell_provider = substrate.shell
    substrate.bind_runtime_git_config(selected_git)

    assert substrate.git is bound_git_provider
    assert substrate.shell is bound_shell_provider
    assert bound_git_provider.config is selected_git
    assert bound_shell_provider._git_config is selected_git

    with pytest.raises(ValidationError, match="already bound"):
        substrate.bind_runtime_git_config(conflicting_git)

    assert substrate.git is bound_git_provider
    assert substrate.shell is bound_shell_provider
    assert bound_git_provider.config is selected_git
    assert bound_shell_provider._git_config is selected_git


def test_builder_registers_fallback_git_provider_for_effect_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    substrate = LocalResourceProviderSubstrate(root, git_config=DEFAULT_CONFIG.git)
    del substrate.git
    builder_module = importlib.import_module("agent_libos.runtime.builder")
    original = builder_module.reconcile_pending_external_effects
    captured: dict[str, Any] = {}

    def capture_recovery(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs.get("provider_overrides") or {})
        return original(*args, **kwargs)

    monkeypatch.setattr(
        builder_module,
        "reconcile_pending_external_effects",
        capture_recovery,
    )
    runtime = Runtime.open(
        ":memory:",
        config=_runtime_config(),
        substrate=substrate,
        module_manifests=(),
    )
    try:
        assert isinstance(runtime.git.provider, LocalGitProvider)
        assert captured["git"] is runtime.git.provider
    finally:
        runtime.close()


def test_unsupported_git_version_is_lazy_and_stable(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    git_config = replace(DEFAULT_CONFIG.git, minimum_version="999.0.0")
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect old Git")
        _grant_git_authority(runtime, pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.repository_info(pid)
        assert exc_info.value.code == GitErrorCode.UNSUPPORTED_GIT_VERSION.value
    finally:
        runtime.close()


def test_task_authority_git_effect_family_denies_mutation_before_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="read Git without mutation authority",
            authority_manifest={"permitted_effects": ["git.read"]},
        )
        _grant_git_authority(runtime, pid)
        status = runtime.git.status(pid)
        (root / "authority.txt").write_text("must remain unstaged\n", encoding="utf-8")

        with pytest.raises(CapabilityDenied, match="does not permit effect class"):
            runtime.git.stage(pid, ["authority.txt"], status.state.token)

        assert _git(root, "diff", "--cached", "--name-only", "--").strip() == b""
        effects = runtime.store.list_external_effects(pid=pid)
        assert [effect.operation for effect in effects] == ["read"]
    finally:
        runtime.close()


def test_provider_never_discovers_a_parent_repository(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    _init_repository(parent)
    child = parent / "nested-workspace"
    child.mkdir()

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(child).repository_layout()
    assert exc_info.value.code == GitErrorCode.NOT_REPOSITORY.value


def test_provider_rejects_symlink_git_entry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _init_repository(source)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        (workspace / ".git").symlink_to(source / ".git", target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(workspace).repository_layout()
    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_detects_windows_reparse_attribute_without_following_it() -> None:
    git_provider = importlib.import_module("agent_libos.substrate.git")
    metadata = type(
        "ReparseMetadata",
        (),
        {
            "st_mode": stat.S_IFDIR,
            "st_file_attributes": 0x400,
        },
    )()

    assert git_provider._is_link_or_reparse(metadata)


def test_repository_layout_rejects_unavailable_directory_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    git_directory = root / ".git"
    original_stat = Path.stat

    def zero_git_directory_identity(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        observed = original_stat(path, *args, **kwargs)
        if path == git_directory:
            values = list(observed)
            values[1] = 0
            return os.stat_result(values)
        return observed

    monkeypatch.setattr(Path, "stat", zero_git_directory_identity)

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).repository_layout()

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_rejects_managed_worktree_selector_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    real_worktree = managed_root / "wt_real"
    alias_worktree = managed_root / "wt_alias"
    _git(root, "worktree", "add", "--detach", str(real_worktree), "HEAD")
    try:
        alias_worktree.symlink_to(real_worktree, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(GitError) as exc_info:
        provider.repository_layout(worktree=alias_worktree)

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_rejects_gitfile_metadata_ancestor_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_linked"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    gitfile = worktree / ".git"
    git_dir = Path(gitfile.read_text(encoding="utf-8").removeprefix("gitdir: ").strip())
    alias = root / ".git" / "worktree-metadata-alias"
    try:
        alias.symlink_to(git_dir.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    gitfile.write_text(f"gitdir: {alias / git_dir.name}\n", encoding="utf-8")

    with pytest.raises(GitError) as exc_info:
        provider.repository_layout(worktree=worktree)

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_accepts_normal_linked_worktree_backpointer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_normal"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")

    layout = provider.repository_layout(worktree=worktree)
    backpointer = Path(
        (layout.git_dir / "gitdir").read_text(encoding="utf-8").strip()
    )

    assert layout.linked_worktree
    assert layout.git_dir != layout.common_dir
    assert Path(os.path.abspath(backpointer)) == worktree / ".git"


def test_provider_accepts_relative_linked_worktree_metadata_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_relative"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    gitfile = worktree / ".git"
    git_dir = Path(
        gitfile.read_text(encoding="utf-8")
        .removeprefix("gitdir: ")
        .strip()
    )
    gitfile.write_text(
        f"gitdir: {os.path.relpath(git_dir, worktree)}\n",
        encoding="utf-8",
    )
    (git_dir / "gitdir").write_text(
        f"{os.path.relpath(gitfile, git_dir)}\n",
        encoding="utf-8",
    )

    layout = provider.repository_layout(worktree=worktree)

    assert layout.linked_worktree
    assert layout.git_dir == git_dir
    assert layout.common_dir == root / ".git"


def test_provider_rooted_at_linked_worktree_accepts_explicit_trusted_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    linked = tmp_path / "linked-root"
    _git(root, "worktree", "add", "--detach", str(linked), "HEAD")
    config = replace(
        DEFAULT_CONFIG.git,
        trusted_metadata_roots=(str(root / ".git"),),
    )

    layout = LocalGitProvider(linked, config=config).repository_layout()

    assert layout.linked_worktree
    assert layout.root == linked
    assert layout.git_dir != layout.common_dir
    assert layout.common_dir == root / ".git"


def test_list_worktrees_rejects_linked_gitfile_reusing_primary_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_primary_alias"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    (worktree / ".git").write_text(
        f"gitdir: {root / '.git'}\n",
        encoding="utf-8",
    )
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject primary metadata worktree alias",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )

        with pytest.raises(GitError) as exc_info:
            runtime.git.list_worktrees(pid)

        assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    finally:
        runtime.close()


def test_list_worktrees_rejects_linked_gitfile_aliasing_another_worktree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    first = managed_root / "wt_first"
    alias = managed_root / "wt_other_alias"
    _git(root, "worktree", "add", "--detach", str(first), "HEAD")
    _git(root, "worktree", "add", "--detach", str(alias), "HEAD")
    first_git_dir = Path(
        (first / ".git")
        .read_text(encoding="utf-8")
        .removeprefix("gitdir: ")
        .strip()
    )
    (alias / ".git").write_text(
        f"gitdir: {first_git_dir}\n",
        encoding="utf-8",
    )
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject cross-worktree metadata alias",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )

        with pytest.raises(GitError) as exc_info:
            runtime.git.list_worktrees(pid)

        assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    finally:
        runtime.close()


def test_provider_rejects_symlinked_linked_worktree_backpointer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_backpointer_link"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    git_dir = Path(
        (worktree / ".git")
        .read_text(encoding="utf-8")
        .removeprefix("gitdir: ")
        .strip()
    )
    marker = git_dir / "gitdir"
    external = tmp_path / "external-gitdir-marker"
    external.write_bytes(marker.read_bytes())
    marker.unlink()
    try:
        marker.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(GitError) as exc_info:
        provider.repository_layout(worktree=worktree)

    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_rejects_object_alternates(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    alternates = root / ".git" / "objects" / "info" / "alternates"
    alternates.write_text(str(tmp_path / "objects"), encoding="utf-8")

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).repository_layout()
    assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value


def test_provider_rejects_symlinked_repository_config(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    config_path = root / ".git" / "config"
    external_config = tmp_path / "external-config"
    config_path.replace(external_config)
    try:
        config_path.symlink_to(external_config)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).repository_state()
    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value


def test_provider_rejects_repository_config_includes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    included = tmp_path / "included-git-config"
    included.write_text("[user]\n\tname = Included Identity\n", encoding="utf-8")
    _git(root, "config", "include.path", str(included))

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).repository_state()
    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value


def test_provider_disables_host_git_trace_sinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    trace_output = tmp_path / "git-trace-events.json"
    trace_target = trace_output.as_posix()
    (fake_home / ".gitconfig").write_text(
        "[trace2]\n"
        f"\teventTarget = {trace_target}\n"
        f"\tnormalTarget = {trace_target}\n"
        f"\tperfTarget = {trace_target}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(fake_home))

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect Git safely")
        _grant_git_authority(runtime, pid)
        runtime.git.status(pid)
        assert not trace_output.exists()
    finally:
        runtime.close()


def test_active_external_filter_is_rejected_before_status_can_execute_it(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "filter-ran"
    (root / ".gitattributes").write_text("*.txt filter=hostile\n", encoding="utf-8")
    _git(root, "config", "filter.hostile.clean", f"touch {sentinel}")
    _git(root, "config", "filter.hostile.smudge", "cat")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject hostile filter")
        _grant_git_authority(runtime, pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.status(pid)
        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
        assert not sentinel.exists()
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_ignored_attribute_tree_does_not_activate_external_filter(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    ignored = root / "ignored-environment"
    ignored.mkdir()
    (root / ".gitignore").write_text("ignored-environment/\n", encoding="utf-8")
    (ignored / ".gitattributes").write_text(
        "*.txt filter=hostile\n",
        encoding="utf-8",
    )
    (ignored / "payload.txt").write_text("ignored\n", encoding="utf-8")
    sentinel = tmp_path / "filter-ran"
    _git(root, "config", "filter.hostile.clean", f"touch {sentinel}")
    _git(root, "config", "filter.hostile.smudge", "cat")

    result = LocalGitProvider(root).run(["status", "--porcelain=v2", "-z"])

    assert result.returncode == 0
    assert not sentinel.exists()


def test_index_only_filter_is_rejected_before_status_can_execute_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "index-filter-ran"
    attributes = root / ".gitattributes"
    attributes.write_text("*.txt filter=hostile\n", encoding="utf-8")
    _git(root, "add", "--", ".gitattributes")
    attributes.unlink()
    _git(root, "config", "filter.hostile.clean", f"touch {sentinel}")
    _git(root, "config", "filter.hostile.smudge", "cat")
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(["status", "--porcelain=v2", "-z"])

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()


def test_active_textconv_driver_is_rejected_before_blame_can_execute_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "textconv-ran"
    (root / ".gitattributes").write_text(
        "tracked.txt diff=hostile\n",
        encoding="utf-8",
    )
    _git(root, "config", "diff.hostile.textconv", f"touch {sentinel}")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(["blame", "HEAD", "--", "tracked.txt"])

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()


@pytest.mark.parametrize("driver_name", ["hostile/name", "hostile.name"])
def test_active_external_filter_with_structured_name_is_rejected_before_execution(
    tmp_path: Path,
    driver_name: str,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "filter-ran"
    (root / ".gitattributes").write_text(
        f"  *.txt filter={driver_name}  \n",
        encoding="utf-8",
    )
    _git(root, "config", f"filter.{driver_name}.clean", f"touch {sentinel}")
    _git(root, "config", f"filter.{driver_name}.smudge", "cat")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject structured filter name")
        _grant_git_authority(runtime, pid)

        with pytest.raises(GitError) as exc_info:
            runtime.git.status(pid)

        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
        assert not sentinel.exists()
        assert runtime.store.list_external_effects(pid=pid) == []
    finally:
        runtime.close()


def test_attribute_pattern_named_like_driver_is_not_treated_as_active_driver(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "filter-ran"
    (root / ".gitattributes").write_text(
        "filter=hostile/name binary\n",
        encoding="utf-8",
    )
    _git(root, "config", "filter.hostile/name.clean", f"touch {sentinel}")
    _git(root, "config", "filter.hostile/name.smudge", "cat")

    provider = LocalGitProvider(root)
    result = provider.run(["status", "--porcelain=v2", "-z"])

    assert result.returncode == 0
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("configured_path", "location"),
    [("attributes", "workspace"), ("~/attributes", "home")],
)
def test_configured_attributes_file_uses_git_path_resolution(
    tmp_path: Path,
    configured_path: str,
    location: str,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "filter-ran"
    attributes = (
        root / "attributes"
        if location == "workspace"
        else Path(os.environ["HOME"]) / "attributes"
    )
    attributes.write_text("*.txt filter=hostile\n", encoding="utf-8")
    _git(root, "config", "core.attributesFile", configured_path)
    _git(root, "config", "filter.hostile.clean", f"touch {sentinel}")
    _git(root, "config", "filter.hostile.smudge", "cat")
    (root / "tracked.txt").write_text("changed\n", encoding="utf-8")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(["status", "--porcelain=v2", "-z"])

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()


def test_checkout_rejects_filter_activated_only_by_target_tree(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "hostile")
    (root / ".gitattributes").write_text(
        "tracked.txt filter=hostile\n",
        encoding="utf-8",
    )
    (root / "tracked.txt").write_text("target\n", encoding="utf-8")
    _git(root, "add", "--", ".gitattributes", "tracked.txt")
    _git(root, "commit", "-q", "-m", "target attributes")
    _git(root, "switch", "-q", "main")
    sentinel = tmp_path / "filter-ran"
    _git(root, "config", "filter.hostile.clean", "cat")
    _git(root, "config", "filter.hostile.smudge", f"touch {sentinel}")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject target filter")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token

        with pytest.raises(GitError) as exc_info:
            runtime.git.switch(pid, "hostile", state)

        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
        assert not sentinel.exists()
        assert _git(root, "branch", "--show-current").strip() == b"main"
    finally:
        runtime.close()


def test_binary_target_attributes_are_discovered_before_checkout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "binary-attributes")
    (root / ".gitattributes").write_bytes(
        b"\0ignored binary line\ntracked.txt filter=hostile\n"
    )
    (root / "tracked.txt").write_text("target\n", encoding="utf-8")
    _git(
        root,
        "-c",
        "filter.hostile.clean=cat",
        "add",
        "--",
        ".gitattributes",
        "tracked.txt",
    )
    _git(root, "commit", "-q", "-m", "binary target attributes")
    _git(root, "switch", "-q", "main")
    sentinel = tmp_path / "binary-attribute-filter-ran"
    _git(root, "config", "filter.hostile.clean", "cat")
    _git(root, "config", "filter.hostile.smudge", f"touch {sentinel}")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(
            ["switch", "binary-attributes"],
            read_only=False,
        )

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()
    assert _git(root, "branch", "--show-current").strip() == b"main"


def test_target_attribute_discovery_does_not_enumerate_the_entire_tree(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "large")
    for index in range(800):
        (root / f"ordinary-file-{index:03d}-with-a-long-name.txt").write_text(
            f"{index}\n",
            encoding="utf-8",
        )
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "large tree without attributes")
    _git(root, "switch", "-q", "main")
    _git(root, "config", "filter.inactive.smudge", "cat")
    limited = replace(
        DEFAULT_CONFIG.git,
        output_max_bytes=65_536,
        output_hard_limit_bytes=65_536,
    )

    result = LocalGitProvider(root, config=limited).run(
        ["switch", "large"],
        read_only=False,
    )

    assert result.returncode == 0
    assert _git(root, "branch", "--show-current").strip() == b"large"


def test_rebase_rejects_filter_activated_by_an_intermediate_replayed_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "topic")
    nested = root / "nested"
    nested.mkdir()
    (nested / ".gitattributes").write_text(
        "tracked.txt filter=hostile\n",
        encoding="utf-8",
    )
    (nested / "tracked.txt").write_text("topic one\n", encoding="utf-8")
    _git(
        root,
        "-c",
        "filter.hostile.clean=cat",
        "add",
        "--",
        "nested/.gitattributes",
        "nested/tracked.txt",
    )
    _git(root, "commit", "-q", "-m", "activate filter")
    (nested / ".gitattributes").unlink()
    (nested / "tracked.txt").write_text("topic two\n", encoding="utf-8")
    _git(root, "-c", "filter.hostile.clean=cat", "add", "-A")
    _git(root, "commit", "-q", "-m", "remove filter")
    _git(root, "switch", "-q", "main")
    (root / "base.txt").write_text("base\n", encoding="utf-8")
    _git(root, "add", "--", "base.txt")
    _git(root, "commit", "-q", "-m", "advance base")
    _git(root, "switch", "-q", "topic")
    sentinel = tmp_path / "rebase-filter-ran"
    _git(root, "config", "filter.hostile.clean", "cat")
    _git(root, "config", "filter.hostile.smudge", f"touch {sentinel}")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(
            ["rebase", "--no-autostash", "main"],
            read_only=False,
        )

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()
    assert _git(root, "branch", "--show-current").strip() == b"topic"


def test_provider_rejects_branch_merge_options_before_typed_merge(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "side")
    (root / "side.txt").write_text("side\n", encoding="utf-8")
    _git(root, "add", "--", "side.txt")
    _git(root, "commit", "-q", "-m", "side")
    side_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "main")
    (root / "main.txt").write_text("main\n", encoding="utf-8")
    _git(root, "add", "--", "main.txt")
    _git(root, "commit", "-q", "-m", "main")
    before_oid = _git(root, "rev-parse", "HEAD").strip()
    _git(root, "config", "branch.main.mergeOptions", "--no-commit")

    provider = LocalGitProvider(root)
    with pytest.raises(GitError) as exc_info:
        provider.run(
            ["merge", "--no-edit", "--no-gpg-sign", side_oid],
            read_only=False,
        )

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert _git(root, "rev-parse", "HEAD").strip() == before_oid
    assert _git(root, "status", "--porcelain") == b""
    merge_head = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert merge_head.returncode == 1


def test_repository_hook_is_disabled_for_typed_commit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "hook-ran"
    hook = root / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch '{sentinel}'\nexit 1\n", encoding="utf-8")
    hook.chmod(0o700)
    (root / "hook-safe.txt").write_text("safe\n", encoding="utf-8")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="commit without repository hooks")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        staged = runtime.git.stage(pid, ["hook-safe.txt"], state)
        committed = runtime.git.commit(pid, "hook-safe commit", staged.after.token)
        assert committed.created_oid is not None
        assert not sentinel.exists()
    finally:
        runtime.close()


def test_partial_clone_config_is_rejected_to_prevent_lazy_fetch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "config", "remote.origin.promisor", "true")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject partial clone")
        _grant_git_authority(runtime, pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.repository_info(pid)
        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    finally:
        runtime.close()


def test_git_path_token_preserves_non_utf8_bytes() -> None:
    raw_name = b"invalid-\xff.txt"
    encoded = GitPrimitive._git_path(raw_name)
    assert encoded.lossy
    assert encoded.display == "invalid-\ufffd.txt"
    assert GitPrimitive._decode_path(encoded) == raw_name


def test_status_preserves_newline_paths_and_async_facade(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows filenames cannot contain newline characters")
    root = tmp_path / "repo"
    _init_repository(root)
    raw_name = "line\nbreak.txt"
    (root / raw_name).write_text("bytes\n", encoding="utf-8")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect byte paths")
        _grant_git_authority(runtime, pid)
        status = asyncio.run(runtime.git.astatus(pid))
        entry = next(item for item in status.entries if item.path.display == raw_name)
        assert not entry.path.lossy
        assert runtime.git._decode_path(entry.path) == os.fsencode(raw_name)
        assert status.bytes > 0
        assert len(status.sha256) == 64
        assert not status.truncated
    finally:
        runtime.close()


def test_unborn_and_detached_head_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root, "config", "user.name", "Agent libOS Test")
    _git(root, "config", "user.email", "agent-libos@example.test")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="handle Git HEAD states")
        _grant_git_authority(runtime, pid)
        unborn = runtime.git.status(pid)
        assert unborn.branch == "main"
        assert unborn.head_oid is None

        (root / "first.txt").write_text("first\n", encoding="utf-8")
        dirty_unborn = runtime.git.status(pid)
        assert dirty_unborn.head_oid is None
        staged = runtime.git.stage(pid, ["first.txt"], dirty_unborn.state.token)
        committed = runtime.git.commit(pid, "first", staged.after.token)
        assert committed.created_oid is not None
        detached = runtime.git.switch(
            pid,
            committed.created_oid,
            committed.after.token,
            detach=True,
        )
        status = runtime.git.status(pid)
        assert status.branch is None
        assert status.head_oid == committed.created_oid == detached.created_oid
    finally:
        runtime.close()


def test_status_and_diff_cover_rename_binary_and_symlink_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "binary.bin").write_bytes(b"\x00initial\xff")
    symlink_supported = True
    try:
        (root / "tracked-link").symlink_to("tracked.txt")
    except OSError:
        symlink_supported = False
    _git(root, "add", "--all", "--", ".")
    _git(root, "commit", "-q", "-m", "binary and symlink fixture")

    _git(root, "mv", "--", "tracked.txt", "renamed.txt")
    (root / "binary.bin").write_bytes(b"\x00changed\xfe")
    if symlink_supported:
        (root / "tracked-link").unlink()
        (root / "tracked-link").symlink_to("renamed.txt")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect Git file kinds")
        _grant_git_authority(runtime, pid)
        status = runtime.git.status(pid)
        by_path = {entry.path.display: entry for entry in status.entries}
        assert by_path["renamed.txt"].kind.value == "renamed"
        assert "binary.bin" in by_path
        if symlink_supported:
            assert "tracked-link" in by_path

        diff = runtime.git.diff(pid)
        patch = base64.b64decode(diff.patch_b64, validate=True)
        assert b"GIT binary patch" in patch
        changed = {path.display for path in diff.changed_paths}
        assert "binary.bin" in changed
        staged_diff = runtime.git.diff(pid, scope="staged")
        staged_paths = {path.display for path in staged_diff.changed_paths}
        assert {"renamed.txt", "tracked.txt"} <= staged_paths
    finally:
        runtime.close()


def test_unmerged_status_and_typed_abort(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "side")
    (root / "tracked.txt").write_text("side\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "side")
    _git(root, "switch", "-q", "main")
    (root / "tracked.txt").write_text("main\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "main")
    merge = subprocess.run(
        ["git", "merge", "--no-edit", "side"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert merge.returncode != 0

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="abort a Git conflict")
        _grant_git_authority(runtime, pid)
        conflicted = runtime.git.status(pid)
        assert any(entry.kind.value == "unmerged" for entry in conflicted.entries)
        aborted = _with_auto_approvals(
            runtime,
            lambda: runtime.git.integrate(
                pid,
                "abort",
                conflicted.state.token,
                abort_kind="merge",
            ),
        )
        assert aborted.details["integration"] == "abort"
        assert not any(
            entry.kind.value == "unmerged"
            for entry in runtime.git.status(pid).entries
        )
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "main\n"
    finally:
        runtime.close()


def test_sha256_repository_object_ids_are_supported_when_host_git_supports_them(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sha256-repo"
    root.mkdir()
    initialized = subprocess.run(
        ["git", "init", "-q", "--object-format=sha256"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if initialized.returncode != 0:
        pytest.skip("Host Git does not support SHA-256 repositories")
    _git(root, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(root, "config", "user.name", "Agent libOS Test")
    _git(root, "config", "user.email", "agent-libos@example.test")
    (root / "tracked.txt").write_text("sha256\n", encoding="utf-8")
    _git(root, "add", "--", "tracked.txt")
    _git(root, "commit", "-q", "-m", "sha256 initial")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect SHA-256 Git")
        _grant_git_authority(runtime, pid)
        info = runtime.git.repository_info(pid)
        assert info.object_format == "sha256"
        assert info.state.head_oid is not None and len(info.state.head_oid) == 64
        shown = runtime.git.show(pid, info.state.head_oid)
        assert shown["commit"].oid == info.state.head_oid
    finally:
        runtime.close()


def test_show_returns_complete_per_parent_merge_diffs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "side")
    (root / "side.txt").write_text("side\n", encoding="utf-8")
    _git(root, "add", "--", "side.txt")
    _git(root, "commit", "-q", "-m", "side")
    _git(root, "switch", "-q", "main")
    (root / "main.txt").write_text("main\n", encoding="utf-8")
    _git(root, "add", "--", "main.txt")
    _git(root, "commit", "-q", "-m", "main")
    _git(root, "merge", "-q", "--no-ff", "side", "-m", "merge side")
    merge_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect a merge")
        _grant_git_authority(runtime, pid)

        shown = runtime.git.show(pid, merge_oid)

        assert len(shown["commit"].parents) == 2
        assert [item["parent_oid"] for item in shown["parent_diffs"]] == shown[
            "commit"
        ].parents
        assert shown["patch_base_oid"] == shown["commit"].parents[0]
        assert shown["patch"] == shown["parent_diffs"][0]["patch"]
        assert shown["patch_b64"] == shown["parent_diffs"][0]["patch_b64"]
        assert not shown["parent_diffs_truncated"]
        assert all(not item["truncated"] for item in shown["parent_diffs"])
        assert all(item["bytes"] > 0 for item in shown["parent_diffs"])
        assert {
            path.display for path in shown["parent_diffs"][0]["changed_paths"]
        } == {"side.txt"}
        assert {
            path.display for path in shown["parent_diffs"][1]["changed_paths"]
        } == {"main.txt"}
    finally:
        runtime.close()


def test_diff_truncation_and_hard_output_limit_are_explicit(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "tracked.txt").write_text("initial\n" + ("large line\n" * 2_000), encoding="utf-8")
    git_config = replace(
        DEFAULT_CONFIG.git,
        output_max_bytes=1_024,
        output_hard_limit_bytes=4_096,
        patch_max_bytes=1_024,
        patch_hard_limit_bytes=4_096,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bound Git output")
        _grant_git_authority(runtime, pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.diff(pid, max_bytes=1_024)
        assert exc_info.value.code == GitErrorCode.OUTPUT_TOO_LARGE.value
    finally:
        runtime.close()

    moderate = tmp_path / "moderate"
    _init_repository(moderate)
    (moderate / "tracked.txt").write_text("initial\n" + ("changed\n" * 80), encoding="utf-8")
    second = _open_runtime(moderate)
    try:
        pid = second.process.spawn(image="base-agent:v0", goal="truncate Git output")
        _grant_git_authority(second, pid)
        diff = second.git.diff(pid, max_bytes=128)
        assert diff.truncated
        assert diff.bytes > 128
        assert len(diff.patch_b64) > 0
        assert len(diff.sha256) == 64
    finally:
        second.close()


def test_high_output_git_failure_charges_bounded_subprocess_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "tracked.txt").write_text(
        "initial\n" + ("large changed line\n" * 12_000),
        encoding="utf-8",
    )
    git_config = replace(
        DEFAULT_CONFIG.git,
        output_max_bytes=65_536,
        output_hard_limit_bytes=131_072,
        patch_max_bytes=65_536,
        patch_hard_limit_bytes=131_072,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="account for bounded high-output Git",
        )
        _grant_git_authority(runtime, pid)

        with pytest.raises(GitError) as exc_info:
            runtime.git.diff(pid, max_bytes=65_536)

        assert exc_info.value.code == GitErrorCode.OUTPUT_TOO_LARGE.value
        assert exc_info.value.details["effect"] == "none"
        assert exc_info.value.details["limit_kind"] == "subprocess_stdout_bytes"
        assert exc_info.value.details["metrics"]["wall_seconds"] > 0
        usage = runtime.process.get(pid).resource_usage
        assert usage.subprocess_wall_seconds > 0
        assert usage.subprocess_peak_memory_bytes > 0
    finally:
        runtime.close()


def test_git_subprocess_budget_terminates_process_before_unbounded_dispatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="enforce Git subprocess budget",
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=1e-6),
        )
        _grant_git_authority(runtime, pid)

        if not runtime.git.provider.supports_subprocess_limits:
            with pytest.raises(ValidationError, match="SubprocessLimits"):
                runtime.git.status(pid)
            process = runtime.process.get(pid)
            assert process.status is ProcessStatus.RUNNABLE
            assert process.resource_usage.subprocess_wall_seconds == 0
            return

        with pytest.raises(ResourceLimitExceeded):
            runtime.git.status(pid)

        process = runtime.process.get(pid)
        assert process.status is ProcessStatus.KILLED
        assert process.resource_usage.subprocess_wall_seconds > 0
        assert any(
            record.action == "resource.limit_exceeded"
            for record in runtime.audit.trace()
        )
    finally:
        runtime.close()


def test_git_approval_retry_rechecks_exhausted_budget_before_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="recheck Git budget after approval wait",
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=10.0),
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.WRITE],
            effect="ask",
            issued_by="git-provider-test",
        )
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="git-provider-test",
        )
        if not runtime.git.provider.supports_subprocess_limits:
            def forbidden_invoke(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("unsupported budget reached provider dispatch")

            monkeypatch.setattr(runtime.git.provider, "_invoke", forbidden_invoke)
            with pytest.raises(ValidationError, match="SubprocessLimits"):
                runtime.git.status(pid)
            return
        state = runtime.git.status(pid).state.token

        with pytest.raises(HumanApprovalRequired):
            runtime.git.worktree(pid, "create", state)
        assert runtime.human.drain_terminal_queue(auto_approve=True)

        remaining = runtime.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        )
        assert remaining is not None and remaining > 0
        runtime.resources.charge(
            pid,
            ResourceUsage(subprocess_wall_seconds=remaining),
            source="test.git.approval_budget_exhaustion",
            kill_on_exceed=False,
        )
        assert runtime.process.get(pid).status is ProcessStatus.RUNNABLE
        assert runtime.resources.remaining_cumulative(
            pid,
            "max_subprocess_wall_seconds",
            "subprocess_wall_seconds",
        ) <= 0

        def forbidden_invoke(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("Git provider dispatch started with an exhausted budget")

        monkeypatch.setattr(runtime.git.provider, "_invoke", forbidden_invoke)
        with pytest.raises(ResourceLimitExceeded):
            runtime.git.worktree(pid, "create", state)
    finally:
        runtime.close()


def test_local_git_provider_enforces_process_tree_memory_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    assert isinstance(provider, GitSubprocessScopeProvider)
    assert isinstance(provider, GitLimitedRunProvider)
    if not provider.supports_subprocess_limits:
        pytest.skip("Host platform cannot enforce Git SubprocessLimits")

    bounded = provider.run_with_limits(
        ["status", "--porcelain=v2", "-z"],
        limits=SubprocessLimits(wall_seconds=5.0),
    )
    assert bounded.returncode == 0

    with provider.subprocess_scope(
        limits=SubprocessLimits(memory_bytes=1),
    ) as scope:
        with pytest.raises(SubprocessLimitExceeded) as exc_info:
            provider.repository_state()

    assert exc_info.value.metrics.limit_kind == "subprocess_memory_bytes"
    assert exc_info.value.metrics.killed
    assert scope.metrics.peak_memory_bytes > 1


def test_local_git_provider_rejects_non_finite_timeout_policy(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(
        root,
        config=replace(DEFAULT_CONFIG.git, local_timeout_s=float("nan")),
    )

    with pytest.raises(GitError) as exc_info:
        provider.repository_state()

    assert exc_info.value.code == GitErrorCode.TIMEOUT.value
    assert "invalid Git timeout" in str(exc_info.value)


def test_local_git_provider_supervises_unread_stdin_under_deadline(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(
        root,
        config=replace(DEFAULT_CONFIG.git, executable=sys.executable),
    )

    started = time.monotonic()
    with pytest.raises(GitError) as exc_info:
        provider._invoke(
            ["-c", "import time; time.sleep(10)"],
            timeout=0.05,
            stdin=b"x" * 1_048_576,
            max_output_bytes=4096,
            read_only=True,
            operation="test-unread-stdin",
        )

    assert time.monotonic() - started < 2.0
    assert exc_info.value.code == GitErrorCode.TIMEOUT.value
    assert exc_info.value.details["metrics"]["killed"] is True


def test_local_git_provider_reaps_child_when_stdin_delivery_raises_baseexception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(
        root,
        config=replace(DEFAULT_CONFIG.git, executable=sys.executable),
    )
    git_substrate = importlib.import_module("agent_libos.substrate.git")
    real_popen = git_substrate.subprocess.Popen
    spawned: list[subprocess.Popen[bytes]] = []

    class InterruptingStdin:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def write(self, _payload: bytes) -> int:
            raise KeyboardInterrupt("simulated stdin interruption")

        def close(self) -> None:
            self._stream.close()

    def launch(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process = real_popen(*args, **kwargs)
        assert process.stdin is not None
        process.stdin = InterruptingStdin(process.stdin)
        spawned.append(process)
        return process

    monkeypatch.setattr(git_substrate.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt, match="simulated stdin interruption"):
        provider._invoke(
            ["-c", "import time; time.sleep(10)"],
            timeout=5.0,
            stdin=b"payload",
            max_output_bytes=4096,
            read_only=True,
            operation="test-stdin-interruption",
        )

    assert len(spawned) == 1
    assert spawned[0].poll() is not None


def test_stage_commit_and_state_token_cas(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "new.txt").write_text("new\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="commit a file")
        _grant_git_authority(runtime, pid)
        before = runtime.git.status(pid)
        staged = runtime.git.stage(pid, ["new.txt"], before.state.token)
        assert staged.after.token != before.state.token
        with pytest.raises(GitError) as exc_info:
            runtime.git.stage(pid, ["new.txt"], before.state.token)
        assert exc_info.value.code == GitErrorCode.STALE_STATE.value

        committed = runtime.git.commit(pid, "add new file", staged.after.token)
        assert committed.created_oid == _git(root, "rev-parse", "HEAD").strip().decode("ascii")
        assert runtime.git.status(pid).entries == []
    finally:
        runtime.close()


def test_directory_pathspec_requires_subtree_filesystem_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    source = root / "src"
    source.mkdir()
    (source / "inside.txt").write_text("inside\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="stage one exact path")
        _grant_git_repository_authority(runtime, pid)
        runtime.capability.issue_trusted(
            pid,
            runtime.filesystem.resource_for("src"),
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )
        state = runtime.git.status(pid).state.token
        effects_before_denial = runtime.store.list_external_effects(pid=pid)

        with pytest.raises((CapabilityDenied, HumanApprovalRequired)):
            runtime.git.stage(pid, ["src"], state)
        assert _git(root, "diff", "--cached", "--name-only", "--").strip() == b""
        directory_resource = runtime.filesystem.directory_resource_for("src")
        assert any(
            record.actor == pid
            and record.action == "capability.authorize"
            and record.target == directory_resource
            and record.decision is not None
            and record.decision.get("allowed") is False
            for record in runtime.audit.trace(actor=pid)
        )
        assert runtime.store.list_external_effects(pid=pid) == effects_before_denial

        runtime.capability.issue_trusted(
            pid,
            runtime.filesystem.resource_for("src/inside.txt"),
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )
        staged = runtime.git.stage(pid, ["src/inside.txt"], state)
        assert staged.changed_paths[0].display == "src/inside.txt"
        assert _git(root, "diff", "--cached", "--name-only", "--").strip() == b"src/inside.txt"
    finally:
        runtime.close()


def test_clean_approval_is_invalidated_when_ignored_content_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    _git(root, "add", "--", ".gitignore")
    _git(root, "commit", "-q", "-m", "ignore fixture")
    ignored = root / "ignored.txt"
    ignored.write_text("first\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="clean an approved snapshot")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token

        with pytest.raises(HumanApprovalRequired):
            runtime.git.clean(pid, state, ignored=True)
        ignored.write_text("changed after request\n", encoding="utf-8")
        assert runtime.human.drain_terminal_queue(auto_approve=True)

        with pytest.raises(HumanApprovalRequired):
            runtime.git.clean(pid, state, ignored=True)
        assert ignored.read_text(encoding="utf-8") == "changed after request\n"

        assert runtime.human.drain_terminal_queue(auto_approve=True)
        runtime.git.clean(pid, state, ignored=True)
        assert not ignored.exists()
    finally:
        runtime.close()


def test_git_commit_lineage_blocks_cross_process_secret_push(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    secret_path = root / "secret.txt"
    secret_path.write_text("classified\n", encoding="utf-8")
    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="write secret source")
        stager = runtime.process.spawn(image="base-agent:v0", goal="stage secret bytes")
        committer = runtime.process.spawn(image="base-agent:v0", goal="commit staged bytes")
        pusher = runtime.process.spawn(image="base-agent:v0", goal="push committed bytes")
        for pid in (writer, stager, committer, pusher):
            _grant_git_authority(runtime, pid, remote="origin")
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="git-lineage-test"),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="secret.txt",
            content_sha256=hashlib.sha256(secret_path.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(stager).state.token
        staged = runtime.git.stage(stager, ["secret.txt"], state)
        committed = runtime.git.commit(committer, "secret lineage", staged.after.token)
        with pytest.raises(CapabilityDenied, match="data-flow denied"):
            runtime.git.push(
                pusher,
                "origin",
                "refs/heads/main",
                committed.after.token,
                local_ref="main",
            )
        assert _git(remote, "for-each-ref").strip() == b""
        denied = runtime.store.list_data_flow_decisions(pid=pusher, outcome="deny")
        assert len(denied) == 1
        assert denied[0].sink == runtime.git.remote_resource("origin")
        assert denied[0].labels.sensitivity.value == "secret"
        assert any(
            record.action == "data_flow.egress"
            and record.target == denied[0].sink
            and record.decision is not None
            and record.decision.get("decision_id") == denied[0].decision_id
            and record.decision.get("outcome") == "deny"
            for record in runtime.audit.trace(actor=pusher)
        )
        assert any(
            event.type == EventType.DATA_FLOW_DECISION
            and event.payload.get("decision_id") == denied[0].decision_id
            and event.payload.get("outcome") == "deny"
            for event in runtime.events.list(
                target=f"data_flow_sink:{denied[0].sink}"
            )
        )
    finally:
        runtime.close()


def test_git_reads_and_patch_artifacts_recover_commit_and_index_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    secret_path = root / "secret.txt"
    secret_path.write_text("classified from Git\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label secret bytes")
        stager = runtime.process.spawn(image="base-agent:v0", goal="stage secret bytes")
        committer = runtime.process.spawn(image="base-agent:v0", goal="commit secret bytes")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read secret commit")
        for pid in (writer, stager, committer, reader):
            _grant_git_authority(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-read-lineage-test",
            ),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="secret.txt",
            content_sha256=hashlib.sha256(secret_path.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        with runtime.data_flow.activate(DataFlowContext()):
            runtime.git.show(reader, base_oid)
            normal_context = runtime.data_flow.current_context()
        assert normal_context.labels.sensitivity.value == "normal"
        assert normal_context.labels.trust_level == "untrusted"

        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(stager).state.token
            staged = runtime.git.stage(stager, ["secret.txt"], state)
        binding = runtime.store.get_file_label_binding("secret.txt")
        assert binding is not None
        runtime.data_flow.tombstone_file(
            pid=writer,
            normalized_path="secret.txt",
            expected_binding_id=binding.binding_id,
            expected_generation=binding.generation,
        )
        assert runtime.data_flow.file_context("secret.txt").labels.sensitivity.value == "normal"

        with runtime.data_flow.activate(DataFlowContext()):
            staged_diff = runtime.git.diff(reader, scope="staged")
            staged_context = runtime.data_flow.current_context()
        assert "classified from Git" in staged_diff.patch
        assert staged_context.labels.sensitivity.value == "secret"
        staged_parents, _staged_refs = runtime.data_flow.provenance_sources(
            staged_context
        )
        assert source.oid in staged_parents

        with runtime.data_flow.activate(DataFlowContext()):
            committed = runtime.git.commit(
                committer,
                "commit classified bytes",
                staged.after.token,
            )
        assert committed.created_oid is not None

        with runtime.data_flow.activate(DataFlowContext()):
            shown = runtime.git.show(reader, committed.created_oid)
            shown_context = runtime.data_flow.current_context()
        assert "classified from Git" in shown["patch"]
        assert shown_context.labels.sensitivity.value == "secret"
        shown_parents, _shown_refs = runtime.data_flow.provenance_sources(
            shown_context
        )
        assert source.oid in shown_parents

        with runtime.data_flow.activate(DataFlowContext()):
            artifact = runtime.git.create_patch(
                reader,
                scope="range",
                base=base_oid,
                head=committed.created_oid,
            )
        stored = runtime.store.get_object(artifact.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids
    finally:
        runtime.close()


def test_git_read_rejects_carrier_label_generation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_text("classified race\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label secret bytes")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read racing diff")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, reader)
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-read-race-test",
            ),
        )
        source_context = runtime.data_flow.context_from_source_oids(
            writer,
            [source.oid],
        )
        digest = hashlib.sha256(tracked.read_bytes()).hexdigest()
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=digest,
            context=source_context,
        )
        original = runtime.git._diff_result

        def racing_diff(**kwargs: Any) -> tuple[Any, dict[str, Any]]:
            runtime.data_flow.bind_written_file_digest(
                pid=writer,
                normalized_path="tracked.txt",
                content_sha256=digest,
                context=source_context,
            )
            return original(**kwargs)

        monkeypatch.setattr(runtime.git, "_diff_result", racing_diff)
        with runtime.data_flow.activate(DataFlowContext()):
            with pytest.raises(CapabilityDenied, match="carrier changed"):
                runtime.git.diff(reader, paths=["tracked.txt"])
            raced_context = runtime.data_flow.current_context()
        assert raced_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_repository_content_prebind_survives_post_effect_settlement_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_text("classified staged bytes\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="stage secret bytes")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read unknown stage")
        for pid in (writer, reader):
            _grant_git_authority(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret stage"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-repository-prebind-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(writer, [source.oid])
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256(tracked.read_bytes()).hexdigest(),
            context=secret_context,
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(writer).state.token

        original = runtime.git._settle_git_lineage

        def fail_settlement(**_kwargs: Any) -> None:
            raise RuntimeError("repository lineage settlement failed")

        monkeypatch.setattr(runtime.git, "_settle_git_lineage", fail_settlement)
        with pytest.raises(RuntimeError, match="repository lineage settlement failed"):
            runtime.git.stage(writer, ["tracked.txt"], state)
        monkeypatch.setattr(runtime.git, "_settle_git_lineage", original)

        with runtime.data_flow.activate(DataFlowContext()):
            staged = runtime.git.diff(reader, scope="staged")
            context = runtime.data_flow.current_context()
        assert "classified staged bytes" in staged.patch
        assert context.labels.sensitivity.value == "secret"
        parents, _source_refs = runtime.data_flow.provenance_sources(context)
        assert source.oid in parents
    finally:
        runtime.close()


def test_range_patch_artifact_inherits_unrelated_current_index_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    (root / "normal.txt").write_text("normal range change\n", encoding="utf-8")
    _git(root, "add", "--", "normal.txt")
    _git(root, "commit", "-q", "-m", "normal range change")
    head_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    (root / "secret-index.txt").write_text("unrelated staged secret\n", encoding="utf-8")
    _git(root, "add", "--", "secret-index.txt")
    runtime = _open_runtime(root)
    try:
        labeler = runtime.process.spawn(image="base-agent:v0", goal="label index")
        reader = runtime.process.spawn(image="base-agent:v0", goal="create range patch")
        for pid in (labeler, reader):
            _grant_git_authority(runtime, pid)
        source = runtime.memory.create_object(
            labeler,
            ObjectType.EVIDENCE,
            {"classification": "secret current index"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-range-patch-index-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(labeler, [source.oid])
        state = runtime.git.provider.repository_state()
        runtime.git._bind_git_lineage(
            pid=labeler,
            state=state,
            carrier_kind="index",
            carrier_id=state.index_sha256,
            context=secret_context,
        )

        with runtime.data_flow.activate(DataFlowContext()):
            ranged = runtime.git.diff(
                reader,
                scope="range",
                base=base_oid,
                head=head_oid,
            )
            range_context = runtime.data_flow.current_context()
        assert "normal range change" in ranged.patch
        assert range_context.labels.sensitivity.value == "normal"

        with runtime.data_flow.activate(DataFlowContext()):
            artifact = runtime.git.create_patch(
                reader,
                scope="range",
                base=base_oid,
                head=head_oid,
            )
        stored = runtime.store.get_object(artifact.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids
    finally:
        runtime.close()


def test_worktree_read_holds_file_label_lock_through_final_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_text("before concurrent write\n", encoding="utf-8")
    runtime = _open_runtime(root)
    writer_thread: threading.Thread | None = None
    try:
        reader = runtime.process.spawn(image="base-agent:v0", goal="read stable diff")
        writer = runtime.process.spawn(image="base-agent:v0", goal="write after refresh")
        _grant_git_authority(runtime, reader)
        _grant_git_authority(runtime, writer)
        refresh_count = 0
        writer_started = threading.Event()
        writer_finished = threading.Event()
        original = runtime.git._aggregate_flow_snapshots

        def write_after_refresh() -> None:
            writer_started.set()
            runtime.filesystem.write_text(
                writer,
                "tracked.txt",
                "after concurrent write\n",
            )
            writer_finished.set()

        def aggregate_with_barrier(
            snapshots: Any,
        ) -> Any:
            nonlocal refresh_count, writer_thread
            refresh_count += 1
            if refresh_count == 2:
                writer_thread = threading.Thread(target=write_after_refresh)
                writer_thread.start()
                assert writer_started.wait(timeout=2)
                assert not writer_finished.wait(timeout=0.1)
            return original(snapshots)

        monkeypatch.setattr(
            runtime.git,
            "_aggregate_flow_snapshots",
            aggregate_with_barrier,
        )
        result = runtime.git.diff(reader, paths=["tracked.txt"])
        assert "before concurrent write" in result.patch
        assert writer_thread is not None
        writer_thread.join(timeout=5)
        assert writer_finished.is_set()
        assert tracked.read_text(encoding="utf-8") == "after concurrent write\n"
    finally:
        if writer_thread is not None:
            writer_thread.join(timeout=5)
        runtime.close()


def test_log_and_blame_aggregate_every_returned_commit_carrier(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    history = root / "history.txt"
    history.write_text("ancestor classified line\n", encoding="utf-8")
    _git(root, "add", "--", "history.txt")
    _git(root, "commit", "-q", "-m", "ancestor secret marker")
    ancestor_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    history.write_text(
        "ancestor classified line\ndescendant normal line\n",
        encoding="utf-8",
    )
    _git(root, "commit", "-q", "-am", "descendant marker")
    descendant_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    runtime = _open_runtime(root)
    try:
        labeler = runtime.process.spawn(image="base-agent:v0", goal="label commit carriers")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read commit history")
        _grant_git_authority(runtime, labeler)
        _grant_git_authority(runtime, reader)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(labeler).state.token
            normal_tag = runtime.git.tag(
                labeler,
                "create",
                "descendant-normal",
                state,
                target=descendant_oid,
            )
        source = runtime.memory.create_object(
            labeler,
            ObjectType.EVIDENCE,
            {"classification": "secret ancestor"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-history-lineage-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            labeler,
            [source.oid],
        )
        with runtime.data_flow.activate(secret_context):
            runtime.git.tag(
                labeler,
                "create",
                "ancestor-secret",
                normal_tag.after.token,
                target=ancestor_oid,
            )

        with runtime.data_flow.activate(DataFlowContext()):
            logged = runtime.git.log(reader, ref=descendant_oid)
            log_context = runtime.data_flow.current_context()
        assert [item.subject for item in logged["commits"]][:2] == [
            "descendant marker",
            "ancestor secret marker",
        ]
        assert log_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            ranged = runtime.git.diff(
                reader,
                scope="range",
                base=base_oid,
                head=descendant_oid,
            )
            range_context = runtime.data_flow.current_context()
        assert "ancestor classified line" in ranged.patch
        assert range_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            blamed = runtime.git.blame(
                reader,
                "history.txt",
                ref=descendant_oid,
            )
            blame_context = runtime.data_flow.current_context()
        assert "ancestor classified line" in blamed["content"]
        assert blame_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_default_blame_does_not_inherit_unread_dirty_worktree_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_text("dirty secret line\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label dirty file")
        reader = runtime.process.spawn(image="base-agent:v0", goal="blame committed file")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, reader)
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret dirty bytes"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-blame-control-test",
            ),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256(tracked.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        with runtime.data_flow.activate(DataFlowContext()):
            blamed = runtime.git.blame(reader, "tracked.txt")
            context = runtime.data_flow.current_context()
        assert "initial" in blamed["content"]
        assert "dirty secret line" not in blamed["content"]
        assert context.labels.sensitivity.value == "normal"
        assert context.labels.trust_level == "untrusted"
    finally:
        runtime.close()


def test_worktree_git_write_rebinds_stale_trusted_file_label(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "untrusted")
    (root / "tracked.txt").write_bytes(b"replacement\n")
    _git(root, "commit", "-q", "-am", "replacement")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="bind trusted file")
        switcher = runtime.process.spawn(image="base-agent:v0", goal="switch worktree")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, switcher)
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"integrity": "trusted"},
            metadata=ObjectMetadata(
                trust_level="trusted",
                integrity="verified",
                origin="trusted-fixture",
            ),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256((root / "tracked.txt").read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(switcher).state.token
        runtime.git.switch(switcher, "untrusted", state)
        binding = runtime.store.get_file_label_binding("tracked.txt")
        assert binding is not None
        assert binding.content_sha256 == hashlib.sha256(b"replacement\n").hexdigest()
        assert binding.labels.trust_level == "untrusted"
        assert binding.labels.integrity == "untrusted"
    finally:
        runtime.close()


def test_failed_worktree_git_write_rebinds_partially_modified_file_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "conflict")
    (root / "tracked.txt").write_text("conflict branch\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "conflicting branch")
    _git(root, "switch", "-q", "main")
    (root / "tracked.txt").write_text("main branch\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "conflicting main")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="bind trusted file")
        integrator = runtime.process.spawn(image="base-agent:v0", goal="merge a conflict")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, integrator)
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"integrity": "trusted"},
            metadata=ObjectMetadata(
                trust_level="trusted",
                integrity="verified",
                origin="trusted-fixture",
            ),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256((root / "tracked.txt").read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(integrator).state.token
        with pytest.raises(GitError):
            runtime.git.integrate(integrator, "merge", state, ref="conflict")

        current = (root / "tracked.txt").read_bytes()
        assert b"<<<<<<<" in current
        binding = runtime.store.get_file_label_binding("tracked.txt")
        assert binding is not None
        assert binding.content_sha256 == hashlib.sha256(current).hexdigest()
        assert binding.labels.trust_level == "untrusted"
        assert binding.labels.integrity == "untrusted"
        effect = runtime.store.list_external_effects(pid=integrator)[-1]
        assert effect.transaction_state == "unknown"
    finally:
        runtime.close()


def test_worktree_git_write_preserves_unchanged_secret_file_label(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "replacement")
    (root / "tracked.txt").write_text("replacement\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "replacement")
    _git(root, "switch", "-q", "main")
    secret_path = root / "secret.txt"
    secret_path.write_text("secret\n", encoding="utf-8")
    secret_digest = hashlib.sha256(secret_path.read_bytes()).hexdigest()
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="bind secret file")
        switcher = runtime.process.spawn(image="base-agent:v0", goal="switch worktree")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, switcher)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="secret-fixture"),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="secret.txt",
            content_sha256=secret_digest,
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(switcher).state.token
        runtime.git.switch(switcher, "replacement", state)

        binding = runtime.store.get_file_label_binding("secret.txt")
        assert binding is not None
        assert binding.content_sha256 == secret_digest
        assert binding.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_switch_propagates_secret_commit_lineage_to_materialized_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "secret-branch")
    secret_path = root / "secret.txt"
    secret_path.write_bytes(b"classified\n")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label secret bytes")
        stager = runtime.process.spawn(image="base-agent:v0", goal="stage secret bytes")
        committer = runtime.process.spawn(image="base-agent:v0", goal="commit secret bytes")
        switcher = runtime.process.spawn(image="base-agent:v0", goal="materialize Git bytes")
        for pid in (writer, stager, committer, switcher):
            _grant_git_authority(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="git-lineage-test"),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="secret.txt",
            content_sha256=hashlib.sha256(secret_path.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(stager).state.token
        staged = runtime.git.stage(stager, ["secret.txt"], state)
        committed = runtime.git.commit(committer, "secret branch", staged.after.token)
        on_main = runtime.git.switch(switcher, "main", committed.after.token)
        assert not secret_path.exists()

        runtime.git.switch(switcher, "secret-branch", on_main.after.token)
        binding = runtime.store.get_file_label_binding("secret.txt")
        assert binding is not None
        assert binding.content_sha256 == hashlib.sha256(b"classified\n").hexdigest()
        assert binding.labels.sensitivity.value == "secret"
        assert binding.labels.trust_level == "untrusted"
    finally:
        runtime.close()


def test_internal_git_lineage_carriers_are_excluded_from_workspace_tree_labels(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="inspect Git lineage")
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="git-lineage-test"),
        )
        internal_path = ".git/agent-libos-flow/test/commit/carrier"
        runtime.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path=internal_path,
            content_sha256=hashlib.sha256(b"carrier").hexdigest(),
            context=runtime.data_flow.context_from_source_oids(pid, [source.oid]),
        )

        exact, _exact_version = runtime.data_flow.file_snapshot(internal_path)
        workspace, _workspace_version = runtime.data_flow.file_tree_snapshot(".")
        assert exact.labels.sensitivity.value == "secret"
        assert workspace.labels.sensitivity.value == "normal"
    finally:
        runtime.close()


def test_stash_round_trip_preserves_secret_worktree_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    tracked = root / "tracked.txt"
    tracked.write_bytes(b"classified stash\n")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label stash bytes")
        stasher = runtime.process.spawn(image="base-agent:v0", goal="round-trip stash")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, stasher)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="stash-lineage-test"),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256(tracked.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(stasher).state.token
        stashed = runtime.git.stash(stasher, "push", state)
        assert stashed.created_oid is not None
        applied = runtime.git.stash(stasher, "apply", stashed.after.token)

        assert tracked.read_bytes() == b"classified stash\n"
        binding = runtime.store.get_file_label_binding("tracked.txt")
        assert binding is not None
        assert binding.content_sha256 == hashlib.sha256(
            b"classified stash\n"
        ).hexdigest()
        assert binding.labels.sensitivity.value == "secret"
        assert applied.after.token != stashed.after.token
    finally:
        runtime.close()


def test_managed_worktree_materialization_preserves_secret_commit_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "secret-worktree")
    secret_path = root / "secret.txt"
    secret_path.write_bytes(b"classified worktree\n")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label secret bytes")
        creator = runtime.process.spawn(image="base-agent:v0", goal="create worktree")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, creator)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            writer,
            ObjectType.EVIDENCE,
            {"classification": "secret"},
            metadata=ObjectMetadata(sensitivity="secret", origin="git-lineage-test"),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=writer,
            normalized_path="secret.txt",
            content_sha256=hashlib.sha256(secret_path.read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(writer, [source.oid]),
        )

        state = runtime.git.status(writer).state.token
        staged = runtime.git.stage(writer, ["secret.txt"], state)
        committed = runtime.git.commit(writer, "secret worktree", staged.after.token)
        created = runtime.git.worktree(
            creator,
            "create",
            committed.after.token,
            ref="secret-worktree",
        )
        worktree_id = str(created.details["managed_worktree_id"])
        managed_root = Path(runtime.git.provider.managed_worktree_root)
        normalized = (managed_root / worktree_id / "secret.txt").relative_to(root).as_posix()
        binding = runtime.store.get_file_label_binding(normalized)
        assert binding is not None
        assert binding.content_sha256 == hashlib.sha256(
            b"classified worktree\n"
        ).hexdigest()
        assert binding.labels.sensitivity.value == "secret"
        assert binding.labels.trust_level == "untrusted"
    finally:
        runtime.close()


def test_restore_source_tree_requires_subtree_filesystem_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "tree-source")
    (root / "node").mkdir()
    (root / "node" / "child.txt").write_text("child\n", encoding="utf-8")
    _git(root, "add", "--", "node/child.txt")
    _git(root, "commit", "-q", "-m", "tree source")
    _git(root, "switch", "-q", "main")
    (root / "node").write_text("single file\n", encoding="utf-8")
    _git(root, "add", "--", "node")
    _git(root, "commit", "-q", "-m", "file target")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="restore a tree")
        _grant_git_repository_authority(runtime, pid)
        runtime.filesystem.grant_path(
            pid,
            "node",
            [CapabilityRight.WRITE, CapabilityRight.DELETE],
            issued_by="git-provider-test",
        )
        state = runtime.git.status(pid).state.token

        with pytest.raises(CapabilityDenied):
            _with_auto_approvals(
                runtime,
                lambda: runtime.git.restore(
                    pid,
                    ["node"],
                    state,
                    staged=True,
                    source="tree-source",
                ),
            )
        assert (root / "node").read_text(encoding="utf-8") == "single file\n"
        trusted_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"integrity": "trusted"},
            metadata=ObjectMetadata(
                trust_level="trusted",
                integrity="verified",
                origin="trusted-fixture",
            ),
        )
        runtime.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path="node",
            content_sha256=hashlib.sha256((root / "node").read_bytes()).hexdigest(),
            context=runtime.data_flow.context_from_source_oids(
                pid,
                [trusted_source.oid],
            ),
        )

        runtime.filesystem.grant_directory(
            pid,
            "node",
            [CapabilityRight.WRITE, CapabilityRight.DELETE],
            issued_by="git-provider-test",
        )
        restored = _with_auto_approvals(
            runtime,
            lambda: runtime.git.restore(
                pid,
                ["node"],
                state,
                staged=True,
                source="tree-source",
            ),
        )
        assert restored.changed_paths[0].display == "node"
        assert (root / "node" / "child.txt").read_text(encoding="utf-8") == "child\n"
        assert runtime.store.get_file_label_binding("node") is None
        child_binding = runtime.store.get_file_label_binding("node/child.txt")
        assert child_binding is not None
        assert child_binding.labels.trust_level == "untrusted"
    finally:
        runtime.close()


def test_worktree_restore_holds_root_label_lock_for_settlement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    for name in ("a.txt", "b.txt"):
        (root / name).write_text(f"{name} original\n", encoding="utf-8")
    _git(root, "add", "--", "a.txt", "b.txt")
    _git(root, "commit", "-q", "-m", "add restore files")
    for name in ("a.txt", "b.txt"):
        (root / name).write_text(f"{name} dirty\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="restore one dirty file")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        observed: list[tuple[str, ...]] = []
        original = runtime.filesystem.hold_file_label_io_paths

        @contextlib.contextmanager
        def capture(paths: Any):
            selected = tuple(paths)
            observed.append(selected)
            with original(selected):
                yield

        monkeypatch.setattr(runtime.filesystem, "hold_file_label_io_paths", capture)
        _with_auto_approvals(
            runtime,
            lambda: runtime.git.restore(pid, ["a.txt"], state),
        )
        assert observed[0] == (".",)
    finally:
        runtime.close()


def test_staged_only_restore_needs_no_filesystem_authority(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    for name in ("a.txt", "b.txt"):
        (root / name).write_text(f"{name} original\n", encoding="utf-8")
    _git(root, "add", "--", "a.txt", "b.txt")
    _git(root, "commit", "-q", "-m", "add staged restore files")
    for name in ("a.txt", "b.txt"):
        (root / name).write_text(f"{name} staged\n", encoding="utf-8")
    _git(root, "add", "--", "a.txt", "b.txt")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="restore index only")
        _grant_git_repository_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        result = _with_auto_approvals(
            runtime,
            lambda: runtime.git.restore(
                pid,
                ["a.txt"],
                state,
                staged=True,
                worktree=False,
            ),
        )
        assert [path.display for path in result.changed_paths] == ["a.txt"]
        assert (root / "a.txt").read_text(encoding="utf-8") == "a.txt staged\n"
        assert (root / "b.txt").read_text(encoding="utf-8") == "b.txt staged\n"
        staged_paths = set(
            _git(root, "diff", "--cached", "--name-only").decode("utf-8").splitlines()
        )
        assert "a.txt" not in staged_paths
        assert "b.txt" in staged_paths
    finally:
        runtime.close()


def test_local_branch_switch_stash_integrate_restore_reset_and_clean(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="exercise typed local Git")
        _grant_git_authority(runtime, pid)

        token = runtime.git.status(pid).state.token
        branch = runtime.git.branch(pid, "create", "topic", token)
        renamed = _with_auto_approvals(
            runtime,
            lambda: runtime.git.branch(
                pid,
                "rename",
                "topic",
                branch.after.token,
                new_name="topic-renamed",
            ),
        )
        switched = runtime.git.switch(pid, "topic-renamed", renamed.after.token)
        (root / "tracked.txt").write_text("topic change\n", encoding="utf-8")
        dirty = runtime.git.status(pid)
        stashed = runtime.git.stash(pid, "push", dirty.state.token)
        assert runtime.git.status(pid).entries == []
        applied = runtime.git.stash(pid, "apply", stashed.after.token)
        staged = runtime.git.stage(pid, ["tracked.txt"], applied.after.token)
        committed = runtime.git.commit(pid, "topic change", staged.after.token)
        tagged = runtime.git.tag(pid, "create", "v-local-test", committed.after.token)
        on_main = runtime.git.switch(pid, "main", tagged.after.token)
        merged = runtime.git.integrate(pid, "merge", on_main.after.token, ref="topic-renamed")
        assert merged.created_oid == committed.created_oid

        (root / "tracked.txt").write_text("discard me\n", encoding="utf-8")
        restore_state = runtime.git.status(pid).state.token
        restored = _with_auto_approvals(
            runtime,
            lambda: runtime.git.restore(pid, ["tracked.txt"], restore_state),
        )
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "topic change\n"

        (root / "temporary.txt").write_text("temporary\n", encoding="utf-8")
        temporary_state = runtime.git.status(pid).state.token
        temporary_staged = runtime.git.stage(pid, ["temporary.txt"], temporary_state)
        unstaged = runtime.git.unstage(pid, ["temporary.txt"], temporary_staged.after.token)
        cleaned = _with_auto_approvals(
            runtime,
            lambda: runtime.git.clean(pid, unstaged.after.token, paths=["temporary.txt"]),
        )
        assert not (root / "temporary.txt").exists()

        old_oid = merged.created_oid
        assert old_oid is not None
        (root / "tracked.txt").write_text("new commit\n", encoding="utf-8")
        new_state = runtime.git.status(pid).state.token
        new_staged = runtime.git.stage(pid, ["tracked.txt"], new_state)
        newer = runtime.git.commit(pid, "newer", new_staged.after.token)
        reset = _with_auto_approvals(
            runtime,
            lambda: runtime.git.reset(pid, old_oid, newer.after.token, mode="hard"),
        )
        assert reset.created_oid == old_oid

        dropped_branch = _with_auto_approvals(
            runtime,
            lambda: runtime.git.branch(
                pid,
                "delete",
                "topic-renamed",
                reset.after.token,
            ),
        )
        dropped_tag = _with_auto_approvals(
            runtime,
            lambda: runtime.git.tag(
                pid,
                "delete",
                "v-local-test",
                dropped_branch.after.token,
            ),
        )
        cleared = _with_auto_approvals(
            runtime,
            lambda: runtime.git.stash(pid, "clear", dropped_tag.after.token),
        )
        assert cleared.after.token != dropped_tag.after.token
    finally:
        runtime.close()


def test_typed_merge_disables_configured_autostash(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "side")
    (root / "tracked.txt").write_text("side\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "side")
    side_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "main")
    before_oid = _git(root, "rev-parse", "HEAD").strip()
    (root / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    _git(root, "config", "merge.autoStash", "true")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="merge without autostash")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token

        with pytest.raises(GitError) as exc_info:
            runtime.git.integrate(pid, "merge", state, ref=side_oid)

        assert exc_info.value.code == GitErrorCode.DIRTY_WORKTREE.value
        assert _git(root, "rev-parse", "HEAD").strip() == before_oid
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "dirty\n"
        stash = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", "refs/stash"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert stash.returncode == 1
        assert not any(
            entry.kind.value == "unmerged"
            for entry in runtime.git.status(pid).entries
        )
    finally:
        runtime.close()


def test_typed_merge_rejects_a_non_terminal_success(
    tmp_path: Path,
) -> None:
    class NonTerminalMergeProvider(LocalGitProvider):
        def __init__(self, workspace_root: Path) -> None:
            super().__init__(workspace_root)
            self.merge_args: tuple[str, ...] | None = None

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
            selected = list(args)
            if selected and selected[0] == "merge":
                self.merge_args = tuple(selected)
                selected = [item for item in selected if item != "--commit"]
                selected.insert(1, "--no-commit")
            return super().run(
                selected,
                worktree=worktree,
                timeout=timeout,
                stdin=stdin,
                max_output_bytes=max_output_bytes,
                read_only=read_only,
                remote=remote,
                expected_remote_fingerprint=expected_remote_fingerprint,
                verify_after=verify_after,
            )

    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "side")
    (root / "side.txt").write_text("side\n", encoding="utf-8")
    _git(root, "add", "--", "side.txt")
    _git(root, "commit", "-q", "-m", "side")
    side_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "main")
    (root / "main.txt").write_text("main\n", encoding="utf-8")
    _git(root, "add", "--", "main.txt")
    _git(root, "commit", "-q", "-m", "main")
    before_oid = _git(root, "rev-parse", "HEAD").strip()
    substrate = LocalResourceProviderSubstrate(root)
    provider = NonTerminalMergeProvider(root)
    substrate.git = provider
    runtime = Runtime.open(
        ":memory:",
        config=_runtime_config(),
        substrate=substrate,
        module_manifests=(),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject partial merge")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token

        with pytest.raises(GitError) as exc_info:
            runtime.git.integrate(pid, "merge", state, ref=side_oid)

        assert exc_info.value.code == GitErrorCode.CONFLICT.value
        assert provider.merge_args is not None
        assert {"--commit", "--no-squash"} <= set(provider.merge_args)
        assert "--no-autostash" not in provider.merge_args
        assert _git(root, "rev-parse", "HEAD").strip() == before_oid
        assert _git(root, "rev-parse", "--verify", "MERGE_HEAD").strip() == side_oid.encode(
            "ascii"
        )
    finally:
        runtime.close()


def test_patch_artifact_round_trip_and_lineage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "tracked.txt").write_text("initial\nchanged\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="round-trip a patch")
        _grant_git_authority(runtime, pid)
        artifact = runtime.git.create_patch(pid)
        assert artifact.bytes > 0
        assert len(artifact.patch_sha256) == 64
        _git(root, "restore", "--", "tracked.txt")
        clean = runtime.git.status(pid)
        applied = runtime.git.apply_patch(pid, artifact.oid, clean.state.token)
        assert applied.details["patch_sha256"] == artifact.patch_sha256
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "initial\nchanged\n"
        stored = runtime.store.get_object(artifact.oid)
        assert stored is not None
        assert stored.type.value == "code_patch"
        assert stored.immutable
        assert stored.payload["patch_sha256"] == artifact.patch_sha256
    finally:
        runtime.close()


def test_patch_artifact_and_applied_files_preserve_source_data_labels(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "tracked.txt").write_text("secret-derived change\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="preserve patch labels")
        _grant_git_authority(runtime, pid)
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {"source": "classified"},
            metadata=ObjectMetadata(sensitivity="secret", origin="test-secret"),
        )
        source_context = runtime.data_flow.context_from_source_oids(pid, [source.oid])
        runtime.data_flow.bind_written_file_digest(
            pid=pid,
            normalized_path="tracked.txt",
            content_sha256=hashlib.sha256((root / "tracked.txt").read_bytes()).hexdigest(),
            context=source_context,
        )
        artifact = runtime.git.create_patch(pid)
        stored = runtime.store.get_object(artifact.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids

        _git(root, "restore", "--", "tracked.txt")
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        clean = runtime.git.status(pid)
        runtime.git.apply_patch(pid, artifact.oid, clean.state.token)
        binding = runtime.store.get_file_label_binding("tracked.txt")
        assert binding is not None
        assert binding.labels.sensitivity.value == "secret"
        assert any(reference.oid == artifact.oid for reference in binding.source_refs)
    finally:
        runtime.close()


def test_managed_worktree_is_generated_inside_ignored_runtime_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="create managed worktree")
        _grant_git_authority(runtime, pid)
        before = runtime.git.status(pid)
        created = runtime.git.worktree(pid, "create", before.state.token)
        worktree_id = created.details["managed_worktree_id"]
        assert worktree_id.startswith("wt_")
        worktree_path = root / DEFAULT_CONFIG.git.worktree_root / worktree_id
        assert worktree_path.is_dir()
        assert (worktree_path / ".git").is_file()
        exclude = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert f"/{DEFAULT_CONFIG.git.worktree_root}/" in exclude
        assert runtime.git.status(pid).entries == []
        listed = runtime.git.list_worktrees(pid)
        assert any(item.worktree_id == worktree_id and item.managed for item in listed["worktrees"])

        observed_locks: list[tuple[str, ...]] = []

        @contextlib.contextmanager
        def capture_locks(paths: Any) -> Any:
            observed_locks.append(tuple(paths))
            yield

        monkeypatch.setattr(
            runtime.filesystem,
            "hold_file_label_io_paths",
            capture_locks,
        )
        (worktree_path / "managed.txt").write_text("managed\n", encoding="utf-8")
        managed_state = runtime.git.status(pid, worktree_id=worktree_id)
        staged = runtime.git.stage(
            pid,
            ["managed.txt"],
            managed_state.state.token,
            worktree_id=worktree_id,
        )
        assert observed_locks[-1] == (
            f"{DEFAULT_CONFIG.git.worktree_root}/{worktree_id}/managed.txt",
        )
        runtime.git.commit(
            pid,
            "managed worktree commit",
            staged.after.token,
            worktree_id=worktree_id,
        )
        main_state = runtime.git.status(pid).state.token
        removed = _with_auto_approvals(
            runtime,
            lambda: runtime.git.worktree(
                pid,
                "remove",
                main_state,
                managed_worktree_id=worktree_id,
            ),
        )
        assert removed.details["managed_worktree_id"] == worktree_id
        assert not worktree_path.exists()
    finally:
        runtime.close()


def test_managed_worktree_create_reuses_generated_id_after_human_approval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="approve managed worktree")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.WRITE],
            effect="ask",
            issued_by="git-provider-test",
        )
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="git-provider-test",
        )
        state = runtime.git.status(pid).state.token

        with pytest.raises(HumanApprovalRequired):
            runtime.git.worktree(pid, "create", state)
        assert runtime.human.drain_terminal_queue(auto_approve=True)

        created = runtime.git.worktree(pid, "create", state)
        worktree_id = created.details["managed_worktree_id"]
        assert worktree_id.startswith("wt_")
        assert (root / DEFAULT_CONFIG.git.worktree_root / worktree_id).is_dir()
        assert any(
            record.action == "primitive.git.worktree"
            and record.target == "git:workspace"
            for record in runtime.audit.trace(actor=pid)
        )
        assert any(
            event.type == EventType.EXTERNAL_WRITE
            and event.source == pid
            and event.payload.get("operation") == "worktree"
            for event in runtime.events.list(target="git:workspace")
        )
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("ref", "new_branch"),
    (
        ("missing-worktree-ref", None),
        (None, "main"),
    ),
    ids=("missing-ref", "duplicate-branch"),
)
def test_managed_worktree_create_failure_does_not_reconcile_absent_target(
    tmp_path: Path,
    ref: str | None,
    new_branch: str | None,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject worktree create")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token

        with pytest.raises(GitError):
            runtime.git.worktree(
                pid,
                "create",
                state,
                ref=ref,
                new_branch=new_branch,
            )

        managed_root = Path(runtime.git.provider.managed_worktree_root)
        assert not managed_root.exists() or not any(managed_root.iterdir())
        exclude = (root / ".git" / "info" / "exclude").read_text(encoding="utf-8")
        assert f"/{DEFAULT_CONFIG.git.worktree_root}/" in exclude
    finally:
        runtime.close()


def test_list_worktrees_omits_unmanaged_external_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    external = tmp_path / "outside-runtime-worktree"
    _init_repository(root)
    _git(root, "worktree", "add", "--detach", str(external), "HEAD")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="list managed worktrees")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )

        listed = runtime.git.list_worktrees(pid)

        assert [(item.worktree_id, item.path) for item in listed["worktrees"]] == [
            ("main", str(root.resolve()))
        ]
        assert str(external.resolve()) not in repr(listed)
        assert str(external.resolve()) not in repr(runtime.audit.trace(actor=pid))
        assert str(external.resolve()) not in repr(
            runtime.events.list(target="git:workspace")
        )
    finally:
        runtime.close()


def test_list_worktrees_rejects_managed_root_symlink_drift(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    external_root = tmp_path / "outside-managed-root"
    external_worktree = external_root / "wt_external"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="list managed worktrees")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )
        external_root.mkdir()
        _git(root, "worktree", "add", "--detach", str(external_worktree), "HEAD")
        managed_root = Path(runtime.git.provider.managed_worktree_root)
        managed_root.parent.mkdir(parents=True, exist_ok=True)
        try:
            managed_root.symlink_to(external_root, target_is_directory=True)
        except OSError:
            pytest.skip("symlinks are unavailable on this platform")

        listed = runtime.git.list_worktrees(pid)

        assert [(item.worktree_id, item.path) for item in listed["worktrees"]] == [
            ("main", str(root.resolve()))
        ]
        assert str(external_worktree.resolve()) not in repr(listed)
        assert str(external_worktree.resolve()) not in repr(
            runtime.audit.trace(actor=pid)
        )
        assert str(external_worktree.resolve()) not in repr(
            runtime.events.list(target="git:workspace")
        )
    finally:
        runtime.close()


def test_list_worktrees_rejects_managed_gitfile_with_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    managed_root = Path(provider.managed_worktree_root)
    managed_root.mkdir(parents=True)
    worktree = managed_root / "wt_untrusted"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    gitfile = worktree / ".git"
    git_dir = Path(gitfile.read_text(encoding="utf-8").removeprefix("gitdir: ").strip())
    alias = root / ".git" / "worktree-list-alias"
    try:
        alias.symlink_to(git_dir.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    gitfile.write_text(f"gitdir: {alias / git_dir.name}\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject structurally unsafe managed worktree",
        )
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
        )

        with pytest.raises(GitError) as exc_info:
            runtime.git.list_worktrees(pid)

        assert exc_info.value.code == GitErrorCode.UNSAFE_REPOSITORY.value
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "lease_oid",
    ("0" * 40, "0" * 64),
    ids=("sha1-zero-oid", "sha256-zero-oid"),
)
def test_push_rejects_all_zero_force_with_lease_before_remote_preflight(
    tmp_path: Path,
    lease_oid: str,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reject an invalid force-with-lease sentinel",
        )

        with pytest.raises(GitError) as exc_info:
            runtime.git.push(
                pid,
                "origin",
                "refs/heads/main",
                "a" * 64,
                local_ref="main",
                force_with_lease_oid=lease_oid,
            )

        assert exc_info.value.code == GitErrorCode.INVALID_REF.value
        assert runtime.store.list_external_effects(pid=pid) == []
        assert not any(
            record.actor == pid and record.action == "primitive.git.push"
            for record in runtime.audit.trace(actor=pid)
        )
        assert not any(
            event.type == EventType.EXTERNAL_WRITE
            and event.source == pid
            and event.target == "git_remote:workspace:origin"
            for event in runtime.events.list()
        )
    finally:
        runtime.close()


def test_file_remote_push_and_fetch_use_only_configured_remote(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    _git(root, "tag", "-a", "v-implicit", "-m", "must not follow the branch push")
    _git(root, "config", "push.followTags", "true")
    _git(root, "config", "push.gpgSign", "true")
    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="use configured remote")
        _grant_git_authority(runtime, pid, remote="origin")
        before = runtime.git.status(pid)
        pushed = runtime.git.push(
            pid,
            "origin",
            "refs/heads/main",
            before.state.token,
            local_ref="refs/heads/main",
        )
        assert pushed.details["remote"] == "origin"
        assert _git(remote, "rev-parse", "refs/heads/main").strip() == pushed.created_oid.encode("ascii")
        implicit_tag = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/tags/v-implicit"],
            cwd=remote,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert implicit_tag.returncode != 0

        assert pushed.created_oid is not None
        _git(root, "update-ref", "refs/remotes/origin/stale", pushed.created_oid)
        _git(root, "config", "fetch.prune", "true")
        _git(root, "config", "remote.origin.prune", "true")
        fetched_state = runtime.git.status(pid).state.token
        fetched = runtime.git.fetch(pid, "origin", fetched_state)
        assert fetched.details["remote"] == "origin"
        assert _git(root, "rev-parse", "refs/remotes/origin/stale").strip() == pushed.created_oid.encode("ascii")
        remote_info = runtime.git.list_remotes(pid)["remotes"][0]
        assert remote_info.fetch_url.startswith("<redacted:")
        assert str(remote) not in remote_info.fetch_url
        assert remote_info.fetch_refspecs == [
            "+refs/heads/*:refs/remotes/origin/*"
        ]
        with pytest.raises(GitError) as exc_info:
            runtime.git.fetch(pid, remote.as_uri(), fetched.after.token)
        assert exc_info.value.code == GitErrorCode.INVALID_REF.value
    finally:
        runtime.close()


def test_file_remote_push_preserves_annotated_tag_object(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    _git(root, "tag", "-a", "v1", "-m", "annotated release")
    local_tag_oid = _git(root, "rev-parse", "refs/tags/v1").strip().decode("ascii")
    assert _git(root, "cat-file", "-t", local_tag_oid).strip() == b"tag"

    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="push annotated tag")
        _grant_git_authority(runtime, pid, remote="origin")
        state = runtime.git.status(pid).state.token

        pushed = runtime.git.push(
            pid,
            "origin",
            "refs/tags/v1",
            state,
            local_ref="refs/tags/v1",
        )

        remote_tag_oid = _git(remote, "rev-parse", "refs/tags/v1").strip().decode("ascii")
        assert pushed.created_oid == pushed.details["local_oid"] == local_tag_oid
        assert remote_tag_oid == local_tag_oid
        assert _git(remote, "cat-file", "-t", remote_tag_oid).strip() == b"tag"
    finally:
        runtime.close()


def test_remote_git_mutation_timeout_is_unknown_and_not_retryable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    hook = remote / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nsleep 2\nexit 0\n", encoding="utf-8")
    hook.chmod(0o700)
    _git(root, "remote", "add", "origin", remote.as_uri())
    git_config = replace(
        DEFAULT_CONFIG.git,
        allow_file_remotes=True,
        remote_timeout_s=0.05,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="retain ambiguous remote timeout",
        )
        _grant_git_authority(runtime, pid, remote="origin")
        state = runtime.git.status(pid).state.token

        with pytest.raises(GitError) as exc_info:
            runtime.git.push(
                pid,
                "origin",
                "refs/heads/main",
                state,
                local_ref="refs/heads/main",
            )

        assert exc_info.value.code == GitErrorCode.TIMEOUT.value
        assert exc_info.value.retryable is False
        assert exc_info.value.details["effect"] == "unknown"
        assert exc_info.value.details["limit_kind"] == "subprocess_timeout"
        assert exc_info.value.details["metrics"]["killed"] is True
        effect = runtime.store.list_external_effects(pid=pid)[-1]
        assert effect.provider == "git"
        assert effect.transaction_state == "unknown"
    finally:
        runtime.close()


def test_remote_url_userinfo_query_and_custom_protocol_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(root)
    for index, unsafe in enumerate((
        "https://user@example.test/repository.git",
        "https://example.test/repository.git?token=secret",
        "ext::sh -c exploit",
        "ftp://example.test/repository.git",
    )):
        if index == 0:
            _git(root, "remote", "add", "origin", unsafe)
        else:
            _git(root, "remote", "set-url", "origin", unsafe)
        with pytest.raises(GitError) as exc_info:
            provider.remote_fingerprint("origin")
        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value


def test_remote_scheme_comparison_uses_normalized_config_values(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    git_config = replace(
        DEFAULT_CONFIG.git,
        allowed_remote_schemes=("HTTPS", "SSH"),
    )

    fingerprint = LocalGitProvider(root, config=git_config).remote_fingerprint(
        "origin"
    )

    assert fingerprint["remote"] == "origin"


def test_multiple_remote_urls_and_escaping_fetch_refspec_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    _git(root, "remote", "set-url", "--add", "--push", "origin", "https://push-one.example.test/repository.git")
    _git(root, "remote", "set-url", "--add", "--push", "origin", "https://push-two.example.test/repository.git")

    with pytest.raises(GitError) as multiple_error:
        LocalGitProvider(root).remote_fingerprint("origin")
    assert multiple_error.value.code == GitErrorCode.UNSAFE_CONFIG.value

    _git(root, "config", "--unset-all", "remote.origin.pushurl")
    _git(root, "config", "--unset-all", "remote.origin.fetch")
    _git(root, "config", "--add", "remote.origin.fetch", "+refs/heads/*:refs/heads/*")
    with pytest.raises(GitError) as refspec_error:
        LocalGitProvider(root).remote_fingerprint("origin")
    assert refspec_error.value.code == GitErrorCode.UNSAFE_CONFIG.value
    with pytest.raises(GitError) as list_error:
        LocalGitProvider(root).remote_configuration("origin")
    assert list_error.value.code == GitErrorCode.UNSAFE_CONFIG.value


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("http.proxy", "http://attacker.example.test:8080"),
        (
            "http.https://example.test/.proxy",
            "http://attacker.example.test:8080",
        ),
        ("http.sslVerify", "false"),
        ("http.sslVersion", "tlsv1"),
        ("remote.origin.proxy", "http://attacker.example.test:8080"),
    ),
    ids=("proxy", "url-proxy", "tls-verify", "tls-version", "remote-proxy"),
)
def test_repository_http_transport_overrides_are_rejected(
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    _git(root, "config", key, value)

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).remote_fingerprint("origin")

    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value


def test_repository_http_transport_override_fails_before_remote_effect(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    _git(root, "config", "http.sslVerify", "false")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject unsafe HTTP config")
        _grant_git_authority(runtime, pid, remote="origin")
        state = runtime.git.status(pid).state.token
        effects_before = runtime.store.list_external_effects(pid=pid)
        remote_resource = runtime.git.remote_resource("origin")
        events_before = runtime.events.list(target=remote_resource)

        with pytest.raises(GitError) as exc_info:
            runtime.git.fetch(pid, "origin", state)

        assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
        assert runtime.store.list_external_effects(pid=pid) == effects_before
        assert runtime.events.list(target=remote_resource) == events_before
        assert not any(
            record.action == "primitive.git.fetch"
            for record in runtime.audit.trace(actor=pid)
        )
        assert any(
            operation.name == "runtime.git.fetch"
            and operation.outcome is not None
            and operation.outcome.value == "failed"
            for operation in runtime.store.list_operations(pid=pid)
        )
    finally:
        runtime.close()


def test_host_global_http_transport_config_remains_supported(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    _git(
        root,
        "config",
        "--global",
        "http.proxy",
        "http://host-proxy.example.test:8080",
    )

    fingerprint = LocalGitProvider(root).remote_fingerprint("origin")

    assert fingerprint["remote"] == "origin"


def test_windows_system_credential_helper_resolves_from_git_install_bin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("Git for Windows installation layout regression")
    root = tmp_path / "repo"
    _init_repository(root)
    install_root = tmp_path / "host-git"
    git_path = install_root / "cmd" / "git.exe"
    helper = install_root / "mingw64" / "bin" / "git-credential-manager.exe"
    git_path.parent.mkdir(parents=True)
    helper.parent.mkdir(parents=True)
    git_path.write_bytes(b"git executable")
    helper.write_bytes(b"credential helper executable")
    provider = LocalGitProvider(root)
    monkeypatch.setattr(
        "agent_libos.substrate.git.shutil.which",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        provider,
        "_resolve_executable",
        lambda: (git_path, (1, 2, 3, 4, 5), "0" * 64),
    )
    monkeypatch.setattr(
        provider,
        "_invoke",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=str(
                install_root / "mingw64" / "libexec" / "git-core"
            ).encode(),
            stderr=b"",
        ),
    )

    resolved, digest = provider._resolve_helper("manager")

    assert Path(resolved) == helper.resolve()
    assert digest == hashlib.sha256(helper.read_bytes()).hexdigest()


def test_shell_credential_helper_is_rejected_without_execution(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    sentinel = tmp_path / "credential-helper-ran"
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    _git(root, "config", "credential.helper", f"!touch {sentinel}")

    with pytest.raises(GitError) as exc_info:
        LocalGitProvider(root).remote_fingerprint("origin")
    assert exc_info.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not sentinel.exists()


def test_scoped_credential_helper_and_askpass_are_rejected_without_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "remote", "add", "origin", "https://example.test/repository.git")
    scoped_sentinel = tmp_path / "scoped-helper-ran"
    askpass_sentinel = tmp_path / "askpass-ran"
    scoped_key = "credential.https://example.test.helper"
    _git(root, "config", scoped_key, f"!touch {scoped_sentinel}")

    with pytest.raises(GitError) as scoped_error:
        LocalGitProvider(root).remote_fingerprint("origin")
    assert scoped_error.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not scoped_sentinel.exists()

    _git(root, "config", "--unset-all", scoped_key)
    _git(root, "config", "core.askPass", f"touch {askpass_sentinel}")
    with pytest.raises(GitError) as askpass_error:
        LocalGitProvider(root).remote_fingerprint("origin")
    assert askpass_error.value.code == GitErrorCode.UNSAFE_CONFIG.value
    assert not askpass_sentinel.exists()


def test_missing_remote_authority_denies_before_remote_metadata_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="deny remote enumeration")
        _grant_git_authority(runtime, pid)
        token = runtime.git.status(pid).state.token
        effects_before = runtime.store.list_external_effects(pid=pid)

        def forbidden_lookup(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
            raise AssertionError("remote metadata was read before authority")

        monkeypatch.setattr(runtime.git.provider, "remote_fingerprint", forbidden_lookup)
        with pytest.raises(CapabilityDenied):
            runtime.git.fetch(pid, "origin", token)
        assert runtime.store.list_external_effects(pid=pid) == effects_before
    finally:
        runtime.close()


def test_remote_fingerprint_change_is_rejected_before_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="reject remote race")
        _grant_git_authority(runtime, pid, remote="origin")
        original = runtime.git.provider.remote_fingerprint
        calls = 0

        def changed_fingerprint(*args: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            result = dict(original(*args, **kwargs))
            if calls > 1:
                result["fingerprint"] = "0" * 64
            return result

        monkeypatch.setattr(runtime.git.provider, "remote_fingerprint", changed_fingerprint)
        token = runtime.git.status(pid).state.token
        with pytest.raises(GitError) as exc_info:
            runtime.git.fetch(pid, "origin", token)
        assert exc_info.value.code == GitErrorCode.STALE_STATE.value
        assert _git(remote, "for-each-ref").strip() == b""
    finally:
        runtime.close()


def test_post_dispatch_git_failure_is_retained_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "ambiguous.txt").write_text("ambiguous\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="retain unknown Git effect")
        _grant_git_authority(runtime, pid)
        original = runtime.git.provider.run

        def fail_after_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            command = args[0] if args else kwargs.get("args")
            if command and command[0] == "add":
                raise GitError(
                    GitErrorCode.UNKNOWN_EFFECT.value,
                    "post-dispatch Git outcome is unknown",
                    operation="stage",
                )
            return result

        monkeypatch.setattr(runtime.git.provider, "run", fail_after_dispatch)
        token = runtime.git.status(pid).state.token
        with pytest.raises(GitError) as exc_info:
            runtime.git.stage(pid, ["ambiguous.txt"], token)
        assert exc_info.value.code == GitErrorCode.UNKNOWN_EFFECT.value
        assert _git(root, "diff", "--cached", "--name-only").strip() == b"ambiguous.txt"
        effect = runtime.store.list_external_effects(pid=pid)[-1]
        assert effect.provider == "git"
        assert effect.operation == "mutate"
        assert effect.transaction_state == "unknown"
    finally:
        runtime.close()


def test_post_dispatch_stale_state_is_retained_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "stale-after-dispatch.txt").write_text("staged\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="reconcile post-dispatch stale Git effect",
        )
        _grant_git_authority(runtime, pid)
        original = runtime.git.provider.run

        def stale_after_dispatch(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            command = args[0] if args else kwargs.get("args")
            if command and command[0] == "add":
                raise GitError(
                    GitErrorCode.STALE_STATE.value,
                    "repository identity changed after dispatch",
                    operation="stage",
                    retryable=True,
                )
            return result

        monkeypatch.setattr(runtime.git.provider, "run", stale_after_dispatch)
        token = runtime.git.status(pid).state.token

        with pytest.raises(GitError) as exc_info:
            runtime.git.stage(pid, ["stale-after-dispatch.txt"], token)

        assert exc_info.value.code == GitErrorCode.STALE_STATE.value
        assert _git(root, "diff", "--cached", "--name-only").strip() == b"stale-after-dispatch.txt"
        effect = runtime.store.list_external_effects(pid=pid)[-1]
        assert effect.provider == "git"
        assert effect.operation == "mutate"
        assert effect.transaction_state == "unknown"
    finally:
        runtime.close()


def test_non_fast_forward_push_and_exact_force_with_lease(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    _init_repository(root)
    initial_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="push with an exact lease")
        _grant_git_authority(runtime, pid, remote="origin")
        first = runtime.git.status(pid)
        initial_push = runtime.git.push(
            pid,
            "origin",
            "refs/heads/main",
            first.state.token,
            local_ref="main",
        )
        (root / "tracked.txt").write_text("remote advance\n", encoding="utf-8")
        dirty = runtime.git.status(pid)
        staged = runtime.git.stage(pid, ["tracked.txt"], dirty.state.token)
        advanced = runtime.git.commit(pid, "advance remote", staged.after.token)
        advanced_push = runtime.git.push(
            pid,
            "origin",
            "refs/heads/main",
            advanced.after.token,
            local_ref="main",
        )
        remote_oid = advanced_push.created_oid
        assert remote_oid is not None and remote_oid != initial_oid

        _git(root, "reset", "--hard", initial_oid)
        behind = runtime.git.status(pid)
        with pytest.raises(GitError) as exc_info:
            runtime.git.push(
                pid,
                "origin",
                "refs/heads/main",
                behind.state.token,
                local_ref="main",
            )
        assert exc_info.value.code == GitErrorCode.NON_FAST_FORWARD.value

        lease_state = runtime.git.status(pid).state.token
        forced = _with_auto_approvals(
            runtime,
            lambda: runtime.git.push(
                pid,
                "origin",
                "refs/heads/main",
                lease_state,
                local_ref="main",
                force_with_lease_oid=remote_oid,
            ),
        )
        assert forced.details["expected_remote_oid"] == remote_oid
        assert _git(remote, "rev-parse", "refs/heads/main").strip().decode("ascii") == initial_oid
        assert initial_push.created_oid == initial_oid
    finally:
        runtime.close()


@pytest.mark.parametrize("strategy", ["ff_only", "merge"])
def test_fast_forward_pull_from_configured_bare_remote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    other = tmp_path / "other"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    _git(root, "push", "-q", "origin", "main:refs/heads/main")

    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "config", "user.name", "Remote Test")
    _git(other, "config", "user.email", "remote@example.test")
    _git(other, "remote", "add", "origin", remote.as_uri())
    _git(other, "fetch", "-q", "origin", "main")
    _git(other, "checkout", "-q", "-b", "main", "FETCH_HEAD")
    (other / "remote.txt").write_text("from remote\n", encoding="utf-8")
    _git(other, "add", "--", "remote.txt")
    _git(other, "commit", "-q", "-m", "remote change")
    _git(other, "push", "-q", "origin", "main:refs/heads/main")
    remote_oid = _git(other, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "config", "merge.autoStash", "true")

    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fast-forward pull")
        _grant_git_authority(runtime, pid, remote="origin")
        merge_args: list[tuple[str, ...]] = []
        original_run = runtime.git.provider.run

        def record_merge_args(
            args: Sequence[str],
            **kwargs: Any,
        ) -> GitCommandResult:
            if args and args[0] == "merge":
                merge_args.append(tuple(args))
            return original_run(args, **kwargs)

        monkeypatch.setattr(runtime.git.provider, "run", record_merge_args)
        state = runtime.git.status(pid).state.token
        pulled = runtime.git.pull(
            pid,
            "origin",
            state,
            branch="main",
            strategy=strategy,
        )
        assert pulled.created_oid == remote_oid
        assert (root / "remote.txt").read_text(encoding="utf-8") == "from remote\n"
        assert len(merge_args) == 1
        # merge --no-autostash was introduced after the supported Git 2.26
        # floor; the provider-level config pin supplies the same safety policy.
        assert "--no-autostash" not in merge_args[0]
        assert "merge.autoStash=false" in runtime.git.provider._repo_prefix(
            runtime.git.provider.repository_layout()
        )
        assert "core.longpaths=true" in runtime.git.provider._repo_prefix(
            runtime.git.provider.repository_layout()
        )
        if strategy == "ff_only":
            assert "--ff-only" in merge_args[0]
        else:
            assert {"--commit", "--no-squash", "--no-gpg-sign"} <= set(
                merge_args[0]
            )
    finally:
        runtime.close()


def test_pull_defaults_to_the_capability_scoped_current_branch(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    remote = tmp_path / "remote.git"
    other = tmp_path / "other"
    _init_repository(root)
    remote.mkdir()
    _git(remote, "init", "--bare", "-q")
    _git(root, "remote", "add", "origin", remote.as_uri())
    _git(root, "push", "-q", "origin", "main:refs/heads/main")

    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "config", "user.name", "Remote Test")
    _git(other, "config", "user.email", "remote@example.test")
    _git(other, "remote", "add", "origin", remote.as_uri())
    _git(other, "fetch", "-q", "origin", "main")
    _git(other, "checkout", "-q", "-b", "main", "FETCH_HEAD")
    (other / "remote.txt").write_text("from remote\n", encoding="utf-8")
    _git(other, "add", "--", "remote.txt")
    _git(other, "commit", "-q", "-m", "remote main change")
    _git(other, "push", "-q", "origin", "main:refs/heads/main")
    _git(other, "switch", "-q", "-c", "secret")
    (other / "secret.txt").write_text("remote secret\n", encoding="utf-8")
    _git(other, "add", "--", "secret.txt")
    _git(other, "commit", "-q", "-m", "remote secret change")
    _git(other, "push", "-q", "origin", "secret:refs/heads/secret")
    _git(root, "update-ref", "-d", "refs/remotes/origin/main")
    _git(root, "update-ref", "-d", "refs/remotes/origin/secret")

    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="pull only authorized main")
        _grant_git_authority(runtime, pid)
        runtime.capability.issue_trusted(
            pid,
            "git_remote:workspace:origin",
            [CapabilityRight.READ],
            issued_by="git-provider-test",
            constraints={"git_allowed_refs": ["refs/heads/main"]},
        )
        state = runtime.git.status(pid).state.token

        runtime.git.pull(pid, "origin", state, strategy="ff_only")

        assert (root / "remote.txt").read_text(encoding="utf-8") == "from remote\n"
        secret_ref = subprocess.run(
            ["git", "rev-parse", "--verify", "refs/remotes/origin/secret"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert secret_ref.returncode != 0
    finally:
        runtime.close()


def test_repository_content_high_water_covers_all_patch_scopes_and_renames(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    origin_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "mv", "tracked.txt", "renamed.txt")
    _git(root, "commit", "-q", "-m", "rename tracked file")
    (root / "noise.txt").write_text("noise\n", encoding="utf-8")
    _git(root, "add", "--", "noise.txt")
    _git(root, "commit", "-q", "-m", "unrelated base")
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    (root / "renamed.txt").write_text("replacement\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "replace renamed content")
    head_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    runtime = _open_runtime(root)
    try:
        labeler = runtime.process.spawn(image="base-agent:v0", goal="label origin")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read patches")
        for pid in (labeler, reader):
            _grant_git_authority(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            labeler,
            ObjectType.EVIDENCE,
            {"classification": "secret byte origin"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-repository-high-water-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            labeler,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(labeler).state.token
        with runtime.data_flow.activate(secret_context):
            runtime.git.tag(
                labeler,
                "create",
                "secret-origin",
                state,
                target=origin_oid,
            )

        with runtime.data_flow.activate(DataFlowContext()):
            ranged = runtime.git.diff(
                reader,
                scope="range",
                base=base_oid,
                head=head_oid,
            )
            range_context = runtime.data_flow.current_context()
        assert "-initial" in ranged.patch
        assert range_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            same = runtime.git.diff(
                reader,
                scope="range",
                base=head_oid,
                head=head_oid,
            )
            same_context = runtime.data_flow.current_context()
        assert same.patch == ""
        assert same_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            shown = runtime.git.show(reader, head_oid)
            show_context = runtime.data_flow.current_context()
        assert "-initial" in shown["patch"]
        assert show_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            artifact = runtime.git.create_patch(
                reader,
                scope="range",
                base=base_oid,
                head=head_oid,
            )
        stored = runtime.store.get_object(artifact.oid)
        assert stored is not None
        assert stored.metadata.sensitivity == "secret"
        assert source.oid in stored.provenance.parent_oids

        (root / "renamed.txt").write_text("worktree replacement\n", encoding="utf-8")
        with runtime.data_flow.activate(DataFlowContext()):
            worktree = runtime.git.diff(reader, paths=["renamed.txt"])
            worktree_context = runtime.data_flow.current_context()
        assert worktree_context.labels.sensitivity.value == "secret"
        with runtime.data_flow.activate(DataFlowContext()):
            staged = runtime.git.stage(
                reader,
                ["renamed.txt"],
                worktree.state.token,
            )
        with runtime.data_flow.activate(DataFlowContext()):
            staged_diff = runtime.git.diff(reader, scope="staged")
            staged_context = runtime.data_flow.current_context()
        assert staged.after.token == staged_diff.state.token
        assert staged_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_range_reads_ignore_unrelated_long_history(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    commit_oids: list[str] = []
    for index in range(8):
        (root / f"noise-{index}.txt").write_text(f"{index}\n", encoding="utf-8")
        _git(root, "add", "--", f"noise-{index}.txt")
        _git(root, "commit", "-q", "-m", f"noise {index}")
        commit_oids.append(_git(root, "rev-parse", "HEAD").strip().decode("ascii"))
    git_config = replace(
        DEFAULT_CONFIG.git,
        log_entry_limit=2,
        log_entry_hard_limit=2,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        reader = runtime.process.spawn(image="base-agent:v0", goal="read short range")
        _grant_git_authority(runtime, reader)
        adjacent = runtime.git.diff(
            reader,
            scope="range",
            base=commit_oids[-2],
            head=commit_oids[-1],
        )
        same = runtime.git.diff(
            reader,
            scope="range",
            base=commit_oids[-1],
            head=commit_oids[-1],
        )
        assert "noise-7.txt" in adjacent.patch
        assert same.patch == ""
    finally:
        runtime.close()


def test_repository_content_lineage_survives_resource_alias_and_path_spelling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    database = tmp_path / "runtime.db"
    first_config = _runtime_config()
    first = Runtime.open(
        database,
        config=first_config,
        substrate=LocalResourceProviderSubstrate(root),
        module_manifests=(),
    )
    try:
        labeler = first.process.spawn(image="base-agent:v0", goal="label repository")
        _grant_git_authority(first, labeler)
        first.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = first.memory.create_object(
            labeler,
            ObjectType.EVIDENCE,
            {"classification": "secret repository alias"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-repository-alias-test",
            ),
        )
        secret_context = first.data_flow.context_from_source_oids(labeler, [source.oid])
        with first.data_flow.activate(DataFlowContext()):
            state = first.git.status(labeler).state.token
        with first.data_flow.activate(secret_context):
            first.git.tag(labeler, "create", "secret-alias", state)
    finally:
        first.close()

    aliased_git = replace(
        DEFAULT_CONFIG.git,
        repository_resource="git:alternate",
    )
    second = Runtime.open(
        database,
        config=_runtime_config(git=aliased_git),
        substrate=LocalResourceProviderSubstrate(root, git_config=aliased_git),
        module_manifests=(),
    )
    try:
        reader = second.process.spawn(image="base-agent:v0", goal="read aliased repository")
        _grant_git_authority(second, reader)
        second.capability.issue_trusted(
            reader,
            "git:alternate",
            [CapabilityRight.READ, CapabilityRight.DIFF],
            issued_by="git-provider-test",
        )
        with second.data_flow.activate(DataFlowContext()):
            refs = second.git.list_refs(reader, kind="tags")
            context = second.data_flow.current_context()
        assert "secret-alias" in {ref.short_name for ref in refs["refs"]}
        assert context.labels.sensitivity.value == "secret"
        parents, _source_refs = second.data_flow.provenance_sources(context)
        assert source.oid in parents

        composed = tmp_path / "Caf\N{LATIN SMALL LETTER E WITH ACUTE}" / "Repo"
        decomposed = tmp_path / "CAFE\N{COMBINING ACUTE ACCENT}" / "repo"
        assert GitPrimitive._canonical_workspace_identity(
            composed
        ) == GitPrimitive._canonical_workspace_identity(decomposed)
    finally:
        second.close()


def test_show_log_and_blame_include_emitted_parent_carriers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    (root / "tracked.txt").write_text("parent classified line\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "classified parent")
    parent_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "-c", "side")
    (root / "side.txt").write_text("side\n", encoding="utf-8")
    _git(root, "add", "--", "side.txt")
    _git(root, "commit", "-q", "-m", "side parent")
    side_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "main")
    (root / "tracked.txt").write_text("child line\n", encoding="utf-8")
    _git(root, "commit", "-q", "-am", "child")
    child_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "merge", "-q", "--no-ff", "side", "-m", "merge side")
    merge_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    runtime = _open_runtime(root)
    try:
        labeler = runtime.process.spawn(image="base-agent:v0", goal="late label parents")
        reader = runtime.process.spawn(image="base-agent:v0", goal="read emitted parents")
        for pid in (labeler, reader):
            _grant_git_authority(runtime, pid)
        source = runtime.memory.create_object(
            labeler,
            ObjectType.EVIDENCE,
            {"classification": "secret parent carrier"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-parent-carrier-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(labeler, [source.oid])
        state = runtime.git.provider.repository_state()
        for oid in (parent_oid, side_oid):
            runtime.git._bind_git_lineage(
                pid=labeler,
                state=state,
                carrier_kind="commit",
                carrier_id=oid,
                context=secret_context,
            )

        with runtime.data_flow.activate(DataFlowContext()):
            shown = runtime.git.show(reader, child_oid)
            shown_context = runtime.data_flow.current_context()
        assert "parent classified line" in shown["patch"]
        assert shown_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            blamed = runtime.git.blame(reader, "tracked.txt", ref=child_oid)
            blame_context = runtime.data_flow.current_context()
        assert f"previous {parent_oid}" in blamed["content"]
        assert blame_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            logged = runtime.git.log(reader, ref=merge_oid, limit=1)
            log_context = runtime.data_flow.current_context()
        assert side_oid in logged["commits"][0].parents
        assert log_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_simulated_pull_request_create_review_close_and_merge_requires_approval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="manage simulated pull request")
        _grant_git_authority(runtime, pid)
        before = runtime.git.status(pid)
        created = runtime.git.create_pull_request(
            pid,
            "Feature",
            "Adds feature.txt",
            "main",
            "feature",
            before.state.token,
        )
        pull_request = created["pull_request"]
        assert pull_request.status is GitPullRequestStatus.OPEN
        assert runtime.git.inspect_pull_request(pid, pull_request.pr_id).head_oid == pull_request.head_oid
        listed = runtime.git.list_pull_requests(pid)
        assert [item.pr_id for item in listed["pull_requests"]] == [pull_request.pr_id]

        reviewed = runtime.git.review_pull_request(
            pid,
            pull_request.pr_id,
            "comment",
            "looks good",
            created["operation"].after.token,
        )
        assert reviewed["pull_request"].reviews[0].decision.value == "comment"
        with pytest.raises(HumanApprovalRequired):
            runtime.git.merge_pull_request(
                pid,
                pull_request.pr_id,
                reviewed["operation"].after.token,
            )
        closed = runtime.git.close_pull_request(
            pid,
            pull_request.pr_id,
            reviewed["operation"].after.token,
        )
        assert closed["pull_request"].status is GitPullRequestStatus.CLOSED
    finally:
        runtime.close()


def test_pull_request_create_preflights_metadata_count_before_snapshot_refs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    git_config = replace(
        DEFAULT_CONFIG.git,
        status_entry_limit=1,
        status_entry_hard_limit=1,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bound PR metadata count")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        first = runtime.git.create_pull_request(
            pid,
            "First feature",
            "first body",
            "main",
            "feature",
            state,
        )
        refs_before = _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/agent-libos/pull-requests",
        ).splitlines()

        with pytest.raises(GitError) as exc_info:
            runtime.git.create_pull_request(
                pid,
                "Second feature",
                "second body",
                "main",
                "feature",
                first["operation"].after.token,
            )

        assert exc_info.value.code == GitErrorCode.OUTPUT_TOO_LARGE.value
        assert _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/agent-libos/pull-requests",
        ).splitlines() == refs_before
        assert len(runtime.git.provider.list_pull_request_metadata(limit=1)) == 1
        complete = runtime.git.list_pull_requests(pid, limit=1)
        assert [item.pr_id for item in complete["pull_requests"]] == [
            first["pull_request"].pr_id
        ]
        assert complete["truncated"] is False
        runtime.git.status(pid)
    finally:
        runtime.close()


def test_pull_request_list_is_truncated_only_above_requested_limit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="list bounded PR metadata",
        )
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        first = runtime.git.create_pull_request(
            pid,
            "First feature",
            "first body",
            "main",
            "feature",
            state,
        )
        second = runtime.git.create_pull_request(
            pid,
            "Second feature",
            "second body",
            "main",
            "feature",
            first["operation"].after.token,
        )

        limited = runtime.git.list_pull_requests(pid, limit=1)
        assert len(limited["pull_requests"]) == 1
        assert limited["truncated"] is True

        complete = runtime.git.list_pull_requests(pid, limit=2)
        assert {item.pr_id for item in complete["pull_requests"]} == {
            first["pull_request"].pr_id,
            second["pull_request"].pr_id,
        }
        assert complete["truncated"] is False
    finally:
        runtime.close()


def test_pull_request_create_preflights_aggregate_bytes_before_snapshot_refs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    git_config = replace(
        DEFAULT_CONFIG.git,
        output_max_bytes=65_536,
        output_hard_limit_bytes=65_536,
        patch_max_bytes=65_536,
        patch_hard_limit_bytes=65_536,
    )
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bound PR metadata bytes")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        first = runtime.git.create_pull_request(
            pid,
            "First large feature",
            "a" * 35_000,
            "main",
            "feature",
            state,
        )
        refs_before = _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/agent-libos/pull-requests",
        ).splitlines()

        with pytest.raises(GitError) as exc_info:
            runtime.git.create_pull_request(
                pid,
                "Second large feature",
                "b" * 35_000,
                "main",
                "feature",
                first["operation"].after.token,
            )

        assert exc_info.value.code == GitErrorCode.OUTPUT_TOO_LARGE.value
        assert _git(
            root,
            "for-each-ref",
            "--format=%(refname)",
            "refs/agent-libos/pull-requests",
        ).splitlines() == refs_before
        assert len(runtime.git.provider.list_pull_request_metadata(limit=10)) == 1
        runtime.git.status(pid)
    finally:
        runtime.close()


def test_pull_request_metadata_listing_rejects_collection_over_hard_count(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    provider = LocalGitProvider(
        root,
        config=replace(
            DEFAULT_CONFIG.git,
            status_entry_limit=1,
            status_entry_hard_limit=1,
        ),
    )
    directory = root / ".git" / "agent-libos" / "pull_requests"
    directory.mkdir(parents=True)
    (directory / "pr_a.json").write_bytes(b"{}")
    (directory / "pr_b.json").write_bytes(b"{}")

    with pytest.raises(GitError) as exc_info:
        provider.list_pull_request_metadata(limit=1)

    assert exc_info.value.code == GitErrorCode.OUTPUT_TOO_LARGE.value


def test_pull_request_metadata_persists_and_restores_data_flow_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create secret PR")
        reviewer = runtime.process.spawn(image="base-agent:v0", goal="review secret PR")
        inspector = runtime.process.spawn(image="base-agent:v0", goal="inspect secret PR")
        for pid in (creator, reviewer, inspector):
            _grant_git_authority(runtime, pid)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git_pr:workspace:*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            creator,
            ObjectType.EVIDENCE,
            {"classification": "secret pull request"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-lineage-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            creator,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token
        with runtime.data_flow.activate(secret_context):
            created = runtime.git.create_pull_request(
                creator,
                "Classified feature",
                "Contains secret review context",
                "main",
                "feature",
                state,
            )
        pull_request = created["pull_request"]

        for kind in ("pull_requests", "all"):
            with runtime.data_flow.activate(DataFlowContext()):
                listed_refs = runtime.git.list_refs(inspector, kind=kind)
                refs_context = runtime.data_flow.current_context()
            assert {
                ref.name for ref in listed_refs["refs"]
            }.issuperset(set(runtime.git._pull_request_snapshot_refs(pull_request.pr_id)))
            assert refs_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            branch_refs = runtime.git.list_refs(inspector, kind="branches")
        assert {ref.short_name for ref in branch_refs["refs"]}.issuperset(
            {"main", "feature"}
        )

        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git_pr:workspace:*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
            ),
            actor="git-provider-test",
            replace=True,
            require_capability=False,
        )
        with runtime.data_flow.activate(DataFlowContext()):
            with pytest.raises(CapabilityDenied, match="data-flow denied"):
                runtime.git.review_pull_request(
                    reviewer,
                    pull_request.pr_id,
                    "comment",
                    "normal review",
                    created["operation"].after.token,
                )

        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git_pr:workspace:*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            replace=True,
            require_capability=False,
        )
        with runtime.data_flow.activate(DataFlowContext()):
            reviewed = runtime.git.review_pull_request(
                reviewer,
                pull_request.pr_id,
                "comment",
                "normal review",
                created["operation"].after.token,
            )

        with runtime.data_flow.activate(DataFlowContext()):
            inspected = runtime.git.inspect_pull_request(
                inspector,
                pull_request.pr_id,
            )
            inspect_context = runtime.data_flow.current_context()
        assert inspected.body == "Contains secret review context"
        assert inspected.reviews == reviewed["pull_request"].reviews
        assert inspect_context.labels.sensitivity.value == "secret"
        inspect_parents, _inspect_refs = runtime.data_flow.provenance_sources(
            inspect_context
        )
        assert source.oid in inspect_parents

        with runtime.data_flow.activate(DataFlowContext()):
            listed = runtime.git.list_pull_requests(inspector)
            list_context = runtime.data_flow.current_context()
        assert [item.pr_id for item in listed["pull_requests"]] == [
            pull_request.pr_id
        ]
        assert list_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_pull_request_prebind_survives_post_write_lineage_settlement_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create PR")
        reviewer = runtime.process.spawn(image="base-agent:v0", goal="review secret PR")
        inspector = runtime.process.spawn(image="base-agent:v0", goal="inspect unknown PR")
        for pid in (creator, reviewer, inspector):
            _grant_git_authority(runtime, pid)
        for pattern in ("git:workspace", "git_pr:workspace:*"):
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                ),
                actor="git-provider-test",
                require_capability=False,
            )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token
            created = runtime.git.create_pull_request(
                creator,
                "Feature",
                "Normal body",
                "main",
                "feature",
                state,
            )
        source = runtime.memory.create_object(
            reviewer,
            ObjectType.EVIDENCE,
            {"classification": "secret review"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-prebind-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            reviewer,
            [source.oid],
        )
        original = runtime.git._bind_git_lineage
        individual_bind_count = 0

        def fail_second_individual_bind(**kwargs: Any) -> None:
            nonlocal individual_bind_count
            if kwargs.get("carrier_kind") == "pull_request":
                individual_bind_count += 1
                if individual_bind_count == 2:
                    raise RuntimeError("post-write lineage settlement failed")
            original(**kwargs)

        monkeypatch.setattr(
            runtime.git,
            "_bind_git_lineage",
            fail_second_individual_bind,
        )
        with runtime.data_flow.activate(secret_context):
            with pytest.raises(
                RuntimeError,
                match="post-write lineage settlement failed",
            ):
                runtime.git.review_pull_request(
                    reviewer,
                    created["pull_request"].pr_id,
                    "comment",
                    "classified review",
                    created["operation"].after.token,
                )
        monkeypatch.setattr(runtime.git, "_bind_git_lineage", original)

        with runtime.data_flow.activate(DataFlowContext()):
            inspected = runtime.git.inspect_pull_request(
                inspector,
                created["pull_request"].pr_id,
            )
            inspect_context = runtime.data_flow.current_context()
        assert len(inspected.reviews) == 1
        assert inspect_context.labels.sensitivity.value == "secret"

        with runtime.data_flow.activate(DataFlowContext()):
            listed = runtime.git.list_pull_requests(inspector)
            list_context = runtime.data_flow.current_context()
        assert len(listed["pull_requests"][0].reviews) == 1
        assert list_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_pull_request_create_requires_repository_sink_clearance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create secret PR")
        _grant_git_authority(runtime, creator)
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git_pr:workspace:*",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="secret",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
            ),
            actor="git-provider-test",
            require_capability=False,
        )
        source = runtime.memory.create_object(
            creator,
            ObjectType.EVIDENCE,
            {"classification": "secret pull request"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-multi-sink-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            creator,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token
        with runtime.data_flow.activate(secret_context):
            with pytest.raises(CapabilityDenied, match="data-flow denied"):
                runtime.git.create_pull_request(
                    creator,
                    "Classified feature",
                    "Secret body",
                    "main",
                    "feature",
                    state,
                )
        assert _git(
            root,
            "for-each-ref",
            "refs/agent-libos/pull-requests",
        ).strip() == b""
        assert not runtime.git.provider.list_pull_request_metadata(limit=10)
        assert runtime.git.provider.repository_state().head_oid == base_oid
    finally:
        runtime.close()


def test_pull_request_prebind_failure_leaves_no_snapshot_refs_or_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create secret PR")
        _grant_git_authority(runtime, creator)
        for pattern in ("git:workspace", "git_pr:workspace:*"):
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                ),
                actor="git-provider-test",
                require_capability=False,
            )
        source = runtime.memory.create_object(
            creator,
            ObjectType.EVIDENCE,
            {"classification": "secret pull request"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-prebind-failure-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            creator,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token

        def fail_prebind(_pr_id: str) -> None:
            raise RuntimeError("pull request lineage prebind failed")

        monkeypatch.setattr(
            runtime.git,
            "_prebind_pull_request_lineage",
            fail_prebind,
        )
        with runtime.data_flow.activate(secret_context):
            with pytest.raises(
                RuntimeError,
                match="pull request lineage prebind failed",
            ):
                runtime.git.create_pull_request(
                    creator,
                    "Classified feature",
                    "Secret body",
                    "main",
                    "feature",
                    state,
                )
        assert _git(
            root,
            "for-each-ref",
            "refs/agent-libos/pull-requests",
        ).strip() == b""
        assert not runtime.git.provider.list_pull_request_metadata(limit=10)
    finally:
        runtime.close()


def test_pull_request_collection_prebind_never_downgrades_prior_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "secret-feature")
    (root / "secret-feature.txt").write_text("secret feature\n", encoding="utf-8")
    _git(root, "add", "--", "secret-feature.txt")
    _git(root, "commit", "-q", "-m", "secret feature")
    _git(root, "switch", "-q", "main")
    _git(root, "switch", "-q", "-c", "normal-feature")
    (root / "normal-feature.txt").write_text("normal feature\n", encoding="utf-8")
    _git(root, "add", "--", "normal-feature.txt")
    _git(root, "commit", "-q", "-m", "normal feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create two PRs")
        inspector = runtime.process.spawn(image="base-agent:v0", goal="list two PRs")
        _grant_git_authority(runtime, creator)
        _grant_git_authority(runtime, inspector)
        for pattern in ("git:workspace", "git_pr:workspace:*"):
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                ),
                actor="git-provider-test",
                require_capability=False,
            )
        source = runtime.memory.create_object(
            creator,
            ObjectType.EVIDENCE,
            {"classification": "secret pull request"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-collection-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            creator,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token
        with runtime.data_flow.activate(secret_context):
            secret_pr = runtime.git.create_pull_request(
                creator,
                "Secret feature",
                "Secret body",
                "main",
                "secret-feature",
                state,
            )
        with runtime.data_flow.activate(DataFlowContext()):
            normal_pr = runtime.git.create_pull_request(
                creator,
                "Normal feature",
                "Normal body",
                "main",
                "normal-feature",
                secret_pr["operation"].after.token,
            )
        assert normal_pr["pull_request"].pr_id != secret_pr["pull_request"].pr_id

        with runtime.data_flow.activate(DataFlowContext()):
            listed = runtime.git.list_pull_requests(inspector)
            list_context = runtime.data_flow.current_context()
        assert len(listed["pull_requests"]) == 2
        assert list_context.labels.sensitivity.value == "secret"
    finally:
        runtime.close()


def test_pull_request_merge_requires_repository_sink_clearance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    base_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        creator = runtime.process.spawn(image="base-agent:v0", goal="create secret PR")
        merger = runtime.process.spawn(image="base-agent:v0", goal="merge secret PR")
        _grant_git_authority(runtime, creator)
        _grant_git_authority(runtime, merger)
        for pattern in ("git:workspace", "git_pr:workspace:*"):
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity="secret",
                ),
                actor="git-provider-test",
                require_capability=False,
            )
        source = runtime.memory.create_object(
            creator,
            ObjectType.EVIDENCE,
            {"classification": "secret pull request"},
            metadata=ObjectMetadata(
                sensitivity="secret",
                origin="git-pr-merge-sink-test",
            ),
        )
        secret_context = runtime.data_flow.context_from_source_oids(
            creator,
            [source.oid],
        )
        with runtime.data_flow.activate(DataFlowContext()):
            state = runtime.git.status(creator).state.token
        with runtime.data_flow.activate(secret_context):
            created = runtime.git.create_pull_request(
                creator,
                "Classified feature",
                "Secret body",
                "main",
                "feature",
                state,
            )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern="git:workspace",
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity="normal",
            ),
            actor="git-provider-test",
            replace=True,
            require_capability=False,
        )
        pr_id = created["pull_request"].pr_id
        with pytest.raises(HumanApprovalRequired):
            runtime.git.merge_pull_request(
                merger,
                pr_id,
                created["operation"].after.token,
            )
        assert runtime.human.drain_terminal_queue(auto_approve=True)
        with runtime.data_flow.activate(DataFlowContext()):
            with pytest.raises(CapabilityDenied, match="data-flow denied"):
                runtime.git.merge_pull_request(
                    merger,
                    pr_id,
                    created["operation"].after.token,
                )
        assert _git(root, "rev-parse", "HEAD").strip().decode("ascii") == base_oid
        assert not (root / "feature.txt").exists()
        metadata = runtime.git.provider.read_pull_request_metadata(pr_id)
        assert metadata is not None
        pull_request, _bodies = runtime.git._parse_pull_request_metadata(
            metadata[0],
            expected_pr_id=pr_id,
        )
        assert pull_request.status is GitPullRequestStatus.OPEN
    finally:
        runtime.close()


def test_pull_request_create_reuses_generated_id_after_human_approval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="approve pull request creation")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="git-provider-test",
        )
        runtime.capability.issue_trusted(
            pid,
            "git_pr:workspace:*",
            [CapabilityRight.WRITE],
            effect="ask",
            issued_by="git-provider-test",
        )
        state = runtime.git.status(pid).state.token

        with pytest.raises(HumanApprovalRequired):
            runtime.git.create_pull_request(
                pid,
                "Feature",
                "Adds feature.txt",
                "main",
                "feature",
                state,
            )
        assert runtime.human.drain_terminal_queue(auto_approve=True)

        created = runtime.git.create_pull_request(
            pid,
            "Feature",
            "Adds feature.txt",
            "main",
            "feature",
            state,
        )
        pr_id = created["pull_request"].pr_id
        assert pr_id.startswith("pr_")
        assert any(
            record.action == "primitive.git.create_pull_request"
            and record.target == runtime.git.pull_request_resource(pr_id)
            for record in runtime.audit.trace(actor=pid)
        )
        assert any(
            event.type == EventType.EXTERNAL_WRITE
            and event.source == pid
            and event.payload.get("operation") == "create_pull_request"
            for event in runtime.events.list(
                target=runtime.git.pull_request_resource(pr_id)
            )
        )
    finally:
        runtime.close()


def test_pull_request_metadata_failure_after_write_is_retained_as_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text("feature\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", "feature")
    _git(root, "switch", "-q", "main")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="retain ambiguous PR write")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        created = runtime.git.create_pull_request(
            pid,
            "Feature",
            "Adds feature.txt",
            "main",
            "feature",
            state,
        )
        pr_id = created["pull_request"].pr_id
        original = runtime.git.provider.write_pull_request_metadata

        def write_then_report_wrong_digest(*args: Any, **kwargs: Any) -> str:
            original(*args, **kwargs)
            return "0" * 64

        monkeypatch.setattr(
            runtime.git.provider,
            "write_pull_request_metadata",
            write_then_report_wrong_digest,
        )

        with pytest.raises(GitError) as exc_info:
            runtime.git.review_pull_request(
                pid,
                pr_id,
                "comment",
                "persisted despite error",
                created["operation"].after.token,
            )

        assert exc_info.value.code == GitErrorCode.UNKNOWN_EFFECT.value
        assert not isinstance(exc_info.value, GitProviderEffectNotStarted)
        metadata = runtime.git.provider.read_pull_request_metadata(pr_id)
        assert metadata is not None and b"persisted despite error" in metadata[0]
        effect = runtime.store.list_external_effects(pid=pid)[-1]
        assert effect.provider == "git"
        assert effect.transaction_state == "unknown"
    finally:
        runtime.close()


@pytest.mark.parametrize("strategy", ["fast_forward", "merge", "squash"])
def test_simulated_pull_request_merge_strategies(
    tmp_path: Path,
    strategy: str,
) -> None:
    root = tmp_path / strategy
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "feature")
    (root / "feature.txt").write_text(f"{strategy}\n", encoding="utf-8")
    _git(root, "add", "--", "feature.txt")
    _git(root, "commit", "-q", "-m", f"feature {strategy}")
    feature_oid = _git(root, "rev-parse", "HEAD").strip().decode("ascii")
    _git(root, "switch", "-q", "main")

    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal=f"merge PR by {strategy}")
        _grant_git_authority(runtime, pid)
        state = runtime.git.status(pid).state.token
        created = runtime.git.create_pull_request(
            pid,
            f"Feature {strategy}",
            "strategy coverage",
            "main",
            "feature",
            state,
        )
        pr_id = created["pull_request"].pr_id
        merged = _with_auto_approvals(
            runtime,
            lambda: runtime.git.merge_pull_request(
                pid,
                pr_id,
                created["operation"].after.token,
                strategy=strategy,
            ),
        )
        pull_request = merged["pull_request"]
        assert pull_request.status is GitPullRequestStatus.MERGED
        assert pull_request.merged_oid == _git(root, "rev-parse", "HEAD").strip().decode("ascii")
        assert (root / "feature.txt").read_text(encoding="utf-8") == f"{strategy}\n"
        parent_count = len(_git(root, "show", "-s", "--format=%P", "HEAD").split())
        if strategy == "fast_forward":
            assert pull_request.merged_oid == feature_oid
            assert parent_count == 1
        elif strategy == "merge":
            assert pull_request.merged_oid != feature_oid
            assert parent_count == 2
        else:
            assert pull_request.merged_oid != feature_oid
            assert parent_count == 1
    finally:
        runtime.close()


def test_filesystem_and_direct_raw_shell_git_respect_typed_git_controls(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    shell_config = replace(DEFAULT_CONFIG.shell, default_policy_level="always_allow")
    config = replace(_runtime_config(), shell=shell_config)
    runtime = Runtime.open(
        ":memory:",
        config=config,
        substrate=LocalResourceProviderSubstrate(root, git_config=config.git),
        module_manifests=(),
    )
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="attempt Git bypasses")
        runtime.filesystem.grant_directory(
            pid,
            ".",
            [CapabilityRight.READ, CapabilityRight.WRITE],
            issued_by="git-provider-test",
        )
        runtime.capability.issue_trusted(
            pid,
            "shell:git",
            [CapabilityRight.EXECUTE],
            issued_by="git-provider-test",
        )
        with pytest.raises(CapabilityDenied):
            runtime.git.status(pid)
        with pytest.raises(CapabilityDenied, match="Git metadata"):
            runtime.filesystem.read_text(pid, ".git/config")
        with pytest.raises(ValidationError, match="typed git_"):
            runtime.shell.run(pid, ["GiT.ExE", "reset", "--hard"])
        hardened = runtime.shell.run(pid, ["GIT.EXE", "diff"])
        assert hardened.returncode == 0
        assert hardened.argv == ["git", "diff"]
        if os.name != "nt":
            runtime.shell.grant_policy(
                pid,
                runtime.config.shell.always_allow_level,
                issued_by="git-provider-test",
            )
            with pytest.raises(ValidationError, match="typed git_"):
                runtime.shell.run(pid, ["env", "git", "branch", "wrapper-created"])
            wrapped_ref = subprocess.run(
                ["git", "show-ref", "--verify", "refs/heads/wrapper-created"],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert wrapped_ref.returncode != 0
    finally:
        runtime.close()


def test_git_model_tools_are_visible_but_still_require_capability(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="use typed Git tools")
        denied = runtime.tools.call(pid, "git_status", {})
        assert not denied.ok
        _grant_git_authority(runtime, pid)
        status = runtime.tools.call(pid, "git_status", {})
        assert status.ok, status.error
        assert status.payload["state"]["token"]

        invalid = runtime.tools.call(
            pid,
            "git_fetch",
            {
                "remote": "origin",
                "expected_state_token": status.payload["state"]["token"],
                "url": "https://example.test/secret-token",
            },
        )
        assert not invalid.ok
        assert "secret-token" not in str(invalid.error)
    finally:
        runtime.close()
