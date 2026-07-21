from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import importlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TypeVar

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, GitDefaults
from agent_libos.models import (
    CapabilityRight,
    EventType,
    GitErrorCode,
    GitPullRequestStatus,
    ObjectMetadata,
    ObjectType,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    GitError,
    HumanApprovalRequired,
    ValidationError,
)
from agent_libos.primitives.git import GitPrimitive
from agent_libos.substrate import (
    GitProviderEffectNotStarted,
    LocalGitProvider,
    LocalResourceProviderSubstrate,
)


_T = TypeVar("_T")


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


def test_worktree_git_write_rebinds_stale_trusted_file_label(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    _git(root, "switch", "-q", "-c", "untrusted")
    (root / "tracked.txt").write_text("replacement\n", encoding="utf-8")
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
    secret_path.write_text("classified\n", encoding="utf-8")
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
    tracked.write_text("classified stash\n", encoding="utf-8")
    runtime = _open_runtime(root)
    try:
        writer = runtime.process.spawn(image="base-agent:v0", goal="label stash bytes")
        stasher = runtime.process.spawn(image="base-agent:v0", goal="round-trip stash")
        _grant_git_authority(runtime, writer)
        _grant_git_authority(runtime, stasher)
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
    secret_path.write_text("classified worktree\n", encoding="utf-8")
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


@pytest.mark.timeout(240 if os.name == "nt" else 120)
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


def test_fast_forward_pull_from_configured_bare_remote(tmp_path: Path) -> None:
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

    git_config = replace(DEFAULT_CONFIG.git, allow_file_remotes=True)
    runtime = _open_runtime(root, git=git_config)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="fast-forward pull")
        _grant_git_authority(runtime, pid, remote="origin")
        state = runtime.git.status(pid).state.token
        pulled = runtime.git.pull(
            pid,
            "origin",
            state,
            branch="main",
            strategy="ff_only",
        )
        assert pulled.created_oid == remote_oid
        assert (root / "remote.txt").read_text(encoding="utf-8") == "from remote\n"
    finally:
        runtime.close()


def test_pull_fetches_only_the_capability_scoped_branch(tmp_path: Path) -> None:
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

        runtime.git.pull(pid, "origin", state, branch="main", strategy="ff_only")

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


def test_filesystem_and_raw_shell_cannot_bypass_typed_git_boundary(tmp_path: Path) -> None:
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
