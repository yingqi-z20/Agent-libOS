"""Tool failures expose identifier-shaped diagnostic codes, never exception text.

A model that only sees ``validation_error: ToolExecutionError (correlation_id=...)``
cannot tell which argument was wrong and retries the same malformed call; one
real run repeated a rejected ``git_diff`` seven times.  The boundary keeps the
exception text hashed but lets tools attach short machine codes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import CapabilityRight
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import public_identifier_details


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


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


def _open_runtime(root: Path) -> Runtime:
    config = replace(
        DEFAULT_CONFIG,
        modules=replace(
            DEFAULT_CONFIG.modules,
            manifest_paths=(),
            trusted_modules=(),
            trusted_sha256=(),
        ),
    )
    return Runtime.open(
        ":memory:",
        config=config,
        substrate=LocalResourceProviderSubstrate(root, git_config=DEFAULT_CONFIG.git),
        module_manifests=(),
    )


def test_public_identifier_details_keep_codes_and_drop_text() -> None:
    details = {
        "git_error_code": "invalid_ref",
        "hint": "worktree_scope_requires_null_base_and_head",
        "operation": None,
        "message": "worktree diff does not accept base/head",
        "path": "src/a b.py",
        "stderr": "fatal: bad revision",
        "correlation_id": "corr_x",
        "count": 3,
    }

    assert public_identifier_details(details) == {
        "git_error_code": "invalid_ref",
        "hint": "worktree_scope_requires_null_base_and_head",
    }
    assert public_identifier_details({"x" * 65: "ok", "ok": "y" * 65}) == {}


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,
    reason="git is required",
)
def test_git_diff_scope_failure_tells_the_model_why(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="inspect the diff")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ, CapabilityRight.DIFF],
            issued_by="tool-failure-test",
        )
        runtime.filesystem.grant_directory(
            pid, ".", [CapabilityRight.READ], issued_by="tool-failure-test"
        )

        failed = runtime.tools.call(
            pid, "git_diff", {"scope": "worktree", "base": "HEAD", "head": "HEAD"}
        )

        assert not failed.ok
        rendered = json.dumps(failed.payload, sort_keys=True)
        assert "worktree_scope_requires_null_base_and_head" in rendered
        assert "invalid_ref" in rendered
        assert "does not accept" not in rendered, "exception text stays hashed"

        repaired = runtime.tools.call(
            pid, "git_diff", {"scope": "worktree", "base": None, "head": None}
        )
        assert repaired.ok, repaired.error
    finally:
        runtime.close()


@pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,
    reason="git is required",
)
def test_reported_main_worktree_identity_is_accepted_as_worktree_id(tmp_path: Path) -> None:
    """``git_status`` reports the main worktree's identity digest; reuse must work.

    A model copied that digest into ``git_diff.worktree_id`` and was rejected
    with ``invalid_path`` twice in one real run.
    """

    root = tmp_path / "repo"
    _init_repository(root)
    runtime = _open_runtime(root)
    try:
        pid = runtime.process.spawn(image="coding-agent:v0", goal="inspect the diff")
        runtime.capability.issue_trusted(
            pid,
            "git:workspace",
            [CapabilityRight.READ, CapabilityRight.DIFF],
            issued_by="tool-failure-test",
        )
        runtime.filesystem.grant_directory(
            pid, ".", [CapabilityRight.READ], issued_by="tool-failure-test"
        )
        status = runtime.tools.call(pid, "git_status", {})
        assert status.ok, status.error
        identity = status.payload["worktree_id"]
        assert identity != "main" and len(identity) == 32

        by_identity = runtime.tools.call(
            pid, "git_diff", {"scope": "worktree", "worktree_id": identity}
        )
        assert by_identity.ok, by_identity.error

        rejected = runtime.tools.call(
            pid, "git_diff", {"scope": "worktree", "worktree_id": "0" * 32}
        )
        assert not rejected.ok
        rendered = json.dumps(rejected.payload, sort_keys=True)
        assert "invalid_path" in rendered
        assert "worktree_id_must_be_main_or_a_managed_wt_id" in rendered
    finally:
        runtime.close()
