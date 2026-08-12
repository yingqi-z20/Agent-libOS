from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import check_changed_whitespace


_ZERO_SHA = "0" * 40


def _git(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, message: str) -> None:
    _git(repository, "add", "--all")
    _git(
        repository,
        "-c",
        "user.name=Whitespace Test",
        "-c",
        "user.email=whitespace@example.invalid",
        "commit",
        "-m",
        message,
    )


def test_zero_before_uses_default_branch_merge_base_not_empty_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "symbolic-ref", "HEAD", "refs/heads/main")
    tracked = repository / "tracked.txt"
    tracked.write_text("historical trailing whitespace   \n", encoding="utf-8")
    _commit(repository, "historical base")
    base = _git(repository, "rev-parse", "HEAD")

    _git(repository, "switch", "--create", "feature")
    _git(repository, "update-ref", "refs/remotes/origin/main", base)
    (repository / "clean.txt").write_text("clean change\n", encoding="utf-8")
    _commit(repository, "clean feature change")

    assert (
        check_changed_whitespace.main(
            [
                "--repository",
                str(repository),
                "--base-sha",
                _ZERO_SHA,
                "--default-branch",
                "main",
            ]
        )
        == 0
    )

    (repository / "clean.txt").write_text("new trailing whitespace   \n", encoding="utf-8")
    _commit(repository, "bad feature change")
    assert (
        check_changed_whitespace.main(
            [
                "--repository",
                str(repository),
                "--base-sha",
                _ZERO_SHA,
                "--default-branch",
                "main",
            ]
        )
        != 0
    )


def test_explicit_unavailable_base_fails_closed(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init")
    _git(repository, "symbolic-ref", "HEAD", "refs/heads/main")
    (repository / "tracked.txt").write_text("clean\n", encoding="utf-8")
    _commit(repository, "root")

    assert (
        check_changed_whitespace.main(
            [
                "--repository",
                str(repository),
                "--base-sha",
                "f" * 40,
                "--default-branch",
                "main",
            ]
        )
        == 2
    )
