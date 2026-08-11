from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


_FULL_OBJECT_ID = re.compile(r"[0-9a-fA-F]{40,64}")


def _git(
    repository: Path,
    *args: str,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
        input=input_text,
    )


def _commit(repository: Path, revision: str) -> str | None:
    resolved = _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.returncode != 0:
        return None
    return resolved.stdout.strip()


def resolve_diff_base(
    repository: Path,
    *,
    base_sha: str,
    default_branch: str,
) -> str:
    """Resolve a bounded baseline for a changed-lines whitespace check.

    GitHub reports an all-zero ``before`` object for the first push of a new
    branch. In that case, comparing with the empty tree would re-lint every
    historical line. Prefer the default branch merge base, then HEAD's parent,
    and use the empty tree only for a genuine root commit.
    """

    selected_base = base_sha.strip()
    if selected_base and set(selected_base) != {"0"}:
        if _FULL_OBJECT_ID.fullmatch(selected_base) is None:
            raise ValueError("base SHA must be a full hexadecimal object id")
        resolved = _commit(repository, selected_base)
        if resolved is None:
            raise ValueError("base SHA is not an available commit")
        return resolved

    head = _commit(repository, "HEAD")
    if head is None:
        raise ValueError("HEAD is not an available commit")

    branch = default_branch.strip()
    if branch:
        for candidate in (
            f"refs/remotes/origin/{branch}",
            f"refs/heads/{branch}",
        ):
            if _commit(repository, candidate) is None:
                continue
            merge_base = _git(repository, "merge-base", candidate, "HEAD")
            if merge_base.returncode == 0:
                resolved = merge_base.stdout.strip()
                if resolved and resolved != head:
                    return resolved

    parent = _commit(repository, "HEAD^")
    if parent is not None:
        return parent

    empty_tree = _git(repository, "mktree", input_text="")
    if empty_tree.returncode != 0 or not empty_tree.stdout.strip():
        raise ValueError("could not create the empty-tree baseline")
    return empty_tree.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whitespace only in lines changed from a safe Git baseline."
    )
    parser.add_argument("--base-sha", default="")
    parser.add_argument("--default-branch", default="")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    repository = args.repository.resolve()
    try:
        base = resolve_diff_base(
            repository,
            base_sha=args.base_sha,
            default_branch=args.default_branch,
        )
    except ValueError as exc:
        print(f"whitespace baseline error: {exc}", file=sys.stderr)
        return 2

    check = subprocess.run(
        ["git", "diff", "--check", base, "HEAD", "--"],
        cwd=repository,
        check=False,
    )
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())
