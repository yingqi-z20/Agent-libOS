from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import GitError
from agent_libos.substrate.git import LocalGitProvider


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MAX_UNTRACKED_FILE_BYTES = 16 * 1024 * 1024
_MAX_UNTRACKED_TOTAL_BYTES = 64 * 1024 * 1024


def capture_source_provenance() -> dict[str, Any]:
    """Bind a live report to one bounded Git working-tree identity."""

    provider = LocalGitProvider(REPOSITORY_ROOT, config=DEFAULT_CONFIG.git)
    try:
        for operation in ("repository_info", "list_refs", "status", "diff"):
            provider.validate_read_only_operation(operation)
        commit = _git(provider, "rev-parse", "HEAD").strip()
        status = _git(
            provider,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        diff = _git(
            provider,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "HEAD",
            "--",
        )
        untracked = _git(
            provider,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        digest = hashlib.sha256()
        digest.update(commit)
        digest.update(b"\0status\0")
        digest.update(status)
        digest.update(b"\0diff\0")
        digest.update(diff)
        total = 0
        for raw_name in sorted(item for item in untracked.split(b"\0") if item):
            relative = raw_name.decode("utf-8", errors="surrogateescape")
            source = REPOSITORY_ROOT / relative
            digest.update(b"\0untracked\0")
            digest.update(raw_name)
            if source.is_symlink():
                digest.update(b"\0symlink\0")
                digest.update(os.readlink(source).encode("utf-8", errors="surrogateescape"))
                continue
            if not source.is_file():
                raise RuntimeError("untracked provenance entry is not a regular file")
            size = source.stat().st_size
            if size > _MAX_UNTRACKED_FILE_BYTES:
                raise RuntimeError("untracked provenance file exceeds the size limit")
            total += size
            if total > _MAX_UNTRACKED_TOTAL_BYTES:
                raise RuntimeError("untracked provenance exceeds the aggregate size limit")
            digest.update(source.read_bytes())
        return {
            "schema_version": 1,
            "available": True,
            "commit": commit.decode("ascii") or None,
            "dirty": bool(status),
            "working_tree_sha256": digest.hexdigest(),
        }
    except (GitError, OSError, RuntimeError, UnicodeError):
        return {
            "schema_version": 1,
            "available": False,
            "commit": None,
            "dirty": None,
            "working_tree_sha256": None,
        }


def build_source_provenance(
    start: dict[str, Any],
    end: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "start": start,
        "end": end,
        "stable": start == end and start.get("available") is True,
    }


def valid_stable_source_provenance(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return False
    start = value.get("start")
    end = value.get("end")
    return bool(
        value.get("stable") is True
        and isinstance(start, dict)
        and isinstance(end, dict)
        and start == end
        and start.get("available") is True
        and isinstance(start.get("commit"), str)
        and len(start["commit"]) in {40, 64}
        and isinstance(start.get("dirty"), bool)
        and isinstance(start.get("working_tree_sha256"), str)
        and len(start["working_tree_sha256"]) == 64
    )


def _git(provider: LocalGitProvider, *args: str) -> bytes:
    result = provider.run(
        args,
        read_only=True,
        max_output_bytes=DEFAULT_CONFIG.git.output_hard_limit_bytes,
    )
    if result.returncode != 0:
        raise RuntimeError("Git provenance command failed")
    return result.stdout
