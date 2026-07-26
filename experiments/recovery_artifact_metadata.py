from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_libos.utils.ids import new_id, utc_now
from experiments.run_benchmark import (
    REPO_ROOT,
    _ProvenanceBudget,
    _git_provenance,
    _sha256_file,
)


@dataclass(frozen=True)
class RecoverySourceEntry:
    path: str
    sha256: str


@dataclass(frozen=True)
class RecoveryRunIdentity:
    run_id: str
    started_at: str
    source_entries: tuple[RecoverySourceEntry, ...]
    source_sha256: str
    git_provenance: dict[str, Any]


def new_recovery_run_identity(
    *,
    source_paths: tuple[Path, ...],
) -> RecoveryRunIdentity:
    """Capture identity and immutable source digests before measured work."""

    selected_source_paths: set[Path] = set()
    for raw_path in (Path(__file__), *source_paths):
        absolute = raw_path.absolute()
        if absolute.is_symlink():
            raise RuntimeError(
                f"recovery benchmark source may not be a symlink: {absolute}"
            )
        selected_source_paths.add(absolute.resolve(strict=True))
    source_entries: list[RecoverySourceEntry] = []
    source_budget = _ProvenanceBudget()
    for path in sorted(selected_source_paths, key=lambda item: item.as_posix()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"recovery benchmark source is not a regular file: {path}")
        try:
            relative = path.relative_to(REPO_ROOT).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"recovery benchmark source is outside the repository: {path}"
            ) from exc
        source_entries.append(
            RecoverySourceEntry(
                path=relative,
                sha256=_sha256_file(path, budget=source_budget),
            )
        )
    aggregate = hashlib.sha256()
    for entry in source_entries:
        aggregate.update(entry.path.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(entry.sha256.encode("ascii"))
        aggregate.update(b"\0")
    return RecoveryRunIdentity(
        run_id=new_id("benchmark_run"),
        started_at=utc_now(),
        source_entries=tuple(source_entries),
        source_sha256=aggregate.hexdigest(),
        git_provenance=_git_provenance(),
    )


def build_recovery_artifact_metadata(
    *,
    benchmark_id: str,
    identity: RecoveryRunIdentity,
    selected_profile: str,
    profile_defaults: dict[str, int],
    explicit_overrides: dict[str, int],
    effective_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build non-secret identity, invocation, source, and environment evidence."""

    source_entries = [
        {
            "path": entry.path,
            "sha256": entry.sha256,
        }
        for entry in identity.source_entries
    ]

    effective_profile_parameters = {
        key: effective_parameters[key] for key in profile_defaults
    }
    profile_matches_parameters = effective_profile_parameters == profile_defaults
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "run_id": identity.run_id,
        "started_at": identity.started_at,
        "completed_at": utc_now(),
        "invocation": {
            "selected_profile": selected_profile,
            "classification": (
                "named-profile" if not explicit_overrides else "custom-overrides"
            ),
            "named_profile_evidence": not explicit_overrides
            and profile_matches_parameters,
            "profile_matches_parameters": profile_matches_parameters,
            "profile_defaults": profile_defaults,
            "explicit_overrides": explicit_overrides,
            "effective_parameters": effective_parameters,
        },
        "provenance": {
            "schema_version": 1,
            "git": identity.git_provenance,
            "benchmark_sources": source_entries,
            "benchmark_source_sha256": identity.source_sha256,
            "environment": {
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "agent_libos_version": _distribution_version("agent-libos"),
            },
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
