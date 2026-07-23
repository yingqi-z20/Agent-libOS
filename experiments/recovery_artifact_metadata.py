from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

from agent_libos.utils.ids import new_id, utc_now
from experiments.run_benchmark import REPO_ROOT, _git_provenance


def new_recovery_run_identity() -> tuple[str, str]:
    """Return an opaque run id and its UTC start timestamp."""

    return new_id("benchmark_run"), utc_now()


def build_recovery_artifact_metadata(
    *,
    benchmark_id: str,
    run_id: str,
    started_at: str,
    selected_profile: str,
    profile_defaults: dict[str, int],
    explicit_overrides: dict[str, int],
    effective_parameters: dict[str, Any],
    source_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Build non-secret identity, invocation, source, and environment evidence."""

    selected_source_paths = {Path(__file__).resolve()}
    selected_source_paths.update(path.resolve() for path in source_paths)
    source_entries = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(selected_source_paths, key=lambda item: item.as_posix())
    ]
    aggregate = hashlib.sha256()
    for entry in source_entries:
        aggregate.update(entry["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(entry["sha256"].encode("ascii"))
        aggregate.update(b"\0")

    effective_profile_parameters = {
        key: effective_parameters[key] for key in profile_defaults
    }
    profile_matches_parameters = effective_profile_parameters == profile_defaults
    return {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "started_at": started_at,
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
            "git": _git_provenance(),
            "benchmark_sources": source_entries,
            "benchmark_source_sha256": aggregate.hexdigest(),
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
