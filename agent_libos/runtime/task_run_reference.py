from __future__ import annotations

from typing import Any


TASK_RUN_REFERENCE_KEY = "$task_run_ref"
TASK_RUN_REFERENCE_SCHEMA_VERSION = 1
_TASK_RUN_REFERENCE_FIELDS = frozenset(
    {"run_id", "payload_sha256", "schema_version"}
)
_LOWER_HEX = frozenset("0123456789abcdef")


def is_task_run_reference_payload(value: Any) -> bool:
    """Return whether *value* is the exact non-secret TaskRun goal marker.

    This predicate intentionally accepts only strict JSON dictionaries and an
    exact field set.  Checkpoint forks and committed images use it to remove a
    source-run reference without accidentally treating arbitrary user payloads
    containing a similarly named key as runtime metadata.
    """

    if type(value) is not dict or set(value) != {TASK_RUN_REFERENCE_KEY}:
        return False
    reference = value.get(TASK_RUN_REFERENCE_KEY)
    if type(reference) is not dict or set(reference) != _TASK_RUN_REFERENCE_FIELDS:
        return False
    run_id = reference.get("run_id")
    payload_sha256 = reference.get("payload_sha256")
    schema_version = reference.get("schema_version")
    return (
        type(run_id) is str
        and bool(run_id)
        and run_id == run_id.strip()
        and type(payload_sha256) is str
        and len(payload_sha256) == 64
        and all(character in _LOWER_HEX for character in payload_sha256)
        and type(schema_version) is int
        and schema_version == TASK_RUN_REFERENCE_SCHEMA_VERSION
    )


__all__ = [
    "TASK_RUN_REFERENCE_KEY",
    "TASK_RUN_REFERENCE_SCHEMA_VERSION",
    "is_task_run_reference_payload",
]
