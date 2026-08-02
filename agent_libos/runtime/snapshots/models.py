"""Compatibility exports for snapshot domain models.

New persistence and orchestration code should import these value objects from
``agent_libos.models.snapshot``. Keeping this module avoids breaking existing
runtime-local imports while preserving the storage-to-runtime dependency
boundary.
"""

from agent_libos.models.snapshot import (
    ExecRollbackState,
    SNAPSHOT_SCHEMA_VERSION,
    ProcessSnapshot,
    SnapshotHeader,
    SnapshotRows,
)

__all__ = [
    "ExecRollbackState",
    "SNAPSHOT_SCHEMA_VERSION",
    "ProcessSnapshot",
    "SnapshotHeader",
    "SnapshotRows",
]
