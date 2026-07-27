from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.snapshots.models import SNAPSHOT_SCHEMA_VERSION, ProcessSnapshot
from agent_libos.utils.serde import bounded_json_loads, dumps


class SnapshotVersionError(ValidationError):
    pass


class SnapshotCodec:
    """Strict codec for the 0.3 typed process snapshot format."""

    schema_version = SNAPSHOT_SCHEMA_VERSION
    max_document_bytes = 16_777_216

    @classmethod
    def decode_mapping(cls, value: Mapping[str, Any]) -> ProcessSnapshot:
        version = value.get("version")
        if version != cls.schema_version:
            raise SnapshotVersionError(
                f"unsupported snapshot version: {version!r}; expected {cls.schema_version}"
            )
        return ProcessSnapshot.from_mapping(value)

    @classmethod
    def encode_mapping(cls, snapshot: ProcessSnapshot) -> dict[str, Any]:
        if snapshot.header.schema_version != cls.schema_version:
            raise SnapshotVersionError(
                "cannot encode snapshot version "
                f"{snapshot.header.schema_version}; expected {cls.schema_version}"
            )
        return snapshot.to_mapping()

    @classmethod
    def canonicalize_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> tuple[ProcessSnapshot, dict[str, Any]]:
        """Validate once and transfer the codec-owned canonical values."""

        snapshot = cls.decode_mapping(value)
        return snapshot, snapshot.to_mapping(copy_values=False)

    @classmethod
    def dumps(cls, snapshot: ProcessSnapshot) -> str:
        return dumps(cls.encode_mapping(snapshot))

    @classmethod
    def loads(cls, value: str) -> ProcessSnapshot:
        try:
            decoded = bounded_json_loads(
                value,
                max_bytes=cls.max_document_bytes,
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"invalid snapshot document JSON: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise ValidationError("snapshot document must be an object")
        return cls.decode_mapping(decoded)
