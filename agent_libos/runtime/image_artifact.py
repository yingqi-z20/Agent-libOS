from __future__ import annotations

import hashlib
from typing import Any

from agent_libos.config import AgentLibOSConfig
from agent_libos.models import AgentImage
from agent_libos.models.exceptions import NotFound
from agent_libos.storage.repositories import ExtensionRepository
from agent_libos.utils.serde import dumps


class ImageArtifactLoader:
    """Validate and load immutable image boot artifacts from persistence."""

    def __init__(
        self,
        extensions: ExtensionRepository,
        config: AgentLibOSConfig,
    ) -> None:
        self._extensions = extensions
        self._config = config

    def load(
        self,
        image: AgentImage,
        *,
        expected_kind: str | None = None,
    ) -> dict[str, Any]:
        artifact_id = str(image.boot.get("artifact_id") or "")
        expected_sha256 = str(image.boot.get("artifact_sha256") or "")
        selected_kind = expected_kind or str(image.boot.get("kind") or "")
        found = self._extensions.get_image_artifact(artifact_id)
        if found is None:
            raise NotFound(f"image artifact not found: {artifact_id}")
        artifact, metadata = found
        actual_kind = str(artifact.get("kind") or "")
        persisted_id = str(metadata.get("artifact_id") or "")
        persisted_kind = str(metadata.get("kind") or "")
        if persisted_id != artifact_id:
            raise RuntimeError(
                f"image artifact id mismatch: {persisted_id} != {artifact_id}"
            )
        if persisted_kind != actual_kind:
            raise RuntimeError(
                "image artifact persisted kind mismatch: "
                f"{persisted_kind} != {actual_kind}"
            )
        if selected_kind and (
            actual_kind != selected_kind or persisted_kind != selected_kind
        ):
            raise RuntimeError(
                f"image artifact kind mismatch: {actual_kind} != {selected_kind}"
            )
        expected_version = (
            self._config.image_commit.artifact_version
            if actual_kind == "checkpoint_commit"
            else 1
        )
        if artifact.get("artifact_version") != expected_version:
            raise RuntimeError(
                "image artifact version mismatch: "
                f"{artifact.get('artifact_version')} != {expected_version}"
            )
        actual_sha256 = hashlib.sha256(
            dumps(artifact).encode("utf-8")
        ).hexdigest()
        persisted_sha256 = str(metadata.get("sha256") or "")
        if expected_sha256 and persisted_sha256 != expected_sha256:
            raise RuntimeError(f"image artifact hash mismatch for {artifact_id}")
        if persisted_sha256 != actual_sha256:
            raise RuntimeError(
                f"image artifact content hash mismatch for {artifact_id}"
            )
        return artifact


__all__ = ["ImageArtifactLoader"]
