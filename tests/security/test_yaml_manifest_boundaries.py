from __future__ import annotations

from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models.exceptions import ValidationError


def test_skill_manifest_rejects_non_string_mapping_key_without_type_error(
    tmp_path: Path,
) -> None:
    package = tmp_path / "bad-key-skill"
    package.mkdir()
    package.joinpath("SKILL.md").write_text(
        """---
name: bad-key-skill
description: Reject non-string YAML keys.
metadata:
  1: invalid
---
Instructions.
""",
        encoding="utf-8",
    )
    runtime = Runtime.open(tmp_path / "skill.sqlite")
    try:
        with pytest.raises(
            ValidationError,
            match="mapping keys must be strings",
        ):
            runtime.skills.validate_package_path(package)
        assert runtime.store.get_skill("bad-key-skill") is None
    finally:
        runtime.close()


def test_image_manifest_rejects_non_string_mapping_key_without_type_error(
    tmp_path: Path,
) -> None:
    package = tmp_path / "bad-key-image"
    package.mkdir()
    package.joinpath("IMAGE.yaml").write_text(
        """image:
  1: invalid
""",
        encoding="utf-8",
    )
    runtime = Runtime.open(tmp_path / "image.sqlite")
    try:
        with pytest.raises(
            ValidationError,
            match="mapping keys must be strings",
        ):
            runtime.image_registry.validate_package_path(package)
        assert runtime.store.list_image_artifacts() == []
    finally:
        runtime.close()


def test_mcp_manifest_rejects_non_string_mapping_key_without_type_error(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "mcp.sqlite")
    try:
        with pytest.raises(
            ValidationError,
            match="mapping keys must be strings",
        ):
            runtime.mcp.register_server_from_yaml_text(
                "mcp_server:\n  1: invalid\n",
                actor="cli",
                require_capability=False,
            )
        assert runtime.store.list_mcp_servers() == []
    finally:
        runtime.close()
