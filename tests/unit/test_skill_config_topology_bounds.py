from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG, load_config_file


def test_skill_topology_and_catalog_bounds_have_release_defaults() -> None:
    assert DEFAULT_CONFIG.skills.max_package_directories == 256
    assert DEFAULT_CONFIG.skills.max_package_depth == 32
    assert DEFAULT_CONFIG.skills.catalog_scan_limit == 1_000


@pytest.mark.parametrize(
    "field",
    ("max_package_directories", "max_package_depth", "catalog_scan_limit"),
)
@pytest.mark.parametrize("invalid", (True, 0, -1))
def test_skill_topology_and_catalog_bounds_reject_bool_and_nonpositive_values(
    field: str,
    invalid: object,
) -> None:
    with pytest.raises((PydanticValidationError, ValueError), match=field):
        replace(
            DEFAULT_CONFIG,
            skills=replace(DEFAULT_CONFIG.skills, **{field: invalid}),
        )


def test_skill_topology_and_catalog_bounds_load_as_strict_yaml_integers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skills.yaml"
    path.write_text(
        "\n".join(
            (
                "skills:",
                "  max_package_directories: 17",
                "  max_package_depth: 9",
                "  catalog_scan_limit: 321",
            )
        ),
        encoding="utf-8",
    )

    config = load_config_file(path)

    assert config.skills.max_package_directories == 17
    assert config.skills.max_package_depth == 9
    assert config.skills.catalog_scan_limit == 321


@pytest.mark.parametrize(
    "field",
    ("max_package_directories", "max_package_depth", "catalog_scan_limit"),
)
def test_skill_topology_and_catalog_bounds_reject_yaml_bool(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / f"invalid-{field}.yaml"
    path.write_text(f"skills:\n  {field}: true\n", encoding="utf-8")

    with pytest.raises(PydanticValidationError) as exc_info:
        load_config_file(path)

    assert any(
        error["loc"] == ("skills", field) and error["type"] == "int_type"
        for error in exc_info.value.errors()
    )


def test_skill_topology_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "unknown-skill-bound.yaml"
    path.write_text("skills:\n  package_directory_limit: 5\n", encoding="utf-8")

    with pytest.raises(PydanticValidationError, match="package_directory_limit"):
        load_config_file(path)
