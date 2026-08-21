from __future__ import annotations

from dataclasses import fields
import re
from pathlib import Path
from typing import get_type_hints

from pydantic import StrictBool, StrictFloat, StrictInt

from agent_libos.config import (
    DEFAULT_CONFIG,
    McpDefaults,
    SemanticDefaults,
    load_config_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DOC = ROOT / "docs" / "configuration.md"
CLI_DOC = ROOT / "docs" / "cli.md"
_SEMANTIC_STRICT_FIELD = re.compile(r"`semantic\.([a-z][a-z0-9_]*)`")


def test_configuration_guide_delegates_exact_inventory_to_generated_reference() -> None:
    text = CONFIG_DOC.read_text(encoding="utf-8")

    assert "[exact configuration reference](configuration_reference.md)" in text
    assert "single exhaustive field inventory" in text
    assert "| Group | Fields |" not in text
    assert "| `llm.profiles.<profile_id>` |" not in text


def test_configuration_reference_lists_every_strict_semantic_field() -> None:
    text = CONFIG_DOC.read_text(encoding="utf-8")
    strict_section = text.split("strict set comprises:", 1)[1].split(
        "Strict integer fields", 1
    )[0]
    annotations = get_type_hints(SemanticDefaults, include_extras=True)

    documented = _SEMANTIC_STRICT_FIELD.findall(strict_section)
    expected = [
        field.name
        for field in fields(SemanticDefaults)
        if annotations[field.name] in {StrictInt, StrictFloat}
    ]

    assert documented == expected


def test_configuration_reference_lists_mcp_strict_fields_by_annotation() -> None:
    text = CONFIG_DOC.read_text(encoding="utf-8")
    strict_section = text.split("MCP's strict declarations", 1)[1].split(
        "Strict integer fields", 1
    )[0]
    normalized = " ".join(strict_section.split())
    declarations = list(
        re.finditer(
            r"MCP `(?P<kind>StrictInt|StrictFloat|StrictBool)` "
            r"\((?P<count>[0-9]+) fields\):",
            normalized,
        )
    )
    annotations = get_type_hints(McpDefaults, include_extras=True)
    strict_annotations = {
        "StrictInt": StrictInt,
        "StrictFloat": StrictFloat,
        "StrictBool": StrictBool,
    }

    assert [match.group("kind") for match in declarations] == list(
        strict_annotations
    )
    for index, declaration in enumerate(declarations):
        body_end = (
            declarations[index + 1].start()
            if index + 1 < len(declarations)
            else len(normalized)
        )
        documented = re.findall(
            r"`mcp\.([a-z][a-z0-9_]*)`",
            normalized[declaration.end() : body_end],
        )
        expected = [
            field.name
            for field in fields(McpDefaults)
            if annotations[field.name]
            == strict_annotations[declaration.group("kind")]
        ]

        assert int(declaration.group("count")) == len(expected)
        assert documented == expected


def test_documented_semantic_default_yaml_loads(tmp_path: Path) -> None:
    text = CONFIG_DOC.read_text(encoding="utf-8")
    section = text.split("### Semantic Phase 2–4 configuration", 1)[1]
    match = re.search(r"```yaml\n(?P<yaml>.*?)\n```", section, flags=re.DOTALL)
    assert match is not None
    config_path = tmp_path / "semantic-defaults.yaml"
    config_path.write_text(match.group("yaml"), encoding="utf-8")

    loaded = load_config_file(config_path)

    assert loaded.semantic == DEFAULT_CONFIG.semantic


def test_cli_capability_list_documents_configured_default_and_maximum() -> None:
    text = CLI_DOC.read_text(encoding="utf-8")

    assert "`capabilities list --limit` defaults to `capability.list_limit`" in text
    assert "(100 by default)" in text
    assert "same configured value is the maximum" in text
