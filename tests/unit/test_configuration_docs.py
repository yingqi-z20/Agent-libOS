from __future__ import annotations

from dataclasses import fields
import re
from pathlib import Path
from typing import get_type_hints

from pydantic import StrictFloat, StrictInt

from agent_libos.config import (
    DEFAULT_CONFIG,
    LLMProfile,
    SemanticDefaults,
    load_config_file,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DOC = ROOT / "docs" / "configuration.md"
CLI_DOC = ROOT / "docs" / "cli.md"
_ROW = re.compile(r"^\| `(?P<group>[a-z_]+)` \| (?P<fields>.+) \|$")
_FIELD = re.compile(r"`([a-z][a-z0-9_]*)`")
_LLM_PROFILE_ROW = re.compile(r"^\| `llm\.profiles\.<profile_id>` \| (?P<fields>.+) \|$")
_SEMANTIC_STRICT_FIELD = re.compile(r"`semantic\.([a-z][a-z0-9_]*)`")


def test_configuration_reference_lists_every_default_dataclass_field() -> None:
    documented: dict[str, list[str]] = {}
    for line in CONFIG_DOC.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match:
            documented[match.group("group")] = _FIELD.findall(match.group("fields"))

    expected = {
        field.name: [nested.name for nested in fields(getattr(DEFAULT_CONFIG, field.name))]
        for field in fields(DEFAULT_CONFIG)
    }

    assert documented == expected


def test_configuration_reference_lists_every_llm_profile_field() -> None:
    profile_fields: list[str] | None = None
    for line in CONFIG_DOC.read_text(encoding="utf-8").splitlines():
        match = _LLM_PROFILE_ROW.match(line)
        if match:
            profile_fields = _FIELD.findall(match.group("fields"))

    assert profile_fields == [field.name for field in fields(LLMProfile)]


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
