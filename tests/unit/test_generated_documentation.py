from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
import json
from pathlib import Path
import re
import types
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints

import pytest

from agent_libos.api.cli import _build_cli_parser
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, McpDefaults
from agent_libos.runtime.image_registry import (
    IMAGE_PACKAGE_MANIFEST_FIELDS,
    IMAGE_PACKAGE_REQUIRED_MANIFEST_FIELDS,
)
from scripts.generate_cli_reference import (
    _render_parser,
    _walk_parsers as _generated_walk_parsers,
    render_cli_reference,
)
from scripts.generate_config_reference import (
    _dataclass_template as _generated_dataclass_template,
    _default_text,
    _markdown,
    _type_name,
    render_config_reference,
)
from scripts.generate_pypi_readme import render_pypi_readme


ROOT = Path(__file__).resolve().parents[2]


def _walk_parsers(parser: argparse.ArgumentParser) -> list[argparse.ArgumentParser]:
    parsers: list[argparse.ArgumentParser] = []
    seen: set[int] = set()

    def visit(current: argparse.ArgumentParser) -> None:
        if id(current) in seen:
            return
        seen.add(id(current))
        parsers.append(current)
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for child in action.choices.values():
                    visit(child)

    visit(parser)
    return parsers


def _template_dataclass(
    annotation: Any,
) -> tuple[type[Any], tuple[str, ...]] | None:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _template_dataclass(args[0])
    if origin in {Union, types.UnionType}:
        candidates = {
            candidate
            for item in args
            if (candidate := _template_dataclass(item)) is not None
        }
        return next(iter(candidates)) if len(candidates) == 1 else None
    if origin is None:
        if isinstance(annotation, type) and is_dataclass(annotation):
            return annotation, ()
        return None
    if origin in {dict, Mapping} or (
        isinstance(origin, type) and issubclass(origin, Mapping)
    ):
        nested = _template_dataclass(args[-1]) if args else None
        if nested is None:
            return None
        cls, segments = nested
        return cls, ("<key>", *segments)
    if origin in {tuple, list, set, frozenset}:
        nested = _template_dataclass(args[0]) if args else None
        if nested is None:
            return None
        cls, segments = nested
        return cls, ("<item>", *segments)
    return None


def _field_value(field: Any, instance: Any | None) -> Any:
    if instance is not None:
        return getattr(instance, field.name)
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


def _scalar_default(value: Any) -> str | None:
    if value is MISSING:
        return "required"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return None


def _expected_config_rows(
    cls: type[Any],
    *,
    prefix: str,
    instance: Any | None,
) -> list[tuple[str, str | None]]:
    hints = get_type_hints(cls, include_extras=True)
    rows: list[tuple[str, str | None]] = []
    for field in fields(cls):
        path = f"{prefix}.{field.name}" if prefix else field.name
        value = _field_value(field, instance)
        if is_dataclass(value):
            rows.extend(
                _expected_config_rows(type(value), prefix=path, instance=value)
            )
            continue
        rows.append((path, _scalar_default(value)))
        template = _template_dataclass(hints[field.name])
        if template is None:
            continue
        template_cls, segments = template
        try:
            template_instance = template_cls()
        except (TypeError, ValueError):
            template_instance = None
        rows.extend(
            _expected_config_rows(
                template_cls,
                prefix=".".join((path, *segments)),
                instance=template_instance,
            )
        )
    return rows


def _config_table_rows(rendered: str) -> list[tuple[str, str, str]]:
    matches = re.finditer(
        r"^\| `(?P<path>[^`]+)` \| `(?P<type>[^`]*)` \| "
        r"`(?P<default>[^`]*)` \|",
        rendered,
        flags=re.MULTILINE,
    )
    return [
        tuple(
            match.group(name).replace("\\|", "|").replace("\\`", "`")
            for name in ("path", "type", "default")
        )
        for match in matches
    ]


def _expected_argument_syntax(action: argparse.Action) -> str:
    if action.metavar is None:
        if action.choices is not None:
            metavars = (
                "{" + ",".join(str(choice) for choice in action.choices) + "}",
            )
        else:
            default = action.dest.upper() if action.option_strings else action.dest
            metavars = (default,)
    elif isinstance(action.metavar, tuple):
        metavars = tuple(str(item) for item in action.metavar)
    else:
        metavars = (str(action.metavar),)

    if action.nargs == 0:
        value = ""
    elif action.nargs is None:
        assert len(metavars) == 1
        value = metavars[0]
    elif action.nargs == "?":
        assert len(metavars) == 1
        value = f"[{metavars[0]}]"
    elif action.nargs == "*":
        assert len(metavars) == 1
        value = f"[{metavars[0]} ...]"
    elif action.nargs == "+":
        assert len(metavars) == 1
        value = f"{metavars[0]} [{metavars[0]} ...]"
    elif action.nargs == argparse.REMAINDER:
        assert len(metavars) == 1
        value = f"{metavars[0]} ..."
    else:
        assert isinstance(action.nargs, int)
        assert len(metavars) in {1, action.nargs}
        values = metavars * action.nargs if len(metavars) == 1 else metavars
        value = " ".join(values)

    if not action.option_strings:
        return value
    options = ", ".join(action.option_strings)
    return f"{options} {value}" if value else options


def _generated_index(rendered: str, heading: str) -> list[tuple[str, str]]:
    marker = f"## {heading}\n"
    assert rendered.count(marker) == 1
    body = rendered.split(marker, 1)[1].split("\n## ", 1)[0]
    nonempty_lines = [line for line in body.splitlines() if line]
    entries = re.findall(
        r"^- \[`(?P<label>[^`]+)`\]\(#(?P<anchor>[a-z0-9_-]+)\)$",
        body,
        flags=re.MULTILINE,
    )
    assert nonempty_lines == [
        f"- [`{label}`](#{anchor})" for label, anchor in entries
    ]
    return entries


def test_generated_cli_reference_matches_every_parser_command_and_option() -> None:
    rendered = render_cli_reference()
    checked_in = (ROOT / "docs" / "cli_reference.md").read_text(encoding="utf-8")
    assert checked_in == rendered

    parsers = _walk_parsers(_build_cli_parser())
    section_matches = list(
        re.finditer(
            r"^## `(?P<prog>[^`]+)`\n(?P<body>.*?)(?=^## `|\Z)",
            rendered,
            flags=re.MULTILINE | re.DOTALL,
        )
    )
    sections = {
        match.group("prog"): match.group("body") for match in section_matches
    }
    assert len(section_matches) == len(sections) == len(parsers)
    assert set(sections) == {parser.prog for parser in parsers}

    for parser in parsers:
        section = sections[parser.prog]
        expected_actions = [
            _expected_argument_syntax(action)
            .replace("|", "\\|")
            .replace("`", "\\`")
            for action in parser._actions
            if not isinstance(action, argparse._SubParsersAction)
        ]
        ordinary_actions = [
            action
            for action in parser._actions
            if not isinstance(action, argparse._SubParsersAction)
        ]
        argument_rows = [
            line[2:-2].split(" | ", 3)
            for line in section.splitlines()
            if re.match(r"^\| `.*` \| (?:yes|no) \|", line)
        ]
        assert [row[0][1:-1] for row in argument_rows] == expected_actions

        for action, row in zip(ordinary_actions, argument_rows, strict=True):
            expected_required = bool(action.required) if action.option_strings else (
                action.nargs not in {"?", "*"}
            )
            expected_default = (
                "—"
                if action.default is argparse.SUPPRESS
                else "`"
                + repr(action.default).replace("|", "\\|").replace("`", "\\`")
                + "`"
            )
            assert row[1] == ("yes" if expected_required else "no")
            assert row[2] == expected_default

        for action in parser._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            for command in action.choices:
                escaped = str(command).replace("|", "\\|").replace("`", "\\`")
                assert f"| `{escaped}` |" in section


def test_generated_cli_index_matches_unique_top_level_sections() -> None:
    rendered = render_cli_reference()
    parser = _build_cli_parser()
    expected_parsers = [parser]
    seen = {id(parser)}
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for child in action.choices.values():
            if id(child) in seen:
                continue
            seen.add(id(child))
            expected_parsers.append(child)

    entries = _generated_index(rendered, "Command index")
    labels = [label for label, _anchor in entries]
    anchors = [anchor for _label, anchor in entries]
    expected_labels = [current.prog for current in expected_parsers]
    section_labels = re.findall(r"^## `([^`]+)`$", rendered, flags=re.MULTILINE)

    assert labels == expected_labels
    assert len(labels) == len(set(labels))
    assert len(anchors) == len(set(anchors))
    assert all(re.fullmatch(r"[a-z0-9_-]+(?: [a-z0-9_-]+)*", label) for label in labels)
    assert anchors == [label.replace(" ", "-") for label in labels]
    assert all(label in section_labels for label in labels)
    assert rendered.index("## Command index") < rendered.index(
        f"## `{expected_labels[0]}`"
    )


def test_cli_generator_deduplicates_alias_parsers_and_preserves_help() -> None:
    parser = argparse.ArgumentParser(prog="sample")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", aliases=("show",), help="Inspect one item")

    walked = _generated_walk_parsers(parser)
    rendered = "\n".join(_render_parser(parser))

    assert [item.prog for item in walked] == ["sample", "sample inspect"]
    assert "| `inspect` | Inspect one item |" in rendered
    assert "| `show` | Inspect one item |" in rendered


def test_cli_generator_is_independent_of_terminal_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_cli_parser()

    monkeypatch.setenv("COLUMNS", "40")
    narrow = "\n".join(_render_parser(parser))
    monkeypatch.setenv("COLUMNS", "240")
    wide = "\n".join(_render_parser(parser))

    assert narrow == wide


def test_cli_generator_does_not_use_argparse_formatter_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_cli_parser()

    def reject_formatter_use(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("generated reference must not use argparse formatter output")

    monkeypatch.setattr(argparse.ArgumentParser, "_get_formatter", reject_formatter_use)
    rendered = "\n".join(_render_parser(parser))
    subcommands = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    command_syntax = "{" + ",".join(subcommands.choices) + "} ..."

    assert command_syntax in rendered
    assert "| `--config CONFIG` |" in rendered


def test_cli_generator_renders_aliases_and_mutex_groups_canonically() -> None:
    parsers = {parser.prog: parser for parser in _walk_parsers(_build_cli_parser())}
    task_start = "\n".join(_render_parser(parsers["agent-libos task-run start"]))
    object_start = "\n".join(_render_parser(parsers["agent-libos object-task start"]))
    exec_command = "\n".join(_render_parser(parsers["agent-libos exec"]))
    task_usage = task_start.split("```text\n", 1)[1].split("\n```", 1)[0]

    assert "| `--title, --display-title TITLE` | yes |" in task_start
    assert "--title TITLE" in task_usage
    assert "--display-title" not in task_usage
    assert "(--owner-oid OWNER_OID | --owner-name OWNER_NAME)" in object_start
    assert "[--run | --no-run]" in exec_command


def test_generated_configuration_reference_is_current_and_exhaustive() -> None:
    rendered = render_config_reference()
    checked_in = (ROOT / "docs" / "configuration_reference.md").read_text(
        encoding="utf-8"
    )
    assert checked_in == rendered

    documented_rows = _config_table_rows(rendered)
    expected_rows = _expected_config_rows(
        AgentLibOSConfig,
        prefix="",
        instance=DEFAULT_CONFIG,
    )
    assert [row[0] for row in documented_rows] == [row[0] for row in expected_rows]
    assert len(documented_rows) == len({row[0] for row in documented_rows})

    documented_defaults = {path: default for path, _type, default in documented_rows}
    for path, expected_default in expected_rows:
        if expected_default is not None:
            assert documented_defaults[path] == expected_default

    for required in (
        "llm.profiles.<key>.max_tokens",
        "shell.whitelist.<item>.argv",
        "semantic.policy_epoch.epoch_id",
        "semantic.policy_epoch.auto_approval_rules.<item>.rule_id",
        "semantic.policy_epoch.hard_deny_rules.<item>.reason_code",
    ):
        assert required in documented_defaults


def test_generated_configuration_index_matches_all_top_level_groups() -> None:
    rendered = render_config_reference()
    entries = _generated_index(rendered, "Configuration group index")
    labels = [label for label, _anchor in entries]
    anchors = [anchor for _label, anchor in entries]
    expected_labels = [field.name for field in fields(AgentLibOSConfig)]
    section_labels = re.findall(r"^## `([^`]+)`$", rendered, flags=re.MULTILINE)

    assert labels == expected_labels
    assert len(labels) == len(set(labels))
    assert len(anchors) == len(set(anchors))
    assert anchors == expected_labels
    assert section_labels == expected_labels
    assert rendered.index("## Configuration group index") < rendered.index(
        f"## `{expected_labels[0]}`"
    )


@pytest.mark.parametrize(
    ("renderer", "index_heading"),
    (
        (render_cli_reference, "Command index"),
        (render_config_reference, "Configuration group index"),
    ),
)
def test_generated_reference_links_back_to_documentation_home_before_index(
    renderer: Any,
    index_heading: str,
) -> None:
    rendered = renderer()
    home_link = "[documentation home](index.md)"

    assert rendered.count(home_link) == 1
    assert rendered.index(home_link) < rendered.index(f"## {index_heading}")


def test_generated_configuration_type_names_are_stable_and_explicit() -> None:
    hints = get_type_hints(McpDefaults, include_extras=True)

    assert _type_name(hints["server_page_limit"]) == "StrictInt"
    assert _type_name(hints["schema_regex_match_timeout_s"]) == "StrictFloat"
    assert _type_name(hints["oauth_enabled"]) == "StrictBool"
    assert _type_name(str | None) == "str | None"
    assert _type_name(dict[str, tuple[McpDefaults, ...]]) == (
        "dict[str, tuple[McpDefaults, ...]]"
    )


def test_generated_configuration_scalar_strings_preserve_exact_whitespace() -> None:
    value = "two  spaces\nand | a pipe"
    encoded = json.dumps(value, ensure_ascii=False)

    assert _default_text(value) == encoded
    assert _markdown(encoded) == encoded.replace("|", "\\|")


def test_config_generator_rejects_ambiguous_dataclass_templates() -> None:
    @dataclass
    class First:
        value: str = "first"

    @dataclass
    class Second:
        value: str = "second"

    with pytest.raises(TypeError, match="ambiguous dataclass template"):
        _generated_dataclass_template(First | Second)


@pytest.mark.parametrize(
    "path",
    (
        "scripts/generate_config_reference.py",
        "scripts/generate_cli_reference.py",
    ),
)
def test_documentation_generators_parse_as_python_311(path: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(source, filename=path, feature_version=11)


def test_agent_image_authoring_table_matches_the_runtime_manifest_contract() -> None:
    document = (ROOT / "docs" / "agent_images.md").read_text(encoding="utf-8")
    section = document.split("## Manifest fields", 1)[1].split("## Prompt modes", 1)[0]
    documented = set(
        re.findall(r"^\| `([a-z][a-z0-9_]*)` \|", section, flags=re.MULTILINE)
    )
    assert documented == IMAGE_PACKAGE_MANIFEST_FIELDS
    for required in IMAGE_PACKAGE_REQUIRED_MANIFEST_FIELDS:
        row = next(line for line in section.splitlines() if line.startswith(f"| `{required}` |"))
        assert "Required" in row


def test_pypi_readme_rewrites_only_repository_relative_and_mutable_links() -> None:
    source = """# Example

[Guide](docs/guide.md#start)
[Same page](#example)
[Old main](https://github.com/yingqi-z20/Agent-libOS/blob/main/SECURITY.md)
[External](https://example.com/docs)
"""
    rendered = render_pypi_readme(source, version="1.2.3")

    assert (
        "https://github.com/yingqi-z20/Agent-libOS/blob/v1.2.3/"
        "docs/guide.md#start"
    ) in rendered
    assert (
        "https://github.com/yingqi-z20/Agent-libOS/blob/v1.2.3/SECURITY.md"
        in rendered
    )
    assert "](#example)" in rendered
    assert "](https://example.com/docs)" in rendered
    assert "/blob/main/" not in rendered


@pytest.mark.parametrize("target", ("../outside.md", "/absolute.md", "bad\\path.md"))
def test_pypi_readme_rejects_nonportable_repository_links(target: str) -> None:
    with pytest.raises(ValueError, match="portable|safe"):
        render_pypi_readme(f"[bad]({target})\n", version="1.2.3")
