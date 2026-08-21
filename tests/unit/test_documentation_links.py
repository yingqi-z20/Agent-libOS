from __future__ import annotations

from contextlib import redirect_stdout
from functools import lru_cache
from io import StringIO
import re
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit

from agent_libos.api.cli import _parse_cli_args


ROOT = Path(__file__).resolve().parents[2]


def _repository_markdown_documents() -> tuple[Path, ...]:
    completed = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.md",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    documents = []
    for relative in completed.stdout.split("\0"):
        if not relative or Path(relative).suffix.lower() != ".md":
            continue
        path = ROOT / relative
        if not path.is_file():
            raise AssertionError(f"repository Markdown document is missing: {relative}")
        documents.append(path)
    return tuple(sorted(documents))


DOCUMENTS = _repository_markdown_documents()
_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

_IMPORTANT_COMMAND_SECTIONS = {
    "explain": "Explainable Operations",
    "workflow": "Workflow Run",
    "object-task": "Object Tasks",
    "checkpoint": "Checkpoint Commands",
    "skills": "Skill Commands",
    "capabilities": "Capability Commands",
    "images": "Image Commands",
    "jsonrpc": "JSON-RPC Commands",
    "mcp": "MCP Commands",
    "modules": "Runtime Module Commands",
}

_IMPORTANT_LEAF_PARAMETERS = {
    ("workflow", "run"): (
        "Workflow Run",
        ("--args-json", "--image", "--goal"),
    ),
    ("object-task", "start"): (
        "Object Tasks",
        ("--pid", "--owner-oid", "--wait"),
    ),
    ("exec",): (
        "Process Builtins",
        ("--pid", "--preserve-capabilities", "--run"),
    ),
    ("skills", "activate"): (
        "Skill Commands",
        ("--expected-package-sha256",),
    ),
    ("capabilities", "grant"): (
        "Capability Commands",
        ("--rights",),
    ),
    ("images", "commit"): (
        "Image Commands",
        ("--name",),
    ),
    ("jsonrpc", "call"): (
        "JSON-RPC Commands",
        ("--params-json",),
    ),
    ("mcp", "call"): (
        "MCP Commands",
        ("--arguments-json",),
    ),
}

_ACTOR_SCOPED_GROUPS = (
    "checkpoint",
    "skills",
    "capabilities",
    "images",
    "jsonrpc",
    "mcp",
)


def _without_code(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"~~~.*?~~~", "", text, flags=re.DOTALL)
    return re.sub(r"`[^`\n]*`", "", text)


def _target(raw: str) -> str:
    selected = raw.strip()
    if selected.startswith("<") and ">" in selected:
        return selected[1 : selected.index(">")]
    return selected.split(maxsplit=1)[0]


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in _HEADING.findall(path.read_text(encoding="utf-8")):
        plain = re.sub(r"<[^>]+>", "", heading)
        plain = re.sub(r"[`*_~]", "", plain).strip().lower()
        slug = "".join(
            character
            for character in plain
            if character.isalnum() or character in {" ", "-", "_"}
        )
        slug = re.sub(r"\s+", "-", slug)
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


@lru_cache(maxsize=None)
def _cli_help(command: tuple[str, ...]) -> str:
    stdout = StringIO()
    with redirect_stdout(stdout):
        try:
            _parse_cli_args([*command, "--help"])
        except SystemExit as exc:
            if exc.code != 0:
                raise AssertionError(
                    f"CLI help failed for {' '.join(command)} with {exc.code}"
                ) from exc
        else:
            raise AssertionError(f"CLI help did not exit for {' '.join(command)}")
    return stdout.getvalue()


def _markdown_section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"docs/cli.md is missing the {heading!r} section"
    return match.group("body")


def _nested_commands(help_text: str) -> tuple[str, ...]:
    match = re.search(r"\{(?P<commands>[a-z0-9,-]+)\}", help_text)
    assert match is not None, "CLI group help does not expose a subcommand list"
    return tuple(match.group("commands").split(","))


def test_local_documentation_links_and_anchors_resolve() -> None:
    failures: list[str] = []
    for document in DOCUMENTS:
        text = _without_code(document.read_text(encoding="utf-8"))
        for raw in _LINK.findall(text):
            selected = _target(raw)
            parsed = urlsplit(selected)
            if parsed.scheme or parsed.netloc:
                continue
            linked = document if not parsed.path else (document.parent / unquote(parsed.path)).resolve()
            if not linked.exists():
                failures.append(f"{document.relative_to(ROOT)} -> missing {selected}")
                continue
            if parsed.fragment and linked.suffix.lower() == ".md":
                anchor = unquote(parsed.fragment).lower()
                if anchor not in _anchors(linked):
                    failures.append(
                        f"{document.relative_to(ROOT)} -> missing anchor {selected}"
                    )

    assert not failures, "\n" + "\n".join(failures)


def test_document_inventory_covers_tracked_and_untracked_markdown_files() -> None:
    relative_documents = {path.relative_to(ROOT) for path in DOCUMENTS}
    builtin_skills = {
        path
        for path in relative_documents
        if path.parts[:3] == ("agent_libos", "skills", "builtin")
        and path.name == "SKILL.md"
    }
    experiment_documents = {
        path for path in relative_documents if path.parts[0] == "experiments"
    }

    assert builtin_skills
    assert experiment_documents
    assert Path("docs/events.md") in relative_documents
    assert Path("docs/index.md") in relative_documents
    assert Path("docs/glossary.md") in relative_documents
    assert Path("docs/troubleshooting.md") in relative_documents
    assert Path("docs/agent_images.md") in relative_documents
    assert Path("docs/cli_reference.md") in relative_documents
    assert Path("docs/configuration_reference.md") in relative_documents
    assert Path("benchmarks/runtime_safety/README.md") in relative_documents


def test_current_document_heading_levels_do_not_skip() -> None:
    documents = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
    failures: list[str] = []

    for document in documents:
        cleaned = _without_code(document.read_text(encoding="utf-8"))
        previous_level = 0
        for match in re.finditer(
            r"^(?P<marks>#{1,6})\s+",
            cleaned,
            flags=re.MULTILINE,
        ):
            level = len(match.group("marks"))
            line = cleaned.count("\n", 0, match.start()) + 1
            if previous_level == 0 and level != 1:
                failures.append(
                    f"{document.relative_to(ROOT)}:{line} starts at heading level {level}"
                )
            elif previous_level and level > previous_level + 1:
                failures.append(
                    f"{document.relative_to(ROOT)}:{line} jumps from H{previous_level} to H{level}"
                )
            previous_level = level

    assert not failures, "\n" + "\n".join(failures)


def test_long_current_contract_documents_have_navigation_and_home_links() -> None:
    excluded = {
        # Generated references have generator-checked indexes of their own.
        "cli_reference.md",
        "configuration_reference.md",
        # This commit-bound research report is explicitly historical.
        "semantic_permission_and_dataflow_research.md",
    }
    failures: list[str] = []

    for document in sorted((ROOT / "docs").glob("*.md")):
        text = document.read_text(encoding="utf-8")
        if document.name in excluded or len(text.splitlines()) < 500:
            continue
        if "## In this guide" not in text:
            failures.append(f"{document.relative_to(ROOT)} -> missing task navigation")
        if re.search(r"\[[Dd]ocumentation home\]\(index\.md\)", text) is None:
            failures.append(f"{document.relative_to(ROOT)} -> missing documentation-home link")

    assert not failures, "\n" + "\n".join(failures)


def test_readme_and_docs_home_link_narrative_and_generated_entrypoints() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    cli_guide = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")

    assert "github.com/yingqi-z20/Agent-libOS/blob/main/" not in readme
    for target in (
        "docs/index.md",
        "docs/cli.md",
        "docs/cli_reference.md",
        "docs/configuration.md",
        "docs/configuration_reference.md",
        "docs/glossary.md",
        "docs/troubleshooting.md",
        "docs/agent_images.md",
    ):
        assert f"]({target})" in readme

    for target in (
        "cli.md",
        "cli_reference.md",
        "configuration.md",
        "configuration_reference.md",
        "glossary.md",
        "troubleshooting.md",
        "agent_images.md",
    ):
        assert f"]({target})" in docs_home

    assert cli_guide.startswith("# CLI Guide\n")
    assert "](cli_reference.md)" in cli_guide
    assert "same installed version's `agent-libos ... --help`" in cli_guide


def test_cli_reference_tracks_every_top_level_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "agent_libos.api.cli", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    help_match = re.search(r"\{([^{}]+)\}", completed.stdout)
    assert help_match is not None
    actual = help_match.group(1).split(",")

    cli_reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    documented_match = re.search(
        r"## Top-Level Commands\s+```text\n(?P<body>.*?)\n```",
        cli_reference,
        flags=re.DOTALL,
    )
    assert documented_match is not None
    documented = [
        line.split(maxsplit=1)[0]
        for line in documented_match.group("body").splitlines()
        if line.strip()
    ]

    assert documented == actual


def test_cli_reference_tracks_important_nested_commands() -> None:
    cli_reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    failures: list[str] = []

    for group, heading in _IMPORTANT_COMMAND_SECTIONS.items():
        section = _markdown_section(cli_reference, heading)
        for command in _nested_commands(_cli_help((group,))):
            if not re.search(
                rf"(?<![\w-]){re.escape(command)}(?![\w-])",
                section,
            ):
                failures.append(f"{group} {command} is absent from {heading!r}")

    assert not failures, "\n" + "\n".join(failures)


def test_cli_reference_tracks_important_leaf_parameters() -> None:
    cli_reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    failures: list[str] = []

    for command, (heading, parameters) in _IMPORTANT_LEAF_PARAMETERS.items():
        help_text = _cli_help(command)
        section = _markdown_section(cli_reference, heading)
        for parameter in parameters:
            if parameter not in help_text:
                failures.append(
                    f"agent-libos {' '.join(command)} help is missing {parameter}"
                )
            if parameter not in section:
                failures.append(
                    f"{heading!r} does not document {' '.join(command)} {parameter}"
                )

    assert not failures, "\n" + "\n".join(failures)


def test_cli_reference_tracks_actor_scoped_command_groups() -> None:
    cli_reference = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    actor_contract = re.search(
        r"`--actor-pid` is a command-group option.*?(?=\n\n)",
        cli_reference,
        flags=re.DOTALL,
    )
    assert actor_contract is not None

    failures: list[str] = []
    for group in _ACTOR_SCOPED_GROUPS:
        if "--actor-pid" not in _cli_help((group,)):
            failures.append(f"agent-libos {group} help is missing --actor-pid")
        if f"`{group}`" not in actor_contract.group(0):
            failures.append(
                f"docs/cli.md actor-scope contract does not name {group}"
            )

    assert not failures, "\n" + "\n".join(failures)
