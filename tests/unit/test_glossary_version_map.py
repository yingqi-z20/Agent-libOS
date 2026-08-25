from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def _version_map_section() -> str:
    text = (DOCS / "glossary.md").read_text(encoding="utf-8")
    start = text.find("## Version map")
    assert start != -1, "docs/glossary.md must keep a '## Version map' section"
    after = text[start:]
    nxt = re.search(r"\n## ", after[3:])
    end = (3 + nxt.start()) if nxt else len(after)
    return after[:end]


def _maintained_docs() -> list[Path]:
    files = sorted(ROOT.glob("*.md")) + sorted(DOCS.glob("*.md"))
    # Skip generated artifacts: their content is pinned by generators/tests.
    skip = {"README.pypi.md", "cli_reference.md", "configuration_reference.md"}
    return [f for f in files if f.name not in skip]


def test_version_map_documents_namespaces_used_by_docs() -> None:
    """The version map is the contract-mandated disambiguation point
    (docs/index.md: before interpreting an unqualified v1/v2/v3/v7, consult the
    version map). It must therefore list the version namespaces the docs actually
    reference with bare tokens. This ratchets the prompt-layout/prompt-cache
    namespace and the frozen RuntimeStore schema, GUI snapshot, and public
    semantic-status projections so a future edit cannot silently drop them."""
    vm = _version_map_section().lower()
    assert "prompt" in vm and "cache" in vm  # llm.prompt_layout / prompt-cache v1/v2
    assert "runtimestore schema" in vm  # persisted SQL store shape (frozen value 7)
    assert "snapshot" in vm  # GUI snapshot envelope
    assert "semantic" in vm  # public semantic-status projection (value 3)


def test_no_stale_prerelease_schema_tokens_in_docs() -> None:
    """The RuntimeStore schema is frozen at v7; pre-1.0 product-version tokens
    such as '0.3' must not survive as schema/visibility qualifiers anywhere in
    the maintained docs. This ratchets the stale-token fix so a regression
    (e.g. re-introducing 'the 0.3 schema' or 'required 0.3 visibility state')
    fails CI rather than shipping as an unmappable version."""
    stale = re.compile(r"(?<!\d)0\.\d+\s+(?:schema|visibility)", re.IGNORECASE)
    failures: list[str] = []
    for doc in _maintained_docs():
        text = doc.read_text(encoding="utf-8")
        for match in stale.finditer(text):
            failures.append(
                f"{doc.relative_to(ROOT)}: stale pre-release schema/visibility "
                f"token {match.group()!r}"
            )
    assert not failures, "\n".join(failures)
