"""Pure prompt-projection helpers that bound long-horizon prompt growth."""

from __future__ import annotations

import json

from agent_libos.llm.prompt import (
    FEEDBACK_STUB_RECORD_TYPE,
    PROMPT_LAYOUT_LEGACY_V1,
    _compact_materialized_context_text,
    _context_metadata_section,
    _slim_legacy_capability_rows,
    compact_skill_tool_guide,
)
from agent_libos.memory import object_memory
from agent_libos.models import MaterializedContext


_SKILL_BODY = """# Edit the workspace

Intro paragraph.

## Tool guide

### `write_text_file`

- Input: path and content.
- Output: bytes_written.

### `delete_file`

- Input: path.

## Recommended workflow

1. Read first.

## Failure and recovery

- Never guess.
"""


def test_stub_record_type_is_shared_between_memory_and_prompt_layers() -> None:
    assert FEEDBACK_STUB_RECORD_TYPE == object_memory.FEEDBACK_STUB_RECORD_TYPE


def test_compact_skill_tool_guide_removes_only_exercised_subsections() -> None:
    compacted = compact_skill_tool_guide(_SKILL_BODY, ["write_text_file"])

    assert "### `write_text_file`" not in compacted
    assert "bytes_written" not in compacted
    assert "### `delete_file`" in compacted
    assert "Guides for `write_text_file` are omitted" in compacted
    assert "## Recommended workflow" in compacted
    assert "## Failure and recovery" in compacted
    assert compacted.index("## Tool guide") < compacted.index("## Recommended workflow")


def test_compact_skill_tool_guide_keeps_body_without_matching_tools() -> None:
    assert compact_skill_tool_guide(_SKILL_BODY, ["read_text_file"]) == _SKILL_BODY
    assert compact_skill_tool_guide("no guide here", ["write_text_file"]) == "no guide here"


def test_compact_skill_tool_guide_replaces_whole_guide_when_all_tools_used() -> None:
    compacted = compact_skill_tool_guide(_SKILL_BODY, ["write_text_file", "delete_file"])

    assert "### " not in compacted
    assert "Guides for `delete_file`, `write_text_file` are omitted" in compacted
    assert compacted.count("## Tool guide") == 1
    assert "1. Read first." in compacted


def _legacy_row(resource: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "cap_id": f"cap_{abs(hash(resource)) % 10_000}",
        "resource": resource,
        "rights": ["read"],
        "effect": "allow",
        "status": "active",
        "policy": "always_allow",
        "uses_remaining": None,
        "delegable": False,
        "delegation_depth": 0,
        "issuer": "memory",
        "parent_cap_id": None,
        "expires_at": None,
    }
    row.update(overrides)
    return row


def test_legacy_capability_rows_fold_object_grants_and_drop_constant_fields() -> None:
    rows = [
        _legacy_row("object:obj_1", rights=["materialize", "read"]),
        _legacy_row("object:obj_2", rights=["materialize", "read"]),
        _legacy_row("object:obj_3", rights=["materialize", "read"]),
        _legacy_row("filesystem:workspace:*", rights=["read", "write"]),
        _legacy_row("shell:*", uses_remaining=3),
    ]

    slim = _slim_legacy_capability_rows(rows, tools=[])
    encoded = json.dumps(slim, sort_keys=True)

    folded = [row for row in slim if row["resource"] == "object:materialized"]
    assert len(folded) == 1
    assert folded[0]["object_count"] == 3
    assert "obj_1" not in encoded
    assert "cap_" not in encoded
    assert '"status"' not in encoded
    assert '"issuer"' not in encoded
    assert '"parent_cap_id"' not in encoded
    assert '"delegation_depth"' not in encoded
    shell = next(row for row in slim if row["resource"] == "shell:*")
    assert shell["uses_remaining"] == 3
    assert len(slim) == 3


def test_legacy_capability_rows_keep_ids_when_a_visible_tool_accepts_them() -> None:
    tools = [
        {
            "name": "inspect_capability",
            "spec_json": json.dumps(
                {"input_schema": {"type": "object", "properties": {"cap_id": {"type": "string"}}}}
            ),
        }
    ]
    rows = [_legacy_row("filesystem:workspace:*", delegation_depth=2, parent_cap_id="cap_parent")]

    slim = _slim_legacy_capability_rows(rows, tools=tools)

    assert slim[0]["cap_id"] == rows[0]["cap_id"]
    assert slim[0]["delegation_depth"] == 2
    assert slim[0]["parent_cap_id"] == "cap_parent"


def test_v2_compaction_strips_object_ids_from_feedback_stubs() -> None:
    stub = {
        "record_type": FEEDBACK_STUB_RECORD_TYPE,
        "object_oid": "obj_secret",
        "name": "tool_result:obj_secret",
        "stub_reason": "superseded",
        "summary": {"tool_name": "read_text_file", "path": "src/a.py"},
        "type": "tool_result",
    }
    text = json.dumps(stub, sort_keys=True)

    compact = _compact_materialized_context_text(text, include_object_ids=False)
    kept = _compact_materialized_context_text(text, include_object_ids=True)

    assert "obj_secret" not in json.loads(compact)["summary"]
    assert "object_oid" not in json.loads(compact)
    assert json.loads(kept)["object_oid"] == "obj_secret"
    assert json.loads(compact)["stub_reason"] == "superseded"


def test_legacy_metadata_explains_compacted_feedback_stubs() -> None:
    context = MaterializedContext(
        text="goal",
        object_refs=["obj-goal", "obj-old"],
        token_count=40,
        omitted_objects=[],
        policy_used="working_set",
        object_manifest=[
            {"oid": "obj-goal", "disposition": "included", "transform": "verbatim"},
            {"oid": "obj-old", "disposition": "included", "transform": "compacted"},
        ],
    )

    section = _context_metadata_section(context, prompt_layout=PROMPT_LAYOUT_LEGACY_V1)

    assert "compacted_feedback_stubs: 1" in section
    assert FEEDBACK_STUB_RECORD_TYPE in section
    assert "do not re-read files merely to re-establish context" in section

    plain = MaterializedContext(
        text="goal",
        object_refs=["obj-goal"],
        token_count=4,
        omitted_objects=[],
        policy_used="working_set",
        object_manifest=[{"oid": "obj-goal", "disposition": "included", "transform": "verbatim"}],
    )
    assert "compacted_feedback_stubs" not in _context_metadata_section(
        plain, prompt_layout=PROMPT_LAYOUT_LEGACY_V1
    )
