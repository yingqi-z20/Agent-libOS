from __future__ import annotations

from agent_libos.llm.prompt import (
    PROMPT_LAYOUT_CACHE_OPTIMIZED_V2,
    PROMPT_LAYOUT_LEGACY_V1,
    _context_metadata_section,
)
from agent_libos.models import MaterializedContext


def _context(*, omitted: list[str], manifest: list[dict[str, object]]) -> MaterializedContext:
    return MaterializedContext(
        text="goal text",
        object_refs=["obj-goal"],
        token_count=12,
        omitted_objects=omitted,
        policy_used="working_set",
        object_manifest=manifest,
    )


def test_legacy_layout_explains_omitted_objects_by_reason() -> None:
    context = _context(
        omitted=["obj-old-1", "obj-old-2", "obj-big"],
        manifest=[
            {"oid": "obj-goal", "disposition": "included", "reason": "selected"},
            {"oid": "obj-old-1", "disposition": "omitted", "reason": "capability_denied"},
            {"oid": "obj-old-2", "disposition": "omitted", "reason": "capability_denied"},
            {"oid": "obj-big", "disposition": "omitted", "reason": "token_budget"},
        ],
    )

    section = _context_metadata_section(context, prompt_layout=PROMPT_LAYOUT_LEGACY_V1)

    assert "capability_denied=2" in section
    assert "Runtime reopen" in section
    assert "token_budget=1" in section
    assert "materialization budget" in section
    assert "Re-observe with a fresh read" in section
    assert "do not treat an omission as proof" in section


def test_legacy_layout_without_manifest_still_guides_re_observation() -> None:
    context = _context(omitted=["obj-unknown"], manifest=[])

    section = _context_metadata_section(context, prompt_layout=PROMPT_LAYOUT_LEGACY_V1)

    assert "omitted_object_reasons: not materialized in this quantum" in section
    assert "omitted_object_guidance" in section


def test_no_omissions_adds_no_guidance() -> None:
    context = _context(omitted=[], manifest=[{"oid": "obj-goal", "disposition": "included", "reason": "selected"}])

    section = _context_metadata_section(context, prompt_layout=PROMPT_LAYOUT_LEGACY_V1)

    assert "omitted_object_reasons" not in section
    assert "omitted_object_guidance" not in section
    assert section.endswith("- omitted_objects: []")


def test_cache_optimized_layout_keeps_its_compact_count_only_warning() -> None:
    context = _context(
        omitted=["obj-old-1"],
        manifest=[{"oid": "obj-old-1", "disposition": "omitted", "reason": "capability_denied"}],
    )

    section = _context_metadata_section(context, prompt_layout=PROMPT_LAYOUT_CACHE_OPTIMIZED_V2)

    assert section == "Materialized context warning:\n- omitted_object_count: 1"
