from __future__ import annotations

import pytest

from agent_libos.api.semantic_public import project_flow_status


_DIGEST = "b" * 64
_NOW = "2026-08-10T00:00:00+00:00"


def _flow_status() -> dict[str, object]:
    return {
        "schema_version": 1,
        "available": True,
        "counts": {
            "entities": 0,
            "activities": 0,
            "edges": 0,
            "label_assertions": 0,
        },
        "coverage": {
            "complete": 0,
            "partial": 0,
            "unknown": 0,
            "conflict": 0,
            "stale": 0,
        },
        "capture_failures": 0,
        "legacy_history": {
            "present": True,
            "source_schema_version": 5,
            "assessment_count": 7,
            "coverage": "unknown",
            "evidence_sha256": _DIGEST,
            "created_at": _NOW,
        },
    }


def test_flow_status_carries_v5_history_only_as_unknown_coverage() -> None:
    projected = project_flow_status(_flow_status())

    assert projected["legacy_history"] == {
        "present": True,
        "source_schema_version": 5,
        "assessment_count": 7,
        "coverage": "unknown",
        "evidence_sha256": _DIGEST,
        "created_at": _NOW,
    }


def test_flow_status_rejects_non_unknown_legacy_coverage_without_echo() -> None:
    value = _flow_status()
    legacy = value["legacy_history"]
    assert isinstance(legacy, dict)
    sentinel = "RAW_SECRET_LEGACY_COVERAGE_SENTINEL"
    legacy["coverage"] = sentinel

    with pytest.raises(TypeError) as raised:
        project_flow_status(value)

    assert sentinel not in str(raised.value)
