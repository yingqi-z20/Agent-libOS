from __future__ import annotations

import pytest

from agent_libos.api.gui.server import (
    _BoundedRunRevisions,
    GuiServerError,
    _task_run_conflict_envelope,
    _task_run_mutation_identity,
    _task_run_page_payload,
    _task_run_ledger_item_payload,
    _task_run_requirement_payload,
    _task_run_summary_payload,
)
from agent_libos.models import (
    TaskRunCursor,
    TaskRunPage,
    TaskRunStatus,
)
from agent_libos.models.exceptions import TaskRunRevisionConflict


def _summary(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "schema_version": 1,
        "run_id": "run_gui_1",
        "revision": 7,
        "status": TaskRunStatus.NEEDS_ATTENTION,
        "display_title": "Durable repair",
        "root_pid": "pid_root",
        "active_pid": "pid_root",
        "step_count": 3,
        "completed_step_count": 2,
        "requirement_count": 2,
        "satisfied_requirement_count": 1,
        "blockers": [
            {
                "kind": "unknown_effect",
                "effect_id": "effect_1",
                "secret": "must-not-cross-summary-boundary",
                "message": "provider-authored detail must not cross",
                "public_error": {
                    "code": "provider_error",
                    "error_type": "ProviderError",
                    "correlation_id": "corr_1",
                    "message": "attacker-controlled replacement",
                },
            },
            {"kind": "not_a_registered_blocker", "secret": "drop-me"},
        ],
        "allowed_actions": ["recover", "cancel", "invented_action"],
        "retention": "purge_on_terminal",
        "payloads_purged": False,
        "created_at": "2030-01-01T00:00:00Z",
        "updated_at": "2030-01-01T00:01:00Z",
    }
    values.update(changes)
    return values


def test_task_run_summary_projection_is_versioned_and_closed() -> None:
    projected = _task_run_summary_payload(_summary())

    assert projected["schema_version"] == 1
    assert projected["run_id"] == "run_gui_1"
    assert projected["revision"] == 7
    assert projected["status"] == "needs_attention"
    assert projected["payloads_purged"] is False
    assert projected["allowed_actions"] == ["recover", "cancel"]
    assert projected["blockers"] == [
        {
            "kind": "unknown_effect",
            "effect_id": "effect_1",
            "code": "provider_error",
            "message": "provider_error: ProviderError (correlation_id=corr_1)",
        }
    ]
    encoded = repr(projected)
    assert "must-not-cross-summary-boundary" not in encoded
    assert "provider-authored detail" not in encoded
    assert "attacker-controlled replacement" not in encoded
    assert "invented_action" not in encoded

    with pytest.raises(TypeError, match="invalid public identity"):
        _task_run_summary_payload(_summary(schema_version=2))
    with pytest.raises(TypeError, match="payload retention state"):
        _task_run_summary_payload(_summary(payloads_purged="false"))


def test_task_run_page_uses_items_and_opaque_wire_cursor() -> None:
    page = TaskRunPage(
        records=(_summary(),),  # type: ignore[arg-type]
        next_cursor=TaskRunCursor(
            created_at="2030-01-01T00:00:00Z",
            run_id="run_gui_1",
        ),
    )

    projected = _task_run_page_payload(page, summary_items=True)

    assert projected["items"][0]["run_id"] == "run_gui_1"
    assert projected["has_more"] is True
    assert isinstance(projected["next_cursor"], str)
    assert projected["next_cursor"]
    assert "created_at" not in projected["next_cursor"]


def test_task_run_requirement_projection_omits_payload_and_body() -> None:
    projected = _task_run_requirement_payload(
        {
            "requirement_id": "requirement_1",
            "run_id": "run_gui_1",
            "ordinal": 2,
            "kind": "follow_up",
            "status": "pending",
            "requirement_sha256": "a" * 64,
            "label": "Customer confirmation",
            "created_by": "host",
            "created_at": "2030-01-01T00:00:00Z",
            "updated_at": "2030-01-01T00:00:00Z",
            "payload_id": "payload_private_1",
            "payload": {"secret": "PRIVATE_REQUIREMENT_PAYLOAD"},
            "body": "PRIVATE_REQUIREMENT_BODY",
            "waiver_reason": "PRIVATE_WAIVER_REASON",
        }
    )

    assert projected["requirement_id"] == "requirement_1"
    assert projected["status"] == "pending"
    assert projected["content_available"] is False
    assert projected["content_retention"] == "hash_only"
    assert projected["content_sha256"] == "a" * 64
    assert "payload_id" not in projected
    assert "PRIVATE_" not in repr(projected)

    plaintext = _task_run_requirement_payload(
        {
            "requirement_id": "requirement_2",
            "run_id": "run_gui_1",
            "ordinal": 3,
            "kind": "follow_up",
            "status": "pending",
            "requirement_sha256": "b" * 64,
            "label": "Retained follow-up",
            "created_by": "host",
            "created_at": "2030-01-01T00:00:00Z",
            "updated_at": "2030-01-01T00:00:00Z",
            "started_at": None,
            "completed_at": None,
            "waived_by": None,
            "content_retention": "plaintext",
            "content_available": True,
            "content_text": "0123456789",
        },
        content_max_chars=5,
    )
    assert plaintext["content_available"] is True
    assert plaintext["content_retention"] == "plaintext"
    assert plaintext["content_text"] == "01234"
    assert plaintext["content_truncated"] is True

    with pytest.raises(TypeError, match="invalid public identity"):
        _task_run_requirement_payload(
            {
                **plaintext,
                "requirement_sha256": None,
            }
        )
    with pytest.raises(TypeError, match="invalid public identity"):
        _task_run_requirement_payload({**plaintext, "schema_version": 2})


def test_task_run_ledger_projection_is_versioned_and_drops_secret_metadata() -> None:
    projected = _task_run_ledger_item_payload(
        {
            "item_id": "ledger_1",
            "run_id": "run_gui_1",
            "seq": 3,
            "kind": "effect",
            "status": "unknown",
            "label": "effect outcome",
            "occurred_at": "2030-01-01T00:00:00Z",
            "effect_id": "effect_1",
            "payload_id": "payload_private_1",
            "provider_secret": "PRIVATE_TOP_LEVEL_SECRET",
            "metadata": {
                "effect_state": "unknown",
                "binding_hash": "a" * 64,
                "task_run_payload_ref": "payload_private_1",
                "api_key": "PRIVATE_METADATA_SECRET",
                "provider_response": {"token": "PRIVATE_PROVIDER_RESPONSE"},
            },
        }
    )

    assert projected == {
        "schema_version": 1,
        "kind": "effect",
        "seq": 3,
        "item_id": "ledger_1",
        "run_id": "run_gui_1",
        "status": "unknown",
        "effect_id": "effect_1",
        "label": "effect outcome",
        "occurred_at": "2030-01-01T00:00:00Z",
        "metadata": {
            "effect_state": "unknown",
            "binding_hash": "a" * 64,
        },
    }
    assert "PRIVATE_" not in repr(projected)


@pytest.mark.parametrize("revision", [None, -1, True, 1.0, "1"])
def test_task_run_mutation_identity_rejects_non_revision_values(
    revision: object,
) -> None:
    with pytest.raises(GuiServerError, match="expected_revision"):
        _task_run_mutation_identity(
            {"expected_revision": revision, "command_id": "command_1"}
        )


def test_task_run_revision_conflict_has_stable_http_code() -> None:
    error = TaskRunRevisionConflict("revision mismatch")
    error.run_id = "run_gui_1"
    error.expected_revision = 4
    error.actual_revision = 5

    assert _task_run_conflict_envelope(error) == {
        "type": "TaskRunRevisionConflict",
        "code": "task_run_revision_conflict",
        "message": "revision mismatch",
        "run_id": "run_gui_1",
        "expected_revision": 4,
        "actual_revision": 5,
    }


def test_task_run_sse_revision_filter_rejects_duplicate_and_lower_updates() -> None:
    revisions = _BoundedRunRevisions(2)

    assert revisions.accept("run_1", 5) is True
    assert revisions.accept("run_1", 5) is False
    assert revisions.accept("run_1", 4) is False
    assert revisions.accept("run_1", 6) is True
    assert revisions.accept("run_2", 1) is True
    assert revisions.accept("run_3", 1) is True
    assert len(revisions) == 2
    # run_1 was the least recently touched key after accepting run_2/run_3.
    assert revisions.accept("run_1", 1) is True
