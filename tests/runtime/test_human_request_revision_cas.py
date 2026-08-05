from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos.models import HumanRequest, HumanRequestStatus
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import SQLiteStore


def _pending_request(request_id: str = "human-revision-cas") -> HumanRequest:
    return HumanRequest(
        request_id=request_id,
        pid="pid-human-revision-cas",
        human="operator",
        payload={"kind": "question", "prompt": "approve?"},
        status=HumanRequestStatus.PENDING,
        decision=None,
        blocking=True,
        created_at="2026-08-05T00:00:00Z",
        updated_at="2026-08-05T00:00:00Z",
    )


def test_human_request_revision_defaults_to_zero_and_insert_round_trips() -> None:
    request = _pending_request()
    assert request.revision == 0

    store = SQLiteStore(":memory:")
    try:
        store.insert_human_request(request)

        persisted = store.get_human_request(request.request_id)
        assert persisted == request
        assert persisted is not None
        assert persisted.revision == 0
    finally:
        store.close()


def test_human_request_compare_and_set_has_one_revision_and_status_winner() -> None:
    store = SQLiteStore(":memory:")
    try:
        request = _pending_request()
        store.insert_human_request(request)
        expected = store.get_human_request(request.request_id)
        assert expected is not None

        approved = replace(
            expected,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:01Z",
            revision=expected.revision + 1,
        )
        rejected = replace(
            expected,
            status=HumanRequestStatus.REJECTED,
            decision={"approved": False},
            updated_at="2026-08-05T00:00:02Z",
            revision=expected.revision + 1,
        )

        assert store.compare_and_set_human_request(expected, approved) is True
        assert store.compare_and_set_human_request(expected, rejected) is False

        persisted = store.get_human_request(request.request_id)
        assert persisted == approved
        assert persisted is not None
        assert persisted.revision == 1

        # A matching revision is insufficient when the durable status changed.
        wrong_status = replace(
            persisted,
            status=HumanRequestStatus.PENDING,
            decision=None,
        )
        status_target = replace(
            wrong_status,
            status=HumanRequestStatus.REJECTED,
            decision={"approved": False},
            updated_at="2026-08-05T00:00:03Z",
            revision=wrong_status.revision + 1,
        )
        assert store.compare_and_set_human_request(wrong_status, status_target) is False
        assert store.get_human_request(request.request_id) == approved
    finally:
        store.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_id", "human-revision-cas-other"),
        ("pid", "pid-human-revision-cas-other"),
        ("human", "another-operator"),
        ("blocking", False),
        ("created_at", "2026-08-05T00:00:01Z"),
    ],
)
def test_human_request_compare_and_set_rejects_identity_changes_before_write(
    field: str,
    value: object,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        expected = _pending_request()
        store.insert_human_request(expected)
        target = replace(
            expected,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:01Z",
            revision=1,
            **{field: value},
        )

        with pytest.raises(ValidationError):
            store.compare_and_set_human_request(expected, target)

        assert store.get_human_request(expected.request_id) == expected
    finally:
        store.close()


def test_human_request_compare_and_set_requires_exact_next_revision() -> None:
    store = SQLiteStore(":memory:")
    try:
        expected = _pending_request()
        store.insert_human_request(expected)
        target = replace(
            expected,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:01Z",
            revision=expected.revision + 2,
        )

        with pytest.raises(ValidationError):
            store.compare_and_set_human_request(expected, target)

        assert store.get_human_request(expected.request_id) == expected
    finally:
        store.close()


def test_human_request_revision_prevents_pending_status_aba() -> None:
    store = SQLiteStore(":memory:")
    try:
        original = _pending_request("human-status-aba")
        store.insert_human_request(original)
        delivered = replace(
            original,
            status=HumanRequestStatus.DELIVERED,
            decision={"delivery_committed": True},
            updated_at="2026-08-05T00:00:01Z",
            revision=1,
        )
        restored_pending = replace(
            delivered,
            status=HumanRequestStatus.PENDING,
            decision={"provider_not_started": True},
            updated_at="2026-08-05T00:00:02Z",
            revision=2,
        )
        stale_approval = replace(
            original,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:03Z",
            revision=1,
        )

        assert store.compare_and_set_human_request(original, delivered) is True
        assert store.compare_and_set_human_request(delivered, restored_pending) is True
        assert store.compare_and_set_human_request(original, stale_approval) is False

        persisted = store.get_human_request(original.request_id)
        assert persisted == restored_pending
        assert persisted is not None
        assert persisted.status is HumanRequestStatus.PENDING
        assert persisted.revision == 2
    finally:
        store.close()


def test_legacy_human_request_update_is_a_revision_cas() -> None:
    store = SQLiteStore(":memory:")
    try:
        original = _pending_request()
        store.insert_human_request(original)
        approved = replace(
            original,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:01Z",
        )
        stale_rejection = replace(
            original,
            status=HumanRequestStatus.REJECTED,
            decision={"approved": False},
            updated_at="2026-08-05T00:00:02Z",
        )

        assert store.update_human_request(approved) is True
        assert approved.revision == 1
        assert store.update_human_request(stale_rejection) is False
        assert stale_rejection.revision == 0

        persisted = store.get_human_request(original.request_id)
        assert persisted == approved
        assert persisted is not None
        assert persisted.revision == 1
    finally:
        store.close()


def test_human_request_revision_cas_survives_sqlite_reopen(tmp_path: Path) -> None:
    database = tmp_path / "human-request-revision.sqlite"
    request = _pending_request()

    store = SQLiteStore(database)
    try:
        store.insert_human_request(request)
        approved = replace(
            request,
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True},
            updated_at="2026-08-05T00:00:01Z",
            revision=1,
        )
        assert store.compare_and_set_human_request(request, approved) is True
    finally:
        store.close()

    reopened = SQLiteStore(database)
    try:
        approved = reopened.get_human_request(request.request_id)
        assert approved is not None
        assert approved.revision == 1
        delivered = replace(
            approved,
            status=HumanRequestStatus.DELIVERED,
            updated_at="2026-08-05T00:00:02Z",
            revision=2,
        )
        assert reopened.compare_and_set_human_request(approved, delivered) is True
    finally:
        reopened.close()

    reopened_again = SQLiteStore(database)
    try:
        delivered = reopened_again.get_human_request(request.request_id)
        assert delivered is not None
        assert delivered.status is HumanRequestStatus.DELIVERED
        assert delivered.revision == 2
    finally:
        reopened_again.close()
