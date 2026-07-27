from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.evidence.external_effects import (
    iter_external_effect_recovery,
    prepare_external_effect_intent,
    reconcile_pending_external_effects,
)
from agent_libos.models import (
    ExternalEffectCursor,
    ExternalEffectPage,
    ExternalEffectRecord,
    ExternalEffectRecoveryQuery,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.sdk.protected_operations import ProtectedOperationSDK
from agent_libos.storage import SQLiteStore


def _effect(
    effect_id: str,
    created_at: str,
    *,
    effect_state: str = "pending",
    transaction_state: str = "dispatched",
    idempotency_key: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid="pid_recovery",
        provider="provider",
        operation="write",
        target="target",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=True,
        information_flow=False,
        provider_metadata=dict(metadata or {}),
        created_at=created_at,
        effect_state=effect_state,
        transaction_state=transaction_state,
        idempotency_key=idempotency_key,
        updated_at=created_at,
    )


class _PagedEffects:
    def __init__(self, records: list[ExternalEffectRecord]) -> None:
        self.records = {record.effect_id: record for record in records}
        self.operation_records: dict[str, Any] = {}
        self.operation_evidence: list[Any] = []
        self.queries: list[ExternalEffectRecoveryQuery] = []
        self.idempotency_lookups: list[tuple[str, str]] = []
        self.abandoned: list[str] = []

    def query_external_effect_recovery(
        self,
        query: ExternalEffectRecoveryQuery,
    ) -> ExternalEffectPage:
        self.queries.append(query)
        eligible = sorted(
            (
                record
                for record in self.records.values()
                if record.effect_state == query.effect_state
                and (
                    not query.transaction_states
                    or record.transaction_state in query.transaction_states
                )
                and (
                    query.after is None
                    or ExternalEffectCursor(record.created_at, record.effect_id)
                    > query.after
                )
            ),
            key=lambda record: (record.created_at, record.effect_id),
        )
        selected = tuple(eligible[: query.limit])
        cursor = None
        if len(eligible) > query.limit:
            last = selected[-1]
            cursor = ExternalEffectCursor(last.created_at, last.effect_id)
        return ExternalEffectPage(records=selected, next_cursor=cursor)

    def list_external_effects(self, **_kwargs: Any) -> list[ExternalEffectRecord]:
        raise AssertionError("startup recovery must not scan external-effect history")

    def get_external_effect(self, effect_id: str) -> ExternalEffectRecord | None:
        return self.records.get(effect_id)

    def get_external_effect_by_idempotency(
        self,
        pid: str,
        idempotency_key: str,
    ) -> ExternalEffectRecord | None:
        self.idempotency_lookups.append((pid, idempotency_key))
        return next(
            (
                record
                for record in self.records.values()
                if record.pid == pid and record.idempotency_key == idempotency_key
            ),
            None,
        )

    def insert_external_effect(self, record: ExternalEffectRecord) -> None:
        self.records[record.effect_id] = record

    def finalize_external_effect(
        self,
        effect_id: str,
        record: ExternalEffectRecord,
    ) -> bool:
        if effect_id not in self.records:
            return False
        self.records[effect_id] = record
        return True

    def transition_external_effect(
        self,
        effect_id: str,
        *,
        transaction_state: str,
        provider_metadata: dict[str, Any],
        provider_receipt: dict[str, Any] | None = None,
        updated_at: str,
        **_kwargs: Any,
    ) -> bool:
        current = self.records.get(effect_id)
        if current is None:
            return False
        self.records[effect_id] = replace(
            current,
            transaction_state=transaction_state,
            provider_metadata=provider_metadata,
            provider_receipt=dict(provider_receipt or {}),
            updated_at=updated_at,
        )
        return True

    def abandon_external_effect_intent(self, effect_id: str) -> bool:
        if effect_id not in self.records:
            return False
        self.records.pop(effect_id)
        self.abandoned.append(effect_id)
        return True

    def get_capability_use_reservation(self, _reservation_id: str) -> None:
        return None

    def list_operation_evidence(
        self,
        *,
        operation_ids: tuple[str, ...] | None = None,
        evidence_types: tuple[str, ...] | None = None,
        evidence_id: str | None = None,
        limit: int | None = None,
        **_kwargs: Any,
    ) -> list[Any]:
        selected = [
            link
            for link in self.operation_evidence
            if (operation_ids is None or link.operation_id in operation_ids)
            and (evidence_types is None or link.evidence_type in evidence_types)
            and (evidence_id is None or link.evidence_id == evidence_id)
        ]
        return selected if limit is None else selected[:limit]

    def get_operation(self, operation_id: str) -> Any | None:
        return self.operation_records.get(operation_id)

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        del include_object_payloads
        yield self


class _OperationLinks:
    def __init__(self) -> None:
        self.links: list[tuple[Any, ...]] = []

    def link_evidence(self, *args: Any, **kwargs: Any) -> None:
        self.links.append((*args, kwargs))


def test_sqlite_recovery_query_is_filtered_keyset_paged_and_directly_indexed() -> None:
    store = SQLiteStore(":memory:")
    try:
        for record in (
            _effect(
                "effect_final",
                "2026-01-01T00:00:00Z",
                effect_state="finalized",
                transaction_state="committed",
            ),
            _effect(
                "effect_prepared",
                "2026-01-01T00:00:01Z",
                transaction_state="prepared",
            ),
            _effect(
                "effect_dispatch_1",
                "2026-01-01T00:00:02Z",
                idempotency_key="stable-key",
            ),
            _effect("effect_dispatch_2", "2026-01-01T00:00:03Z"),
        ):
            store.insert_external_effect(record)

        first = store.query_external_effect_recovery(
            ExternalEffectRecoveryQuery(
                transaction_states=("dispatched",),
                limit=1,
            )
        )
        assert [record.effect_id for record in first.records] == ["effect_dispatch_1"]
        assert first.next_cursor == ExternalEffectCursor(
            "2026-01-01T00:00:02Z",
            "effect_dispatch_1",
        )
        second = store.query_external_effect_recovery(
            ExternalEffectRecoveryQuery(
                transaction_states=("dispatched",),
                after=first.next_cursor,
                limit=1,
            )
        )
        assert [record.effect_id for record in second.records] == ["effect_dispatch_2"]
        assert second.next_cursor is None
        plan = store._query(
            "EXPLAIN QUERY PLAN SELECT * FROM external_effects "
            "WHERE effect_state = ? AND transaction_state IN (?) "
            "AND (created_at, effect_id) > (?, ?) "
            "ORDER BY created_at, effect_id LIMIT ?",
            (
                "pending",
                "dispatched",
                first.next_cursor.created_at,
                first.next_cursor.effect_id,
                1,
            ),
        )
        details = "\n".join(str(row["detail"]) for row in plan)
        assert "idx_external_effects_recovery_transaction" in details
        assert "(created_at,effect_id)>" in details.replace(" ", "")
        assert store.get_external_effect_by_idempotency(
            "pid_recovery",
            "stable-key",
        ) == first.records[0]
        with pytest.raises(ValidationError, match="hard limit"):
            store.query_external_effect_recovery(
                ExternalEffectRecoveryQuery(limit=5_001)
            )
    finally:
        store.close()


def test_recovery_iterator_walks_pages_without_full_history_scan() -> None:
    effects = _PagedEffects(
        [
            _effect("effect_1", "2026-01-01T00:00:01Z"),
            _effect("effect_2", "2026-01-01T00:00:02Z"),
            _effect("effect_3", "2026-01-01T00:00:03Z"),
        ]
    )

    recovered = list(
        iter_external_effect_recovery(
            effects,
            ExternalEffectRecoveryQuery(limit=1),
        )
    )

    assert [record.effect_id for record in recovered] == [
        "effect_1",
        "effect_2",
        "effect_3",
    ]
    assert [query.after for query in effects.queries] == [
        None,
        ExternalEffectCursor("2026-01-01T00:00:01Z", "effect_1"),
        ExternalEffectCursor("2026-01-01T00:00:02Z", "effect_2"),
    ]


def test_prepared_sdk_recovery_uses_filtered_pages() -> None:
    effect = _effect(
        "effect_prepared",
        "2026-01-01T00:00:01Z",
        transaction_state="prepared",
        metadata={
            "protected_operation": {
                "contract_name": "primitive.test.write",
                "actor": "pid_recovery",
                "reservation_ids": [],
            }
        },
    )
    effects = _PagedEffects([effect])
    operation_id = "operation_prepared"
    effects.operation_records[operation_id] = SimpleNamespace(
        operation_id=operation_id,
        name="primitive.test.write",
        actor="pid_recovery",
        pid="pid_recovery",
    )
    effects.operation_evidence.append(
        SimpleNamespace(
            operation_id=operation_id,
            evidence_type="external_effect",
            evidence_id=effect.effect_id,
            role="effect",
            metadata={
                "effect_state": "pending",
                "provider": effect.provider,
                "operation": effect.operation,
            },
        )
    )
    operations = _OperationLinks()
    sdk = ProtectedOperationSDK(
        effects=effects,
        authority_policy=SimpleNamespace(),
        capabilities=SimpleNamespace(restore_reserved_use=lambda *_args, **_kwargs: None),
        audit=SimpleNamespace(),
        events=SimpleNamespace(),
        resources=None,
        operations=operations,
        require_recovery_lease=lambda: None,
    )

    summary = sdk.recover_prepared(page_size=1)
    assert summary.total_count == 1
    assert summary.sample_effect_ids == ("effect_prepared",)
    assert effects.abandoned == ["effect_prepared"]
    assert effects.queries == [
        ExternalEffectRecoveryQuery(transaction_states=("prepared",), limit=1)
    ]


def test_provider_reconciliation_uses_bounded_pages() -> None:
    effects = _PagedEffects(
        [
            _effect("effect_1", "2026-01-01T00:00:01Z"),
            _effect("effect_2", "2026-01-01T00:00:02Z"),
        ]
    )
    provider = SimpleNamespace(
        reconcile_external_effect=lambda record: {
            "state": "committed",
            "provider_receipt": {"effect_id": record.effect_id},
        }
    )

    reconciled = reconcile_pending_external_effects(
        effects,
        SimpleNamespace(provider=provider),
        require_recovery_lease=lambda: None,
        page_size=1,
    )

    assert reconciled.total_count == 2
    assert reconciled.sample_effect_ids == ("effect_1",)
    assert reconciled.truncated
    assert all(
        record.effect_state == "finalized"
        for record in effects.records.values()
    )
    assert [query.limit for query in effects.queries] == [1, 1]


def test_intent_idempotency_uses_direct_lookup() -> None:
    effects = _PagedEffects([])

    record = prepare_external_effect_intent(
        effects,
        pid="pid_recovery",
        provider="provider",
        operation="write",
        target="target",
        state_mutation=True,
        information_flow=False,
        idempotency_key="direct-key",
        canonical_args={"value": 1},
    )

    assert record.idempotency_key == "direct-key"
    assert effects.idempotency_lookups == [("pid_recovery", "direct-key")]
