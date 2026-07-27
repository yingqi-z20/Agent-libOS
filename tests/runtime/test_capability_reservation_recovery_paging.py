from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    Capability,
    CapabilityRight,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.storage import PostgresStore, SQLiteStore, UnitOfWork


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_capability_reservation_recovery_is_paged_bounded_and_cursor_stable(
    backend: str,
) -> None:
    with _store_for_backend(backend, page_size=2) as store:
        created_at = "2026-01-01T00:00:00Z"
        reservation_ids = tuple(
            f"reservation-{index:02d}" for index in range(5)
        )
        with store.transaction() as cursor:
            cursor.executemany(
                """
                INSERT INTO capability_use_reservations (
                    reservation_id, cap_id, count, status, reserved_by,
                    reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        reservation_id,
                        f"cap-{reservation_id}",
                        1,
                        "reserved",
                        "crashed-worker",
                        "injected stale reservation",
                        created_at,
                        created_at,
                    )
                    for reservation_id in reversed(reservation_ids)
                ],
            )
            cursor.execute(
                """
                INSERT INTO capability_use_reservations (
                    reservation_id, cap_id, count, status, reserved_by,
                    reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "committed-reservation",
                    "committed-cap",
                    1,
                    "committed",
                    "completed-worker",
                    "terminal reservation must be ignored",
                    created_at,
                    created_at,
                ),
            )

        summary = UnitOfWork(
            store
        ).authority.abandon_stale_capability_use_reservations(
            require_recovery_lease=lambda: None,
        )

        assert summary.total_count == 5
        assert summary.sample_reservation_ids == reservation_ids[:2]
        assert summary.truncated
        rows = store.select_table_rows(
            "capability_use_reservations",
            order_by="reservation_id",
        )
        status_by_id = {
            str(row["reservation_id"]): str(row["status"])
            for row in rows
        }
        assert {
            status_by_id[reservation_id]
            for reservation_id in reservation_ids
        } == {"abandoned"}
        assert status_by_id["committed-reservation"] == "committed"


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_prepared_effect_recovery_precedes_stale_reservation_abandonment(
    backend: str,
    tmp_path: Path,
) -> None:
    with _runtime_target(backend, tmp_path, page_size=1) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            actor = "prepared-recovery-actor"
            contract_name = "primitive.test.prepared_recovery_order"
            effect_id = "effect-prepared-recovery-order"
            now = "2026-01-01T00:00:00Z"
            operation = runtime.operations.start(
                kind="primitive",
                name=contract_name,
                actor=actor,
                pid=actor,
            )
            with runtime.operations.attach(operation.operation_id):
                with runtime.store.transaction():
                    prepared_capability, prepared_reservation_id = (
                        _reserve_capability_use(
                            runtime,
                            subject=actor,
                            resource="test:prepared-recovery-order",
                            used_by=actor,
                            reason=(
                                "protected operation reserved authority for "
                                f"{contract_name}"
                            ),
                        )
                    )
                    runtime.store.insert_external_effect(
                        ExternalEffectRecord(
                            effect_id=effect_id,
                            record_id=None,
                            event_id=None,
                            pid=actor,
                            provider="test",
                            operation="prepared_recovery_order",
                            target="test:prepared-recovery-order",
                            rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                            rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                            state_mutation=True,
                            information_flow=False,
                            provider_metadata={
                                "protected_operation": {
                                    "contract_name": contract_name,
                                    "actor": actor,
                                    "reservation_ids": [prepared_reservation_id],
                                    "prepared_recovery": None,
                                }
                            },
                            created_at=now,
                            effect_state="pending",
                            transaction_state="prepared",
                            updated_at=now,
                        )
                    )
                    runtime.operations.expect("effect")
                    assert runtime.operations.link_evidence(
                        "external_effect",
                        effect_id,
                        "effect",
                        metadata={
                            "effect_state": "pending",
                            "provider": "test",
                            "operation": "prepared_recovery_order",
                        },
                    ) is not None
                    assert runtime.operations.link_evidence(
                        "capability_reservation",
                        prepared_reservation_id,
                        "effect_reservation",
                        metadata={
                            "effect_id": effect_id,
                            "capability_id": prepared_capability.cap_id,
                            "count": 1,
                            "contract_name": contract_name,
                            "actor": actor,
                        },
                    ) is not None

            stale_capability, stale_reservation_id = _reserve_capability_use(
                runtime,
                subject="crashed-worker",
                resource="test:stale-recovery-order",
                used_by="crashed-worker",
                reason="injected unrelated stale reservation",
            )
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.recovered_prepared_operations.total_count == 1
            assert reopened.recovered_prepared_operations.sample_effect_ids == (
                effect_id,
            )
            assert reopened.recovered_capability_use_reservations.total_count == 1
            assert (
                reopened.recovered_capability_use_reservations.sample_reservation_ids
                == (stale_reservation_id,)
            )

            restored = reopened.store.get_capability(prepared_capability.cap_id)
            stale = reopened.store.get_capability(stale_capability.cap_id)
            assert restored is not None and restored.uses_remaining == 1
            assert stale is not None and stale.uses_remaining == 0
            assert reopened.store.get_external_effect(effect_id) is None
            prepared_reservation = reopened.store.get_capability_use_reservation(
                prepared_reservation_id
            )
            stale_reservation = reopened.store.get_capability_use_reservation(
                stale_reservation_id
            )
            assert prepared_reservation is not None
            assert prepared_reservation["status"] == "restored"
            assert stale_reservation is not None
            assert stale_reservation["status"] == "abandoned"
        finally:
            reopened.close()


def test_open_runtime_rejects_capability_reservation_recovery_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open()
    transaction_attempted = False

    def unexpected_transaction(*_args: object, **_kwargs: object) -> object:
        nonlocal transaction_attempted
        transaction_attempted = True
        raise AssertionError("recovery queried storage before validating its lease")

    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(runtime.store, "transaction", unexpected_transaction)
            with pytest.raises(
                RuntimeError,
                match="active startup recovery lease",
            ):
                runtime.store.abandon_stale_capability_use_reservations(
                    require_recovery_lease=runtime.lifecycle.require_recovery_lease,
                )
            assert not transaction_attempted
    finally:
        runtime.close()


def test_sqlite_capability_reservation_recovery_uses_covering_index() -> None:
    store = SQLiteStore(":memory:", config=_config(page_size=2))
    try:
        initial = list(
            store.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT reservation_id, created_at
                  FROM capability_use_reservations
                 WHERE status = ?
                 ORDER BY created_at, reservation_id
                 LIMIT ?
                """,
                ("reserved", 2),
            )
        )
        resumed = list(
            store.conn.execute(
                """
                EXPLAIN QUERY PLAN
                SELECT reservation_id, created_at
                 FROM capability_use_reservations
                 WHERE status = ?
                   AND (created_at, reservation_id) > (?, ?)
                 ORDER BY created_at, reservation_id
                 LIMIT ?
                """,
                (
                    "reserved",
                    "2026-01-01T00:00:00Z",
                    "reservation-01",
                    2,
                ),
            )
        )

        assert "idx_capability_reservations_recovery" in "\n".join(
            str(row["detail"]) for row in initial
        )
        assert "idx_capability_reservations_recovery" in "\n".join(
            str(row["detail"]) for row in resumed
        )
        assert "(created_at,reservation_id)>" in "\n".join(
            str(row["detail"]) for row in resumed
        ).replace(" ", "")
    finally:
        store.close()


def _reserve_capability_use(
    runtime: Runtime,
    *,
    subject: str,
    resource: str,
    used_by: str,
    reason: str,
) -> tuple[Capability, str]:
    capability = runtime.capability.issue_trusted(
        subject,
        resource,
        [CapabilityRight.READ],
        issued_by="recovery-contract-test",
        uses_remaining=1,
    )
    decision = runtime.capability.authorize(
        subject,
        resource,
        CapabilityRight.READ,
    )
    reservation_id = runtime.capability.reserve_decision_use(
        decision,
        used_by=used_by,
        reason=reason,
    )
    assert reservation_id is not None
    return capability, reservation_id


@contextlib.contextmanager
def _store_for_backend(
    backend: str,
    *,
    page_size: int,
) -> Iterator[SQLiteStore | PostgresStore]:
    postgres_context = contextlib.nullcontext(None)
    if backend == "sqlite":
        store: SQLiteStore | PostgresStore | None = SQLiteStore(
            ":memory:",
            config=_config(page_size=page_size),
        )
    elif backend == "postgres":
        postgres_context = _postgres_schema_dsn()
        store = None
    else:
        raise AssertionError(f"unknown backend: {backend}")

    with postgres_context as dsn:
        if dsn is not None:
            store = PostgresStore(
                dsn,
                config=_config(dsn=dsn, page_size=page_size),
            )
        assert store is not None
        try:
            yield store
        finally:
            store.close()


@contextlib.contextmanager
def _runtime_target(
    backend: str,
    tmp_path: Path,
    *,
    page_size: int,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if backend == "sqlite":
        yield (
            tmp_path / "capability-reservation-recovery.sqlite",
            _config(page_size=page_size),
        )
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        yield dsn, _config(dsn=dsn, page_size=page_size)


def _config(
    *,
    page_size: int,
    dsn: str | None = None,
) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        runtime=RuntimeDefaults(
            store_backend="postgres" if dsn is not None else "sqlite",
            store_dsn=dsn,
            capability_use_reservation_recovery_page_size=page_size,
            capability_use_reservation_recovery_page_hard_limit=max(
                page_size,
                3,
            ),
        )
    )


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_cap_res_recovery_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )
