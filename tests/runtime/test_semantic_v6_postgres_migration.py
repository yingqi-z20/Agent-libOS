from __future__ import annotations

import threading

import pytest

import agent_libos.storage.semantic_v6_migration as semantic_v6_migration
from agent_libos.storage import (
    SemanticControlStateRecord,
    SemanticHumanOutcomeLinkRecord,
    SemanticMachineOutcomeRecord,
    SemanticMachineSettlementRecord,
)
from agent_libos.storage.postgres import PostgresStore, _postgres_runtime_lock_key
from agent_libos.storage.semantic_v6_migration import (
    StoreV6MigrationError,
    apply_store_v6_migration,
    plan_store_v6_migration,
)
from agent_libos.storage.mcp_v7_migration import (
    apply_store_v7_migration,
    plan_store_v7_migration,
)
from agent_libos.storage.v6_schema_contract import V6_TABLES
from agent_libos.storage.v7_schema_contract import V7_TABLES
from tests.runtime.test_semantic_v5_postgres_migration import _postgres_schema_dsn


def _issued_settlement(identity: str) -> SemanticMachineSettlementRecord:
    return SemanticMachineSettlementRecord(
        settlement_id=identity,
        assessment_id=None,
        job_id=None,
        request_id=f"request-{identity}",
        request_revision=0,
        pid="pid-postgres-recovery",
        operation_id=None,
        effect_id=f"effect-{identity}",
        epoch_id="epoch-postgres-recovery",
        policy_sha256="a" * 64,
        tenant_bucket_sha256="b" * 64,
        action_id="filesystem.read",
        outcome="issued",
        capability_id=f"capability-{identity}",
        binding_sha256="c" * 64,
        decision_sha256="d" * 64,
        matched_rule_id="rule-postgres-recovery",
        reason_codes=("policy_match",),
        created_at="2026-08-07T00:00:00+00:00",
    )


def _downgrade_to_v5(dsn: str, *, assessment_count: int = 0) -> None:
    import psycopg
    from psycopg import sql

    PostgresStore(dsn).close()
    with psycopg.connect(dsn, autocommit=True) as connection:
        for table in sorted(V7_TABLES | V6_TABLES):
            connection.execute(
                sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
            )
        connection.execute(
            "UPDATE runtime_schema SET schema_version = 5 WHERE singleton = 1"
        )
        for index in range(assessment_count):
            connection.execute(
                "INSERT INTO semantic_assessments "
                "(assessment_id, job_id, kind, status, domain, action_id, "
                "tenant_bucket_sha256, pid, request_id, operation_id, effect_id, "
                "shadow_outcome, ood, record_json, created_at, completed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                (
                    f"legacy-pg-assessment-{index}",
                    f"legacy-pg-job-{index}",
                    "approval",
                    "succeeded",
                    "filesystem",
                    "filesystem.read",
                    "a" * 64,
                    f"legacy-pg-pid-{index}",
                    f"legacy-pg-request-{index}",
                    None,
                    None,
                    "require_human",
                    0,
                    "{}",
                    "2026-08-07T00:00:00+00:00",
                    "2026-08-07T00:00:00+00:00",
                ),
            )


@pytest.mark.postgres
def test_postgres_v5_to_v6_migration_round_trip() -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for table in sorted(V7_TABLES | V6_TABLES):
                connection.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
                )
            connection.execute(
                "UPDATE runtime_schema SET schema_version = 5 WHERE singleton = 1"
            )
            connection.execute(
                "INSERT INTO semantic_assessments "
                "(assessment_id, job_id, kind, status, domain, action_id, "
                "tenant_bucket_sha256, pid, request_id, operation_id, effect_id, "
                "shadow_outcome, ood, record_json, created_at, completed_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "%s, %s, %s, %s, %s)",
                (
                    "legacy-pg-assessment",
                    "legacy-pg-job",
                    "approval",
                    "succeeded",
                    "filesystem",
                    "filesystem.read",
                    "a" * 64,
                    "legacy-pg-pid",
                    "legacy-pg-request",
                    None,
                    None,
                    "require_human",
                    0,
                    "{}",
                    "2026-08-07T00:00:00+00:00",
                    "2026-08-07T00:00:00+00:00",
                ),
            )

        plan = plan_store_v6_migration(dsn)
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (5,)
            assert connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'semantic_flow_entities'"
            ).fetchone() == (0,)
        result = apply_store_v6_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )

        assert result.applied
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (6,)
        v7_plan = plan_store_v7_migration(dsn)
        apply_store_v7_migration(
            dsn,
            expected_plan_sha256=v7_plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )
        reopened = PostgresStore(dsn)
        try:
            assert reopened.conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == {"schema_version": 7}
            legacy = reopened.get_semantic_legacy_coverage()
            assert legacy is not None
            assert legacy.assessment_count == 1
            assert legacy.coverage == "unknown"
            human_link = SemanticHumanOutcomeLinkRecord(
                link_id="human-link-pg",
                request_id="human-request-pg",
                request_revision=2,
                pid="human-pid-pg",
                assessment_id=None,
                job_id=None,
                settlement_id=None,
                outcome="rejected",
                source="human",
                decision_sha256="b" * 64,
                created_at="2026-08-07T00:00:00+00:00",
            )
            assert reopened.append_semantic_human_outcome_link(human_link) == human_link
            assert reopened.get_semantic_human_outcome_link_for_request(
                human_link.request_id
            ) == human_link
        finally:
            reopened.close()


@pytest.mark.postgres
def test_postgres_v5_to_v6_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for table in sorted(V7_TABLES | V6_TABLES):
                connection.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
                )
            connection.execute(
                "UPDATE runtime_schema SET schema_version = 5 WHERE singleton = 1"
            )
        plan = plan_store_v6_migration(dsn)
        real_execute = semantic_v6_migration._execute_v6_ddl

        def fail_after_ddl(connection: object) -> None:
            real_execute(connection)
            raise StoreV6MigrationError("injected PostgreSQL migration failure")

        monkeypatch.setattr(
            semantic_v6_migration,
            "_execute_v6_ddl",
            fail_after_ddl,
        )
        with pytest.raises(StoreV6MigrationError, match="injected PostgreSQL"):
            apply_store_v6_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )

        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (5,)
            assert connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'semantic_flow_entities'"
            ).fetchone() == (0,)


@pytest.mark.postgres
@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_postgres_v6_reconciles_exact_target_after_uncertain_commit(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v5(dsn, assessment_count=2)
        plan = plan_store_v6_migration(dsn)
        if fault == "commit_ack":
            real_commit = semantic_v6_migration._PostgresConnection.commit

            def lost_commit_ack(connection: object) -> None:
                real_commit(connection)
                raise RuntimeError("injected lost commit ACK")

            monkeypatch.setattr(
                semantic_v6_migration._PostgresConnection,
                "commit",
                lost_commit_ack,
            )
            expected_error = "lost commit ACK"
        else:
            real_require = semantic_v6_migration._require_canonical_v6
            calls = 0

            def fail_post_commit_readback(
                backend: object,
                connection: object,
            ) -> None:
                nonlocal calls
                real_require(backend, connection)
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected post-commit readback failure")

            monkeypatch.setattr(
                semantic_v6_migration,
                "_require_canonical_v6",
                fail_post_commit_readback,
            )
            expected_error = "post-commit readback"

        with pytest.raises(RuntimeError, match=expected_error):
            apply_store_v6_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (6,)

        result = apply_store_v6_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )
        assert result.applied is False
        assert result.already_applied is True
        assert result.plan == plan


@pytest.mark.postgres
def test_postgres_v6_plan_rejects_a_different_database_schema() -> None:
    with _postgres_schema_dsn() as first_dsn, _postgres_schema_dsn() as second_dsn:
        _downgrade_to_v5(first_dsn, assessment_count=1)
        _downgrade_to_v5(second_dsn, assessment_count=1)
        first_plan = plan_store_v6_migration(first_dsn)
        second_plan = plan_store_v6_migration(second_dsn)

        assert first_plan.source_catalog_sha256 == second_plan.source_catalog_sha256
        assert first_plan.source_digest_sha256 != second_plan.source_digest_sha256
        assert (
            first_plan.database_identity_sha256
            != second_plan.database_identity_sha256
        )
        assert first_plan.plan_sha256 != second_plan.plan_sha256
        with pytest.raises(StoreV6MigrationError, match="plan digest"):
            apply_store_v6_migration(
                second_dsn,
                expected_plan_sha256=first_plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )


@pytest.mark.postgres
def test_postgres_control_fence_serializes_concurrent_rotation() -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        control = SemanticControlStateRecord(
            revision=0,
            generation=0,
            mode="off",
            active_epoch_id=None,
            active_policy_sha256=None,
            tripped=False,
            trip_code=None,
            updated_at="2026-08-07T00:00:00+00:00",
        )
        assert store.compare_and_set_semantic_control_state(None, control)
        started = threading.Event()
        completed = threading.Event()
        errors: list[BaseException] = []

        def rotate() -> None:
            try:
                with psycopg.connect(dsn, autocommit=True) as connection:
                    started.set()
                    connection.execute(
                        "UPDATE semantic_control_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
            except BaseException as exc:
                errors.append(exc)
            finally:
                completed.set()

        contender = threading.Thread(target=rotate, daemon=True)
        with store.transaction():
            assert store.fence_semantic_control_state(control)
            contender.start()
            assert started.wait(timeout=2)
            assert not completed.wait(timeout=0.2)
        assert completed.wait(timeout=5)
        contender.join(timeout=1)
        assert not errors
        assert store.conn.execute(
            "SELECT revision FROM semantic_control_state WHERE singleton = 1"
        ).fetchone() == {"revision": 1}
        store.close()


@pytest.mark.postgres
def test_postgres_v6_apply_requires_snapshot_and_exclusive_advisory_lock() -> None:
    import psycopg
    from psycopg import sql

    with _postgres_schema_dsn() as dsn:
        PostgresStore(dsn).close()
        with psycopg.connect(dsn, autocommit=True) as connection:
            for table in sorted(V7_TABLES | V6_TABLES):
                connection.execute(
                    sql.SQL("DROP TABLE {}").format(sql.Identifier(table))
                )
            connection.execute(
                "UPDATE runtime_schema SET schema_version = 5 WHERE singleton = 1"
            )
        plan = plan_store_v6_migration(dsn)
        with pytest.raises(StoreV6MigrationError, match="snapshot confirmation"):
            apply_store_v6_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
            )

        with psycopg.connect(dsn, autocommit=True) as lock_connection:
            database, schema = lock_connection.execute(
                "SELECT current_database(), current_schema()"
            ).fetchone()
            lease_key = _postgres_runtime_lock_key(str(database), str(schema))
            assert lock_connection.execute(
                "SELECT pg_try_advisory_lock(%s)", (lease_key,)
            ).fetchone() == (True,)
            try:
                with pytest.raises(StoreV6MigrationError, match="already open"):
                    apply_store_v6_migration(
                        dsn,
                        expected_plan_sha256=plan.plan_sha256,
                        postgres_snapshot_confirmed=True,
                    )
            finally:
                assert lock_connection.execute(
                    "SELECT pg_advisory_unlock(%s)", (lease_key,)
                ).fetchone() == (True,)

        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (5,)
            assert connection.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'semantic_flow_entities'"
            ).fetchone() == (0,)


@pytest.mark.postgres
def test_postgres_unresolved_settlement_query_excludes_terminal_outcomes() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            resolved = _issued_settlement("resolved-postgres")
            unresolved = _issued_settlement("unresolved-postgres")
            with store.transaction():
                store.append_semantic_machine_settlement(resolved)
                store.append_semantic_machine_settlement(unresolved)
                store.append_semantic_machine_outcome(
                    SemanticMachineOutcomeRecord(
                        outcome_id="resolved-postgres-succeeded",
                        settlement_id=resolved.settlement_id,
                        effect_id=resolved.effect_id,
                        outcome="succeeded",
                        evidence_sha256="a" * 64,
                        created_at="2026-08-07T00:00:01+00:00",
                    )
                )
            page = store.query_unresolved_semantic_machine_settlements(limit=10)
            assert page.records == (unresolved,)
            assert page.next_cursor is None
        finally:
            store.close()
