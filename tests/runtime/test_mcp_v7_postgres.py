from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

import agent_libos.storage.mcp_v7_migration as mcp_v7_migration
from agent_libos.mcp.manifest import (
    McpResourceSpec,
    McpServerManifestV3,
    canonical_mcp_v3_manifest_json,
)
from agent_libos.models import (
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.mcp import (
    McpProtocolMode,
    McpStdioTransportSpec,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.storage import (
    McpContinuationRecord,
    McpRemoteTaskRecord,
    McpSideEffectPreparationRecord,
    PostgresStore,
    UnitOfWork,
)
from agent_libos.storage.mcp_v7_migration import (
    StoreV7MigrationError,
    apply_store_v7_migration,
    plan_store_v7_migration,
)
from agent_libos.storage.v7_schema_contract import V7_TABLES
from tests.runtime.test_semantic_v5_postgres_migration import _postgres_schema_dsn


def _continuation() -> McpContinuationRecord:
    return McpContinuationRecord(
        continuation_id="continuation-postgres-1",
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=7,
        owner_id="owner-postgres",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        request_sha256="3" * 64,
        effect_id="effect-postgres",
        capability_sha256="4" * 64,
        data_flow_sha256="5" * 64,
        human_request_id="human-postgres",
        broker_ref="broker:postgres-opaque",
        broker_value_sha256="6" * 64,
        status="input_required",
        revision=0,
        expires_at="2026-08-12T00:00:00Z",
        metadata={"automatic_retry_disabled": True},
        created_at="2026-08-11T00:00:00Z",
        updated_at="2026-08-11T00:00:00Z",
    )


def _input_required_task(*, task_ref: str, broker_ref: str) -> McpRemoteTaskRecord:
    return McpRemoteTaskRecord(
        task_ref=task_ref,
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=7,
        owner_id="owner-postgres",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        origin_request_sha256="3" * 64,
        origin_effect_id="effect-postgres",
        human_request_id="human-postgres-task",
        broker_ref=broker_ref,
        remote_id_sha256="4" * 64,
        status="input_required",
        revision=0,
        expires_at="2026-08-12T00:00:00Z",
        poll_interval_ms=0,
        status_message_sha256=None,
        result_ref=None,
        result_sha256=None,
        metadata={"automatic_retry_disabled": True},
        created_at="2026-08-11T00:00:00Z",
        updated_at="2026-08-11T00:00:00Z",
    )


def _side_effect_preparation() -> McpSideEffectPreparationRecord:
    return McpSideEffectPreparationRecord(
        preparation_id="preparation-postgres-1",
        operation_kind="continuation",
        operation_id="continuation-postgres-staged",
        operation_revision=None,
        server_id="server.v3",
        server_spec_sha256="0" * 64,
        server_generation=7,
        owner_id="owner-postgres",
        auth_principal_sha256="1" * 64,
        auth_scope_sha256="2" * 64,
        human_request_id="human-postgres-staged",
        human_preview_sha256="3" * 64,
        broker_ref="broker:postgres-staged",
        broker_value_sha256="6" * 64,
        result_ref=None,
        result_sha256=None,
        status="prepared",
        revision=0,
        expires_at="2026-08-12T00:00:00Z",
        metadata={"cleanup_mode": "abort", "retire_refs": ()},
        created_at="2026-08-11T00:00:00Z",
        updated_at="2026-08-11T00:00:00Z",
    )


def _pending_mcp_effect(*, effect_id: str, pid: str) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider="mcp",
        operation="tools/call",
        target="mcp:server.v3:tool",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=False,
        information_flow=True,
        provider_metadata={},
        created_at="2026-08-11T00:00:00Z",
        effect_state="pending",
        transaction_state="dispatched",
        updated_at="2026-08-11T00:00:00Z",
    )


def _downgrade_to_v6(dsn: str) -> None:
    import psycopg
    from psycopg import sql

    PostgresStore(dsn).close()
    with psycopg.connect(dsn, autocommit=True) as connection:
        for table in sorted(V7_TABLES):
            connection.execute(sql.SQL("DROP TABLE {}").format(sql.Identifier(table)))
        changed = connection.execute(
            "UPDATE runtime_schema SET schema_version = 6 "
            "WHERE singleton = 1 AND schema_version = 7"
        )
        assert changed.rowcount == 1
        connection.execute(
            "INSERT INTO runtime_counters (counter_name, value) VALUES (%s, %s)",
            ("mcp-v7-postgres-sentinel", 7),
        )


def _manifest(*, timeout_s: float = 10.0) -> McpServerManifestV3:
    return McpServerManifestV3(
        schema_version=3,
        server_id="modern.postgres.v3",
        transport="stdio",
        timeout_s=timeout_s,
        max_request_bytes=1024,
        max_response_bytes=4096,
        protocol_mode=McpProtocolMode.REVISION_2026_07_28,
        resources=(
            McpResourceSpec(resource_id="readme", remote_uri="docs://readme"),
        ),
        stdio=McpStdioTransportSpec(command="demo-postgres-v3"),
    )


def _insert_human_request(store: PostgresStore, request_id: str) -> None:
    store.insert_human_request(
        HumanRequest(
            request_id=request_id,
            pid="owner-postgres",
            human="owner",
            payload={"type": "question", "context": {}},
            status=HumanRequestStatus.PENDING,
            decision=None,
            blocking=True,
            created_at="2026-08-11T00:00:00Z",
            updated_at="2026-08-11T00:00:00Z",
        )
    )


@pytest.mark.postgres
def test_postgres_v7_fresh_catalog_reopen_and_atomic_cas() -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        _insert_human_request(store, "human-postgres")
        _insert_human_request(store, "human-postgres-task")
        repository = UnitOfWork(store).mcp_continuations
        original = repository.insert(_continuation())
        replacement = replace(
            original,
            status="dispatching",
            revision=1,
            updated_at="2026-08-11T00:00:01Z",
        )
        assert repository.compare_and_swap(
            original.continuation_id,
            expected_revision=0,
            replacement=replacement,
        )
        assert not repository.compare_and_swap(
            original.continuation_id,
            expected_revision=0,
            replacement=replacement,
        )
        task_repository = UnitOfWork(store).mcp_remote_tasks
        with pytest.raises(ValidationError, match="already bound"):
            task_repository.insert(
                replace(
                    _input_required_task(
                        task_ref="task-postgres-cross-table-conflict",
                        broker_ref="broker:postgres-task-cross-table-conflict",
                    ),
                    human_request_id=original.human_request_id,
                )
            )
        task = task_repository.insert(
            _input_required_task(
                task_ref="task-postgres-1",
                broker_ref="broker:postgres-task-1",
            )
        )
        assert task_repository.count() == 1
        assert task_repository.count(owner_id="owner-postgres") == 1
        assert task_repository.get_by_remote_id_sha256(
            task.server_id,
            task.remote_id_sha256,
        ) == task
        with pytest.raises(ValidationError, match="identity conflicts"):
            task_repository.insert(
                _input_required_task(
                    task_ref="task-postgres-2",
                    broker_ref="broker:postgres-task-2",
                )
            )
        store.close()

        reopened = PostgresStore(dsn)
        try:
            assert UnitOfWork(reopened).mcp_continuations.get(
                original.continuation_id
            ) == replacement
            assert UnitOfWork(reopened).mcp_remote_tasks.get(task.task_ref) == task
            assert reopened.conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == {"schema_version": 7}
        finally:
            reopened.close()

        forbidden = {
            "token", "access_token", "refresh_token", "client_secret",
            "pkce_verifier", "oauth_state", "authorization_code", "raw_content",
            "event_json", "notification_json", "remote_task_id", "request_state",
        }
        with psycopg.connect(dsn, autocommit=True) as connection:
            columns = {
                str(row[0]).casefold()
                for row in connection.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = ANY(%s)",
                    (list(V7_TABLES),),
                )
            }
        assert columns.isdisjoint(forbidden)


@pytest.mark.postgres
def test_postgres_side_effect_preparation_reopens_and_commits_atomically() -> None:
    with _postgres_schema_dsn() as dsn:
        preparation = _side_effect_preparation()
        store = PostgresStore(dsn)
        assert UnitOfWork(store).mcp_side_effects.insert(preparation) == preparation
        store.close()

        reopened = PostgresStore(dsn)
        try:
            unit = UnitOfWork(reopened)
            assert unit.mcp_side_effects.get(preparation.preparation_id) == preparation
            _insert_human_request(reopened, "human-postgres-staged")
            continuation = replace(
                _continuation(),
                continuation_id="continuation-postgres-staged",
                human_request_id="human-postgres-staged",
                broker_ref="broker:postgres-staged",
            )
            assert unit.mcp_side_effects.commit(
                preparation.preparation_id,
                expected_revision=0,
                replacement=continuation,
            )
            retirement = unit.mcp_side_effects.get(preparation.preparation_id)
            assert retirement == replace(
                preparation,
                status="cleaning",
                revision=1,
                metadata={
                    "automatic_retry_disabled": True,
                    "cleanup_mode": "retire",
                    "retire_refs": (),
                },
            )
            assert unit.mcp_continuations.get(continuation.continuation_id) == continuation
        finally:
            reopened.close()

        verified = PostgresStore(dsn)
        try:
            unit = UnitOfWork(verified)
            committed = unit.mcp_continuations.get(
                "continuation-postgres-staged"
            )
            assert committed is not None
            assert unit.mcp_side_effects.list() == (retirement,)
            assert unit.mcp_side_effects.delete(
                retirement.preparation_id,
                expected_revision=retirement.revision,
            )
            dispatching = replace(
                committed,
                status="dispatching",
                revision=1,
                updated_at="2026-08-11T00:00:01Z",
            )
            assert unit.mcp_continuations.compare_and_swap(
                committed.continuation_id,
                expected_revision=0,
                replacement=dispatching,
            )
            complete = replace(
                dispatching,
                status="complete",
                broker_ref=None,
                broker_value_sha256=None,
                revision=2,
                updated_at="2026-08-11T00:00:02Z",
            )
            assert unit.mcp_continuations.compare_and_swap(
                committed.continuation_id,
                expected_revision=1,
                replacement=complete,
            )
            assert unit.mcp_continuations.count_active() == 0
            assert unit.mcp_continuations.list_terminal(limit=1) == (complete,)
            terminal_retirement = unit.mcp_side_effects.insert(
                replace(
                    _side_effect_preparation(),
                    preparation_id="preparation-postgres-terminal",
                    operation_revision=2,
                    human_request_id=None,
                    human_preview_sha256=None,
                    broker_ref=None,
                    broker_value_sha256=None,
                    metadata={
                        "cleanup_mode": "abort",
                        "retire_refs": (),
                        "retire_human_request_id": "human-postgres-staged",
                        "retire_human_preview_sha256": "3" * 64,
                    },
                    created_at="2026-08-11T00:00:03Z",
                    updated_at="2026-08-11T00:00:03Z",
                )
            )
            assert unit.mcp_side_effects.commit_terminal(
                terminal_retirement.preparation_id,
                expected_revision=0,
            )
            cleaning = unit.mcp_side_effects.get(
                terminal_retirement.preparation_id
            )
            assert cleaning is not None and cleaning.status == "cleaning"
            assert unit.mcp_side_effects.delete(
                cleaning.preparation_id,
                expected_revision=cleaning.revision,
            )
            assert verified.get_human_request("human-postgres-staged") is not None
        finally:
            verified.close()


@pytest.mark.postgres
def test_postgres_two_prepared_handoffs_join_one_transaction_and_rollback() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        unit = UnitOfWork(store)
        continuation = replace(
            _continuation(),
            continuation_id="continuation-postgres-composite",
            human_request_id="human-postgres-composite",
            broker_ref="broker:postgres-composite-continuation",
        )
        continuation_preparation = unit.mcp_side_effects.insert(
            replace(
                _side_effect_preparation(),
                preparation_id="preparation-postgres-composite-continuation",
                operation_id=continuation.continuation_id,
                human_request_id=continuation.human_request_id,
                broker_ref=continuation.broker_ref,
            )
        )
        task = replace(
            _input_required_task(
                task_ref="task-postgres-composite",
                broker_ref="broker:postgres-composite-task",
            ),
            human_request_id=None,
            status="working",
        )
        task_preparation = unit.mcp_side_effects.insert(
            replace(
                _side_effect_preparation(),
                preparation_id="preparation-postgres-composite-task",
                operation_kind="remote_task",
                operation_id=task.task_ref,
                human_request_id=None,
                human_preview_sha256=None,
                broker_ref=task.broker_ref,
                broker_value_sha256=task.remote_id_sha256,
            )
        )
        _insert_human_request(store, continuation.human_request_id)
        pending_effect = _pending_mcp_effect(
            effect_id=task.origin_effect_id,
            pid=task.owner_id,
        )
        unit.evidence.insert_external_effect(pending_effect)
        committed_effect = replace(
            pending_effect,
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            effect_state="finalized",
            transaction_state="committed",
            provider_receipt={
                "mcp_durable_result": {
                    "kind": "remote_task",
                    "task_ref": task.task_ref,
                }
            },
            updated_at="2026-08-11T00:00:01Z",
        )

        with pytest.raises(RuntimeError, match="rollback postgres composite"):
            with unit.transaction():
                assert unit.mcp_side_effects.commit(
                    continuation_preparation.preparation_id,
                    expected_revision=0,
                    replacement=continuation,
                )
                assert unit.mcp_side_effects.commit(
                    task_preparation.preparation_id,
                    expected_revision=0,
                    replacement=task,
                )
                assert unit.evidence.finalize_external_effect(
                    pending_effect.effect_id,
                    committed_effect,
                )
                raise RuntimeError("rollback postgres composite")
        assert unit.mcp_continuations.get(continuation.continuation_id) is None
        assert unit.mcp_remote_tasks.get(task.task_ref) is None
        assert unit.mcp_side_effects.get(
            continuation_preparation.preparation_id
        ) == continuation_preparation
        assert unit.mcp_side_effects.get(
            task_preparation.preparation_id
        ) == task_preparation
        assert unit.evidence.get_external_effect(pending_effect.effect_id) == pending_effect
        store.close()

        reopened = PostgresStore(dsn)
        try:
            unit = UnitOfWork(reopened)
            with unit.transaction():
                assert unit.mcp_side_effects.commit(
                    continuation_preparation.preparation_id,
                    expected_revision=0,
                    replacement=continuation,
                )
                assert unit.mcp_side_effects.commit(
                    task_preparation.preparation_id,
                    expected_revision=0,
                    replacement=task,
                )
                assert unit.evidence.finalize_external_effect(
                    pending_effect.effect_id,
                    committed_effect,
                )
            assert unit.mcp_continuations.get(continuation.continuation_id) == continuation
            assert unit.mcp_remote_tasks.get(task.task_ref) == task
            assert unit.evidence.get_external_effect(pending_effect.effect_id) == committed_effect
            for preparation in (continuation_preparation, task_preparation):
                retirement = unit.mcp_side_effects.get(preparation.preparation_id)
                assert retirement is not None and retirement.status == "cleaning"
                assert unit.mcp_side_effects.delete(
                    retirement.preparation_id,
                    expected_revision=retirement.revision,
                )
        finally:
            reopened.close()


@pytest.mark.postgres
def test_postgres_v3_registry_digest_cas_round_trip() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        extension = UnitOfWork(store).extensions
        manifest = _manifest()
        assert extension.compare_and_swap_mcp_v3_server(
            manifest,
            expected_current_sha256=None,
            registered_by="host",
            created_at="2026-08-11T00:00:00Z",
        )
        binding = extension.get_mcp_registry_binding(manifest.server_id)
        assert not extension.compare_and_swap_mcp_v3_server(
            _manifest(timeout_s=11.0),
            expected_current_sha256="f" * 64,
            registered_by="host",
            created_at="2026-08-11T00:00:01Z",
        )
        assert extension.get_mcp_registry_binding(manifest.server_id) == binding
        current_sha256 = hashlib.sha256(
            canonical_mcp_v3_manifest_json(manifest).encode("utf-8")
        ).hexdigest()
        replacement = _manifest(timeout_s=11.0)
        assert extension.compare_and_swap_mcp_v3_server(
            replacement,
            expected_current_sha256=current_sha256,
            registered_by="host",
            created_at="2026-08-11T00:00:01Z",
        )
        store.close()

        reopened = PostgresStore(dsn)
        try:
            persisted = UnitOfWork(reopened).extensions.get_mcp_v3_server(
                manifest.server_id
            )
            assert persisted is not None
            assert persisted[0] == replacement
        finally:
            reopened.close()


@pytest.mark.postgres
def test_postgres_v6_to_v7_plan_apply_and_reopen() -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v6(dsn)
        plan = plan_store_v7_migration(dsn)
        assert plan.backend == "postgres"
        assert plan.from_schema_version == 6
        assert plan.to_schema_version == 7
        with pytest.raises(StoreV7MigrationError, match="snapshot confirmation"):
            apply_store_v7_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (6,)

        result = apply_store_v7_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )

        assert result.applied
        reopened = PostgresStore(dsn)
        try:
            assert reopened.conn.execute(
                "SELECT value FROM runtime_counters WHERE counter_name = ?",
                ("mcp-v7-postgres-sentinel",),
            ).fetchone() == {"value": 7}
            present = {
                str(row["table_name"])
                for row in reopened.conn.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema()"
                )
            }
            assert V7_TABLES <= present
        finally:
            reopened.close()


@pytest.mark.postgres
def test_postgres_v6_to_v7_failure_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v6(dsn)
        plan = plan_store_v7_migration(dsn)
        real_execute = mcp_v7_migration._execute_v7_ddl

        def fail_after_ddl(connection: object) -> None:
            real_execute(connection)
            raise StoreV7MigrationError("injected PostgreSQL schema-v7 failure")

        monkeypatch.setattr(mcp_v7_migration, "_execute_v7_ddl", fail_after_ddl)
        with pytest.raises(StoreV7MigrationError, match="injected PostgreSQL"):
            apply_store_v7_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )

        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (6,)
            present = {
                str(row[0])
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = current_schema()"
                )
            }
        assert present.isdisjoint(V7_TABLES)


@pytest.mark.postgres
@pytest.mark.parametrize("fault", ["commit_ack", "post_commit_readback"])
def test_postgres_v7_reconciles_exact_target_after_uncertain_commit(
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    import psycopg

    with _postgres_schema_dsn() as dsn:
        _downgrade_to_v6(dsn)
        plan = plan_store_v7_migration(dsn)
        if fault == "commit_ack":
            real_commit = mcp_v7_migration._PostgresConnection.commit

            def lost_commit_ack(connection: object) -> None:
                real_commit(connection)
                raise RuntimeError("injected lost commit ACK")

            monkeypatch.setattr(
                mcp_v7_migration._PostgresConnection,
                "commit",
                lost_commit_ack,
            )
            expected_error = "lost commit ACK"
        else:
            real_require = mcp_v7_migration._require_canonical_v7
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
                mcp_v7_migration,
                "_require_canonical_v7",
                fail_post_commit_readback,
            )
            expected_error = "post-commit readback"

        with pytest.raises(RuntimeError, match=expected_error):
            apply_store_v7_migration(
                dsn,
                expected_plan_sha256=plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )
        with psycopg.connect(dsn, autocommit=True) as connection:
            assert connection.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone() == (7,)

        result = apply_store_v7_migration(
            dsn,
            expected_plan_sha256=plan.plan_sha256,
            postgres_snapshot_confirmed=True,
        )
        assert result.applied is False
        assert result.already_applied is True
        assert result.plan == plan


@pytest.mark.postgres
def test_postgres_v7_plan_rejects_a_different_database_schema() -> None:
    with _postgres_schema_dsn() as first_dsn, _postgres_schema_dsn() as second_dsn:
        _downgrade_to_v6(first_dsn)
        _downgrade_to_v6(second_dsn)
        first_plan = plan_store_v7_migration(first_dsn)
        second_plan = plan_store_v7_migration(second_dsn)

        assert first_plan.source_catalog_sha256 == second_plan.source_catalog_sha256
        assert first_plan.source_digest_sha256 != second_plan.source_digest_sha256
        assert (
            first_plan.database_identity_sha256
            != second_plan.database_identity_sha256
        )
        assert first_plan.plan_sha256 != second_plan.plan_sha256
        with pytest.raises(StoreV7MigrationError, match="plan digest"):
            apply_store_v7_migration(
                second_dsn,
                expected_plan_sha256=first_plan.plan_sha256,
                postgres_snapshot_confirmed=True,
            )
