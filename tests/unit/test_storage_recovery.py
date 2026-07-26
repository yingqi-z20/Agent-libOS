from __future__ import annotations

import os
import sqlite3
import stat
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agent_libos.storage.sqlite as sqlite_backend
from agent_libos.models import (
    AgentObject,
    AgentProcess,
    Capability,
    CapabilityStatus,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    MemoryView,
    KilledProcessOutcome,
    ObjectHandle,
    ObjectMetadata,
    ObjectType,
    ProcessStatus,
    Provenance,
    ResourceBudget,
    ResourceUsage,
    ViewMode,
)
from agent_libos.models.exceptions import (
    ProcessRevisionConflict,
    UnsupportedStoreVersion,
    ValidationError,
)
from agent_libos.evidence.external_effects import (
    abandon_external_effect_intent,
    record_external_effect,
)
from agent_libos.storage import (
    ProcessRepository,
    SQLRuntimeStore,
    STORE_SCHEMA_VERSION,
    SQLiteStore,
)
from agent_libos.storage.postgres import PostgresStore
from agent_libos.storage.sql import _V3_REQUIRED_COLUMNS
from agent_libos.process_transition import ProcessTransitionService
from agent_libos.utils.ids import utc_now
from tests.support.external_effects import begin_external_effect_intent


class _ConnectionProxy:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    @property
    def row_factory(self) -> Any:
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._connection.row_factory = value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class TestExternalEffectIntentRecovery:
    def test_intent_finalization_is_identity_bound_idempotent_and_reserves_state_metadata(self) -> None:
        store = SQLiteStore(':memory:')
        classification = ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=False,
            metadata={'effect_state': 'pending', 'provider_value': 'kept'},
        )
        try:
            intent = begin_external_effect_intent(
                store,
                pid='pid_effect',
                provider='test-provider',
                operation='write',
                target='test:target',
                state_mutation=True,
                information_flow=False,
                metadata={'effect_state': 'forged', 'outcome': 'forged'},
            )
            pending = store.list_external_effects(pid='pid_effect')[0]
            assert pending.effect_state == 'pending'
            assert pending.provider_metadata['effect_state'] == 'pending'
            assert pending.provider_metadata['outcome'] == 'unknown_after_provider_boundary'

            finalized = record_external_effect(
                store,
                pid='pid_effect',
                provider='test-provider',
                operation='write',
                target='test:target',
                classification=classification,
                audit_record=None,
                event=None,
                metadata={'effect_state': 'forged'},
                intent_effect_id=intent.effect_id,
            )
            rows = store.list_external_effects(pid='pid_effect')
            assert len(rows) == 1
            assert finalized.effect_id == intent.effect_id == rows[0].effect_id
            assert rows[0].effect_state == 'finalized'
            assert rows[0].provider_metadata['effect_state'] == 'finalized'
            assert rows[0].provider_metadata['provider_value'] == 'kept'

            with pytest.raises(ValidationError, match='record id must match'):
                store.finalize_external_effect(intent.effect_id, replace(finalized, effect_id='wrong_effect'))
            with pytest.raises(ValidationError, match='must be finalized'):
                store.finalize_external_effect(intent.effect_id, replace(finalized, effect_state='pending'))

            with pytest.raises(ValidationError, match='already finalized'):
                record_external_effect(
                    store,
                    pid='pid_effect',
                    provider='test-provider',
                    operation='write',
                    target='test:target',
                    classification=classification,
                    audit_record=None,
                    event=None,
                    intent_effect_id=intent.effect_id,
                )

            mismatched = begin_external_effect_intent(
                store,
                pid='pid_effect',
                provider='test-provider',
                operation='read',
                target='test:other',
                state_mutation=False,
                information_flow=True,
            )
            with pytest.raises(ValidationError, match='did not match'):
                record_external_effect(
                    store,
                    pid='pid_effect',
                    provider='wrong-provider',
                    operation='read',
                    target='test:other',
                    classification=classification,
                    audit_record=None,
                    event=None,
                    intent_effect_id=mismatched.effect_id,
                )
            remaining = [row for row in store.list_external_effects(pid='pid_effect') if row.effect_id == mismatched.effect_id]
            assert len(remaining) == 1 and remaining[0].effect_state == 'pending'
        finally:
            store.close()

    def test_intent_abandon_is_pending_only_and_not_repeatable(self) -> None:
        store = SQLiteStore(':memory:')
        try:
            intent = begin_external_effect_intent(
                store,
                pid='pid_effect',
                provider='test-provider',
                operation='read',
                target='test:target',
                state_mutation=False,
                information_flow=True,
            )
            abandon_external_effect_intent(store, intent.effect_id)
            assert store.list_external_effects(pid='pid_effect') == []
            with pytest.raises(ValidationError, match='missing or already finalized'):
                abandon_external_effect_intent(store, intent.effect_id)
        finally:
            store.close()


class _FinalizeFailureConnection(_ConnectionProxy):
    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        commit_failures: int = 0,
        rollback_failures: int = 0,
        release_failures: int = 0,
        commit_error: BaseException | None = None,
        rollback_error: BaseException | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        super().__init__(connection)
        self.commit_failures = commit_failures
        self.rollback_failures = rollback_failures
        self.release_failures = release_failures
        self.commit_error = commit_error or RuntimeError("injected commit failure")
        self.rollback_error = rollback_error or RuntimeError("injected rollback failure")
        self.release_error = release_error or RuntimeError("injected release failure")

    def commit(self) -> None:
        if self.commit_failures:
            self.commit_failures -= 1
            raise self.commit_error
        self._connection.commit()

    def rollback(self) -> None:
        if self.rollback_failures:
            self.rollback_failures -= 1
            raise self.rollback_error
        self._connection.rollback()

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        if sql.lstrip().upper().startswith("RELEASE SAVEPOINT") and self.release_failures:
            self.release_failures -= 1
            raise self.release_error
        return self._connection.execute(sql, parameters)


class _CommitThenRaiseConnection(_ConnectionProxy):
    def __init__(self, connection: sqlite3.Connection, error: BaseException) -> None:
        super().__init__(connection)
        self.error = error

    def commit(self) -> None:
        self._connection.commit()
        raise self.error


class _ReleaseThenRaiseConnection(_ConnectionProxy):
    def __init__(self, connection: sqlite3.Connection, error: BaseException) -> None:
        super().__init__(connection)
        self.error = error
        self.failed = False

    def execute(self, sql: str, parameters: Any = ()) -> Any:
        result = self._connection.execute(sql, parameters)
        if not self.failed and sql.strip().upper().startswith("RELEASE SAVEPOINT"):
            self.failed = True
            raise self.error
        return result


class _ExecuteFailureConnection(_ConnectionProxy):
    def __init__(self, connection: sqlite3.Connection, *, marker: str) -> None:
        super().__init__(connection)
        self.marker = marker
        self.failed = False

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        if self.marker in sql and not self.failed:
            self.failed = True
            raise RuntimeError(f"injected SQL failure at {self.marker}")
        return self._connection.execute(sql, parameters)

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        cursor = self._connection.cursor(*args, **kwargs)
        owner = self

        class _FailingCursor:
            def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
                if owner.marker in sql and not owner.failed:
                    owner.failed = True
                    raise RuntimeError(f"injected SQL failure at {owner.marker}")
                return cursor.execute(sql, parameters)

            def __getattr__(self, name: str) -> Any:
                return getattr(cursor, name)

        return _FailingCursor()


def _runnable_process(pid: str) -> AgentProcess:
    now = utc_now()
    return AgentProcess(
        pid=pid,
        parent_pid=None,
        image_id="base-agent:v0",
        status=ProcessStatus.RUNNABLE,
        goal_oid=None,
        memory_view=None,
        capabilities=[],
        loaded_skills={},
        tool_table={},
        event_cursor=None,
        checkpoint_head=None,
        resource_budget=ResourceBudget(),
        resource_usage=ResourceUsage(),
        created_at=now,
        updated_at=now,
    )


def _finite_capability(cap_id: str) -> Capability:
    return Capability(
        cap_id=cap_id,
        subject="pid_test",
        resource="clock:now",
        rights={"read"},
        constraints={},
        issued_by="test",
        issued_at=utc_now(),
        uses_remaining=2,
    )


def _runtime_object(
    oid: str,
    payload: Any,
    *,
    version: int = 1,
) -> AgentObject:
    now = utc_now()
    return AgentObject(
        oid=oid,
        namespace="system",
        name=oid,
        type=ObjectType.ARTIFACT,
        schema_version="1",
        payload=payload,
        metadata=ObjectMetadata(),
        provenance=Provenance(created_from_action="test"),
        version=version,
        immutable=False,
        created_by="test",
        created_at=now,
        updated_at=now,
        owner_id="pid_test",
    )


def _create_legacy_objects_table(connection: sqlite3.Connection, table: str = "objects") -> None:
    connection.execute(
        f"""
        CREATE TABLE {table} (
          oid TEXT PRIMARY KEY,
          name TEXT NOT NULL UNIQUE,
          type TEXT NOT NULL,
          schema_version TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          provenance_json TEXT NOT NULL,
          version INTEGER NOT NULL,
          immutable INTEGER NOT NULL,
          created_by TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "obj_legacy",
            "legacy.object",
            "artifact",
            "1",
            "{}",
            "{}",
            "{}",
            1,
            1,
            "pid_legacy",
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )
    connection.commit()


class TestStoreTransactionRecovery:
    def test_payload_reader_waits_for_rollback_without_observing_uncommitted_cache(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")
        payload_mutated = threading.Event()
        release_writer = threading.Event()
        reader_finished = threading.Event()
        writer_errors: list[BaseException] = []
        reader_errors: list[BaseException] = []
        observed: list[Any] = []
        oid = "obj_payload_isolation"
        before = _runtime_object(oid, {"value": "committed"})
        after = replace(
            before,
            payload={"value": "uncommitted"},
            version=2,
            updated_at=utc_now(),
        )

        def mutate_then_rollback() -> None:
            try:
                with pytest.raises(RuntimeError, match="rollback payload"):
                    with store.transaction():
                        assert store.update_object(
                            after,
                            expected_version=before.version,
                        )
                        payload_mutated.set()
                        assert release_writer.wait(timeout=5)
                        raise RuntimeError("rollback payload")
            except BaseException as exc:
                writer_errors.append(exc)

        def read_payload() -> None:
            try:
                assert payload_mutated.wait(timeout=5)
                observed.append(store.object_payload(oid))
            except BaseException as exc:
                reader_errors.append(exc)
            finally:
                reader_finished.set()

        try:
            store.insert_object(before)
            writer = threading.Thread(target=mutate_then_rollback, daemon=True)
            reader = threading.Thread(target=read_payload, daemon=True)
            writer.start()
            assert payload_mutated.wait(timeout=5)
            reader.start()
            leaked_before_rollback = reader_finished.wait(timeout=0.2)
            release_writer.set()
            writer.join(timeout=5)
            reader.join(timeout=5)

            assert not writer.is_alive()
            assert not reader.is_alive()
            assert not leaked_before_rollback
            assert writer_errors == []
            assert reader_errors == []
            assert observed == [{"value": "committed"}]
            assert store.object_payload(oid) == {"value": "committed"}
            restored = store.get_object(oid)
            assert restored is not None
            assert restored.version == 1
            assert restored.payload == {"value": "committed"}
        finally:
            release_writer.set()
            store.close()

    def test_get_object_reads_sql_row_and_payload_from_one_store_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")
        row_selected = threading.Event()
        writer_started = threading.Event()
        writer_finished = threading.Event()
        reader_errors: list[BaseException] = []
        writer_errors: list[BaseException] = []
        observed: list[AgentObject] = []
        oid = "obj_payload_snapshot"
        before = _runtime_object(oid, {"value": "before"})
        after = replace(
            before,
            payload={"value": "after"},
            version=2,
            updated_at=utc_now(),
        )
        try:
            store.insert_object(before)
            original_has_payload = store.has_object_payload

            def coordinated_has_payload(*args: Any, **kwargs: Any) -> bool:
                if threading.current_thread().name == "payload-snapshot-reader":
                    row_selected.set()
                    assert writer_started.wait(timeout=5)
                    # A correct get_object holds the store lock across both
                    # the SQL row and cache read, keeping this writer blocked.
                    writer_finished.wait(timeout=0.2)
                return original_has_payload(*args, **kwargs)

            monkeypatch.setattr(store, "has_object_payload", coordinated_has_payload)

            def read_object() -> None:
                try:
                    selected = store.get_object(oid)
                    assert selected is not None
                    observed.append(selected)
                except BaseException as exc:
                    reader_errors.append(exc)

            def update_object() -> None:
                try:
                    assert row_selected.wait(timeout=5)
                    writer_started.set()
                    assert store.update_object(after, expected_version=before.version)
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    writer_finished.set()

            reader = threading.Thread(
                target=read_object,
                name="payload-snapshot-reader",
                daemon=True,
            )
            writer = threading.Thread(target=update_object, daemon=True)
            reader.start()
            writer.start()
            reader.join(timeout=5)
            writer.join(timeout=5)

            assert not reader.is_alive()
            assert not writer.is_alive()
            assert reader_errors == []
            assert writer_errors == []
            assert len(observed) == 1
            assert observed[0].version == 1
            assert observed[0].payload == {"value": "before"}
            current = store.get_object(oid)
            assert current is not None
            assert current.version == 2
            assert current.payload == {"value": "after"}
        finally:
            store.close()

    def test_process_memory_root_append_is_commutative(self) -> None:
        store = SQLiteStore(":memory:")
        process = _runnable_process("pid_roots")
        process.memory_view = MemoryView(
            view_id="view_roots",
            owner_pid=process.pid,
            roots=[],
            filters=[],
            rights_policy="attenuate",
            created_from=None,
            mode=ViewMode.MUTABLE,
        )
        store.insert_process(process)
        roots = [
            ObjectHandle(oid="oid_a", rights={"read"}, capability_id="cap_a"),
            ObjectHandle(oid="oid_b", rights={"read"}, capability_id="cap_b"),
        ]
        try:
            threads = [
                threading.Thread(
                    target=store.append_process_memory_roots,
                    args=(process.pid, [root]),
                )
                for root in roots
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            stored = store.get_process(process.pid)
            assert stored is not None and stored.memory_view is not None
            assert {root.oid for root in stored.memory_view.roots} == {"oid_a", "oid_b"}
        finally:
            store.close()

    def test_terminal_process_cannot_be_resurrected_by_stale_or_fresh_update(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_process(_runnable_process("pid_terminal"))
            stale = store.get_process("pid_terminal")
            killed = store.get_process("pid_terminal")
            assert stale is not None and killed is not None
            ProcessTransitionService(ProcessRepository(store)).transition(
                killed.pid,
                ProcessStatus.KILLED,
                expected_revision=killed.revision,
                outcome=KilledProcessOutcome(code="test_fixture"),
            )

            stale.tool_table["late"] = "tool_late"
            with pytest.raises(ProcessRevisionConflict):
                store.update_process(stale)
            terminal = store.get_process("pid_terminal")
            assert terminal is not None
            terminal.status = ProcessStatus.RUNNABLE
            with pytest.raises(ProcessRevisionConflict):
                store.update_process(terminal)
            assert store.get_process("pid_terminal").status == ProcessStatus.KILLED  # type: ignore[union-attr]
        finally:
            store.close()

    def test_execution_token_fences_stale_quantum_completion(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_process(_runnable_process("pid_fenced"))
            first = store.claim_execution("pid_fenced", owner_id="runtime_first")
            assert first is not None
            assert store.complete_execution(first)
            second = store.claim_execution("pid_fenced", owner_id="runtime_second")
            assert second is not None
            assert second.generation > first.generation

            assert store.complete_execution(first) is False
            running = store.get_process("pid_fenced")
            assert running is not None
            assert running.status == ProcessStatus.RUNNING
            assert running.execution_lease_id == second.lease_id
            assert store.complete_execution(second)
        finally:
            store.close()

    def test_outer_rollback_restores_payload_mutated_by_committed_inner_transaction(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))

            with pytest.raises(RuntimeError, match="rollback outer"):
                with store.transaction():
                    store.set_object_payload("obj_payload", {"value": "after"})
                    raise RuntimeError("rollback outer")

            assert store.object_payload("obj_payload") == {"value": "before"}
        finally:
            store.close()

    def test_nested_payload_commit_merges_earliest_before_image_into_parent(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))

            with pytest.raises(RuntimeError, match="rollback outer"):
                with store.transaction():
                    store.set_object_payload("obj_payload", {"value": "middle"})
                    with store.transaction():
                        store.set_object_payload("obj_payload", {"value": "after"})
                    raise RuntimeError("rollback outer")

            assert store.object_payload("obj_payload") == {"value": "before"}
        finally:
            store.close()

    def test_set_object_payload_sql_failure_restores_previous_payload(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))
            store.conn = _ExecuteFailureConnection(
                store.conn,
                marker="UPDATE objects SET payload_json",
            )

            with pytest.raises(RuntimeError, match="injected SQL failure"):
                store.set_object_payload("obj_payload", {"value": "after"})

            assert store.object_payload("obj_payload") == {"value": "before"}
        finally:
            store.close()

    def test_set_object_payload_rejects_missing_and_released_rows_without_cache(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")
        missing_oid = "obj_payload_missing"
        released_oid = "obj_payload_released"
        try:
            with pytest.raises(
                ValidationError,
                match="missing or released Object",
            ):
                store.set_object_payload(missing_oid, {"value": "orphan"})
            assert store.get_persisted_object_state(missing_oid) is None
            assert not store.has_object_payload(missing_oid)
            assert missing_oid not in store._object_payloads
            with pytest.raises(KeyError):
                store.object_payload(missing_oid)

            store.insert_object(
                _runtime_object(released_oid, {"value": "before release"})
            )
            assert store.delete_object(released_oid)
            before = store.get_persisted_object_state(released_oid)
            assert before is not None
            assert before.lifecycle_state.value == "released"
            assert not before.payload_present

            with pytest.raises(
                ValidationError,
                match="missing or released Object",
            ):
                store.set_object_payload(released_oid, {"value": "resurrected"})

            after = store.get_persisted_object_state(released_oid)
            assert after == before
            assert not store.has_object_payload(released_oid)
            assert released_oid not in store._object_payloads
            with pytest.raises(KeyError):
                store.object_payload(released_oid)
        finally:
            store.close()

    def test_concurrent_release_and_set_payload_never_caches_a_released_object(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")
        oid = "obj_payload_release_race"
        start = threading.Barrier(3)
        set_outcomes: list[str] = []
        release_outcomes: list[bool] = []
        unexpected: list[BaseException] = []

        def set_payload() -> None:
            try:
                start.wait(timeout=5)
                store.set_object_payload(oid, {"value": "racing"})
                set_outcomes.append("updated")
            except ValidationError:
                set_outcomes.append("rejected")
            except BaseException as exc:
                unexpected.append(exc)

        def release() -> None:
            try:
                start.wait(timeout=5)
                release_outcomes.append(store.delete_object(oid))
            except BaseException as exc:
                unexpected.append(exc)

        try:
            store.insert_object(_runtime_object(oid, {"value": "initial"}))
            setter = threading.Thread(target=set_payload, daemon=True)
            releaser = threading.Thread(target=release, daemon=True)
            setter.start()
            releaser.start()
            start.wait(timeout=5)
            setter.join(timeout=5)
            releaser.join(timeout=5)

            assert not setter.is_alive()
            assert not releaser.is_alive()
            assert unexpected == []
            assert set_outcomes in (["updated"], ["rejected"])
            assert release_outcomes == [True]
            state = store.get_persisted_object_state(oid)
            assert state is not None
            assert state.lifecycle_state.value == "released"
            assert not state.payload_present
            assert not store.has_object_payload(oid)
            assert oid not in store._object_payloads
        finally:
            store.close()

    def test_set_object_payload_ack_loss_poison_reopens_to_released_state(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "set-payload-ack-loss.sqlite"
        oid = "obj_payload_ack_loss"
        store = SQLiteStore(db_path)
        store.insert_object(_runtime_object(oid, {"value": "before"}))
        diagnostic = RuntimeError("injected payload commit acknowledgement loss")
        store.conn = _CommitThenRaiseConnection(store.conn, diagnostic)
        try:
            with pytest.raises(
                ValidationError,
                match="unusable after uncertain transaction commit",
            ):
                store.set_object_payload(oid, {"value": "after"})
            assert store._poisoned_reason == "commit_outcome_uncertain"
            assert store._object_payloads[oid] == {"value": "after"}
        finally:
            store.close()

        reopened = SQLiteStore(db_path)
        try:
            summary = reopened.recover_missing_runtime_object_payloads(
                require_recovery_lease=lambda: None,
            )
            assert summary.total_count == 1
            assert summary.sample_oids == (oid,)
            state = reopened.get_persisted_object_state(oid)
            assert state is not None
            assert state.lifecycle_state.value == "released"
            assert not state.payload_present
            assert state.recovered_after_reopen
            assert not reopened.has_object_payload(oid)
        finally:
            reopened.close()

    def test_set_object_payload_commit_failure_restores_previous_payload(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))
            store.conn = _FinalizeFailureConnection(store.conn, commit_failures=1)

            with pytest.raises(RuntimeError, match="injected commit failure"):
                store.set_object_payload("obj_payload", {"value": "after"})

            assert store.object_payload("obj_payload") == {"value": "before"}
        finally:
            store.close()

    def test_outer_execute_commit_failure_preserves_primary_and_rolls_back_sql(self) -> None:
        store = SQLiteStore(":memory:")
        primary = RuntimeError("injected exact outer commit failure")

        class _ExactCommitFailureConnection(_ConnectionProxy):
            def commit(self) -> None:
                raise primary

        store.conn = _ExactCommitFailureConnection(store.conn)
        try:
            with pytest.raises(RuntimeError) as caught:
                store._execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("outer-commit-failed", None, "{}", "test", "1", "1"),
                )

            assert caught.value is primary
            assert store.select_table_rows(
                "object_namespaces",
                "namespace = ?",
                ("outer-commit-failed",),
            ) == []
        finally:
            store.close()

    def test_commit_failure_rolls_back_sql_and_object_payload_snapshot(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))
            store.conn = _FinalizeFailureConnection(store.conn, commit_failures=1)

            with pytest.raises(RuntimeError, match="injected commit failure"):
                with store.transaction(include_object_payloads=True) as cursor:
                    store.set_object_payload("obj_payload", {"value": "after"})
                    cursor.execute(
                        "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                        ("commit-failed", None, "{}", "test", "1", "1"),
                    )

            assert store.object_payload("obj_payload") == {"value": "before"}
            assert store.select_table_rows(
                "object_namespaces", "namespace = ?", ("commit-failed",)
            ) == []

            with store.transaction() as cursor:
                cursor.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("healthy-after-rollback", None, "{}", "test", "1", "1"),
                )
            assert store.get_namespace("healthy-after-rollback") is not None
        finally:
            store.close()

    def test_commit_error_after_apply_never_serves_mixed_sql_and_payload_state(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "commit-after-apply.sqlite"
        store = SQLiteStore(db_path)
        oid = "obj_commit_after_apply"
        before = _runtime_object(oid, {"value": "before"})
        after = replace(
            before,
            payload={"value": "after"},
            version=2,
            updated_at=utc_now(),
        )
        store.insert_object(before)
        secret = "/private/runtime/tenant.sqlite?password=commit-secret"
        diagnostic = RuntimeError(
            f"injected diagnostic after commit applied at {secret}"
        )
        store.conn = _CommitThenRaiseConnection(store.conn, diagnostic)
        try:
            with pytest.raises(
                ValidationError,
                match="unusable after uncertain transaction commit",
            ) as caught:
                store.update_object(after, expected_version=before.version)

            assert secret not in str(caught.value)
            assert caught.value.__cause__ is None
            assert store._poisoned_reason == "commit_outcome_uncertain"
            assert len(store._poisoned_failure_fingerprints) == 1
            fingerprint = store._poisoned_failure_fingerprints[0]
            assert fingerprint["error_type"] == "RuntimeError"
            assert fingerprint["exception_text"]["bytes"] == len(
                str(diagnostic).encode("utf-8")
            )
            assert len(fingerprint["exception_text"]["sha256"]) == 64
            assert secret not in repr(fingerprint)
            assert store._object_payloads[oid] == {"value": "after"}
            with pytest.raises(
                ValidationError,
                match="unusable after uncertain transaction commit",
            ) as health:
                store.get_object(oid)
            assert secret not in str(health.value)
            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT version FROM objects WHERE oid = ?",
                    (oid,),
                ).fetchone()
            finally:
                connection.close()
            assert row == (2,)
        finally:
            store.close()

    def test_single_oid_transaction_does_not_copy_unrelated_cached_payloads(self) -> None:
        store = SQLiteStore(":memory:")

        class CopyProbe:
            def __init__(self) -> None:
                self.copies = 0

            def __deepcopy__(self, _memo: dict[int, Any]) -> CopyProbe:
                self.copies += 1
                return self

        probe = CopyProbe()
        try:
            store.insert_object(_runtime_object("target", {"value": "before"}))
            with store.locked():
                store._object_payloads["unrelated"] = probe

            with store.transaction(include_object_payloads=True):
                store.set_object_payload("target", {"value": "after"})

            assert probe.copies == 0
            assert store.object_payload("target") == {"value": "after"}
        finally:
            store.close()

    def test_release_error_after_apply_poison_does_not_restore_only_payload_cache(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "release-after-apply.sqlite"
        store = SQLiteStore(db_path)
        oid = "obj_release_after_apply"
        before = _runtime_object(oid, {"value": "before"})
        after = replace(
            before,
            payload={"value": "after"},
            version=2,
            updated_at=utc_now(),
        )
        store.insert_object(before)
        secret = "/private/runtime/savepoint.sqlite?token=savepoint-secret"
        diagnostic = RuntimeError(
            f"injected diagnostic after savepoint release at {secret}"
        )
        store.conn = _ReleaseThenRaiseConnection(store.conn, diagnostic)
        try:
            with pytest.raises(
                ValidationError,
                match="unusable after transaction rollback failure",
            ) as caught:
                with store.transaction():
                    with store.transaction():
                        assert store.update_object(
                            after,
                            expected_version=before.version,
                        )
            assert secret not in str(caught.value)
            assert caught.value.__cause__ is None
            assert store._poisoned_reason == "savepoint_release_recovery_failed"
            assert all(
                secret not in repr(fingerprint)
                for fingerprint in store._poisoned_failure_fingerprints
            )

            # RELEASE took effect inside the still-uncommitted outer
            # transaction. The failed ROLLBACK TO makes its exact nested state
            # unknowable, so the store remains unreadable and does not pretend
            # that restoring the payload alone recovered it.
            assert store._object_payloads[oid] == {"value": "after"}
            with pytest.raises(
                ValidationError,
                match="unusable after transaction rollback failure",
            ) as health:
                store.get_object(oid)
            assert secret not in str(health.value)
            connection = sqlite3.connect(db_path)
            try:
                row = connection.execute(
                    "SELECT version FROM objects WHERE oid = ?",
                    (oid,),
                ).fetchone()
            finally:
                connection.close()
            assert row == (1,)
        finally:
            store.close()

    def test_rollback_failure_poison_closes_store(self) -> None:
        store = SQLiteStore(":memory:")
        commit_secret = "/private/commit.sqlite?password=commit-driver-secret"
        rollback_secret = "/private/rollback.sqlite?password=rollback-driver-secret"
        store.conn = _FinalizeFailureConnection(
            store.conn,
            commit_failures=1,
            rollback_failures=1,
            commit_error=RuntimeError(f"commit failed at {commit_secret}"),
            rollback_error=RuntimeError(f"rollback failed at {rollback_secret}"),
        )
        try:
            with pytest.raises(
                ValidationError,
                match="unusable.*rollback failure",
            ) as caught:
                with store.transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                        ("uncertain", None, "{}", "test", "1", "1"),
                    )
            assert caught.value.__cause__ is None
            assert commit_secret not in str(caught.value)
            assert rollback_secret not in str(caught.value)
            assert store._poisoned_reason == "transaction_commit_rollback_failed"
            assert len(store._poisoned_failure_fingerprints) == 2
            assert all(
                commit_secret not in repr(fingerprint)
                and rollback_secret not in repr(fingerprint)
                and len(fingerprint["exception_text"]["sha256"]) == 64
                for fingerprint in store._poisoned_failure_fingerprints
            )

            with pytest.raises(
                ValidationError,
                match="unusable.*rollback failure",
            ) as health:
                store.list_processes()
            assert commit_secret not in str(health.value)
            assert rollback_secret not in str(health.value)
        finally:
            store.close()

    def test_release_savepoint_failure_rolls_back_nested_sql_and_payload(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_object(_runtime_object("obj_payload", {"value": "before"}))
            store.conn = _FinalizeFailureConnection(store.conn, release_failures=1)

            with store.transaction() as outer:
                outer.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("outer-before", None, "{}", "test", "1", "1"),
                )
                with pytest.raises(RuntimeError, match="injected release failure"):
                    with store.transaction(include_object_payloads=True) as inner:
                        store.set_object_payload("obj_payload", {"value": "after"})
                        inner.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            ("inner-failed", None, "{}", "test", "1", "1"),
                        )
                outer.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("outer-after", None, "{}", "test", "1", "1"),
                )

            assert store.object_payload("obj_payload") == {"value": "before"}
            namespaces = {
                row["namespace"] for row in store.select_table_rows("object_namespaces")
            }
            assert {"outer-before", "outer-after"}.issubset(namespaces)
            assert "inner-failed" not in namespaces
        finally:
            store.close()

    def test_release_failure_with_failed_savepoint_rollback_poison_closes_store(self) -> None:
        store = SQLiteStore(":memory:")
        store.conn = _FinalizeFailureConnection(store.conn, release_failures=2)
        try:
            with pytest.raises(ValidationError, match="unusable.*rollback failure"):
                with store.transaction():
                    with store.transaction() as inner:
                        inner.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            ("nested-uncertain", None, "{}", "test", "1", "1"),
                        )

            with pytest.raises(ValidationError, match="unusable.*rollback failure"):
                store.list_processes()
        finally:
            store.close()

    def test_claim_runnable_process_does_not_commit_outer_transaction(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_process(_runnable_process("pid_claim"))
            with pytest.raises(RuntimeError, match="rollback claim"):
                with store.transaction():
                    claimed = store.claim_runnable_process("pid_claim")
                    assert claimed is not None
                    assert claimed.status == ProcessStatus.RUNNING
                    raise RuntimeError("rollback claim")

            process = store.get_process("pid_claim")
            assert process is not None
            assert process.status == ProcessStatus.RUNNABLE
        finally:
            store.close()

    def test_consume_capability_uses_does_not_commit_outer_transaction(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            store.insert_capability(_finite_capability("cap_nested"))
            with pytest.raises(RuntimeError, match="rollback consume"):
                with store.transaction():
                    consumed = store.consume_capability_uses("cap_nested")
                    assert consumed is not None
                    assert consumed.uses_remaining == 1
                    raise RuntimeError("rollback consume")

            capability = store.get_capability("cap_nested")
            assert capability is not None
            assert capability.uses_remaining == 2
            assert capability.status == CapabilityStatus.ACTIVE
        finally:
            store.close()


class TestSQLiteRuntimeLeaseRecovery:
    def test_runtime_lease_refuses_symlink_without_modifying_target(self, tmp_path: Path) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("O_NOFOLLOW lease path is not used on this platform")
        db_path = tmp_path / "runtime.sqlite"
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        target = tmp_path / "must-not-change.txt"
        target.write_text("sentinel", encoding="utf-8")
        try:
            lease_path.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is not available in this environment")

        opened: SQLiteStore | None = None
        try:
            with pytest.raises(ValidationError, match="unsafe runtime lease"):
                opened = SQLiteStore(db_path)
        finally:
            if opened is not None:
                opened.close()

        assert target.read_text(encoding="utf-8") == "sentinel"

    def test_runtime_lease_requires_regular_file_from_fstat(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if sqlite_backend.fcntl is None:
            pytest.skip("file lease is not used on this platform")
        real_fstat = sqlite_backend.os.fstat

        def non_regular_lease(fd: int) -> Any:
            result = real_fstat(fd)
            return SimpleNamespace(st_mode=stat.S_IFDIR | (result.st_mode & 0o777))

        monkeypatch.setattr(sqlite_backend.os, "fstat", non_regular_lease)
        with pytest.raises(ValidationError, match="regular file"):
            SQLiteStore(tmp_path / "runtime.sqlite")

    def test_fcntl_fallback_ignores_stale_legacy_lockfile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sqlite_backend, "fcntl", None)
        db_path = tmp_path / "runtime.sqlite"
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        lease_path.write_text("2000-01-01T00:00:00+00:00\n999999999\n", encoding="utf-8")

        store = SQLiteStore(db_path)
        try:
            assert store.list_processes() == []
        finally:
            store.close()

    def test_sqlite_connection_uses_same_canonical_path_as_lease(self, tmp_path: Path) -> None:
        target = tmp_path / "canonical.sqlite"
        sqlite3.connect(target).close()
        alias = tmp_path / "alias.sqlite"
        try:
            alias.symlink_to(target)
        except OSError:
            pytest.skip("symlink creation is not available in this environment")

        store = SQLiteStore(alias)
        try:
            database_row = store.conn.execute("PRAGMA database_list").fetchone()
            assert database_row is not None
            assert Path(database_row["file"]) == target.resolve()
            assert store.path == str(alias)
        finally:
            store.close()


class _PostgresResult:
    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row

    def fetchone(self) -> dict[str, Any]:
        return self.row


class _PostgresLeaseConnection:
    def __init__(self, database: str, schema: str) -> None:
        self.database = database
        self.schema = schema
        self.closed = False
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, sql: str, params: Any = ()) -> _PostgresResult:
        selected = tuple(params)
        self.calls.append((sql, selected))
        if "current_database()" in sql:
            return _PostgresResult({"database_name": self.database, "schema_name": self.schema})
        return _PostgresResult({"acquired": True})

    def close(self) -> None:
        self.calls.append(("CLOSE SESSION", ()))
        self.closed = True


class TestPostgresRuntimeLeaseIsolation:
    def test_advisory_lease_key_isolated_by_database_and_schema(self) -> None:
        keys: list[int] = []
        stores: list[tuple[PostgresStore, _PostgresLeaseConnection]] = []
        for database, schema in (("db_a", "schema_a"), ("db_a", "schema_b"), ("db_b", "schema_a")):
            store = PostgresStore.__new__(PostgresStore)
            store._runtime_lease_acquired = False
            connection = _PostgresLeaseConnection(database, schema)
            store._acquire_runtime_lease(connection)  # type: ignore[arg-type]
            stores.append((store, connection))
            lock_calls = [call for call in connection.calls if "pg_try_advisory_lock" in call[0]]
            assert len(lock_calls) == 1
            keys.append(int(lock_calls[0][1][0]))

        assert len(set(keys)) == 3

        for store, connection in stores:
            lease_key = store._runtime_lease_key
            assert lease_key is not None
            store.conn = connection  # type: ignore[assignment]
            store.close()
            assert connection.calls[-1] == ("CLOSE SESSION", ())
            assert store._runtime_lease_acquired is False
            assert store._runtime_lease_key is None


class TestUnsupportedStoreVersion:
    def test_v3_schema_manifest_matches_fresh_store(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            assert set(_V3_REQUIRED_COLUMNS) == {
                "runtime_schema",
                *store.ALLOWED_TABLES,
            }
            for table, expected_columns in _V3_REQUIRED_COLUMNS.items():
                actual_columns = {
                    str(row["name"])
                    for row in store.conn.execute(f"PRAGMA table_info({table})")
                }
                assert actual_columns == expected_columns, table
        finally:
            store.close()

    def test_fresh_schema_rejects_a_second_schema_marker(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            with pytest.raises(sqlite3.IntegrityError):
                store.conn.execute(
                    "INSERT INTO runtime_schema (singleton, schema_version) VALUES (?, ?)",
                    (2, STORE_SCHEMA_VERSION),
                )
        finally:
            store.close()

    def test_interrupted_bootstrap_rolls_back_and_reopens_cleanly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = tmp_path / "interrupted-bootstrap.sqlite"
        original = SQLRuntimeStore._write_store_schema_version

        def interrupt_after_marker(store: SQLRuntimeStore) -> None:
            original(store)
            raise RuntimeError("injected bootstrap interruption")

        monkeypatch.setattr(
            SQLRuntimeStore,
            "_write_store_schema_version",
            interrupt_after_marker,
        )
        with pytest.raises(RuntimeError, match="bootstrap interruption"):
            SQLiteStore(db_path)
        monkeypatch.setattr(
            SQLRuntimeStore,
            "_write_store_schema_version",
            original,
        )

        connection = sqlite3.connect(db_path)
        try:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall() == []
        finally:
            connection.close()

        reopened = SQLiteStore(db_path)
        try:
            row = reopened.conn.execute(
                "SELECT schema_version FROM runtime_schema WHERE singleton = 1"
            ).fetchone()
            assert row is not None
            assert row["schema_version"] == STORE_SCHEMA_VERSION
        finally:
            reopened.close()

    def test_utf16_sqlite_store_is_rejected_before_bootstrap(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = tmp_path / "utf16.sqlite"
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("PRAGMA encoding = 'UTF-16le'")
            connection.execute("CREATE TABLE encoding_seed (value TEXT)")
            connection.execute("DROP TABLE encoding_seed")
            connection.commit()
            assert connection.execute("PRAGMA encoding").fetchone() == ("UTF-16le",)
        finally:
            connection.close()
        before = db_path.read_bytes()

        with pytest.raises(UnsupportedStoreVersion, match=r"SQLite.*requires UTF-8"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before
        assert not lease_path.exists()
        connection = sqlite3.connect(db_path)
        try:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall() == []
            assert connection.execute("PRAGMA encoding").fetchone() == ("UTF-16le",)
        finally:
            connection.close()

    def test_wrong_schema_marker_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "wrong-version.sqlite"
        SQLiteStore(db_path).close()
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("UPDATE runtime_schema SET schema_version = 2")
            connection.commit()
        finally:
            connection.close()
        before = db_path.read_bytes()

        with pytest.raises(UnsupportedStoreVersion, match="expected 3"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before

    def test_incomplete_v3_schema_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "incomplete-v3.sqlite"
        SQLiteStore(db_path).close()
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("DROP TABLE checkpoints")
            connection.commit()
        finally:
            connection.close()
        before = db_path.read_bytes()

        with pytest.raises(UnsupportedStoreVersion, match="incomplete"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before

    def test_incomplete_v3_column_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "incomplete-v3-column.sqlite"
        SQLiteStore(db_path).close()
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("ALTER TABLE checkpoints DROP COLUMN reason")
            connection.commit()
        finally:
            connection.close()
        before = db_path.read_bytes()

        with pytest.raises(UnsupportedStoreVersion, match="incomplete"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before

    def test_nonempty_unversioned_store_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "unrelated.sqlite"
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("CREATE TABLE unrelated_business_data (value TEXT NOT NULL)")
            connection.execute("INSERT INTO unrelated_business_data VALUES ('sentinel')")
            connection.commit()
        finally:
            connection.close()
        db_path.chmod(0o644)
        before = db_path.read_bytes()
        before_mode = stat.S_IMODE(db_path.stat().st_mode)
        opened: SQLiteStore | None = None

        try:
            with pytest.raises(UnsupportedStoreVersion, match="unversioned"):
                opened = SQLiteStore(db_path)
        finally:
            if opened is not None:
                opened.close()

        assert db_path.read_bytes() == before
        assert stat.S_IMODE(db_path.stat().st_mode) == before_mode
        assert not lease_path.exists()

    def test_legacy_objects_store_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "legacy.sqlite"
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        connection = sqlite3.connect(db_path)
        try:
            _create_legacy_objects_table(connection)
        finally:
            connection.close()
        db_path.chmod(0o644)
        before = db_path.read_bytes()
        before_mode = stat.S_IMODE(db_path.stat().st_mode)

        with pytest.raises(UnsupportedStoreVersion, match="archive-only"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before
        assert stat.S_IMODE(db_path.stat().st_mode) == before_mode
        assert not lease_path.exists()
        connection = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            assert tables == {"objects"}
        finally:
            connection.close()

    def test_interrupted_legacy_rebuild_is_rejected_without_mutation(self, tmp_path: Path) -> None:
        db_path = tmp_path / "interrupted.sqlite"
        connection = sqlite3.connect(db_path)
        try:
            _create_legacy_objects_table(connection, table="objects_old")
        finally:
            connection.close()
        before = db_path.read_bytes()

        with pytest.raises(UnsupportedStoreVersion, match="archive-only"):
            SQLiteStore(db_path)

        assert db_path.read_bytes() == before
        connection = sqlite3.connect(db_path)
        try:
            assert connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'objects_old'"
            ).fetchone() == ("objects_old",)
        finally:
            connection.close()
