from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path
import os
import re
import stat
import threading

import pytest

import agent_libos.storage.postgres as postgres_backend
import agent_libos.storage.sqlite as sqlite_backend
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage.factory import _sqlite_target
from agent_libos.storage.postgres import _PostgresCursor, _PostgresDialect
from agent_libos.storage import (
    PostgresStore,
    SQLRuntimeStore,
    SQLiteStore,
    StoreCloseClaimOutcome,
)


ROOT = Path(__file__).resolve().parents[2]
AGENT_LIBOS = ROOT / "agent_libos"
STORAGE_BACKENDS = {
    AGENT_LIBOS / "storage" / "sqlite.py",
    AGENT_LIBOS / "storage" / "postgres.py",
}


class _PostgresCleanupResult:
    def __init__(self, row: dict[str, object]) -> None:
        self.row = row

    def fetchone(self) -> dict[str, object]:
        return self.row


class _PostgresCleanupConnection:
    def __init__(
        self,
        *,
        close_error: BaseException | None = None,
        close_releases_ownership: bool = False,
    ) -> None:
        self.close_error = close_error
        self.close_releases_ownership = close_releases_ownership
        self.closed = False
        self.events: list[str] = []

    def execute(self, sql: str, params: object = ()) -> _PostgresCleanupResult:
        if "current_database()" in sql:
            self.events.append("identity")
            return _PostgresCleanupResult(
                {"database_name": "runtime", "schema_name": "agent_libos"}
            )
        if "pg_try_advisory_lock" in sql:
            self.events.append("lock")
            return _PostgresCleanupResult({"acquired": True})
        raise AssertionError(f"unexpected PostgreSQL cleanup SQL: {sql}")

    def close(self) -> None:
        self.events.append("close")
        if self.close_error is not None:
            if self.close_releases_ownership:
                self.closed = True
            raise self.close_error
        self.closed = True


class TestStorageBackendBoundaries:
    def test_sqlite_database_lease_and_wal_sidecars_are_owner_only(self, tmp_path: Path) -> None:
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "fchmod"):
            pytest.skip("POSIX owner-only file modes are unavailable")
        db_path = tmp_path / "private-runtime.sqlite"
        previous_umask = os.umask(0o022)
        try:
            store = SQLiteStore(db_path)
        finally:
            os.umask(previous_umask)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
            lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
            if sqlite_backend.fcntl is not None and hasattr(os, "O_NOFOLLOW"):
                assert stat.S_IMODE(lease_path.stat().st_mode) == 0o600

            store.conn.execute("PRAGMA journal_mode=WAL")
            store.conn.execute("CREATE TABLE private_mode_probe(value TEXT)")
            store.conn.execute("INSERT INTO private_mode_probe VALUES ('secret')")
            store.conn.commit()
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{db_path}{suffix}")
                assert sidecar.exists()
                assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
        finally:
            store.close()

    def test_sqlite_reopen_tightens_owner_database_and_lease_modes(self, tmp_path: Path) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("POSIX secure runtime lease is unavailable")
        db_path = tmp_path / "existing-runtime.sqlite"
        store = SQLiteStore(db_path)
        store.close()
        lease_path = db_path.with_suffix(db_path.suffix + ".runtime.lock")
        db_path.chmod(0o644)
        lease_path.chmod(0o644)

        reopened = SQLiteStore(db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
            assert stat.S_IMODE(lease_path.stat().st_mode) == 0o600
        finally:
            reopened.close()

    def test_runtime_code_does_not_reach_into_store_connection(self) -> None:
        offenders: list[str] = []
        for path in AGENT_LIBOS.rglob("*.py"):
            if path in STORAGE_BACKENDS:
                continue
            text = path.read_text(encoding="utf-8")
            if "store.conn" in text:
                offenders.append(path.relative_to(ROOT).as_posix())

        assert offenders == []

    def test_runtime_code_does_not_reach_into_store_private_fields(self) -> None:
        offenders: list[str] = []
        allowed_fragments = (
            "runtime.store._jit_sources",
            "runtime.store._handles",
        )
        for path in AGENT_LIBOS.rglob("*.py"):
            if path in STORAGE_BACKENDS:
                continue
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                if "store._" not in line:
                    continue
                if any(fragment in line for fragment in allowed_fragments):
                    continue
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line_number}")

        assert offenders == []

    def test_sqlite_import_is_confined_to_sqlite_backend(self) -> None:
        offenders: list[str] = []
        for path in AGENT_LIBOS.rglob("*.py"):
            if path == AGENT_LIBOS / "storage" / "sqlite.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "import sqlite3" in text or "from sqlite3" in text:
                offenders.append(path.relative_to(ROOT).as_posix())

        assert offenders == []

    def test_postgres_store_does_not_inherit_sqlite_backend(self) -> None:
        assert issubclass(SQLiteStore, SQLRuntimeStore)
        assert issubclass(PostgresStore, SQLRuntimeStore)
        assert not issubclass(PostgresStore, SQLiteStore)

    def test_postgres_backend_does_not_export_global_sql_rewriter(self) -> None:
        text = (AGENT_LIBOS / "storage" / "postgres.py").read_text(encoding="utf-8")

        assert "def _postgres_sql" not in text
        assert "class PostgresStore(SQLRuntimeStore)" in text
        assert "pg_try_advisory_lock" in text
        assert "pg_advisory_unlock" not in text

    def test_postgres_storage_contract_uses_session_close_as_the_only_release_point(
        self,
    ) -> None:
        text = (ROOT / "docs" / "storage.md").read_text(encoding="utf-8")
        normalized = " ".join(text.split())

        assert "attempt both explicit unlock" not in text
        assert "Session close is the single ownership-release point" in normalized
        assert "never attempts a separate explicit advisory unlock" in normalized

    @pytest.mark.parametrize(
        "failure_type",
        [RuntimeError, KeyboardInterrupt, asyncio.CancelledError],
    )
    def test_postgres_init_failure_releases_lease_and_closes_connection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        failure_type: type[BaseException],
    ) -> None:
        connection = _PostgresCleanupConnection()
        primary_error = failure_type("injected PostgreSQL initialization failure")

        def fail_initialization(*_args: object, **_kwargs: object) -> None:
            connection.events.append("initialize")
            raise primary_error

        monkeypatch.setattr(
            postgres_backend,
            "_PostgresConnection",
            lambda _dsn: connection,
        )
        monkeypatch.setattr(PostgresStore, "_init_store", fail_initialization)

        with pytest.raises(failure_type) as caught:
            PostgresStore("postgresql://runtime")

        assert caught.value is primary_error
        assert connection.events == [
            "identity",
            "lock",
            "initialize",
            "close",
        ]

    def test_postgres_init_cleanup_groups_secondary_errors_after_primary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        primary_error = KeyboardInterrupt("injected initialization interrupt")
        close_error = asyncio.CancelledError("injected connection close cancellation")
        connection = _PostgresCleanupConnection(
            close_error=close_error,
        )

        def fail_initialization(*_args: object, **_kwargs: object) -> None:
            connection.events.append("initialize")
            raise primary_error

        monkeypatch.setattr(
            postgres_backend,
            "_PostgresConnection",
            lambda _dsn: connection,
        )
        monkeypatch.setattr(PostgresStore, "_init_store", fail_initialization)

        with pytest.raises(BaseExceptionGroup) as caught:
            PostgresStore("postgresql://runtime")

        assert list(caught.value.exceptions) == [
            primary_error,
            close_error,
        ]
        assert connection.events == [
            "identity",
            "lock",
            "initialize",
            "close",
        ]

    def test_postgres_close_preserves_session_close_interrupt(self) -> None:
        close_error = asyncio.CancelledError("injected connection close cancellation")
        connection = _PostgresCleanupConnection(
            close_error=close_error,
        )
        store = PostgresStore.__new__(PostgresStore)
        store.conn = connection  # type: ignore[assignment]
        store._runtime_lease_acquired = True
        store._runtime_lease_key = 1

        with pytest.raises(BaseExceptionGroup) as caught:
            store.close()

        assert list(caught.value.exceptions) == [close_error]
        assert connection.events == ["close"]

    def test_postgres_admission_guard_release_delegates_to_backend_close(self) -> None:
        connection = _PostgresCleanupConnection()
        store = PostgresStore.__new__(PostgresStore)
        store.conn = connection  # type: ignore[assignment]
        store._runtime_lease_acquired = True
        store._runtime_lease_key = 1
        store._lock = threading.RLock()
        store._transaction_depth = 0

        @contextmanager
        def expected_guard():
            yield

        store._admission_commit_guard = expected_guard

        outcome = store.release_admission_guard_and_close(expected_guard)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        assert store._admission_commit_guard is None
        assert connection.events == ["close"]

    def test_postgres_session_close_diagnostic_after_release_never_restores_guard(
        self,
    ) -> None:
        close_error = asyncio.CancelledError("close raised after session release")
        connection = _PostgresCleanupConnection(
            close_error=close_error,
            close_releases_ownership=True,
        )
        store = PostgresStore.__new__(PostgresStore)
        store.conn = connection  # type: ignore[assignment]
        store._runtime_lease_acquired = True
        store._runtime_lease_key = 1
        store._lock = threading.RLock()
        store._transaction_depth = 0

        @contextmanager
        def expected_guard():
            yield

        store._admission_commit_guard = expected_guard
        outcome = store.release_admission_guard_and_close(expected_guard)

        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert len(outcome.warnings) == 1
        warning = outcome.warnings[0]
        assert isinstance(warning, BaseExceptionGroup)
        assert list(warning.exceptions) == [close_error]
        assert store._admission_commit_guard is None
        assert store._runtime_lease_acquired is False
        assert store._runtime_lease_key is None
        assert store._released_ownership_reason is not None
        assert connection.events == ["close"]

    def test_postgres_session_close_restores_guard_only_while_session_remains_open(
        self,
    ) -> None:
        close_error = asyncio.CancelledError("session remains open")
        connection = _PostgresCleanupConnection(
            close_error=close_error,
        )
        store = PostgresStore.__new__(PostgresStore)
        store.conn = connection  # type: ignore[assignment]
        store._runtime_lease_acquired = True
        store._runtime_lease_key = 1
        store._lock = threading.RLock()
        store._transaction_depth = 0

        @contextmanager
        def expected_guard():
            yield

        store._admission_commit_guard = expected_guard
        with pytest.raises(BaseExceptionGroup) as caught:
            store.release_admission_guard_and_close(expected_guard)

        assert list(caught.value.exceptions) == [close_error]
        assert store._admission_commit_guard is expected_guard
        assert store._runtime_lease_acquired is True
        assert store._runtime_lease_key == 1
        assert connection.closed is False
        assert connection.events == ["close"]

    def test_postgres_fresh_store_probe_enumerates_current_schema_relations(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.sql = ""

            def execute(self, sql: str):
                self.sql = sql
                return iter(({"name": "unrelated_table"}, {"name": "unrelated_sequence"}))

        connection = FakeConnection()

        assert PostgresStore._probe_user_schema_objects(connection) == {
            "unrelated_table",
            "unrelated_sequence",
        }
        assert "pg_catalog.pg_class" in connection.sql
        assert "current_schema()" in connection.sql

    def test_sqlite_runtime_lease_uses_atomic_lockfile_without_fcntl(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sqlite_backend, "fcntl", None)
        db_path = tmp_path / "runtime.sqlite"
        first = SQLiteStore(db_path)
        try:
            with pytest.raises(ValidationError, match="already open"):
                SQLiteStore(db_path)
        finally:
            first.close()

        assert not db_path.with_suffix(db_path.suffix + ".runtime.lock").exists()
        second = SQLiteStore(db_path)
        second.close()

    def test_sqlite_runtime_lease_rejects_symlink_alias(self, tmp_path: Path) -> None:
        db_path = tmp_path / "runtime.sqlite"
        alias = tmp_path / "runtime-alias.sqlite"
        first = SQLiteStore(db_path)
        try:
            try:
                alias.symlink_to(db_path)
            except OSError:
                pytest.skip("symlink creation is not available in this environment")
            with pytest.raises(ValidationError, match="already open"):
                SQLiteStore(alias)
        finally:
            first.close()

        reopened = SQLiteStore(alias)
        reopened.close()

    def test_nested_store_transactions_use_savepoints(self) -> None:
        store = SQLiteStore(":memory:")
        try:
            with store.transaction() as outer:
                outer.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("outer", None, "{}", "test", "1", "1"),
                )
                with pytest.raises(RuntimeError, match="rollback inner"):
                    with store.transaction() as inner:
                        inner.execute(
                            "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                            ("inner", None, "{}", "test", "1", "1"),
                        )
                        raise RuntimeError("rollback inner")
                outer.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("after", None, "{}", "test", "1", "1"),
                )

            namespaces = {
                row["namespace"] for row in store.select_table_rows("object_namespaces", order_by="namespace")
            }
            assert {"after", "outer"}.issubset(namespaces)
            assert "inner" not in namespaces
        finally:
            store.close()

    def test_admission_commit_guard_wraps_only_outermost_transaction(self) -> None:
        store = SQLiteStore(":memory:")
        calls: list[str] = []

        @contextmanager
        def guard():
            calls.append("enter")
            try:
                yield
            finally:
                calls.append("exit")

        store.bind_admission_commit_guard(guard)
        try:
            with store.transaction() as outer:
                outer.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("guarded-outer", None, "{}", "test", "1", "1"),
                )
                with store.transaction() as inner:
                    inner.execute(
                        "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                        ("guarded-inner", None, "{}", "test", "1", "1"),
                    )

            assert calls == ["enter", "exit"]
        finally:
            store.close()

    def test_admission_commit_guard_unbind_is_exact_and_successor_safe(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def failed_guard():
            yield

        @contextmanager
        def live_guard():
            yield

        failed_released = threading.Event()
        live_bound = threading.Event()
        outcomes: list[bool] = []

        def release_failed_owner() -> None:
            outcomes.append(store.unbind_admission_commit_guard(failed_guard))
            failed_released.set()
            assert live_bound.wait(timeout=2)
            outcomes.append(store.unbind_admission_commit_guard(failed_guard))

        def bind_successor() -> None:
            assert failed_released.wait(timeout=2)
            store.bind_admission_commit_guard(live_guard)
            live_bound.set()

        store.bind_admission_commit_guard(failed_guard)
        release_thread = threading.Thread(target=release_failed_owner)
        bind_thread = threading.Thread(target=bind_successor)
        try:
            release_thread.start()
            bind_thread.start()
            release_thread.join(timeout=3)
            bind_thread.join(timeout=3)

            assert not release_thread.is_alive()
            assert not bind_thread.is_alive()
            assert outcomes == [True, False]
            assert store._admission_commit_guard is live_guard
            with pytest.raises(
                RuntimeError,
                match="admission commit guard is already bound",
            ):
                store.bind_admission_commit_guard(failed_guard)
        finally:
            store.close()

    def test_admission_guard_replacement_is_exact_and_reserves_owned_close(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def lifecycle_guard():
            yield

        @contextmanager
        def stale_guard():
            yield

        @contextmanager
        def close_reservation():
            yield

        store.bind_admission_commit_guard(lifecycle_guard)
        try:
            assert (
                store.replace_admission_commit_guard(
                    stale_guard,
                    close_reservation,
                )
                is False
            )
            assert store._admission_commit_guard is lifecycle_guard
            assert (
                store.replace_admission_commit_guard(
                    lifecycle_guard,
                    close_reservation,
                )
                is True
            )
            assert store.unbind_admission_commit_guard(lifecycle_guard) is False
            assert store._admission_commit_guard is close_reservation
            with pytest.raises(
                RuntimeError,
                match="admission commit guard is already bound",
            ):
                store.bind_admission_commit_guard(stale_guard)

            stale_outcome = store.release_admission_guard_and_close(lifecycle_guard)
            assert stale_outcome.guard_matched is False
            assert stale_outcome.ownership_released is False
            assert store._admission_commit_guard is close_reservation

            outcome = store.release_admission_guard_and_close(close_reservation)
            assert outcome.guard_matched is True
            assert outcome.ownership_released is True
            assert outcome.warnings == ()
        finally:
            if not store._runtime_ownership_released():
                store.close()

    def test_pre_lifecycle_close_reservation_and_concurrent_replacement_have_one_owner(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def first_reservation():
            yield

        @contextmanager
        def second_reservation():
            yield

        barrier = threading.Barrier(3)
        outcomes: list[tuple[object, bool]] = []
        outcomes_lock = threading.Lock()

        def compete(replacement: object) -> None:
            barrier.wait(timeout=2)
            result = store.replace_admission_commit_guard(
                None,
                replacement,  # type: ignore[arg-type]
            )
            with outcomes_lock:
                outcomes.append((replacement, result))

        threads = [
            threading.Thread(target=compete, args=(first_reservation,)),
            threading.Thread(target=compete, args=(second_reservation,)),
        ]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert sorted(result for _, result in outcomes) == [False, True]
        winner = next(owner for owner, result in outcomes if result)
        loser = next(owner for owner, result in outcomes if not result)
        loser_outcome = store.release_admission_guard_and_close(  # type: ignore[arg-type]
            loser,
        )
        assert loser_outcome.guard_matched is False
        assert loser_outcome.ownership_released is False
        assert store._admission_commit_guard is winner
        winner_outcome = store.release_admission_guard_and_close(  # type: ignore[arg-type]
            winner,
        )
        assert winner_outcome.guard_matched is True
        assert winner_outcome.ownership_released is True

    def test_admission_guard_replacement_rejects_active_transaction(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def lifecycle_guard():
            yield

        @contextmanager
        def close_reservation():
            yield

        store.bind_admission_commit_guard(lifecycle_guard)
        try:
            with store.transaction():
                with pytest.raises(
                    RuntimeError,
                    match="cannot replace admission commit guard during a store transaction",
                ):
                    store.replace_admission_commit_guard(
                        lifecycle_guard,
                        close_reservation,
                    )
            assert store._admission_commit_guard is lifecycle_guard
        finally:
            store.close()

    def test_close_probe_reports_current_thread_transaction_and_locked_scope_without_claiming(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def expected_guard():
            yield

        @contextmanager
        def wrong_guard():
            yield

        store.bind_admission_commit_guard(expected_guard)
        try:
            assert (
                store.probe_admission_guard_close(expected_guard)
                is StoreCloseClaimOutcome.READY
            )
            assert store._admission_guard_close_claim is None
            assert (
                store.probe_admission_guard_close(wrong_guard)
                is StoreCloseClaimOutcome.GUARD_MISMATCH
            )
            with store.locked():
                assert (
                    store.probe_admission_guard_close(expected_guard)
                    is StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED
                )
                assert (
                    store.claim_admission_guard_close(expected_guard)
                    is StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED
                )
            with store.transaction():
                assert (
                    store.probe_admission_guard_close(expected_guard)
                    is StoreCloseClaimOutcome.ACTIVE_TRANSACTION
                )
                assert (
                    store.claim_admission_guard_close(expected_guard)
                    is StoreCloseClaimOutcome.ACTIVE_TRANSACTION
                )
            assert store._admission_guard_close_claim is None
        finally:
            store.close()

    def test_close_probe_never_blocks_behind_another_thread_store_scope(self) -> None:
        store = SQLiteStore(":memory:")
        entered = threading.Event()
        release = threading.Event()

        @contextmanager
        def expected_guard():
            yield

        def hold_store_lock() -> None:
            with store.locked():
                entered.set()
                assert release.wait(timeout=2)

        store.bind_admission_commit_guard(expected_guard)
        thread = threading.Thread(target=hold_store_lock)
        thread.start()
        try:
            assert entered.wait(timeout=2)
            assert (
                store.probe_admission_guard_close(expected_guard)
                is StoreCloseClaimOutcome.LOCK_BUSY
            )
            assert (
                store.claim_admission_guard_close(expected_guard)
                is StoreCloseClaimOutcome.LOCK_BUSY
            )
            assert (
                store.try_replace_admission_commit_guard(None, expected_guard)
                is StoreCloseClaimOutcome.LOCK_BUSY
            )
        finally:
            release.set()
            thread.join(timeout=2)
            store.close()
        assert not thread.is_alive()

    def test_nonblocking_guard_repair_is_exact_idempotent_and_scope_safe(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def lifecycle_guard():
            yield

        @contextmanager
        def stale_guard():
            yield

        @contextmanager
        def close_reservation():
            yield

        store.bind_admission_commit_guard(lifecycle_guard)
        try:
            assert (
                store.try_replace_admission_commit_guard(
                    stale_guard,
                    close_reservation,
                )
                is StoreCloseClaimOutcome.GUARD_MISMATCH
            )
            with store.locked():
                assert (
                    store.try_replace_admission_commit_guard(
                        lifecycle_guard,
                        close_reservation,
                    )
                    is StoreCloseClaimOutcome.CURRENT_THREAD_LOCKED
                )
            with store.transaction():
                assert (
                    store.try_replace_admission_commit_guard(
                        lifecycle_guard,
                        close_reservation,
                    )
                    is StoreCloseClaimOutcome.ACTIVE_TRANSACTION
                )
            assert store._admission_commit_guard is lifecycle_guard
            assert (
                store.try_replace_admission_commit_guard(
                    lifecycle_guard,
                    close_reservation,
                )
                is StoreCloseClaimOutcome.READY
            )
            assert store._admission_commit_guard is close_reservation
            assert (
                store.try_replace_admission_commit_guard(
                    None,
                    close_reservation,
                )
                is StoreCloseClaimOutcome.READY
            )
        finally:
            store.close()

    def test_atomic_close_claim_blocks_competing_scopes_until_exact_close(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def expected_guard():
            yield

        @contextmanager
        def wrong_guard():
            yield

        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.claim_admission_guard_close(wrong_guard)
            is StoreCloseClaimOutcome.GUARD_MISMATCH
        )

        for scope in (store.locked, store.transaction):
            with pytest.raises(RuntimeError, match="close is pending"):
                with scope():
                    pass
        with pytest.raises(RuntimeError, match="close is pending"):
            store.validate_column_identifier("processes", "pid")

        competing_errors: list[BaseException] = []

        def competing_transaction() -> None:
            try:
                with store.transaction():
                    pass
            except BaseException as exc:
                competing_errors.append(exc)

        thread = threading.Thread(target=competing_transaction)
        thread.start()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert len(competing_errors) == 1
        assert "close is pending" in str(competing_errors[0])

        wrong = store.release_admission_guard_and_close(wrong_guard)
        assert wrong.guard_matched is False
        assert wrong.ownership_released is False
        assert store._admission_guard_close_claim is expected_guard

        outcome = store.release_admission_guard_and_close(expected_guard)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert store._admission_guard_close_claim is None

    def test_retained_claimed_close_restores_diagnostics_and_requires_fresh_claim(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database = tmp_path / "claimed-close-retry.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def expected_guard():
            yield

        original_close = store.close
        close_error = KeyboardInterrupt("close failed before ownership release")
        close_calls = 0

        def fail_once_then_close() -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise close_error
            original_close()

        monkeypatch.setattr(store, "close", fail_once_then_close)
        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )

        with pytest.raises(KeyboardInterrupt) as caught:
            store.release_admission_guard_and_close(expected_guard)

        assert caught.value is close_error
        assert store._admission_commit_guard is expected_guard
        assert store._admission_guard_close_claim is None
        assert store.list_processes() == []
        with store.locked():
            pass
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )
        outcome = store.release_admission_guard_and_close(expected_guard)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert close_calls == 2

        successor = SQLiteStore(database)
        successor.close()

    def test_poisoned_sqlite_data_plane_keeps_exact_file_lease_teardown_available(
        self,
        tmp_path: Path,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / "poisoned-owned-lease.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def lifecycle_guard():
            yield

        @contextmanager
        def first_reservation():
            yield

        @contextmanager
        def final_reservation():
            yield

        @contextmanager
        def wrong_guard():
            yield

        store.bind_admission_commit_guard(lifecycle_guard)
        store._poison("injected rollback uncertainty")

        assert store._sqlite_connection_reports_closed() is True
        assert store._lease_handle is not None
        assert store._runtime_ownership_released() is False
        with pytest.raises(
            ValidationError,
            match="unusable after transaction rollback failure",
        ):
            store.list_processes()
        with pytest.raises(
            ValidationError,
            match="unusable after transaction rollback failure",
        ):
            store.bind_admission_commit_guard(first_reservation)
        with pytest.raises(ValidationError, match="already open"):
            SQLiteStore(database)

        assert (
            store.replace_admission_commit_guard(
                lifecycle_guard,
                first_reservation,
            )
            is True
        )
        assert (
            store.try_replace_admission_commit_guard(
                first_reservation,
                final_reservation,
            )
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.probe_admission_guard_close(final_reservation)
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.claim_admission_guard_close(final_reservation)
            is StoreCloseClaimOutcome.READY
        )

        wrong = store.release_admission_guard_and_close(wrong_guard)
        assert wrong.guard_matched is False
        assert wrong.ownership_released is False
        assert store._lease_handle is not None
        with pytest.raises(ValidationError, match="already open"):
            SQLiteStore(database)

        outcome = store.release_admission_guard_and_close(final_reservation)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        assert store._lease_handle is None

        successor = SQLiteStore(database)
        successor.close()

    def test_poisoned_in_memory_store_reports_terminal_released_ownership(
        self,
    ) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def lifecycle_guard():
            yield

        @contextmanager
        def close_reservation():
            yield

        store.bind_admission_commit_guard(lifecycle_guard)
        store._poison("injected rollback uncertainty")

        assert store._runtime_ownership_released() is True
        with pytest.raises(
            ValidationError,
            match="unusable after transaction rollback failure",
        ):
            store.list_processes()
        with pytest.raises(
            ValidationError,
            match="unusable after transaction rollback failure",
        ):
            store.bind_admission_commit_guard(close_reservation)
        assert (
            store.replace_admission_commit_guard(
                lifecycle_guard,
                close_reservation,
            )
            is False
        )
        assert (
            store.try_replace_admission_commit_guard(
                lifecycle_guard,
                close_reservation,
            )
            is StoreCloseClaimOutcome.OWNERSHIP_RELEASED
        )
        assert (
            store.probe_admission_guard_close(lifecycle_guard)
            is StoreCloseClaimOutcome.OWNERSHIP_RELEASED
        )
        assert (
            store.claim_admission_guard_close(lifecycle_guard)
            is StoreCloseClaimOutcome.OWNERSHIP_RELEASED
        )

        stale = store.release_admission_guard_and_close(close_reservation)
        assert stale.guard_matched is False
        assert stale.ownership_released is True
        assert store._admission_commit_guard is lifecycle_guard

        exact = store.release_admission_guard_and_close(lifecycle_guard)
        assert exact.guard_matched is True
        assert exact.ownership_released is True
        assert exact.warnings == ()
        assert store._admission_commit_guard is None

    @pytest.mark.parametrize("interrupted_step", ["observe", "mark"])
    def test_post_release_diagnostic_interrupt_never_restores_guard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        interrupted_step: str,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / f"post-release-{interrupted_step}.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def expected_guard():
            yield

        diagnostic = KeyboardInterrupt(f"injected {interrupted_step} interrupt")
        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )

        if interrupted_step == "mark":
            def interrupt_mark(_reason: str) -> None:
                raise diagnostic

            monkeypatch.setattr(
                store,
                "_mark_runtime_ownership_released",
                interrupt_mark,
            )
        else:
            original_observer = store._runtime_ownership_released
            interrupted = False

            def interrupt_after_release() -> bool:
                nonlocal interrupted
                if (
                    store._backend_ownership_release_observed
                    and not interrupted
                ):
                    interrupted = True
                    raise diagnostic
                return original_observer()

            monkeypatch.setattr(
                store,
                "_runtime_ownership_released",
                interrupt_after_release,
            )

        outcome = store.release_admission_guard_and_close(expected_guard)

        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == (diagnostic,)
        assert store._admission_commit_guard is None
        assert store._admission_guard_close_claim is None
        assert store._released_ownership_reason is not None
        successor = SQLiteStore(database)
        successor.close()

    def test_sqlite_file_lease_release_uses_descriptor_close_only(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / "descriptor-close-only.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def expected_guard():
            yield

        original_flock = sqlite_backend.fcntl.flock

        def reject_explicit_unlock(fd: int, operation: int) -> None:
            if operation == sqlite_backend.fcntl.LOCK_UN:
                raise AssertionError("runtime lease must not explicitly unlock")
            original_flock(fd, operation)

        monkeypatch.setattr(sqlite_backend.fcntl, "flock", reject_explicit_unlock)
        store.bind_admission_commit_guard(expected_guard)
        outcome = store.release_admission_guard_and_close(expected_guard)

        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        successor = SQLiteStore(database)
        successor.close()

    def test_sqlite_failed_lease_initialization_closes_without_explicit_unlock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / "failed-lease-initialization.sqlite"
        original_flock = sqlite_backend.fcntl.flock
        original_fsync = sqlite_backend.os.fsync
        initialization_error = OSError("injected lease metadata sync failure")

        def reject_explicit_unlock(fd: int, operation: int) -> None:
            if operation == sqlite_backend.fcntl.LOCK_UN:
                raise AssertionError("runtime lease must not explicitly unlock")
            original_flock(fd, operation)

        def fail_metadata_sync(_fd: int) -> None:
            raise initialization_error

        monkeypatch.setattr(sqlite_backend.fcntl, "flock", reject_explicit_unlock)
        monkeypatch.setattr(sqlite_backend.os, "fsync", fail_metadata_sync)

        with pytest.raises(OSError) as caught:
            SQLiteStore(database)

        assert caught.value is initialization_error
        monkeypatch.setattr(sqlite_backend.os, "fsync", original_fsync)
        successor = SQLiteStore(database)
        successor.close()

    def test_sqlite_file_lease_close_then_diagnostic_is_terminal(
        self,
        tmp_path: Path,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / "lease-close-then-diagnostic.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def expected_guard():
            yield

        lease = store._lease_handle
        assert lease is not None
        real_handle = lease.handle
        close_error = KeyboardInterrupt("diagnostic after descriptor close")

        class CloseThenRaise:
            @property
            def closed(self) -> bool:
                return bool(real_handle.closed)

            def close(self) -> None:
                real_handle.close()
                raise close_error

        lease.handle = CloseThenRaise()
        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )

        outcome = store.release_admission_guard_and_close(expected_guard)

        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == (close_error,)
        assert store._admission_commit_guard is None
        assert store._admission_guard_close_claim is None
        assert store._lease_handle is None
        successor = SQLiteStore(database)
        successor.close()

    def test_sqlite_file_lease_pre_close_diagnostic_retains_exact_owner(
        self,
        tmp_path: Path,
    ) -> None:
        if sqlite_backend.fcntl is None or not hasattr(os, "O_NOFOLLOW"):
            pytest.skip("separate SQLite file runtime lease is unavailable")
        database = tmp_path / "lease-pre-close-diagnostic.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def expected_guard():
            yield

        lease = store._lease_handle
        assert lease is not None
        real_handle = lease.handle
        close_error = KeyboardInterrupt("descriptor close did not start")

        class RaiseBeforeClose:
            @property
            def closed(self) -> bool:
                return bool(real_handle.closed)

            def close(self) -> None:
                raise close_error

        lease.handle = RaiseBeforeClose()
        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )

        with pytest.raises(KeyboardInterrupt) as caught:
            store.release_admission_guard_and_close(expected_guard)

        assert caught.value is close_error
        assert store._admission_commit_guard is expected_guard
        assert store._admission_guard_close_claim is None
        assert store._lease_handle is lease
        assert real_handle.closed is False
        with pytest.raises(ValidationError, match="already open"):
            SQLiteStore(database)

        lease.handle = real_handle
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )
        outcome = store.release_admission_guard_and_close(expected_guard)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        successor = SQLiteStore(database)
        successor.close()

    def test_admission_guard_release_wrong_owner_never_closes_store(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def expected_guard():
            yield

        @contextmanager
        def wrong_guard():
            yield

        store.bind_admission_commit_guard(expected_guard)
        try:
            outcome = store.release_admission_guard_and_close(wrong_guard)
            assert outcome.guard_matched is False
            assert outcome.ownership_released is False
            assert outcome.warnings == ()
            assert store._admission_commit_guard is expected_guard
            with store.transaction() as tx:
                tx.execute(
                    "INSERT INTO object_namespaces VALUES (?, ?, ?, ?, ?, ?)",
                    ("still-open", None, "{}", "test", "1", "1"),
                )
        finally:
            store.close()

    def test_admission_guard_release_rejects_active_transaction(self) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def expected_guard():
            yield

        store.bind_admission_commit_guard(expected_guard)
        try:
            with store.transaction():
                with pytest.raises(
                    RuntimeError,
                    match="cannot release admission commit guard during a store transaction",
                ):
                    store.release_admission_guard_and_close(expected_guard)
            assert store._admission_commit_guard is expected_guard
        finally:
            store.close()

    @pytest.mark.parametrize(
        "failure_type",
        [RuntimeError, KeyboardInterrupt, asyncio.CancelledError],
    )
    def test_admission_guard_release_restores_owner_after_close_failure_and_retries(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        failure_type: type[BaseException],
    ) -> None:
        db_path = tmp_path / f"retry-{failure_type.__name__}.sqlite"
        store = SQLiteStore(db_path)

        @contextmanager
        def expected_guard():
            yield

        original_close = store.close
        close_error = failure_type("injected store close failure")
        close_calls = 0

        def fail_once_then_close() -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise close_error
            original_close()

        monkeypatch.setattr(store, "close", fail_once_then_close)
        store.bind_admission_commit_guard(expected_guard)

        with pytest.raises(failure_type) as caught:
            store.release_admission_guard_and_close(expected_guard)

        assert caught.value is close_error
        assert store._admission_commit_guard is expected_guard
        outcome = store.release_admission_guard_and_close(expected_guard)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        assert close_calls == 2
        assert store._admission_commit_guard is None

        reopened = SQLiteStore(db_path)
        reopened.close()

    @pytest.mark.parametrize("file_backed", [False, True])
    def test_sqlite_close_after_real_connection_release_returns_warning_without_guard_restore(
        self,
        tmp_path: Path,
        file_backed: bool,
    ) -> None:
        database = tmp_path / "partial-close.sqlite"
        store = SQLiteStore(database if file_backed else ":memory:")
        close_error = KeyboardInterrupt(
            "adapter diagnostic after real sqlite3 connection close"
        )
        real_connection = store.conn

        class CloseThenRaise:
            def close(self) -> None:
                real_connection.close()
                raise close_error

            def __getattr__(self, name: str) -> object:
                return getattr(real_connection, name)

        @contextmanager
        def expected_guard():
            yield

        store.conn = CloseThenRaise()  # type: ignore[assignment]
        store.bind_admission_commit_guard(expected_guard)
        assert (
            store.claim_admission_guard_close(expected_guard)
            is StoreCloseClaimOutcome.READY
        )

        outcome = store.release_admission_guard_and_close(expected_guard)

        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == (close_error,)
        assert store._admission_commit_guard is None
        assert store._admission_guard_close_claim is None
        assert store._runtime_ownership_released() is True
        with pytest.raises(
            ValidationError,
            match="unusable after backend ownership release",
        ):
            with store.locked():
                pass
        with pytest.raises(
            ValidationError,
            match="unusable after backend ownership release",
        ):
            store.bind_admission_commit_guard(expected_guard)

        if file_backed:
            successor = SQLiteStore(database)
            successor.close()

    @pytest.mark.parametrize(
        "restore_failure_type",
        [KeyboardInterrupt, asyncio.CancelledError],
    )
    def test_claimed_close_restore_interrupt_keeps_exact_retry_owner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        restore_failure_type: type[BaseException],
    ) -> None:
        database = tmp_path / f"claim-restore-{restore_failure_type.__name__}.sqlite"
        store = SQLiteStore(database)

        @contextmanager
        def close_reservation():
            yield

        @contextmanager
        def successor_guard():
            yield

        @contextmanager
        def wrong_guard():
            yield

        original_close = store.close
        close_error = OSError("close failed before descriptor release")
        restore_error = restore_failure_type("exact guard restore interrupted")
        close_calls = 0

        def fail_once_then_close() -> None:
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise close_error
            original_close()

        def interrupt_restore(_guard: object) -> None:
            raise restore_error

        monkeypatch.setattr(store, "close", fail_once_then_close)
        monkeypatch.setattr(
            store,
            "_restore_admission_commit_guard_after_close_failure",
            interrupt_restore,
        )
        store.bind_admission_commit_guard(close_reservation)
        assert (
            store.claim_admission_guard_close(close_reservation)
            is StoreCloseClaimOutcome.READY
        )

        with pytest.raises(BaseExceptionGroup) as caught:
            store.release_admission_guard_and_close(close_reservation)

        assert list(caught.value.exceptions) == [close_error, restore_error]
        assert store._admission_commit_guard is None
        assert store._admission_guard_close_claim is close_reservation
        assert store._runtime_ownership_released() is False
        with pytest.raises(
            RuntimeError,
            match="admission-guard close is pending",
        ):
            store.bind_admission_commit_guard(successor_guard)
        with pytest.raises(
            RuntimeError,
            match="admission-guard close is pending",
        ):
            with store.locked():
                pass
        assert (
            store.probe_admission_guard_close(wrong_guard)
            is StoreCloseClaimOutcome.GUARD_MISMATCH
        )
        wrong = store.release_admission_guard_and_close(wrong_guard)
        assert wrong.guard_matched is False
        assert wrong.ownership_released is False
        with pytest.raises(ValidationError, match="already open"):
            SQLiteStore(database)

        assert (
            store.try_replace_admission_commit_guard(None, close_reservation)
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.probe_admission_guard_close(close_reservation)
            is StoreCloseClaimOutcome.READY
        )
        assert (
            store.claim_admission_guard_close(close_reservation)
            is StoreCloseClaimOutcome.READY
        )
        outcome = store.release_admission_guard_and_close(close_reservation)
        assert outcome.guard_matched is True
        assert outcome.ownership_released is True
        assert outcome.warnings == ()
        assert store._admission_guard_close_claim is None
        assert close_calls == 2
        successor = SQLiteStore(database)
        successor.close()

    def test_admission_guard_release_groups_close_and_restore_failures(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = SQLiteStore(":memory:")

        @contextmanager
        def expected_guard():
            yield

        original_close = store.close
        original_restore = store._restore_admission_commit_guard_after_close_failure
        close_error = KeyboardInterrupt("injected close interrupt")
        restore_error = asyncio.CancelledError("injected guard restore cancellation")

        def fail_close() -> None:
            raise close_error

        def fail_restore(_guard: object) -> None:
            raise restore_error

        store.bind_admission_commit_guard(expected_guard)
        monkeypatch.setattr(store, "close", fail_close)
        monkeypatch.setattr(
            store,
            "_restore_admission_commit_guard_after_close_failure",
            fail_restore,
        )
        try:
            with pytest.raises(BaseExceptionGroup) as caught:
                store.release_admission_guard_and_close(expected_guard)

            assert list(caught.value.exceptions) == [close_error, restore_error]
            assert store._admission_commit_guard is None
            assert store._admission_guard_close_claim is expected_guard
        finally:
            monkeypatch.setattr(store, "close", original_close)
            monkeypatch.setattr(
                store,
                "_restore_admission_commit_guard_after_close_failure",
                original_restore,
            )
            outcome = store.release_admission_guard_and_close(expected_guard)
            assert outcome.guard_matched is True
            assert outcome.ownership_released is True

    def test_shared_outer_mutations_cannot_bypass_transaction_guard(self) -> None:
        source = (AGENT_LIBOS / "storage" / "sql.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        store_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SQLRuntimeStore"
        )
        guarded_methods = {
            "_execute",
            "consume_capability_uses",
            "commit_capability_use_reservation",
            "claim_llm_pending_action",
            "mark_object_tasks_abandoned",
        }
        direct_mutation_owners = {
            "_write_store_schema_version",
            "_execute_script",
            "_rollback_scope",
            "_initialize_v3_schema",
            "transaction",
            "abandon_stale_capability_use_reservations",
            "_release_missing_runtime_object_payloads",
        }
        dynamic_read_owners = {"_query", "validate_column_identifier"}
        bypasses: list[str] = []
        helper_users: set[str] = set()

        for node in store_class.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                target = child.func
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "_join_or_begin_transaction"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and node.name in guarded_methods
                ):
                    helper_users.add(node.name)
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr in {"execute", "executemany", "commit", "rollback"}
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr == "conn"
                    and isinstance(target.value.value, ast.Name)
                    and target.value.value.id == "self"
                ):
                    sql: str | None = None
                    if child.args:
                        try:
                            candidate = ast.literal_eval(child.args[0])
                        except (TypeError, ValueError):
                            candidate = None
                        if isinstance(candidate, str):
                            sql = candidate.lstrip().upper()
                    read_only = bool(
                        target.attr == "execute"
                        and sql is not None
                        and sql.startswith(("SELECT ", "PRAGMA ", "EXPLAIN "))
                    )
                    if (
                        not read_only
                        and node.name not in direct_mutation_owners
                        and node.name not in dynamic_read_owners
                    ):
                        bypasses.append(f"{node.name}:{child.lineno}")

        assert "_commit_if_outermost" not in source
        assert helper_users == guarded_methods
        assert bypasses == []

    def test_sqlite_uri_normalizes_posix_absolute_paths(self) -> None:
        assert _sqlite_target("sqlite:////tmp/agent-libos.sqlite") == "/tmp/agent-libos.sqlite"
        assert _sqlite_target("sqlite:///C:/agent-libos/runtime.sqlite") == "C:/agent-libos/runtime.sqlite"

    def test_postgres_cursor_does_not_pass_empty_params_for_percent_literals(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def execute(self, *args: object) -> None:
                self.calls.append(args)

        fake = FakeCursor()
        cursor = _PostgresCursor(fake, _PostgresDialect())

        cursor.execute("UPDATE objects SET owner_kind = 'process' WHERE created_by LIKE 'process:%'")

        assert fake.calls == [("UPDATE objects SET owner_kind = 'process' WHERE created_by LIKE 'process:%'",)]

    def test_postgres_cursor_escapes_percent_literals_with_params(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def execute(self, *args: object) -> None:
                self.calls.append(args)

        fake = FakeCursor()
        cursor = _PostgresCursor(fake, _PostgresDialect())

        cursor.execute("DELETE FROM capabilities WHERE subject IN (?) AND resource NOT LIKE 'checkpoint:%'", ("pid_1",))

        assert fake.calls == [
            ("DELETE FROM capabilities WHERE subject IN (%s) AND resource NOT LIKE 'checkpoint:%%'", ("pid_1",))
        ]

    def test_postgres_cursor_preserves_question_mark_literals_with_params(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def execute(self, *args: object) -> None:
                self.calls.append(args)

        fake = FakeCursor()
        cursor = _PostgresCursor(fake, _PostgresDialect())

        cursor.execute("SELECT '?' AS literal, ? AS value, 'it''s ?' AS escaped", ("ok",))

        assert fake.calls == [("SELECT '?' AS literal, %s AS value, 'it''s ?' AS escaped", ("ok",))]

    def test_postgres_cursor_preserves_insert_or_ignore_semantics(self) -> None:
        class FakeCursor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, ...]] = []

            def execute(self, *args: object) -> None:
                self.calls.append(args)

        fake = FakeCursor()
        cursor = _PostgresCursor(fake, _PostgresDialect())

        cursor.execute(
            "INSERT OR IGNORE INTO runtime_counters (counter_name, value) VALUES (?, ?)",
            ("external_effect_ledger", 0),
        )

        assert fake.calls == [
            (
                "INSERT INTO runtime_counters (counter_name, value) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                ("external_effect_ledger", 0),
            )
        ]

    def test_shared_sqlite_dialect_surface_is_explicitly_ratcheted(self) -> None:
        raw_text = (AGENT_LIBOS / "storage" / "sql.py").read_text(encoding="utf-8")
        pattern = r"\b(?:PRAGMA\s+[A-Z_]+|INSERT\s+OR\s+[A-Z_]+|COLLATE\s+[A-Z_]+)\b"
        matches = list(re.finditer(pattern, raw_text, re.IGNORECASE))

        assert {match.group(0).upper() for match in matches} == {
            "PRAGMA TABLE_INFO",
            "INSERT OR IGNORE",
            "INSERT OR REPLACE",
            "COLLATE BINARY",
        }
        assert all(
            match.group(0).upper().startswith("PRAGMA ")
            or match.group(0) == match.group(0).upper()
            for match in matches
        )

    def test_postgres_dialect_translates_schema_probe_and_binary_collation(self) -> None:
        dialect = _PostgresDialect()

        table_probe = dialect.prepare("PRAGMA table_info(processes)")
        index_probe = dialect.prepare("PRAGMA index_list(processes)")
        index_info_probe = dialect.prepare("PRAGMA index_info(idx_processes_status_created)")
        prefix_query = dialect.prepare(
            "SELECT normalized_path FROM file_label_bindings "
            "WHERE normalized_path COLLATE BINARY >= ?",
            with_params=True,
        )

        assert "information_schema.columns" in table_probe
        assert "table_schema = current_schema()" in table_probe
        assert "table_name = 'processes'" in table_probe
        assert index_probe == "SELECT NULL::text AS name WHERE false"
        assert index_info_probe == "SELECT NULL::text AS name WHERE false"
        assert prefix_query == (
            'SELECT normalized_path FROM file_label_bindings '
            'WHERE normalized_path COLLATE "C" >= %s'
        )

    def test_postgres_dialect_translates_skill_trust_replace_upsert(self) -> None:
        prepared = _PostgresDialect().prepare(
            "INSERT OR REPLACE INTO skill_trust ("
            "trust_id, source_type, source, package_sha256, trusted_by, "
            "created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            with_params=True,
        )

        assert "INSERT OR REPLACE" not in prepared
        assert prepared.startswith("INSERT INTO skill_trust")
        assert prepared.count("%s") == 7
        assert "ON CONFLICT (source_type, source, package_sha256) DO UPDATE SET" in prepared
        assert "metadata_json = EXCLUDED.metadata_json" in prepared

    def test_postgres_dialect_removes_sqlite_index_selection_hint(self) -> None:
        prepared = _PostgresDialect().prepare(
            "SELECT call_id FROM llm_calls "
            "INDEXED BY idx_llm_calls_retention_eligible "
            "WHERE created_at <= ? LIMIT ?",
            with_params=True,
        )

        assert "INDEXED BY" not in prepared
        assert prepared == (
            "SELECT call_id FROM llm_calls WHERE created_at <= %s LIMIT %s"
        )

    def test_postgres_dialect_preserves_retention_cas_correlation(self) -> None:
        prepared = _PostgresDialect().prepare(
            """
            UPDATE llm_calls
               SET payload_retention_tier = ?
             WHERE call_id = ?
               AND (
                    ? = 0 OR EXISTS (
                      SELECT 1
                        FROM llm_calls AS newer
                             INDEXED BY idx_llm_calls_provider_chain_head
                       WHERE newer.pid = llm_calls.pid
                         AND newer.purpose = llm_calls.purpose
                         AND (
                           newer.created_at COLLATE BINARY,
                           newer.call_id COLLATE BINARY
                         ) > (
                           llm_calls.created_at COLLATE BINARY,
                           llm_calls.call_id COLLATE BINARY
                         )
                    )
               )
            """,
            with_params=True,
        )

        assert "INDEXED BY" not in prepared
        assert prepared.count("%s") == 3
        assert prepared.count('COLLATE "C"') == 4
        assert "newer.pid = llm_calls.pid" in prepared
        assert "newer.purpose = llm_calls.purpose" in prepared
        aliased = _PostgresDialect().prepare(
            "SELECT 1 FROM llm_calls AS newer "
            "INDEXED BY idx_llm_calls_provider_chain_head "
            "WHERE newer.pid = ?",
            with_params=True,
        )
        assert "INDEXED BY" not in aliased
        assert aliased == (
            "SELECT 1 FROM llm_calls AS newer WHERE newer.pid = %s"
        )
