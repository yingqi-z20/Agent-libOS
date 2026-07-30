from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import ToolSpec
from agent_libos.models.exceptions import ValidationError
from agent_libos.process_execution import bind_process_execution
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import PostgresStore, SQLiteStore
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_tool_identity_lookup_is_bounded_by_requested_ids_not_table_size(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _store_for_backend(backend) as store:
        _insert_tools(store, ("tool-target", "tool-second"))
        original_query = store._query
        captured: list[tuple[str, tuple[object, ...], int]] = []

        def capture(
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            rows = original_query(sql, params)
            if "FROM tools WHERE tool_id IN" in sql:
                captured.append((sql, tuple(params), len(rows)))
            return rows

        monkeypatch.setattr(store, "_query", capture)
        requested = ("tool-target", "tool-missing")
        assert store.get_existing_tool_ids(requested) == frozenset({"tool-target"})

        _insert_tools(store, (f"tool-unrelated-{index}" for index in range(2_000)))
        assert store.get_existing_tool_ids(requested) == frozenset({"tool-target"})

        assert len(captured) == 2
        assert captured[0][0] == captured[1][0]
        assert [len(params) for _sql, params, _row_count in captured] == [2, 2]
        assert [row_count for _sql, _params, row_count in captured] == [1, 1]
        assert all("SELECT tool_id" in sql and "SELECT *" not in sql for sql, *_ in captured)
        _assert_tool_primary_key_index(store, backend, original_query)


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_tool_identity_lookup_rejects_ambiguous_or_oversized_inputs(
    backend: str,
) -> None:
    with _store_for_backend(backend) as store:
        with pytest.raises(ValidationError, match="iterable"):
            store.get_existing_tool_ids(cast(Any, "tool-one"))
        for invalid in ("", "tool\x00bad"):
            with pytest.raises(ValidationError, match="non-empty string"):
                store.get_existing_tool_ids((invalid,))
        with pytest.raises(ValidationError, match="hard cap"):
            store.get_existing_tool_ids(
                f"tool-{index}"
                for index in range(
                    store.config.runtime.publication_artifact_lookup_hard_limit + 1
                )
            )
        with pytest.raises(ValidationError, match="hard cap"):
            store.get_existing_tool_ids(
                "tool-repeated"
                for _index in range(
                    store.config.runtime.publication_artifact_lookup_hard_limit + 1
                )
            )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_process_tool_reverse_binding_tracks_patch_restore_and_raw_snapshot_insert(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _runtime_for_backend(backend, tmp_path) as runtime:
        owner_pid = runtime.process.spawn(goal="tool binding owner")
        other_pid = runtime.process.spawn(goal="tool binding other")
        escaped_tool_id = "tool-contract-escaped"
        before_row = runtime.store.select_table_rows(
            "processes",
            "pid = ?",
            (other_pid,),
        )[0]
        before = runtime.process_exec_state.capture(other_pid)
        before_process = runtime.process.get(other_pid)
        publication_id = f"publication-exec-restore-{uuid4().hex}"
        with runtime.store.transaction():
            admission_token = runtime.uow.processes.claim_host_process_exec(
                other_pid,
                owner_id="test.artifact-lookup:process.exec",
                expected_revision=before_process.revision,
                expected_state_generation=before_process.state_generation,
                expected_execution_generation=before_process.execution_generation,
            )
            assert admission_token is not None
            runtime.store.insert_runtime_publication(
                publication_id=publication_id,
                kind="process_exec",
                pid=other_pid,
                owner_instance_id="test.artifact-lookup",
                plan={
                    "pid": other_pid,
                    "before_snapshot": before.snapshot.to_mapping(),
                    "before_tool_ids": sorted(before.tool_ids),
                    "admission_execution_generation": admission_token.generation,
                    "admission_execution_owner_id": admission_token.owner_id,
                    "admission_execution_lease_id": admission_token.lease_id,
                },
            )
        with bind_process_execution(admission_token):
            runtime.uow.processes.patch_process_tool_tables(
                other_pid,
                tool_table={"escaped": escaped_tool_id},
            )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="rollback_pending",
            phase="compensating",
            expected_states={"planning"},
        )
        original_query = runtime.store._query
        captured: list[tuple[str, tuple[object, ...], int]] = []

        def capture(
            sql: str,
            params: Any = (),
        ) -> list[Any]:
            rows = original_query(sql, params)
            if "FROM process_tool_bindings" in sql and "tool_id =" in sql:
                captured.append((sql, tuple(params), len(rows)))
            return rows

        monkeypatch.setattr(runtime.store, "_query", capture)
        assert runtime.uow.processes.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )
        for index in range(8):
            runtime.process.spawn(goal=f"unrelated binding scale {index}")
        assert runtime.uow.processes.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )

        assert len(captured) == 2
        assert captured[0][0] == captured[1][0]
        assert all("processes" not in sql.replace("process_tool_bindings", "") for sql, *_ in captured)
        assert [len(params) for _sql, params, _count in captured] == [2, 2]
        assert [count for _sql, _params, count in captured] == [1, 1]
        _assert_process_tool_reverse_index(runtime.store, backend, original_query)

        current = runtime.process.get(other_pid)
        with bind_process_execution(admission_token):
            assert runtime.store.restore_process_for_exec(
                before_row,
                expected_revision=current.revision,
                publication_id=publication_id,
            )
        assert not runtime.store.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )

        runtime.uow.processes.patch_process_tool_tables(
            other_pid,
            model_tool_table={"model-only": escaped_tool_id},
        )
        assert runtime.store.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )
        runtime.uow.processes.remove_process_tool_bindings(
            other_pid,
            {"model-only": escaped_tool_id},
        )
        assert not runtime.store.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )

        clone = dict(before_row)
        clone_pid = f"pid-clone-{uuid4().hex}"
        clone["pid"] = clone_pid
        clone["parent_pid"] = None
        clone["tool_table_json"] = dumps({"snapshot-tool": escaped_tool_id})
        clone["model_tool_table_json"] = dumps({})
        runtime.store.insert_table_row("processes", clone)
        assert runtime.store.tool_id_referenced_outside_process(
            escaped_tool_id,
            excluding_pid=owner_pid,
        )
        runtime.store.delete_table_rows("processes", "pid = ?", (clone_pid,))
        assert not runtime.store.select_table_rows(
            "process_tool_bindings",
            "pid = ?",
            (clone_pid,),
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_process_tool_reverse_binding_survives_reopen(
    backend: str,
    tmp_path: Path,
) -> None:
    escaped_tool_id = "tool-contract-reopen"
    with _runtime_target_for_backend(backend, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            owner_pid = runtime.process.spawn(goal="tool binding reopen owner")
            other_pid = runtime.process.spawn(goal="tool binding reopen other")
            runtime.uow.processes.patch_process_tool_tables(
                other_pid,
                model_tool_table={"persisted-model-tool": escaped_tool_id},
            )
        finally:
            runtime.close()

        runtime = Runtime.open(target, config=config)
        try:
            assert runtime.store.tool_id_referenced_outside_process(
                escaped_tool_id,
                excluding_pid=owner_pid,
            )
            runtime.uow.processes.remove_process_tool_bindings(
                other_pid,
                {"persisted-model-tool": escaped_tool_id},
            )
        finally:
            runtime.close()

        runtime = Runtime.open(target, config=config)
        try:
            assert not runtime.store.tool_id_referenced_outside_process(
                escaped_tool_id,
                excluding_pid=owner_pid,
            )
        finally:
            runtime.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_root_spawn_initial_goal_lookup_and_redaction_cas_contract(
    backend: str,
    tmp_path: Path,
) -> None:
    secret = f"{backend}-root-goal-recovery-secret"
    with _runtime_for_backend(backend, tmp_path) as runtime:
        pid = runtime.process.spawn(goal=secret)
        publication = runtime.store.get_committed_root_spawn_publication(pid)
        assert publication is not None
        phases = publication["receipt"]["phases"]
        goal_phases = [
            phase
            for phase in phases
            if phase.get("phase") == "goal_created"
            and "initial_goal_recovery" in phase
        ]
        assert len(goal_phases) == 1
        envelope = goal_phases[0]["initial_goal_recovery"]
        assert envelope["retention"] == "full"
        assert secret in dumps(publication["receipt"])

        generic = runtime.store.get_runtime_publication(
            publication["publication_id"]
        )
        assert generic is not None
        assert secret not in dumps(generic["receipt"])
        generic_list = runtime.store.list_runtime_publications(pid=pid)
        assert [item["publication_id"] for item in generic_list] == [
            publication["publication_id"]
        ]
        assert secret not in dumps(generic_list[0]["receipt"])
        assert (
            goal_phases[0]["goal_oid"] == envelope["goal_oid"]
        )

        assert not runtime.store.redact_root_spawn_initial_goal(
            publication["publication_id"],
            pid=pid,
            goal_oid=envelope["goal_oid"],
            expected_payload_sha256=envelope["payload_sha256"],
            expected_states=("applying",),
        )
        unchanged = runtime.store.get_committed_root_spawn_publication(pid)
        assert unchanged is not None
        unchanged_envelope = next(
            phase["initial_goal_recovery"]
            for phase in unchanged["receipt"]["phases"]
            if phase.get("phase") == "goal_created"
        )
        assert unchanged_envelope["retention"] == "full"

        assert runtime.store.redact_root_spawn_initial_goal(
            publication["publication_id"],
            pid=pid,
            goal_oid=envelope["goal_oid"],
            expected_payload_sha256=envelope["payload_sha256"],
            expected_states=("committed",),
        )
        redacted = runtime.store.get_committed_root_spawn_publication(pid)
        assert redacted is not None
        redacted_envelope = next(
            phase["initial_goal_recovery"]
            for phase in redacted["receipt"]["phases"]
            if phase.get("phase") == "goal_created"
        )
        assert redacted_envelope["retention"] == "hash_only"
        assert "payload" not in redacted_envelope
        assert secret not in dumps(redacted["receipt"])
        assert runtime.store.redact_root_spawn_initial_goal(
            publication["publication_id"],
            pid=pid,
            goal_oid=envelope["goal_oid"],
            expected_payload_sha256=envelope["payload_sha256"],
            expected_states=("committed",),
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_snapshot_tool_pruning_uses_reverse_index_and_honors_model_aliases(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_id = "tool-contract-model-alias"
    with _runtime_for_backend(backend, tmp_path) as runtime:
        scoped_pid = runtime.process.spawn(goal="tool prune scope")
        outside_pid = runtime.process.spawn(goal="tool prune outside scope")
        _insert_tools(runtime.store, (tool_id,))
        runtime.uow.processes.patch_process_tool_tables(
            outside_pid,
            model_tool_table={"outside-model-alias": tool_id},
        )

        def forbid_process_scan(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("tool pruning must not scan process rows")

        monkeypatch.setattr(runtime.store, "list_processes", forbid_process_scan)
        monkeypatch.setattr(runtime.store, "select_table_rows", forbid_process_scan)

        assert runtime.uow.snapshots.tool_id_used_outside_scope(
            tool_id,
            scoped_pids={scoped_pid},
        )
        assert not runtime.uow.snapshots.tool_id_used_outside_scope(
            tool_id,
            scoped_pids={scoped_pid, outside_pid},
        )
        assert not runtime.uow.snapshots.delete_tool_if_unreferenced(
            tool_id,
            excluding_pid=scoped_pid,
        )
        assert runtime.store.get_existing_tool_ids((tool_id,)) == frozenset({tool_id})

        runtime.uow.processes.remove_process_tool_bindings(
            outside_pid,
            {"outside-model-alias": tool_id},
        )
        assert runtime.uow.snapshots.delete_tool_if_unreferenced(
            tool_id,
            excluding_pid=scoped_pid,
        )
        assert runtime.store.get_existing_tool_ids((tool_id,)) == frozenset()


def _insert_tools(store: Any, tool_ids: Iterator[str] | tuple[str, ...]) -> None:
    selected = list(tool_ids)
    spec_json = dumps(ToolSpec(name="lookup_contract", description="contract"))
    created_at = utc_now()
    with store.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO tools "
            "(tool_id, name, spec_json, scope, registered_by, created_at, ephemeral) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    tool_id,
                    f"lookup_{index}",
                    spec_json,
                    "global",
                    "test",
                    created_at,
                    0,
                )
                for index, tool_id in enumerate(selected)
            ],
        )


def _assert_tool_primary_key_index(
    store: Any,
    backend: str,
    query: Any,
) -> None:
    if backend == "sqlite":
        plan = query(
            "EXPLAIN QUERY PLAN SELECT tool_id FROM tools WHERE tool_id IN (?, ?)",
            ("tool-target", "tool-missing"),
        )
        assert any("INDEX" in str(row["detail"]).upper() for row in plan)
        return
    indexes = query(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = ?",
        ("tools",),
    )
    assert any(
        "UNIQUE INDEX" in str(row["indexdef"]).upper()
        and "(tool_id)" in str(row["indexdef"])
        for row in indexes
    )


def _assert_process_tool_reverse_index(
    store: Any,
    backend: str,
    query: Any,
) -> None:
    if backend == "sqlite":
        plan = query(
            "EXPLAIN QUERY PLAN SELECT 1 FROM process_tool_bindings "
            "WHERE tool_id = ? AND pid <> ? LIMIT 1",
            ("tool-contract-escaped", "pid-owner"),
        )
        assert any(
            "IDX_PROCESS_TOOL_BINDINGS_TOOL_PID" in str(row["detail"]).upper()
            for row in plan
        )
        return
    indexes = query(
        "SELECT indexdef FROM pg_indexes "
        "WHERE schemaname = current_schema() AND tablename = ?",
        ("process_tool_bindings",),
    )
    assert any(
        "(tool_id, pid, binding_kind, tool_name)" in str(row["indexdef"])
        for row in indexes
    )


@contextlib.contextmanager
def _store_for_backend(backend: str) -> Iterator[Any]:
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(publication_artifact_lookup_hard_limit=4)
    )
    if backend == "sqlite":
        store = SQLiteStore(":memory:", config=config)
        try:
            yield store
        finally:
            store.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        postgres_config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend="postgres",
                store_dsn=dsn,
                publication_artifact_lookup_hard_limit=4,
            )
        )
        store = PostgresStore(dsn, config=postgres_config)
        try:
            yield store
        finally:
            store.close()


@contextlib.contextmanager
def _runtime_for_backend(
    backend: str,
    tmp_path: Path,
) -> Iterator[Runtime]:
    if backend == "sqlite":
        runtime = Runtime.open(tmp_path / "artifact-lookup.sqlite")
        try:
            yield runtime
        finally:
            runtime.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        runtime = Runtime.open(dsn, config=config)
        try:
            yield runtime
        finally:
            runtime.close()


@contextlib.contextmanager
def _runtime_target_for_backend(
    backend: str,
    tmp_path: Path,
) -> Iterator[tuple[str | Path, AgentLibOSConfig | None]]:
    if backend == "sqlite":
        yield tmp_path / "artifact-lookup-reopen.sqlite", None
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        yield dsn, config


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_artifact_lookup_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
