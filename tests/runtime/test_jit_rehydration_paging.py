from __future__ import annotations

import contextlib
import os
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    AgentProcess,
    JITRehydrationArtifact,
    ProcessCursor,
    ProcessToolBindingCursor,
    ProcessToolBindingPage,
    ProcessToolBindingRecord,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    ToolCandidate,
    ToolCandidateStatus,
    ToolHandle,
    ToolSpec,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.storage import PostgresStore, SQLiteStore, UnitOfWork
from agent_libos.tools.jit import JITToolService


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_jit_rehydration_store_queries_are_keyset_stable_exact_and_hard_bounded(
    backend: str,
) -> None:
    with _store_for_backend(backend, page_size=2, hard_limit=3) as store:
        unit = UnitOfWork(store)
        created_at = "2026-01-01T00:00:00Z"
        bindings = {
            0: {"jit_a": "tool-query-a", "jit_b": "tool-query-b"},
            1: {"jit_c": "tool-query-c"},
            2: {"static": "tool-query-static"},
        }
        for index in reversed(range(5)):
            unit.processes.insert_process(
                _process(
                    f"pid-{index}",
                    created_at=created_at,
                    tool_table=bindings.get(index),
                )
            )
        for tool_id, name, scope, ephemeral in (
            ("tool-query-a", "jit_a", "ephemeral_process", True),
            ("tool-query-b", "jit_b", "ephemeral_process", True),
            ("tool-query-c", "jit_c", "ephemeral_process", True),
            ("tool-query-static", "static", "static", False),
        ):
            unit.extensions.insert_tool(
                ToolHandle(
                    tool_id=tool_id,
                    name=name,
                    capability_id=None,
                    scope=scope,
                ),
                ToolSpec(name=name, description="test"),
                registered_by="test",
                created_at=created_at,
                ephemeral=ephemeral,
            )

        first = unit.processes.query_process_tool_bindings(after=None, limit=2)
        second = unit.processes.query_process_tool_bindings(
            after=first.next_cursor,
            limit=2,
        )
        assert [record.pid for record in first.records] == ["pid-0", "pid-0"]
        assert [(record.tool_name, record.tool_id) for record in first.records] == [
            ("jit_a", "tool-query-a"),
            ("jit_b", "tool-query-b"),
        ]
        assert [record.pid for record in second.records] == ["pid-1"]
        assert (
            second.records[0].tool_name,
            second.records[0].tool_id,
        ) == ("jit_c", "tool-query-c")
        assert first.next_cursor == ProcessToolBindingCursor("pid-0", "jit_b")
        assert second.next_cursor is None

        _insert_jit_lookup_rows(store)
        artifacts = unit.extensions.get_jit_rehydration_artifacts(
            pid="pid-0",
            tool_ids=("tool-valid", "tool-missing", "tool-static"),
        )
        by_id = {artifact.tool_id: artifact for artifact in artifacts}
        assert set(by_id) == {"tool-valid", "tool-missing"}
        assert by_id["tool-valid"].candidate_pid == "pid-0"
        assert by_id["tool-valid"].rehydratable
        assert by_id["tool-missing"].candidate_match_count == 1
        assert by_id["tool-missing"].candidate_pid == "pid-other"
        assert by_id["tool-missing"].rehydratable
        assert unit.snapshots.registered_jit_tool_ids_for_processes(
            ("pid-0",)
        ) == frozenset({"tool-valid"})
        assert unit.snapshots.registered_jit_tool_ids_for_processes(
            ("pid-other",)
        ) == frozenset({"tool-missing"})

        for invalid_limit in (0, -1, True, 4):
            with pytest.raises(ValidationError):
                unit.processes.query_process_tool_bindings(
                    after=None,
                    limit=invalid_limit,
                )
        with pytest.raises(ValidationError, match="cursor"):
            unit.processes.query_process_tool_bindings(
                after=object(),  # type: ignore[arg-type]
                limit=1,
            )
        with pytest.raises(ValidationError, match="hard cap"):
            unit.extensions.get_jit_rehydration_artifacts(
                pid="pid-0",
                tool_ids=("a", "b", "c", "d"),
            )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_jit_eligibility_projection_tracks_typed_tool_lifecycle(
    backend: str,
) -> None:
    with _store_for_backend(backend, page_size=4, hard_limit=4) as store:
        unit = UnitOfWork(store)
        pid = "pid-projection-owner"
        created_at = "2026-01-01T00:00:00Z"
        jit_handle = ToolHandle(
            tool_id="tool-projection-jit",
            name="jit_alias",
            capability_id=None,
            scope="ephemeral_process",
        )
        static_handle = ToolHandle(
            tool_id="tool-projection-static",
            name="static_alias",
            capability_id=None,
            scope="static",
        )
        unit.processes.insert_process(
            _process(
                pid,
                created_at=created_at,
                tool_table={
                    jit_handle.name: jit_handle.tool_id,
                    static_handle.name: static_handle.tool_id,
                },
            )
        )
        assert not unit.processes.query_process_tool_bindings(
            after=None,
            limit=4,
        ).records

        unit.extensions.insert_tool(
            static_handle,
            ToolSpec(name=static_handle.name, description="static package tool"),
            registered_by="package:test",
            created_at=created_at,
            ephemeral=False,
        )
        unit.extensions.insert_tool(
            jit_handle,
            ToolSpec(name=jit_handle.name, description="JIT tool"),
            registered_by="test",
            created_at=created_at,
            ephemeral=True,
        )
        page = unit.processes.query_process_tool_bindings(after=None, limit=4)
        assert [(item.tool_name, item.tool_id) for item in page.records] == [
            (jit_handle.name, jit_handle.tool_id)
        ]
        projection = {
            (str(row["binding_kind"]), str(row["tool_name"])): int(
                row["jit_rehydration_eligible"]
            )
            for row in store.select_table_rows(
                "process_tool_bindings",
                "pid = ?",
                (pid,),
            )
        }
        assert projection == {
            ("callable", "jit_alias"): 1,
            ("callable", "static_alias"): 0,
            ("model", "jit_alias"): 0,
            ("model", "static_alias"): 0,
        }

        unit.extensions.update_tool(
            jit_handle,
            ToolSpec(name=jit_handle.name, description="temporarily static"),
            registered_by="test",
            ephemeral=False,
        )
        assert not unit.processes.query_process_tool_bindings(
            after=None,
            limit=4,
        ).records
        unit.extensions.update_tool(
            jit_handle,
            ToolSpec(name=jit_handle.name, description="JIT tool"),
            registered_by="test",
            ephemeral=True,
        )
        unit.extensions.delete_tool(
            jit_handle.tool_id,
            registered_by="wrong-owner",
        )
        assert len(
            unit.processes.query_process_tool_bindings(after=None, limit=4).records
        ) == 1
        unit.extensions.delete_tool(
            jit_handle.tool_id,
            registered_by="test",
        )
        assert not unit.processes.query_process_tool_bindings(
            after=None,
            limit=4,
        ).records

        unit.extensions.insert_tool(
            jit_handle,
            ToolSpec(name=jit_handle.name, description="JIT tool"),
            registered_by="test",
            created_at=created_at,
            ephemeral=True,
        )
        store.delete_table_rows(
            "tools",
            "tool_id = ? AND ephemeral = 1",
            (jit_handle.tool_id,),
        )
        assert not unit.processes.query_process_tool_bindings(
            after=None,
            limit=4,
        ).records
        unit.extensions.insert_tool(
            jit_handle,
            ToolSpec(name=jit_handle.name, description="JIT tool"),
            registered_by="test",
            created_at=created_at,
            ephemeral=True,
        )
        assert unit.snapshots.delete_tool_if_unreferenced(
            jit_handle.tool_id,
            excluding_pid=pid,
        )
        assert not unit.processes.query_process_tool_bindings(
            after=None,
            limit=4,
        ).records
        for forged_row in (
            {
                "pid": pid,
                "binding_kind": "callable",
                "tool_name": "missing-derived-bit",
                "tool_id": jit_handle.tool_id,
            },
            {
                "pid": pid,
                "binding_kind": "callable",
                "tool_name": "forged-static-bit",
                "tool_id": static_handle.tool_id,
                "jit_rehydration_eligible": 1,
            },
        ):
            with pytest.raises(ValidationError, match="derived typed projection"):
                store.insert_table_row("process_tool_bindings", forged_row)
        with pytest.raises(ValidationError, match="derived typed projection"):
            store.delete_table_rows(
                "process_tool_bindings",
                "pid = ?",
                (pid,),
            )
        assert len(
            store.select_table_rows(
                "process_tool_bindings",
                "pid = ?",
                (pid,),
            )
        ) == 4


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_jit_eligibility_refresh_failure_rolls_back_tool_and_projection(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _store_for_backend(backend, page_size=2, hard_limit=2) as store:
        unit = UnitOfWork(store)
        pid = "pid-projection-rollback"
        created_at = "2026-01-01T00:00:00Z"
        handle = ToolHandle(
            tool_id="tool-projection-rollback",
            name="rollback_alias",
            capability_id=None,
            scope="ephemeral_process",
        )
        unit.processes.insert_process(
            _process(
                pid,
                created_at=created_at,
                tool_table={handle.name: handle.tool_id},
            )
        )
        unit.extensions.insert_tool(
            handle,
            ToolSpec(name=handle.name, description="initially static"),
            registered_by="test",
            created_at=created_at,
            ephemeral=False,
        )
        original_refresh = store._refresh_process_binding_jit_eligibility

        def fail_after_refresh(
            cursor: Any,
            *,
            pid: str | None = None,
            tool_id: str | None = None,
        ) -> None:
            original_refresh(cursor, pid=pid, tool_id=tool_id)
            raise RuntimeError("injected post-refresh failure")

        monkeypatch.setattr(
            store,
            "_refresh_process_binding_jit_eligibility",
            fail_after_refresh,
        )
        with pytest.raises(RuntimeError, match="post-refresh failure"):
            unit.extensions.update_tool(
                handle,
                ToolSpec(name=handle.name, description="attempted JIT update"),
                registered_by="test",
                ephemeral=True,
            )

        tool_rows = store.select_table_rows(
            "tools",
            "tool_id = ?",
            (handle.tool_id,),
        )
        binding_rows = store.select_table_rows(
            "process_tool_bindings",
            "pid = ? AND tool_name = ?",
            (pid, handle.name),
            order_by="binding_kind",
        )
        assert len(tool_rows) == 1
        assert int(tool_rows[0]["ephemeral"]) == 0
        assert [int(row["jit_rehydration_eligible"]) for row in binding_rows] == [
            0,
            0,
        ]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_delete_jit_tool_rows_batches_exact_owner_and_repairs_projection(
    backend: str,
) -> None:
    target_ids = tuple(f"tool-delete-{index:04d}" for index in range(501))
    static_id = target_ids[-1]
    keep_id = "tool-delete-keep"
    pid = "pid-delete-owner"
    other_pid = "pid-delete-other"
    created_at = "2026-01-01T00:00:00Z"

    with _store_for_backend(backend, page_size=4, hard_limit=4) as store:
        unit = UnitOfWork(store)
        unit.processes.insert_process(_process(pid, created_at=created_at))
        _insert_jit_deletion_rows(
            store,
            pid=pid,
            tool_ids=target_ids,
            static_id=static_id,
            keep_id=keep_id,
            created_at=created_at,
        )
        recording_connection = _RecordingConnection(store.conn)
        store.conn = recording_connection

        unit.extensions.delete_jit_tool_rows(
            pid,
            (*target_ids, target_ids[0]),
        )

        statements = [
            (" ".join(sql.split()).upper(), params)
            for sql, params in recording_connection.statements
        ]
        projection_updates = [
            params
            for sql, params in statements
            if sql.startswith("UPDATE PROCESS_TOOL_BINDINGS ")
        ]
        tool_deletes = [
            params
            for sql, params in statements
            if sql.startswith("DELETE FROM TOOLS ")
        ]
        candidate_deletes = [
            params
            for sql, params in statements
            if sql.startswith("DELETE FROM TOOL_CANDIDATES ")
        ]
        assert [len(params) for params in projection_updates] == [500, 1]
        assert [len(params) for params in tool_deletes] == [500, 1]
        assert [len(params) for params in candidate_deletes] == [501, 2]
        assert len(statements) == 6

        remaining_tools = {
            str(row["tool_id"]): int(row["ephemeral"])
            for row in store.select_table_rows("tools")
        }
        assert remaining_tools == {static_id: 0, keep_id: 1}

        binding_rows = store.select_table_rows(
            "process_tool_bindings",
            "pid = ?",
            (pid,),
        )
        eligibility = {
            str(row["tool_id"]): int(row["jit_rehydration_eligible"])
            for row in binding_rows
        }
        assert len(eligibility) == len(target_ids) + 1
        assert all(eligibility[tool_id] == 0 for tool_id in target_ids)
        assert eligibility[keep_id] == 1

        candidate_rows = store.select_table_rows("tool_candidates")
        remaining_candidates = {
            (str(row["pid"]), str(row["registered_tool_id"]))
            for row in candidate_rows
        }
        assert remaining_candidates == {
            (pid, keep_id),
            (other_pid, target_ids[0]),
            (other_pid, static_id),
        }


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_delete_jit_tool_rows_commit_failure_rolls_back_every_projection(
    backend: str,
) -> None:
    target_ids = ("tool-delete-rollback-jit", "tool-delete-rollback-static")
    static_id = target_ids[-1]
    keep_id = "tool-delete-rollback-keep"
    pid = "pid-delete-rollback"
    created_at = "2026-01-01T00:00:00Z"

    with _store_for_backend(backend, page_size=2, hard_limit=2) as store:
        unit = UnitOfWork(store)
        unit.processes.insert_process(_process(pid, created_at=created_at))
        _insert_jit_deletion_rows(
            store,
            pid=pid,
            tool_ids=target_ids,
            static_id=static_id,
            keep_id=keep_id,
            created_at=created_at,
        )
        store.conn = _FailNextCommitConnection(store.conn)

        with pytest.raises(RuntimeError, match="injected commit failure"):
            unit.extensions.delete_jit_tool_rows(pid, target_ids)

        tool_rows = store.select_table_rows("tools")
        assert {
            str(row["tool_id"]): int(row["ephemeral"])
            for row in tool_rows
        } == {
            target_ids[0]: 1,
            static_id: 0,
            keep_id: 1,
        }
        binding_rows = store.select_table_rows(
            "process_tool_bindings",
            "pid = ?",
            (pid,),
        )
        assert {
            str(row["tool_id"]): int(row["jit_rehydration_eligible"])
            for row in binding_rows
        } == {
            target_ids[0]: 1,
            static_id: 1,
            keep_id: 1,
        }
        candidate_rows = store.select_table_rows(
            "tool_candidates",
            "pid = ?",
            (pid,),
        )
        assert {str(row["registered_tool_id"]) for row in candidate_rows} == {
            *target_ids,
            keep_id,
        }


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_jit_binding_cursor_has_backend_neutral_unicode_order(backend: str) -> None:
    with _store_for_backend(backend, page_size=2, hard_limit=2) as store:
        unit = UnitOfWork(store)
        pid = "pid-unicode-bindings"
        created_at = "2026-01-01T00:00:00Z"
        names = ("zeta", "éclair", "工具", "alpha")
        tool_table = {
            name: f"tool-unicode-{index}" for index, name in enumerate(names)
        }
        unit.processes.insert_process(
            _process(pid, created_at=created_at, tool_table=tool_table)
        )
        for name, tool_id in tool_table.items():
            unit.extensions.insert_tool(
                ToolHandle(
                    tool_id=tool_id,
                    name=name,
                    capability_id=None,
                    scope="ephemeral_process",
                ),
                ToolSpec(name=name, description="unicode cursor fixture"),
                registered_by="test",
                created_at=created_at,
                ephemeral=True,
            )

        selected_names: list[str] = []
        cursor: ProcessToolBindingCursor | None = None
        while True:
            page = unit.processes.query_process_tool_bindings(
                after=cursor,
                limit=2,
            )
            selected_names.extend(record.tool_name for record in page.records)
            cursor = page.next_cursor
            if cursor is None:
                break
        assert selected_names == sorted(names)


def test_runtime_rehydrate_validates_owner_before_loaded_fast_path_and_bounds_audit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "jit-rehydration.sqlite"
    config = _config(page_size=2, hard_limit=4)
    runtime = Runtime.open(target, config=config)
    try:
        created_at = "2026-01-01T00:00:00Z"
        source = "export function run() { return {}; }"
        handle = ToolHandle(
            tool_id="tool-shared-jit",
            name="shared_jit",
            capability_id=None,
            scope="ephemeral_process",
        )
        runtime.store.insert_tool(
            handle,
            ToolSpec(name="shared_jit", description="test"),
            registered_by="test",
            created_at=created_at,
            ephemeral=True,
        )
        runtime.store.insert_tool_candidate(
            ToolCandidate(
                candidate_id="candidate-shared-jit",
                pid="pid-00-owner",
                spec=ToolSpec(name="shared_jit", description="test"),
                source_code=source,
                tests=[],
                requested_capabilities=[],
                status=ToolCandidateStatus.REGISTERED,
                validation={"ok": True},
                created_at=created_at,
                updated_at=created_at,
                registered_tool_id=handle.tool_id,
            )
        )
        runtime.store.insert_process(
            _process(
                "pid-00-owner",
                created_at=created_at,
                tool_table={"shared_jit": handle.tool_id},
            )
        )
        runtime.store.insert_process(
            _process(
                "pid-01-other",
                created_at=created_at,
                tool_table={"shared_jit": handle.tool_id},
            )
        )
        for index in range(5):
            stale_handle = ToolHandle(
                tool_id=f"tool-stale-{index}",
                name=f"stale_{index}",
                capability_id=None,
                scope="ephemeral_process",
            )
            runtime.store.insert_tool(
                stale_handle,
                ToolSpec(name=stale_handle.name, description="test"),
                registered_by="test",
                created_at=created_at,
                ephemeral=True,
            )
            runtime.store.insert_process(
                _process(
                    f"pid-{index + 2:02d}-stale",
                    created_at=created_at,
                    tool_table={stale_handle.name: stale_handle.tool_id},
                )
            )
    finally:
        runtime.close()

    reopened = Runtime.open(target, config=config)
    try:
        assert reopened.tools.registry.is_jit(handle.tool_id)
        assert reopened.process.get("pid-00-owner").tool_table == {
            "shared_jit": handle.tool_id
        }
        assert reopened.process.get("pid-01-other").tool_table == {}
        for index in range(5):
            assert reopened.process.get(f"pid-{index + 2:02d}-stale").tool_table == {}
        records = [
            record
            for record in reopened.audit.trace()
            if record.action == "runtime.jit.rehydrate"
        ]
        decision = records[-1].decision
        assert decision["restored_total"] == 1
        assert decision["pruned_stale_total"] == 6
        assert len(decision["restored"]) == 1
        assert len(decision["pruned_stale"]) == 2
        assert decision["pruned_stale_truncated"] is True
    finally:
        reopened.close()


def test_open_runtime_rejects_jit_rehydrate_before_durable_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    read_attempted = False

    def unexpected_read(*_args: object, **_kwargs: object) -> ProcessToolBindingPage:
        nonlocal read_attempted
        read_attempted = True
        raise AssertionError("JIT recovery read storage before validating its lease")

    try:
        monkeypatch.setattr(
            runtime.tools.jit.processes,
            "query_process_tool_bindings",
            unexpected_read,
        )
        with pytest.raises(RuntimeError, match="active startup recovery lease"):
            runtime.tools.jit.rehydrate_registered()
        assert not read_attempted
    finally:
        runtime.close()


def test_jit_rehydrate_transient_memory_and_samples_do_not_scale_with_history() -> None:
    total = 100_000
    page_size = 64
    service = object.__new__(JITToolService)
    service.processes = _SyntheticProcesses(total)
    service.extensions = _SyntheticArtifacts()
    registry = _DiscardingRegistry()
    service.registry = registry
    service.audit = _CapturingAudit()
    service.config = _config(page_size=page_size, hard_limit=page_size)
    service._require_recovery_lease = lambda: None

    tracemalloc.start()
    summary = service.rehydrate_registered()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert summary.restored_total == total
    assert registry.published_total == total
    assert len(summary.restored_sample) == page_size
    assert summary.restored_truncated
    assert service.processes.query_count == (total + page_size - 1) // page_size
    assert service.extensions.query_count == service.processes.query_count
    assert peak < 32 * 1024 * 1024
    assert len(service.audit.decision["restored"]) == page_size


def test_sqlite_jit_rehydration_queries_use_indexes() -> None:
    store = SQLiteStore(":memory:", config=_config(page_size=2, hard_limit=3))
    try:
        process_plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM processes "
            "WHERE (created_at, pid) > (?, ?) "
            "ORDER BY created_at, pid LIMIT ?",
            ("2026-01-01", "pid", 2),
        ).fetchall()
        candidate_plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT registered_tool_id, candidate_id, pid, source_code "
            "FROM tool_candidates WHERE registered_tool_id IN (?, ?) "
            "AND status = ? ORDER BY registered_tool_id, pid, candidate_id LIMIT ?",
            ("tool-1", "tool-2", "registered", 3),
        ).fetchall()
        owner_plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT registered_tool_id FROM tool_candidates "
            "WHERE pid IN (?, ?) AND registered_tool_id IS NOT NULL "
            "ORDER BY pid, registered_tool_id, candidate_id, status",
            ("pid-1", "pid-2"),
        ).fetchall()
        binding_plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT binding.pid, binding.tool_name, "
            "binding.tool_id "
            "FROM process_tool_bindings AS binding "
            "WHERE binding.jit_rehydration_eligible = 1 "
            "AND (binding.pid, binding.tool_name) > (?, ?) "
            "ORDER BY binding.pid, binding.tool_name LIMIT ?",
            ("pid-1", "name", 2),
        ).fetchall()
        assert "idx_processes_created" in "\n".join(
            str(row["detail"]) for row in process_plan
        )
        assert "idx_tool_candidates_jit_rehydration" in "\n".join(
            str(row["detail"]) for row in candidate_plan
        )
        assert "idx_tool_candidates_owner_registration" in "\n".join(
            str(row["detail"]) for row in owner_plan
        )
        binding_details = "\n".join(
            str(row["detail"]) for row in binding_plan
        )
        assert (
            "idx_process_tool_bindings_jit_eligible_recovery" in binding_details
        )
        assert "USE TEMP B-TREE" not in binding_details
    finally:
        store.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_process_and_jit_deep_cursors_are_index_lower_bounds(backend: str) -> None:
    with _store_for_backend(backend, page_size=2, hard_limit=3) as store:
        process_sql = (
            "SELECT * FROM processes WHERE (created_at, pid) > (?, ?) "
            "ORDER BY created_at, pid LIMIT ?"
        )
        binding_sql = (
            "SELECT binding.pid, binding.tool_name, "
            "binding.tool_id FROM process_tool_bindings AS binding "
            "WHERE binding.jit_rehydration_eligible = 1 "
            "AND (binding.pid, binding.tool_name) > (?, ?) "
            "ORDER BY binding.pid, binding.tool_name LIMIT ?"
        )
        if backend == "sqlite":
            process_rows = store.conn.execute(
                f"EXPLAIN QUERY PLAN {process_sql}",
                ("2026-01-01", "pid-deep", 2),
            )
            binding_rows = store.conn.execute(
                f"EXPLAIN QUERY PLAN {binding_sql}",
                ("pid-deep", "tool-deep", 2),
            )
            process_details = "\n".join(str(row["detail"]) for row in process_rows)
            binding_details = "\n".join(str(row["detail"]) for row in binding_rows)
            assert "(created_at,pid)>(?,?)" in process_details
            assert "(pid,tool_name)>(?,?)" in binding_details
        else:
            store.conn.execute("SET enable_seqscan = off")  # type: ignore[attr-defined]
            store.conn.execute(  # type: ignore[attr-defined]
                "SET plan_cache_mode = force_generic_plan"
            )
            process_rows = store.conn.execute(  # type: ignore[attr-defined]
                f"EXPLAIN {process_sql}",
                ("2026-01-01", "pid-deep", 2),
            )
            prepared_binding_sql = binding_sql.replace(
                "(?, ?)",
                "($1, $2)",
            ).replace("LIMIT ?", "LIMIT $3")
            store.conn.execute(  # type: ignore[attr-defined]
                "PREPARE agent_libos_jit_binding_page(text, text, bigint) AS "
                f"{prepared_binding_sql}"
            )
            try:
                binding_rows = store.conn.execute(  # type: ignore[attr-defined]
                    "EXPLAIN EXECUTE agent_libos_jit_binding_page("
                    "'pid-deep', 'tool-deep', 2)"
                )
            finally:
                store.conn.execute(  # type: ignore[attr-defined]
                    "DEALLOCATE agent_libos_jit_binding_page"
                )
            process_details = "\n".join(
                str(row["QUERY PLAN"]) for row in process_rows
            )
            binding_details = "\n".join(
                str(row["QUERY PLAN"]) for row in binding_rows
            )
            assert "Index Cond" in process_details
            assert "Index Cond" in binding_details
            assert "Filter" not in binding_details
            collation_rows = store.conn.execute(  # type: ignore[attr-defined]
                "SELECT column_name, collation_name "
                "FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = ? AND column_name IN (?, ?)",
                ("process_tool_bindings", "pid", "tool_name"),
            )
            assert {
                str(row["column_name"]): str(row["collation_name"])
                for row in collation_rows
            } == {"pid": "C", "tool_name": "C"}
        assert "idx_processes_created" in process_details
        assert (
            "idx_process_tool_bindings_jit_eligible_recovery" in binding_details
        )
        assert "Sort" not in binding_details


def test_sparse_jit_eligibility_page_has_bounded_sqlite_vm_work() -> None:
    """A page must scan the eligible projection, not all callable bindings."""

    total = 100_000
    eligible_indexes = frozenset(range(0, total, 10_000))
    store = SQLiteStore(":memory:", config=_config(page_size=4, hard_limit=64))
    try:
        unit = UnitOfWork(store)
        pid = "pid-sparse-jit"
        created_at = "2026-01-01T00:00:00Z"
        unit.processes.insert_process(_process(pid, created_at=created_at))
        with store.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO tools (tool_id, name, spec_json, scope, "
                "registered_by, created_at, ephemeral) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"tool-sparse-{index:06d}",
                        f"binding-{index:06d}",
                        "{}",
                        (
                            "ephemeral_process"
                            if index in eligible_indexes
                            else "static"
                        ),
                        "test",
                        created_at,
                        int(index in eligible_indexes),
                    )
                    for index in range(total)
                ),
            )
            cursor.executemany(
                "INSERT INTO process_tool_bindings "
                "(pid, binding_kind, tool_name, tool_id, "
                "jit_rehydration_eligible) VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        pid,
                        "callable",
                        f"binding-{index:06d}",
                        f"tool-sparse-{index:06d}",
                        int(index in eligible_indexes),
                    )
                    for index in range(total)
                ),
            )

        progress_callbacks = 0

        def count_vm_progress() -> int:
            nonlocal progress_callbacks
            progress_callbacks += 1
            return 0

        store.conn.set_progress_handler(count_vm_progress, 100)
        try:
            first = unit.processes.query_process_tool_bindings(
                after=None,
                limit=4,
            )
            second = unit.processes.query_process_tool_bindings(
                after=first.next_cursor,
                limit=4,
            )
        finally:
            store.conn.set_progress_handler(None, 0)

        expected_names = [
            f"binding-{index:06d}" for index in sorted(eligible_indexes)
        ]
        assert [record.tool_name for record in first.records] == expected_names[:4]
        assert [record.tool_name for record in second.records] == expected_names[4:8]
        assert first.next_cursor == ProcessToolBindingCursor(
            pid,
            expected_names[3],
        )
        assert progress_callbacks < 20
    finally:
        store.close()


def test_jit_rehydrate_100k_binding_fanout_uses_page_level_sql() -> None:
    """A single high-fanout process must not materialize or query per binding."""

    total = 100_000
    page_size = 64
    store = SQLiteStore(
        ":memory:",
        config=_config(page_size=page_size, hard_limit=page_size),
    )
    try:
        unit = UnitOfWork(store)
        pid = "pid-high-fanout"
        created_at = "2026-01-01T00:00:00Z"
        unit.processes.insert_process(_process(pid, created_at=created_at))
        with store.transaction() as cursor:
            cursor.executemany(
                "INSERT INTO tools (tool_id, name, spec_json, scope, "
                "registered_by, created_at, ephemeral) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"jit-tool-{index:06d}",
                        f"binding-{index:06d}",
                        "{}",
                        "ephemeral_process",
                        "test",
                        created_at,
                        1,
                    )
                    for index in range(total)
                ),
            )
            cursor.executemany(
                "INSERT INTO process_tool_bindings "
                "(pid, binding_kind, tool_name, tool_id, "
                "jit_rehydration_eligible) VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        pid,
                        "callable",
                        f"binding-{index:06d}",
                        f"jit-tool-{index:06d}",
                        1,
                    )
                    for index in range(total)
                ),
            )
            cursor.executemany(
                "INSERT INTO tool_candidates (candidate_id, pid, spec_json, "
                "source_code, tests_json, requested_capabilities_json, status, "
                "registered_tool_id, validation_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    (
                        f"candidate-{index:06d}",
                        pid,
                        "{}",
                        "export function run() { return {}; }",
                        "[]",
                        "[]",
                        "registered",
                        f"jit-tool-{index:06d}",
                        "{}",
                        created_at,
                        created_at,
                    )
                    for index in range(total)
                ),
            )

        eligible_projection = store.conn.execute(
            "SELECT COUNT(*) AS eligible_total "
            "FROM process_tool_bindings AS binding "
            "JOIN tools AS tool ON tool.tool_id = binding.tool_id "
            "WHERE binding.jit_rehydration_eligible = 1 "
            "AND binding.binding_kind = 'callable' AND tool.ephemeral = 1"
        ).fetchone()
        assert eligible_projection is not None
        assert int(eligible_projection["eligible_total"]) == total

        service = object.__new__(JITToolService)
        service.processes = unit.processes
        service.extensions = unit.extensions
        registry = _DiscardingRegistry()
        service.registry = registry
        service.audit = _CapturingAudit()
        service.config = store.config
        service._require_recovery_lease = lambda: None

        query_counts = {"bindings": 0, "tools": 0, "candidates": 0}

        def count_rehydration_query(statement: str) -> None:
            if "FROM process_tool_bindings AS binding" in statement:
                query_counts["bindings"] += 1
            elif "FROM tools WHERE ephemeral = 1" in statement:
                query_counts["tools"] += 1
            elif (
                "FROM tool_candidates" in statement
                and "registered_tool_id IN" in statement
            ):
                query_counts["candidates"] += 1

        store.conn.set_trace_callback(count_rehydration_query)
        try:
            summary = service.rehydrate_registered()
        finally:
            store.conn.set_trace_callback(None)

        assert summary.restored_total == total
        assert registry.published_total == total
        assert summary.pruned_stale_total == 0
        expected_pages = (total + page_size - 1) // page_size
        assert query_counts == {
            "bindings": expected_pages,
            "tools": expected_pages,
            "candidates": expected_pages,
        }
    finally:
        store.close()


class _SyntheticProcesses:
    def __init__(self, total: int) -> None:
        self.total = total
        self.query_count = 0

    def query_process_tool_bindings(
        self,
        *,
        after: ProcessToolBindingCursor | None,
        limit: int,
    ) -> ProcessToolBindingPage:
        self.query_count += 1
        start = int(after.pid.rsplit("-", 1)[1]) + 1 if after is not None else 0
        stop = min(self.total, start + limit)
        records = tuple(
            ProcessToolBindingRecord(
                pid=f"pid-{index:06d}",
                tool_name=f"jit_{index}",
                tool_id=f"tool-{index}",
            )
            for index in range(start, stop)
        )
        next_cursor = None
        if stop < self.total:
            last = records[-1]
            next_cursor = ProcessToolBindingCursor(
                last.pid,
                last.tool_name,
            )
        return ProcessToolBindingPage(records=records, next_cursor=next_cursor)

    def remove_process_tool_bindings(
        self,
        _pid: str,
        _bindings: dict[str, str],
    ) -> None:
        raise AssertionError("valid synthetic JIT binding was pruned")


class _SyntheticArtifacts:
    def __init__(self) -> None:
        self.query_count = 0

    def get_jit_rehydration_artifacts_for_tool_ids(
        self,
        tool_ids: frozenset[str],
    ) -> tuple[JITRehydrationArtifact, ...]:
        self.query_count += 1
        return tuple(
            JITRehydrationArtifact(
                tool_id=tool_id,
                name=f"jit_{index}",
                scope="ephemeral_process",
                candidate_match_count=1,
                candidate_id=f"candidate-{index}",
                candidate_pid=f"pid-{index:06d}",
                source_code="export function run() { return {}; }",
            )
            for tool_id in sorted(tool_ids)
            for index in (int(tool_id.rsplit("-", 1)[1]),)
        )


class _DiscardingRegistry:
    def __init__(self) -> None:
        self.published_total = 0

    def implementation(self, _tool_id: str) -> None:
        return None

    def is_jit(self, _tool_id: str) -> bool:
        return False

    def publish_jit(self, _handle: ToolHandle, _source: str) -> None:
        self.published_total += 1


class _CapturingAudit:
    decision: dict[str, Any]

    def record(self, **kwargs: Any) -> None:
        self.decision = kwargs["decision"]


def _process(
    pid: str,
    *,
    created_at: str,
    tool_table: dict[str, str] | None = None,
) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        parent_pid=None,
        image_id="base-agent:v0",
        status=ProcessStatus.RUNNABLE,
        goal_oid=None,
        memory_view=None,
        capabilities=[],
        loaded_skills={},
        tool_table=dict(tool_table or {}),
        event_cursor=None,
        checkpoint_head=None,
        resource_budget=ResourceBudget(),
        resource_usage=ResourceUsage(),
        created_at=created_at,
        updated_at=created_at,
        model_tool_table=dict(tool_table or {}),
    )


def _insert_jit_lookup_rows(store: SQLiteStore | PostgresStore) -> None:
    created_at = "2026-01-01T00:00:00Z"
    with store.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO tools (tool_id, name, spec_json, scope, registered_by, "
            "created_at, ephemeral) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ("tool-valid", "valid", "{}", "ephemeral_process", "test", created_at, 1),
                ("tool-missing", "missing", "{}", "ephemeral_process", "test", created_at, 1),
                ("tool-static", "static", "{}", "static", "test", created_at, 0),
            ),
        )
        cursor.executemany(
            "INSERT INTO tool_candidates (candidate_id, pid, spec_json, source_code, "
            "tests_json, requested_capabilities_json, status, registered_tool_id, "
            "validation_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    "candidate-valid",
                    "pid-0",
                    "{}",
                    "export function run() { return {}; }",
                    "[]",
                    "[]",
                    "registered",
                    "tool-valid",
                    "{}",
                    created_at,
                    created_at,
                ),
                (
                    "candidate-wrong-owner",
                    "pid-other",
                    "{}",
                    "export function run() { return {}; }",
                    "[]",
                    "[]",
                    "registered",
                    "tool-missing",
                    "{}",
                    created_at,
                    created_at,
                ),
            ),
        )


def _insert_jit_deletion_rows(
    store: SQLiteStore | PostgresStore,
    *,
    pid: str,
    tool_ids: tuple[str, ...],
    static_id: str,
    keep_id: str,
    created_at: str,
) -> None:
    other_pid = "pid-delete-other"
    all_tool_ids = (*tool_ids, keep_id)
    with store.transaction() as cursor:
        cursor.executemany(
            "INSERT INTO tools (tool_id, name, spec_json, scope, registered_by, "
            "created_at, ephemeral) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    tool_id,
                    f"name-{tool_id}",
                    "{}",
                    "static" if tool_id == static_id else "ephemeral_process",
                    "test",
                    created_at,
                    0 if tool_id == static_id else 1,
                )
                for tool_id in all_tool_ids
            ),
        )
        cursor.executemany(
            "INSERT INTO process_tool_bindings "
            "(pid, binding_kind, tool_name, tool_id, jit_rehydration_eligible) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (pid, "callable", f"binding-{tool_id}", tool_id, 1)
                for tool_id in all_tool_ids
            ),
        )
        cursor.executemany(
            "INSERT INTO tool_candidates (candidate_id, pid, spec_json, "
            "source_code, tests_json, requested_capabilities_json, status, "
            "registered_tool_id, validation_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    f"candidate-owner-{index:04d}",
                    pid,
                    "{}",
                    "export function run() { return {}; }",
                    "[]",
                    "[]",
                    "registered",
                    tool_id,
                    "{}",
                    created_at,
                    created_at,
                )
                for index, tool_id in enumerate(all_tool_ids)
            ),
        )
        cursor.executemany(
            "INSERT INTO tool_candidates (candidate_id, pid, spec_json, "
            "source_code, tests_json, requested_capabilities_json, status, "
            "registered_tool_id, validation_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    f"candidate-other-{index}",
                    other_pid,
                    "{}",
                    "export function run() { return {}; }",
                    "[]",
                    "[]",
                    "registered",
                    tool_id,
                    "{}",
                    created_at,
                    created_at,
                )
                for index, tool_id in enumerate((tool_ids[0], static_id))
            ),
        )


class _ConnectionProxy:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class _RecordingConnection(_ConnectionProxy):
    def __init__(self, connection: Any) -> None:
        super().__init__(connection)
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        cursor = self._connection.cursor(*args, **kwargs)
        statements = self.statements

        class _RecordingCursor:
            def execute(self, sql: str, parameters: Any = ()) -> Any:
                statements.append((sql, tuple(parameters)))
                return cursor.execute(sql, parameters)

            def __getattr__(self, name: str) -> Any:
                return getattr(cursor, name)

        return _RecordingCursor()


class _FailNextCommitConnection(_ConnectionProxy):
    def __init__(self, connection: Any) -> None:
        super().__init__(connection)
        self._should_fail = True

    def commit(self) -> None:
        if self._should_fail:
            self._should_fail = False
            raise RuntimeError("injected commit failure")
        self._connection.commit()


@contextlib.contextmanager
def _store_for_backend(
    backend: str,
    *,
    page_size: int,
    hard_limit: int,
) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        store: SQLiteStore | PostgresStore = SQLiteStore(
            ":memory:",
            config=_config(page_size=page_size, hard_limit=hard_limit),
        )
        try:
            yield store
        finally:
            store.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(
            dsn,
            config=_config(page_size=page_size, hard_limit=hard_limit, dsn=dsn),
        )
        try:
            yield store
        finally:
            store.close()


def _config(
    *,
    page_size: int,
    hard_limit: int,
    dsn: str | None = None,
) -> AgentLibOSConfig:
    return AgentLibOSConfig(
        runtime=RuntimeDefaults(
            store_backend="postgres" if dsn is not None else "sqlite",
            store_dsn=dsn,
            jit_rehydration_page_size=page_size,
            jit_rehydration_page_hard_limit=hard_limit,
        )
    )


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_jit_rehydration_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
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
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
