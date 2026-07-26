from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pytest

from agent_libos import Runtime


_FULL_TOOLS = ["echo", "get_current_time"]


def _prepare_process(runtime: Runtime) -> str:
    pid = runtime.process.spawn(image="base-agent:v0", goal="tool table atomicity")
    runtime.tools.configure_process_tools(
        pid,
        _FULL_TOOLS,
        assigned_by="test.setup.full",
    )
    runtime.tools.configure_model_tool_projection(
        pid,
        ["echo"],
        assigned_by="test.setup.projection",
    )
    return pid


def _process_snapshot(runtime: Runtime, pid: str) -> tuple[dict[str, str], dict[str, str], int]:
    process = runtime.process.get(pid)
    return (
        dict(process.tool_table),
        dict(process.model_tool_table),
        process.revision,
    )


def _failed_mutation(
    runtime: Runtime,
    pid: str,
    *,
    projection: bool,
    assigned_by: str,
) -> dict[str, str]:
    if projection:
        return runtime.tools.configure_model_tool_projection(
            pid,
            ["get_current_time"],
            assigned_by=assigned_by,
        )
    return runtime.tools.configure_process_tools(
        pid,
        ["get_working_directory"],
        assigned_by=assigned_by,
    )


def _assert_reopened_tables(
    database: Path,
    pid: str,
    expected_tool_table: dict[str, str],
    expected_model_tool_table: dict[str, str],
) -> None:
    reopened = Runtime.open(database)
    try:
        process = reopened.process.get(pid)
        assert process.tool_table == expected_tool_table
        assert process.model_tool_table == expected_model_tool_table
    finally:
        reopened.close()


def _exercise_audit_rollback_and_reopen(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    projection: bool,
) -> None:
    runtime = Runtime.open(database)
    pid = ""
    before: tuple[dict[str, str], dict[str, str], int] | None = None
    action = "process.tools.project" if projection else "process.tools.configure"
    actor = f"test.failure.audit.{action}"
    try:
        pid = _prepare_process(runtime)
        before = _process_snapshot(runtime, pid)
        audit_ids_before = {record.record_id for record in runtime.audit.trace()}
        original_record = runtime.audit.record

        def fail_after_audit_record(**kwargs: Any) -> Any:
            record = original_record(**kwargs)
            if kwargs.get("action") == action:
                raise RuntimeError("injected audit finalization failure")
            return record

        with monkeypatch.context() as scoped:
            scoped.setattr(runtime.audit, "record", fail_after_audit_record)
            with pytest.raises(RuntimeError, match="audit finalization"):
                _failed_mutation(
                    runtime,
                    pid,
                    projection=projection,
                    assigned_by=actor,
                )

        assert _process_snapshot(runtime, pid) == before
        assert {record.record_id for record in runtime.audit.trace()} == audit_ids_before
        assert not any(record.actor == actor for record in runtime.audit.trace())
    finally:
        runtime.close()

    assert before is not None
    _assert_reopened_tables(database, pid, before[0], before[1])


def _exercise_operation_link_rollback_and_reopen(
    database: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    projection: bool,
) -> None:
    runtime = Runtime.open(database)
    pid = ""
    before: tuple[dict[str, str], dict[str, str], int] | None = None
    actor = "test.failure.link.project" if projection else "test.failure.link.configure"
    try:
        pid = _prepare_process(runtime)
        before = _process_snapshot(runtime, pid)
        audit_ids_before = {record.record_id for record in runtime.audit.trace()}
        original_link = runtime.operations.link_evidence

        def fail_after_operation_link(
            evidence_type: str,
            evidence_id: str,
            role: str,
            **kwargs: Any,
        ) -> Any:
            result = original_link(
                evidence_type,
                evidence_id,
                role,
                **kwargs,
            )
            if evidence_type == "audit" and role == "audit":
                raise KeyboardInterrupt("injected operation-link interruption")
            return result

        with runtime.operations.scope(
            kind="runtime",
            name="test.tool_table_link_rollback",
            actor="test",
            pid=pid,
        ) as operation:
            with monkeypatch.context() as scoped:
                scoped.setattr(
                    runtime.operations,
                    "link_evidence",
                    fail_after_operation_link,
                )
                with pytest.raises(KeyboardInterrupt, match="operation-link"):
                    _failed_mutation(
                        runtime,
                        pid,
                        projection=projection,
                        assigned_by=actor,
                    )

        assert _process_snapshot(runtime, pid) == before
        assert {record.record_id for record in runtime.audit.trace()} == audit_ids_before
        assert runtime.store.list_operation_evidence(
            operation_ids=[operation.operation_id]
        ) == []
        assert not any(record.actor == actor for record in runtime.audit.trace())
    finally:
        runtime.close()

    assert before is not None
    _assert_reopened_tables(database, pid, before[0], before[1])


def test_process_tool_configuration_audit_failure_rolls_back_and_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_audit_rollback_and_reopen(
        tmp_path / "configure-audit.sqlite3",
        monkeypatch,
        projection=False,
    )


def test_process_tool_configuration_link_interruption_rolls_back_and_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_operation_link_rollback_and_reopen(
        tmp_path / "configure-link.sqlite3",
        monkeypatch,
        projection=False,
    )


def test_model_tool_projection_audit_failure_rolls_back_and_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_audit_rollback_and_reopen(
        tmp_path / "projection-audit.sqlite3",
        monkeypatch,
        projection=True,
    )


def test_model_tool_projection_link_interruption_rolls_back_and_survives_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _exercise_operation_link_rollback_and_reopen(
        tmp_path / "projection-link.sqlite3",
        monkeypatch,
        projection=True,
    )


def _run_concurrent_configurations(
    first: Callable[[], dict[str, str]],
    second: Callable[[], dict[str, str]],
) -> list[dict[str, str]]:
    ready = threading.Barrier(3)

    def run(operation: Callable[[], dict[str, str]]) -> dict[str, str]:
        ready.wait()
        return operation()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, first), executor.submit(run, second)]
        ready.wait()
        return [future.result(timeout=10) for future in futures]


def test_concurrent_process_tool_configurations_commit_complete_audited_tables() -> None:
    runtime = Runtime.open("local")
    try:
        pid = _prepare_process(runtime)
        before = runtime.process.get(pid)
        actors = {"test.concurrent.full.first", "test.concurrent.full.second"}
        results = _run_concurrent_configurations(
            lambda: runtime.tools.configure_process_tools(
                pid,
                ["echo"],
                assigned_by="test.concurrent.full.first",
            ),
            lambda: runtime.tools.configure_process_tools(
                pid,
                ["get_working_directory"],
                assigned_by="test.concurrent.full.second",
            ),
        )

        process = runtime.process.get(pid)
        assert results[0] != results[1]
        assert process.tool_table in results
        assert process.model_tool_table == process.tool_table
        assert process.revision == before.revision + 2
        audits = [
            record
            for record in runtime.audit.trace()
            if record.action == "process.tools.configure" and record.actor in actors
        ]
        assert {record.actor for record in audits} == actors
        assert {
            tuple(record.decision["tools"])
            for record in audits
            if record.decision is not None
        } == {tuple(sorted(table)) for table in results}
    finally:
        runtime.close()


def test_concurrent_model_tool_projections_commit_complete_audited_tables() -> None:
    runtime = Runtime.open("local")
    try:
        pid = _prepare_process(runtime)
        before = runtime.process.get(pid)
        full_table = dict(before.tool_table)
        actors = {"test.concurrent.project.first", "test.concurrent.project.second"}
        results = _run_concurrent_configurations(
            lambda: runtime.tools.configure_model_tool_projection(
                pid,
                ["echo"],
                assigned_by="test.concurrent.project.first",
            ),
            lambda: runtime.tools.configure_model_tool_projection(
                pid,
                ["get_current_time"],
                assigned_by="test.concurrent.project.second",
            ),
        )

        process = runtime.process.get(pid)
        assert results[0] != results[1]
        assert process.tool_table == full_table
        assert process.model_tool_table in results
        assert process.revision == before.revision + 2
        audits = [
            record
            for record in runtime.audit.trace()
            if record.action == "process.tools.project" and record.actor in actors
        ]
        assert {record.actor for record in audits} == actors
        assert {
            tuple(record.decision["tools"])
            for record in audits
            if record.decision is not None
        } == {tuple(sorted(table)) for table in results}
    finally:
        runtime.close()
