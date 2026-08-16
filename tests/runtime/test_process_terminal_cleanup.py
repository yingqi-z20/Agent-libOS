from __future__ import annotations

import hashlib
import json
import secrets
import threading

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import (
    AgentProcess,
    EventType,
    KilledProcessOutcome,
    ProcessCursor,
    ProcessSignal,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
)
from agent_libos.models.exceptions import (
    ProcessError,
    ProcessTerminalCleanupRequired,
    ValidationError,
)
from agent_libos.models.snapshot import SnapshotRows
from agent_libos.utils.serde import dumps


def _terminal_events(runtime: Runtime, pid: str) -> list[object]:
    return [
        event
        for event in runtime.events.list(target=pid)
        if event.type == EventType.PROCESS_EXITED
    ]


def _terminal_audits(runtime: Runtime, pid: str) -> list[object]:
    return [
        record
        for record in runtime.audit.trace(target=f"process:{pid}")
        if record.action in {"process.exit", "process.signal"}
    ]


def _orphaned_created_process(pid: str, *, created_at: str) -> AgentProcess:
    return AgentProcess(
        pid=pid,
        parent_pid=None,
        image_id="base-agent:v0",
        status=ProcessStatus.CREATED,
        goal_oid=None,
        memory_view=None,
        capabilities=[],
        loaded_skills={},
        tool_table={},
        event_cursor=None,
        checkpoint_head=None,
        resource_budget=ResourceBudget(),
        resource_usage=ResourceUsage(),
        created_at=created_at,
        updated_at=created_at,
    )


@pytest.mark.parametrize(
    "failure_sink",
    ["event_link", "audit"],
)
def test_orphaned_created_recovery_rolls_back_transition_and_evidence(
    monkeypatch: pytest.MonkeyPatch,
    failure_sink: str,
) -> None:
    runtime = Runtime.open("local")
    pid = f"pid_orphan_atomic_{failure_sink}"
    try:
        process = _orphaned_created_process(
            pid,
            created_at="2026-01-01T00:00:00+00:00",
        )
        runtime.store.insert_process(process)
        original_record = runtime.audit.record
        operations = runtime.events.operations
        assert operations is not None
        original_link = operations.link_evidence

        def fail_after_audit(*args: object, **kwargs: object) -> object:
            result = original_record(*args, **kwargs)
            if kwargs.get("action") == "orphaned_launch":
                raise RuntimeError("injected orphan recovery audit failure")
            return result

        def fail_after_event_link(*args: object, **kwargs: object) -> object:
            result = original_link(*args, **kwargs)
            if args[:1] == ("event",):
                raise RuntimeError("injected orphan recovery evidence failure")
            return result

        if failure_sink == "audit":
            monkeypatch.setattr(runtime.audit, "record", fail_after_audit)
        else:
            monkeypatch.setattr(
                operations,
                "link_evidence",
                fail_after_event_link,
            )

        with pytest.raises(RuntimeError, match="injected orphan recovery"):
            runtime.process._fail_orphaned_created_processes()

        rolled_back = runtime.process.get(pid)
        assert rolled_back == process
        assert runtime.store.get_process_terminal_cleanup(pid) is None
        assert not any(
            event.type == EventType.PROCESS_EXITED and event.source == pid
            for event in runtime.events.list()
        )
        assert not any(
            record.action == "orphaned_launch"
            for record in runtime.audit.trace(target=f"process:{pid}")
        )

        monkeypatch.setattr(runtime.audit, "record", original_record)
        monkeypatch.setattr(operations, "link_evidence", original_link)
        runtime.process._fail_orphaned_created_processes()
        recovered = runtime.process.get(pid)
        assert recovered.status == ProcessStatus.FAILED
        assert recovered.outcome is not None
        assert recovered.outcome.code == "orphaned_launch"
        assert runtime.store.get_process_terminal_cleanup(pid) is not None
        events = [
            event
            for event in runtime.events.list()
            if event.type == EventType.PROCESS_EXITED and event.source == pid
        ]
        assert len(events) == 1
        assert events[0].payload["reason"] == "orphaned_launch"
        assert len(
            [
                record
                for record in runtime.audit.trace(target=f"process:{pid}")
                if record.action == "orphaned_launch"
            ]
        ) == 1

        runtime.process._fail_orphaned_created_processes()
        assert len(
            [
                event
                for event in runtime.events.list()
                if event.type == EventType.PROCESS_EXITED and event.source == pid
            ]
        ) == 1
    finally:
        runtime.close()


def test_killed_process_exit_event_is_stable_across_runtime_reopen(
    tmp_path,
) -> None:
    database = tmp_path / "killed-process-event-reopen.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(goal="stable killed event across reopen")
        process = runtime.process.get(pid)
        runtime.process_transitions.transition(
            pid,
            ProcessStatus.KILLED,
            expected_revision=process.revision,
            outcome=KilledProcessOutcome(code="test_fixture"),
        )
        runtime.process.finalize_killed_processes(
            [pid],
            reason="stable resource termination",
        )
        event_ids = {
            event.event_id
            for event in runtime.events.list()
            if event.type == EventType.PROCESS_EXITED and event.source == pid
        }
        assert len(event_ids) == 1
    finally:
        runtime.close()

    reopened = Runtime.open(database)
    try:
        reopened.process.finalize_killed_processes(
            [pid],
            reason="stable resource termination",
        )
        replayed = [
            event
            for event in reopened.events.list()
            if event.type == EventType.PROCESS_EXITED and event.source == pid
        ]
        assert len(replayed) == 1
        assert {event.event_id for event in replayed} == event_ids
    finally:
        reopened.close()


def test_exit_cleanup_failure_is_durable_idempotent_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="durable terminal cleanup")
        original_finalize = runtime.process._finalize_terminal_process
        fail_cleanup = True
        finalize_calls = 0

        def flaky_finalize(*args: object, **kwargs: object) -> None:
            nonlocal fail_cleanup, finalize_calls
            finalize_calls += 1
            if fail_cleanup:
                raise RuntimeError("injected terminal cleanup failure")
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            flaky_finalize,
        )

        with pytest.raises(ProcessError, match="terminal cleanup"):
            runtime.process.exit(pid, message="terminal outcome commits once")

        terminal = runtime.process.get(pid)
        assert terminal.status == ProcessStatus.EXITED
        state = runtime.process.terminal_cleanup_state(pid)
        assert state["state"] == "failed"
        assert state["completed_phases"] == ["terminal_notify"]
        assert state["failed_phase"] == "process_finalize"
        assert state["attempt_count"] == 1
        assert state["last_error"]["error_type"] == "RuntimeError"
        assert set(state["last_error"]["exception_text"]) == {"bytes", "sha256"}
        assert len(_terminal_events(runtime, pid)) == 1
        assert len(_terminal_audits(runtime, pid)) == 1

        fail_cleanup = False
        repaired = runtime.process.retry_terminal_cleanup(pid)

        assert repaired["state"] == "completed"
        assert repaired["completed_phases"] == [
            "terminal_notify",
            "process_finalize",
        ]
        assert repaired["attempt_count"] == 2
        assert len(_terminal_events(runtime, pid)) == 1
        assert len(_terminal_audits(runtime, pid)) == 1
        assert finalize_calls == 2

        assert runtime.process.retry_terminal_cleanup(pid)["state"] == "completed"
        assert finalize_calls == 2
    finally:
        runtime.close()


def test_mcp_revocation_failure_is_durable_and_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="fail closed MCP terminal revocation")
        manager = runtime._mcp_subscription_manager  # noqa: SLF001
        original_invalidate = manager._invalidate_nowait  # noqa: SLF001
        secret = secrets.token_urlsafe(48)

        def fail_local_invalidation(**_kwargs: object) -> None:
            raise RuntimeError(secret)

        monkeypatch.setattr(manager, "_invalidate_nowait", fail_local_invalidation)
        with pytest.raises(ProcessTerminalCleanupRequired):
            runtime.process.exit(pid, message="terminal outcome remains committed")

        assert runtime.process.get(pid).status is ProcessStatus.EXITED
        cleanup = runtime.process.terminal_cleanup_state(pid)
        assert cleanup["state"] == "failed"
        assert cleanup["failed_phase"] == "terminal_notify"
        assert cleanup["completed_phases"] == ["process_finalize"]
        component = cleanup["last_error"]["errors"][0][
            "component_failures"
        ][0]
        assert component["phase"] == "mcp"
        assert component["error_type"] == "RuntimeError"
        assert set(component["exception_text"]) == {"bytes", "sha256"}
        assert secret not in dumps(cleanup)

        monkeypatch.setattr(manager, "_invalidate_nowait", original_invalidate)
        repaired = runtime.process.retry_terminal_cleanup(pid)
        assert repaired["state"] == "completed"
        assert repaired["completed_phases"] == [
            "terminal_notify",
            "process_finalize",
        ]
        assert repaired["attempt_count"] == 2
    finally:
        runtime.close()


def test_process_exit_tool_reports_committed_outcome_when_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="surface committed exit cleanup state",
        )
        original_finalize = runtime.process._finalize_terminal_process
        secret = secrets.token_urlsafe(48)

        def fail_finalize(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError(secret)

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            fail_finalize,
        )

        result = runtime.tools.call(
            pid,
            "process_exit",
            {"payload": {"done": True}},
        )

        process = runtime.process.get(pid)
        assert result.ok, result.error
        assert process.status == ProcessStatus.EXITED
        assert process.outcome is not None
        assert result.payload["status"] == "exited"
        assert result.payload["terminal_committed"] is True
        assert result.payload["result_oid"] == process.outcome.result_oid
        assert result.payload["error"] == {
            "code": "terminal_cleanup_required",
            "error_type": "ProcessTerminalCleanupRequired",
            "message": (
                "The terminal outcome and result committed, but durable terminal "
                "cleanup remains incomplete."
            ),
            "retryable_by_agent": False,
        }
        cleanup = result.payload["cleanup"]
        assert cleanup["state"] == "failed"
        assert cleanup["failed_phase"] == "process_finalize"
        assert cleanup["attempt_count"] == 1
        assert cleanup["completed_phases"] == ["terminal_notify"]
        assert cleanup["recovery"]["owner"] == "host"
        assert cleanup["recovery"]["action"] == "retry_terminal_cleanup"
        assert cleanup["recovery"]["idempotent"] is True
        assert cleanup["recovery"]["retry_process_exit"] is False
        assert secret not in json.dumps(result.payload, sort_keys=True)

        committed_result = runtime.store.get_object(process.outcome.result_oid)
        assert committed_result is not None
        assert committed_result.payload == {"done": True}

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            original_finalize,
        )
        repaired = runtime.process.retry_terminal_cleanup(pid)
        assert repaired["state"] == "completed"
    finally:
        runtime.close()


def test_repeated_cancel_retries_failed_cleanup_without_second_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="retry terminal cancel")
        original_finalize = runtime.process._finalize_terminal_process
        fail_cleanup = True

        def flaky_finalize(*args: object, **kwargs: object) -> None:
            if fail_cleanup:
                raise RuntimeError("injected cancel cleanup failure")
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            flaky_finalize,
        )

        with pytest.raises(ProcessError, match="terminal cleanup"):
            runtime.process.cancel(pid, "first terminal outcome")

        before_terminal_events = _terminal_events(runtime, pid)
        before_audits = list(runtime.audit.trace(target=f"process:{pid}"))
        fail_cleanup = False

        runtime.process.cancel(pid, "retry must not replace outcome")

        assert runtime.process.get(pid).status == ProcessStatus.KILLED
        assert runtime.process.terminal_cleanup_state(pid)["state"] == "completed"
        assert _terminal_events(runtime, pid) == before_terminal_events
        after_audits = runtime.audit.trace(target=f"process:{pid}")
        assert [record for record in after_audits if record.action == "process.signal"] == [
            record for record in before_audits if record.action == "process.signal"
        ]
    finally:
        runtime.close()


def test_cleanup_failure_remains_durable_when_warning_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="durable cleanup failure observation")
        original_record = runtime.audit.record

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("injected cleanup failure before broken audit")
            ),
        )

        def fail_cleanup_warning(*args: object, **kwargs: object):
            if kwargs.get("action") == "process.terminal_cleanup_failed":
                raise RuntimeError("injected cleanup warning audit failure")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", fail_cleanup_warning)

        with pytest.raises(ProcessError, match="terminal cleanup") as raised:
            runtime.process.exit(pid, message="audit cannot erase cleanup state")

        assert raised.value.__cause__ is None
        assert raised.value.__context__ is None
        state = runtime.process.terminal_cleanup_state(pid)
        assert state["state"] == "failed"
        assert state["failed_phase"] == "process_finalize"
        assert state["last_error"]["error_type"] == "RuntimeError"
    finally:
        runtime.close()


def test_concurrent_cleanup_retries_have_one_phase_leader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    release_finalize = threading.Event()
    try:
        pid = runtime.process.spawn(goal="concurrent terminal cleanup retry")
        original_finalize = runtime.process._finalize_terminal_process
        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("seed failed cleanup")
            ),
        )
        with pytest.raises(ProcessError, match="terminal cleanup"):
            runtime.process.exit(pid, message="seed terminal cleanup")

        finalize_entered = threading.Event()
        finalize_calls = 0
        finalize_lock = threading.Lock()

        def blocking_finalize(*args: object, **kwargs: object) -> None:
            nonlocal finalize_calls
            with finalize_lock:
                finalize_calls += 1
            finalize_entered.set()
            assert release_finalize.wait(timeout=2)
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            blocking_finalize,
        )
        start = threading.Barrier(3)
        outcomes: list[dict[str, object]] = []
        errors: list[BaseException] = []

        def retry() -> None:
            try:
                start.wait(timeout=2)
                outcomes.append(runtime.process.retry_terminal_cleanup(pid))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=retry) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait(timeout=2)
        assert finalize_entered.wait(timeout=2)
        with finalize_lock:
            assert finalize_calls == 1

        release_finalize.set()
        for thread in threads:
            thread.join(timeout=2)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert [outcome["state"] for outcome in outcomes] == [
            "completed",
            "completed",
        ]
        assert finalize_calls == 1
        assert runtime.process.terminal_cleanup_state(pid)["attempt_count"] == 2
    finally:
        release_finalize.set()
        runtime.close()


def test_reopen_recovers_a_durable_failed_cleanup(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "terminal-cleanup-recovery.sqlite"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal="recover terminal cleanup after reopen")
    monkeypatch.setattr(
        runtime.process,
        "_finalize_terminal_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("injected cleanup failure before reopen")
        ),
    )
    with pytest.raises(ProcessError, match="terminal cleanup"):
        runtime.process.exit(pid, message="durable cleanup recovery")
    assert runtime.process.terminal_cleanup_state(pid)["state"] == "failed"
    runtime.close()

    reopened = Runtime.open(database)
    try:
        state = reopened.process.terminal_cleanup_state(pid)
        assert state["state"] == "completed"
        assert state["attempt_count"] == 2
        assert reopened.recovered_terminal_cleanups == {
            "pending": 1,
            "recovered_count": 1,
            "failure_count": 0,
            "recovered": [pid],
            "failures": [],
            "diagnostics_truncated": False,
        }
    finally:
        reopened.close()


def test_cleanup_failure_observations_never_retain_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="opaque cleanup failure observation")
        failure_text = secrets.token_urlsafe(48)
        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(failure_text)
            ),
        )

        with pytest.raises(ProcessTerminalCleanupRequired) as caught:
            runtime.process.exit(pid)

        error = caught.value
        assert error.__cause__ is None
        assert error.__context__ is None
        assert set(vars(error)) == {"pid", "phase", "attempt"}

        state = runtime.process.terminal_cleanup_state(pid)
        assert state["last_error"]["exception_text"] == {
            "bytes": len(failure_text.encode("utf-8")),
            "sha256": hashlib.sha256(failure_text.encode("utf-8")).hexdigest(),
        }

        monkeypatch.setattr(
            runtime.process,
            "_require_recovery_lease",
            lambda: None,
        )
        recovery = runtime.process.recover_terminal_cleanups()
        raw_rows = runtime.store.select_table_rows(
            "process_terminal_cleanups",
            "pid = ?",
            (pid,),
        )
        audit_decisions = [
            record.decision
            for record in runtime.audit.trace(target=f"process:{pid}")
        ]
        outward_exception = {
            "text": str(error),
            "repr": repr(error),
            "attributes": vars(error),
            "cause": repr(error.__cause__),
            "context": repr(error.__context__),
        }

        for observed in (
            state,
            raw_rows,
            audit_decisions,
            recovery,
            outward_exception,
        ):
            assert failure_text not in dumps(observed)
    finally:
        runtime.close()


def test_terminal_cleanup_recovery_uses_bounded_keyset_pages(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_size = 2
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            process_terminal_cleanup_recovery_page_size=page_size,
            process_terminal_cleanup_recovery_page_hard_limit=page_size,
        )
    )
    database = tmp_path / "terminal-cleanup-paging.sqlite"
    runtime = Runtime.open(database, config=config)
    pids: list[str] = []
    monkeypatch.setattr(
        runtime.process,
        "_finalize_terminal_process",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("seed cleanup backlog")
        ),
    )
    for index in range(2 * page_size + 1):
        pid = runtime.process.spawn(goal=f"terminal cleanup page {index}")
        pids.append(pid)
        with pytest.raises(ProcessTerminalCleanupRequired):
            runtime.process.exit(pid)

    first_page = runtime.store.list_process_terminal_cleanups(
        after=None,
        limit=page_size,
    )
    assert len(first_page) == page_size
    first_cursor = ProcessCursor(
        str(first_page[-1]["created_at"]),
        str(first_page[-1]["pid"]),
    )
    second_page = runtime.store.list_process_terminal_cleanups(
        after=first_cursor,
        limit=page_size,
    )
    assert len(second_page) == page_size
    assert {row["pid"] for row in first_page}.isdisjoint(
        row["pid"] for row in second_page
    )
    with pytest.raises(ValidationError, match="hard cap"):
        runtime.store.list_process_terminal_cleanups(
            after=None,
            limit=page_size + 1,
        )
    plan = runtime.store._query(
        "EXPLAIN QUERY PLAN SELECT pid FROM process_terminal_cleanups "
        "WHERE state != 'completed' AND (created_at, pid) > (?, ?) "
        "ORDER BY created_at COLLATE BINARY, pid COLLATE BINARY LIMIT ?",
        (first_cursor.created_at, first_cursor.pid, page_size),
    )
    assert any(
        "IDX_PROCESS_TERMINAL_CLEANUP_RECOVERY_KEYSET"
        in str(row["detail"]).upper()
        for row in plan
    )

    store_type = type(runtime.store)
    original_list = store_type.list_process_terminal_cleanups
    runtime.close()
    calls: list[tuple[ProcessCursor | None, int, tuple[str, ...]]] = []

    def tracked_page(
        store: object,
        *,
        after: ProcessCursor | None,
        limit: int,
        include_completed: bool = False,
    ) -> list[dict[str, object]]:
        rows = original_list(
            store,
            after=after,
            limit=limit,
            include_completed=include_completed,
        )
        calls.append((after, limit, tuple(str(row["pid"]) for row in rows)))
        return rows

    monkeypatch.setattr(store_type, "list_process_terminal_cleanups", tracked_page)
    reopened = Runtime.open(database, config=config)
    try:
        summary = reopened.recovered_terminal_cleanups
        assert summary["pending"] == len(pids)
        assert summary["recovered_count"] == len(pids)
        assert summary["failure_count"] == 0
        assert len(summary["recovered"]) == page_size
        assert summary["failures"] == []
        assert summary["diagnostics_truncated"] is True
        assert [len(call[2]) for call in calls] == [page_size, page_size, 1]
        assert all(call[1] == page_size for call in calls)
        cursors = [call[0] for call in calls[1:]]
        assert all(cursor is not None for cursor in cursors)
        assert all(
            left is not None and right is not None and left < right
            for left, right in zip(cursors, cursors[1:])
        )
        assert all(
            reopened.process.terminal_cleanup_state(pid)["state"] == "completed"
            for pid in pids
        )
    finally:
        reopened.close()


def test_checkpoint_fork_publication_reconstructs_terminal_cleanup_intent() -> None:
    runtime = Runtime.open("local")
    try:
        source_pid = runtime.process.spawn(goal="terminal checkpoint fork source")
        runtime.process.exit(source_pid)
        source_row = runtime.store.select_table_rows(
            "processes",
            "pid = ?",
            (source_pid,),
        )[0]
        clone_pid = f"{source_pid}_checkpoint_clone"
        clone_row = {
            **source_row,
            "pid": clone_pid,
            "parent_pid": None,
            "goal_oid": None,
            "memory_view_json": None,
            "capabilities_json": "[]",
            "loaded_skills_json": "{}",
            "tool_table_json": "{}",
            "model_tool_table_json": "{}",
            "revision": 0,
            "state_generation": 0,
            "execution_generation": 0,
            "execution_owner_id": None,
            "execution_lease_id": None,
        }
        rows = SnapshotRows(processes=(clone_row,))

        runtime.uow.snapshots.insert_checkpoint_fork_rows(
            rows,
            object_payloads={},
        )
        assert runtime.process.get(clone_pid).status == ProcessStatus.EXITED
        assert runtime.process.terminal_cleanup_state(clone_pid)["state"] == "pending"

        runtime.uow.snapshots.publish_checkpoint_fork_process_rows(rows)

        state = runtime.process.terminal_cleanup_state(clone_pid)
        assert state["state"] == "pending"
        assert state["terminal_status"] == ProcessStatus.EXITED.value
        assert runtime.process.retry_terminal_cleanup(clone_pid)["state"] == "completed"
    finally:
        runtime.close()


def test_resource_limit_kill_leaves_retryable_cleanup_on_finalizer_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="resource kill durable cleanup")
        original_finalize = runtime.process._finalize_terminal_process
        fail_cleanup = True

        def flaky_finalize(*args: object, **kwargs: object) -> None:
            if fail_cleanup:
                raise RuntimeError("injected resource kill cleanup failure")
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            flaky_finalize,
        )

        runtime.resources.kill_if_exceeded(pid, reason="injected resource limit")

        assert runtime.process.get(pid).status == ProcessStatus.KILLED
        cleanup = runtime.process.terminal_cleanup_state(pid)
        assert cleanup["state"] == "failed"
        assert cleanup["completed_phases"] == ["terminal_notify"]
        warnings = [
            record
            for record in runtime.audit.trace(target=f"process:{pid}")
            if record.action == "resource.limit_finalize_failed"
        ]
        assert len(warnings) == 1

        fail_cleanup = False
        assert runtime.process.retry_terminal_cleanup(pid)["state"] == "completed"
    finally:
        runtime.close()


def test_human_terminal_interrupt_uses_the_same_durable_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="human interrupt durable cleanup")
        original_finalize = runtime.process._finalize_terminal_process
        fail_cleanup = True

        def flaky_finalize(*args: object, **kwargs: object) -> None:
            if fail_cleanup:
                raise RuntimeError("injected human interrupt cleanup failure")
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            flaky_finalize,
        )

        with pytest.raises(ProcessError, match="terminal cleanup"):
            runtime.human.interrupt(
                pid,
                ProcessSignal.CANCEL,
                {"reason": "human cancelled"},
            )

        assert runtime.process.get(pid).status == ProcessStatus.KILLED
        cleanup = runtime.process.terminal_cleanup_state(pid)
        assert cleanup["state"] == "failed"
        assert cleanup["completed_phases"] == ["terminal_notify"]

        fail_cleanup = False
        assert runtime.process.retry_terminal_cleanup(pid)["state"] == "completed"
    finally:
        runtime.close()


def test_scheduler_failure_keeps_original_outcome_when_cleanup_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="scheduler failure durable cleanup")
        original_finalize = runtime.process._finalize_terminal_process
        fail_cleanup = True

        def flaky_finalize(*args: object, **kwargs: object) -> None:
            if fail_cleanup:
                raise RuntimeError("injected scheduler cleanup failure")
            original_finalize(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(
            runtime.process,
            "_finalize_terminal_process",
            flaky_finalize,
        )

        def fail_quantum(_pid: str) -> None:
            raise RuntimeError("original scheduler quantum failure")

        results = runtime.scheduler.run_pid_until_idle(pid, fail_quantum)

        assert len(results) == 1
        assert results[0]["ok"] is False
        assert "scheduler cleanup" not in results[0]["error"]
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.FAILED
        assert process.outcome is not None
        assert process.outcome.code == "scheduler_task_failed"
        cleanup = runtime.process.terminal_cleanup_state(pid)
        assert cleanup["state"] == "failed"
        assert cleanup["completed_phases"] == ["terminal_notify"]

        task_failures = [
            record
            for record in runtime.audit.trace(target=f"process:{pid}")
            if record.action == "scheduler.process_task_failed"
        ]
        assert len(task_failures) == 1
        assert task_failures[0].decision["internal_error"]["error_type"] == "RuntimeError"
        assert (
            task_failures[0].decision["public_error"]["correlation_id"]
            == results[0]["correlation_id"]
        )

        fail_cleanup = False
        assert runtime.process.retry_terminal_cleanup(pid)["state"] == "completed"
    finally:
        runtime.close()


def test_terminal_cleanup_intent_rolls_back_with_failed_terminal_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="cleanup intent transaction rollback")
        original_record = runtime.audit.record

        def fail_exit_audit(*args: object, **kwargs: object):
            if kwargs.get("action") == "process.exit":
                raise RuntimeError("injected terminal transaction rollback")
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, "record", fail_exit_audit)

        with pytest.raises(RuntimeError, match="terminal transaction rollback"):
            runtime.process.exit(pid)

        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert runtime.store.get_process_terminal_cleanup(pid) is None
    finally:
        runtime.close()
