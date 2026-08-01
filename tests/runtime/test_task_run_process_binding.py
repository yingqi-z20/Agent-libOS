from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import TaskRunRecord, TaskRunStatus
from agent_libos.models.exceptions import (
    ProcessError,
    TaskRunRevisionConflict,
    ValidationError,
)
from agent_libos.utils.ids import utc_now


def _task_run_record(run_id: str, epoch: int, pid: str) -> TaskRunRecord:
    now = utc_now()
    return TaskRunRecord(
        run_id=run_id,
        status=TaskRunStatus.RUNNING,
        display_title="bound process tree",
        image_id="base-agent:v0",
        runtime_epoch=epoch,
        root_pid=pid,
        active_pid=pid,
        created_at=now,
        updated_at=now,
        started_at=now,
    )


def _spawn_bound_root(
    runtime: Runtime,
    *,
    run_id: str = "run-process-tree",
    epoch: int | None = None,
    commit_observer: Callable[[tuple[str, str, str, str]], None] | None = None,
) -> str:
    selected_epoch = runtime.task_runs.runtime_epoch if epoch is None else epoch

    def commit(
        pid: str,
        publication_id: str,
        event_id: str,
        audit_id: str,
    ) -> None:
        runtime.store.insert_task_run(_task_run_record(run_id, selected_epoch, pid))
        if commit_observer is not None:
            commit_observer((pid, publication_id, event_id, audit_id))

    return runtime.process.spawn(
        image="base-agent:v0",
        goal="durable root",
        _task_run_id=run_id,
        _task_run_epoch=selected_epoch,
        _task_run_role="root",
        _task_run_commit=commit,
    )


def test_task_run_root_commit_precedes_fence_and_descendants_inherit_binding() -> None:
    runtime = Runtime.open("local")
    try:
        commit_calls: list[tuple[str, str, str, str]] = []
        root = _spawn_bound_root(runtime, commit_observer=commit_calls.append)
        epoch = runtime.task_runs.runtime_epoch

        root_process = runtime.process.get(root)
        assert (
            root_process.task_run_id,
            root_process.task_run_epoch,
            root_process.task_run_role,
        ) == ("run-process-tree", epoch, "root")
        assert len(commit_calls) == 1
        assert all(isinstance(value, str) and value for value in commit_calls[0])

        spawned = runtime.process.spawn_child(root, "fresh child")
        forked = runtime.process.fork(root, "forked child")
        for child_pid in (spawned, forked):
            child = runtime.process.get(child_pid)
            assert child.parent_pid == root
            assert (
                child.task_run_id,
                child.task_run_epoch,
                child.task_run_role,
            ) == ("run-process-tree", epoch, "child")
    finally:
        runtime.close()


@pytest.mark.parametrize(
    "spawn_kwargs",
    (
        {"_task_run_id": "run-missing-commit", "_task_run_epoch": 3},
        {"_task_run_commit": lambda *_args: None},
        {
            "_task_run_id": "run-wrong-role",
            "_task_run_epoch": 3,
            "_task_run_role": "child",
            "_task_run_commit": lambda *_args: None,
        },
    ),
    ids=("binding-without-commit", "commit-without-binding", "non-root-role"),
)
def test_task_run_root_private_launch_contract_rejects_partial_binding(
    spawn_kwargs: dict[str, Any],
) -> None:
    runtime = Runtime.open("local")
    try:
        with pytest.raises(ValidationError):
            runtime.process.spawn(goal="invalid durable root", **spawn_kwargs)
        assert runtime.process.list() == []
    finally:
        runtime.close()


def test_task_run_epoch_change_during_child_launch_prevents_commit() -> None:
    runtime = Runtime.open("local")
    try:
        epoch = runtime.task_runs.runtime_epoch
        root = _spawn_bound_root(runtime, run_id="run-stale-child")

        def advance_epoch(pid: str, _image: str, _publication_id: str) -> None:
            if pid == root:
                return
            record = runtime.store.get_task_run("run-stale-child")
            assert record is not None
            runtime.store.claim_task_run_epoch(
                record.run_id,
                record.revision,
                runtime_epoch=epoch + 1,
            )

        runtime.process.add_after_spawn_hook(advance_epoch)
        with pytest.raises(ProcessError, match="TaskRun"):
            runtime.process.spawn_child(root, "must lose the epoch race")

        processes = runtime.process.list()
        assert [process.pid for process in processes] == [root]
        assert processes[0].task_run_epoch == epoch + 1
    finally:
        runtime.close()


def test_task_run_root_callback_failure_rolls_back_run_and_process() -> None:
    runtime = Runtime.open("local")
    try:
        epoch = runtime.task_runs.runtime_epoch

        def invalid_commit(pid: str, *_evidence_ids: str) -> object:
            runtime.store.insert_task_run(
                _task_run_record("run-callback-rollback", epoch, pid)
            )
            return object()

        with pytest.raises(
            ValidationError,
            match="commit callback must return None",
        ):
            runtime.process.spawn(
                goal="rollback callback writes",
                _task_run_id="run-callback-rollback",
                _task_run_epoch=epoch,
                _task_run_commit=invalid_commit,  # type: ignore[arg-type]
            )

        assert runtime.store.get_task_run("run-callback-rollback") is None
        assert runtime.process.list() == []
    finally:
        runtime.close()


def test_task_run_spawn_claimed_propagates_root_binding_to_execution_token() -> None:
    runtime = Runtime.open("local")
    try:
        epoch = runtime.task_runs.runtime_epoch

        def commit(pid: str, *_evidence_ids: str) -> None:
            runtime.store.insert_task_run(
                _task_run_record("run-claimed-root", epoch, pid)
            )

        pid, token = runtime.process.spawn_claimed(
            owner_id="task-run-controller",
            image="base-agent:v0",
            goal="claimed durable root",
            _task_run_id="run-claimed-root",
            _task_run_epoch=epoch,
            _task_run_commit=commit,
        )

        process = runtime.process.get(pid)
        assert process.task_run_role == "root"
        assert process.task_run_epoch == epoch
        assert token.pid == pid
        assert token.task_run_epoch == epoch
    finally:
        runtime.close()
