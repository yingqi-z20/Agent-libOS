from __future__ import annotations

import contextlib
import os
import sqlite3
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.models import EventType, OperationKind, OperationOutcome, OperationState
from agent_libos.models.exceptions import (
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
    ValidationError,
)
from agent_libos.process_execution import bind_process_execution
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import open_store
from agent_libos.storage.repositories import SnapshotCheckpointRepository
from agent_libos.utils.serde import dumps


PERSISTENT_BACKENDS = [
    "sqlite-file",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


def _release_fenced_runtime_or_close(runtime: Runtime) -> None:
    reason = runtime.lifecycle.shutdown_reason
    if (
        runtime.lifecycle.state == "close_failed"
        and isinstance(reason, str)
        and reason.startswith("runtime.recovery_required:")
    ):
        result = runtime.release_recovery_diagnostics()
        assert result["ok"] is True, result
        assert result["recovery_diagnostics_released"] is True
        return
    runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize(
    (
        "publication_state",
        "preexisting_outcome",
        "expected_publication_state",
        "expected_operation_outcome",
    ),
    [
        pytest.param(
            "applying",
            None,
            "rolled_back",
            OperationOutcome.FAILED,
            id="applying-running",
        ),
        pytest.param(
            "applying",
            OperationOutcome.SUCCEEDED,
            "rolled_back",
            OperationOutcome.FAILED,
            id="applying-succeeded",
        ),
        pytest.param(
            "committed",
            None,
            "committed",
            OperationOutcome.SUCCEEDED,
            id="committed-running",
        ),
        pytest.param(
            "committed",
            OperationOutcome.INTERRUPTED,
            "committed",
            OperationOutcome.SUCCEEDED,
            id="committed-interrupted",
        ),
    ],
)
def test_reopen_converges_exec_publication_and_operation_atomically(
    kind: str,
    publication_state: str,
    preexisting_outcome: OperationOutcome | None,
    expected_publication_state: str,
    expected_operation_outcome: OperationOutcome,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="publication-operation crash contract")
            operation = runtime.operations.start(
                kind="runtime",
                name="process.exec",
                actor=pid,
                pid=pid,
            )
            publication_id = _insert_exec_publication(
                runtime,
                pid=pid,
                operation_id=operation.operation_id,
                state=publication_state,
            )
            if preexisting_outcome is not None:
                finished = runtime.operations.finish(
                    preexisting_outcome,
                    operation_id=operation.operation_id,
                )
                assert finished is not None
                assert finished.outcome == preexisting_outcome
        finally:
            runtime.close()

        first_snapshot: tuple[dict[str, object], object] | None = None
        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                recovered_operation = reopened.store.get_operation(operation.operation_id)

                assert publication is not None
                assert publication["state"] == expected_publication_state
                assert recovered_operation is not None
                assert recovered_operation.state == OperationState.TERMINAL
                assert recovered_operation.outcome == expected_operation_outcome
                assert recovered_operation.metadata["runtime_publication_id"] == publication_id
                assert (
                    recovered_operation.metadata["runtime_publication_state"]
                    == expected_publication_state
                )
                assert recovered_operation.metadata["runtime_publication_reconciled"] is True

                snapshot = (publication, recovered_operation)
                if first_snapshot is None:
                    first_snapshot = snapshot
                else:
                    assert snapshot == first_snapshot
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_compensation_failed_publication_reconciles_operation_to_unknown(
    kind: str,
    tmp_path: Path,
) -> None:
    """A terminal compensation failure cannot claim success or clean rollback."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="uncertain compensation contract")
            operation = runtime.operations.start(
                kind="runtime",
                name="process.exec",
                actor=pid,
                pid=pid,
            )
            publication_id = _insert_exec_publication(
                runtime,
                pid=pid,
                operation_id=operation.operation_id,
                state="applying",
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="rollback_pending",
                phase="compensating",
                expected_states={"applying"},
            )
            assert runtime.store.advance_runtime_publication(
                publication_id,
                state="failed",
                phase="compensation_failed",
                error={"code": "process_exec_compensation_failed"},
                expected_states={"rollback_pending"},
            )

            first = runtime.image_boot.reconcile_terminal_publications()
            first_operation = runtime.store.get_operation(operation.operation_id)
            first_publication = runtime.store.get_runtime_publication(publication_id)
            assert publication_id in first
            assert first_publication is not None
            assert first_publication["state"] == "failed"
            assert first_publication["phase"] == "compensation_failed"
            assert first_operation is not None
            assert first_operation.state == OperationState.TERMINAL
            assert first_operation.outcome == OperationOutcome.UNKNOWN
            assert first_operation.metadata["runtime_publication_id"] == publication_id

            second = runtime.image_boot.reconcile_terminal_publications()
            assert second == []
            assert runtime.store.get_runtime_publication(publication_id) == first_publication
            assert runtime.store.get_operation(operation.operation_id) == first_operation
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_exec_commit_operation_sink_failure_rolls_back_commit_evidence_and_outcome(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The online commit has no publication/operation/evidence split window."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="before operation sink failure",
            )
            before = runtime.process.get(pid)
            _fail_operation_publication_update(
                runtime,
                monkeypatch,
                publication_state="committed",
            )

            with pytest.raises(
                RuntimeError,
                match="injected committed operation update failure",
            ):
                runtime.exec_process(
                    pid,
                    "base-agent:v0",
                    goal="must not publish committed evidence",
                )

            publication, operation = _exec_publication_and_operation(runtime, pid)
            after = runtime.process.get(pid)
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "compensated"
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.FAILED
            assert operation.metadata["runtime_publication_id"] == publication["publication_id"]
            assert operation.metadata["runtime_publication_state"] == "rolled_back"
            assert after.image_id == before.image_id
            assert after.goal_oid == before.goal_oid
            assert "committed" not in {
                phase.get("phase")
                for phase in publication["receipt"]["phases"]
                if isinstance(phase, dict)
            }
            assert not [
                record
                for record in runtime.audit.trace(target=f"process:{pid}")
                if record.action == "process.exec"
            ]
            assert not [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.PROCESS_EXEC
            ]
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_exec_rollback_operation_sink_failure_stays_pending_until_reopen(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rolled-back receipt cannot commit without its linked operation."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="rollback operation sink failure")
            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                _raise_late_exec_failure,
            )
            _fail_operation_publication_update(
                runtime,
                monkeypatch,
                publication_state="rolled_back",
            )

            with pytest.raises(RuntimePublicationPending) as caught:
                runtime.exec_process(
                    pid,
                    "base-agent:v0",
                    goal="deterministic compensation must remain pending",
                )

            publication, operation = _exec_publication_and_operation(runtime, pid)
            assert caught.value.publication_id == publication["publication_id"]
            assert caught.value.operation_id == operation.operation_id
            assert publication["state"] == "rollback_pending"
            assert publication["phase"] == "compensation_applied"
            assert [
                phase
                for phase in publication["receipt"]["phases"]
                if phase.get("phase") == "compensation_applied"
            ] == [{"phase": "compensation_applied", "pid": pid}]
            assert operation.state == OperationState.RUNNING
            assert operation.outcome == OperationOutcome.PENDING
            assert operation.metadata["runtime_publication_id"] == publication["publication_id"]
            assert operation.metadata["runtime_publication_bound"] is True
            assert not [
                record
                for record in runtime.audit.trace(target=f"process:{pid}")
                if record.action == "image.boot.failed"
            ]
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            publication, operation = _exec_publication_and_operation(reopened, pid)
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensation_finalized"
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.FAILED
            assert operation.metadata["runtime_publication_id"] == publication["publication_id"]
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_exec_compensation_failed_operation_sink_failure_stays_pending_until_reopen(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uncertain compensation has no standalone failed terminal operation."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = runtime.process.spawn(goal="uncertain operation sink failure")
            monkeypatch.setattr(
                runtime.image_boot,
                "_configure_skills",
                _raise_late_exec_failure,
            )
            monkeypatch.setattr(
                runtime.process_exec_state,
                "restore",
                _raise_exec_restore_failure,
            )
            _fail_operation_publication_update(
                runtime,
                monkeypatch,
                publication_state="failed",
            )

            with pytest.raises(RuntimeRecoveryRequired) as caught:
                runtime.exec_process(
                    pid,
                    "base-agent:v0",
                    goal="uncertain compensation must remain pending",
                )

            publication, operation = _exec_publication_and_operation(runtime, pid)
            assert caught.value.publication_id == publication["publication_id"]
            assert caught.value.operation_id == operation.operation_id
            assert caught.value.pid == pid
            assert publication["state"] == "rollback_pending"
            assert publication["phase"] == "compensating"
            assert operation.state == OperationState.RUNNING
            assert operation.outcome == OperationOutcome.PENDING
            assert operation.metadata["runtime_publication_id"] == publication["publication_id"]
            assert operation.metadata["runtime_publication_bound"] is True
        finally:
            _release_fenced_runtime_or_close(runtime)

        reopened = Runtime.open(target, config=config)
        try:
            publication, operation = _exec_publication_and_operation(reopened, pid)
            assert publication["state"] == "rolled_back"
            assert publication["phase"] == "startup_compensated"
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.FAILED
            assert operation.metadata["runtime_publication_id"] == publication["publication_id"]
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_publication_planning_binding_failure_rolls_back_both_rows(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning cannot publish a link unless operation prebinding also commits."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            pid = (
                runtime.process.spawn(goal="exec binding transaction setup")
                if publication_kind == "process_exec"
                else None
            )
            before_publications = {
                publication["publication_id"]
                for publication in runtime.store.list_runtime_publications()
            }
            before_processes = {process.pid for process in runtime.store.list_processes()}
            before_operations = {
                operation.operation_id for operation in runtime.store.list_operations()
            }
            _fail_publication_binding_update(runtime, monkeypatch)

            with pytest.raises(RuntimeError, match="injected publication binding failure"):
                if publication_kind == "process_exec":
                    assert pid is not None
                    runtime.exec_process(pid, "base-agent:v0")
                else:
                    runtime.process.spawn(goal="launch binding transaction failure")

            assert {
                publication["publication_id"]
                for publication in runtime.store.list_runtime_publications()
            } == before_publications
            assert {
                process.pid for process in runtime.store.list_processes()
            } == before_processes
            new_operations = [
                operation
                for operation in runtime.store.list_operations()
                if operation.operation_id not in before_operations
            ]
            assert len(new_operations) == 1
            assert new_operations[0].name == (
                "process.exec" if publication_kind == "process_exec" else "process.spawn"
            )
            assert new_operations[0].state == OperationState.TERMINAL
            assert new_operations[0].outcome == OperationOutcome.FAILED
            assert "runtime_publication_id" not in new_operations[0].metadata
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("launch_kind", ["spawn", "fork", "spawn_child"])
def test_launch_commit_operation_sink_failure_compensates_atomically_and_retries(
    kind: str,
    launch_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A public launch never publishes success without its operation outcome."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        parent = (
            runtime.process.spawn(goal=f"{launch_kind} failure parent")
            if launch_kind != "spawn"
            else None
        )
        before_publication_ids = {
            publication["publication_id"]
            for publication in runtime.store.list_runtime_publications()
        }
        expected_operation_name = f"process.{launch_kind}"
        _fail_launch_operation_publication_update(
            runtime,
            monkeypatch,
            publication_state="committed",
            operation_name=expected_operation_name,
            once=True,
        )
        try:
            with pytest.raises(
                RuntimeError,
                match="injected committed launch operation update failure",
            ):
                _invoke_public_launch(runtime, launch_kind, parent)

            failed_publication = _single_new_launch_publication(
                runtime,
                before_publication_ids,
            )
            failed_pid = str(failed_publication["pid"])
            failed_operation = _publication_operation(runtime, failed_publication)
            assert failed_publication["state"] == "rolled_back"
            assert failed_publication["phase"] == "compensated"
            assert failed_operation.state == OperationState.TERMINAL
            assert failed_operation.outcome == OperationOutcome.FAILED
            assert failed_operation.metadata["runtime_publication_id"] == (
                failed_publication["publication_id"]
            )
            assert runtime.store.get_process(failed_pid) is None
            assert not [
                record
                for record in runtime.audit.trace(target=f"process:{failed_pid}")
                if record.action == expected_operation_name
            ]
            expected_event = (
                EventType.PROCESS_FORKED
                if launch_kind == "fork"
                else EventType.PROCESS_CREATED
            )
            assert not [
                event
                for event in runtime.events.list(target=failed_pid)
                if event.type == expected_event
            ]
            assert runtime.lifecycle.state == "open"

            retry_pid = _invoke_public_launch(runtime, launch_kind, parent)
            assert retry_pid != failed_pid
            assert runtime.store.get_process(retry_pid) is not None
            retry_publication = [
                publication
                for publication in runtime.store.list_runtime_publications(pid=retry_pid)
                if publication["kind"] == "process_launch"
            ][-1]
            retry_operation = _publication_operation(runtime, retry_publication)
            assert retry_publication["state"] == "committed"
            assert retry_operation.state == OperationState.TERMINAL
            assert retry_operation.outcome == OperationOutcome.SUCCEEDED
            snapshots = (
                failed_publication,
                failed_operation,
                retry_publication,
                retry_operation,
            )
        finally:
            runtime.close()

        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                assert reopened.store.get_runtime_publication(
                    failed_publication["publication_id"]
                ) == snapshots[0]
                assert reopened.store.get_operation(
                    failed_operation.operation_id
                ) == snapshots[1]
                assert reopened.store.get_runtime_publication(
                    retry_publication["publication_id"]
                ) == snapshots[2]
                assert reopened.store.get_operation(
                    retry_operation.operation_id
                ) == snapshots[3]
                assert reopened.store.get_process(failed_pid) is None
                assert reopened.store.get_process(retry_pid) is not None
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("launch_kind", ["spawn", "fork", "spawn_child"])
def test_launch_rollback_operation_sink_failure_fences_until_reopen(
    kind: str,
    launch_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rollback receipt cannot commit without its linked operation outcome."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        parent = (
            runtime.process.spawn(goal=f"{launch_kind} rollback parent")
            if launch_kind != "spawn"
            else None
        )
        before_publication_ids = {
            publication["publication_id"]
            for publication in runtime.store.list_runtime_publications()
        }
        runtime.process.add_after_spawn_hook(_raise_launch_body_failure)
        _fail_launch_operation_publication_update(
            runtime,
            monkeypatch,
            publication_state="rolled_back",
            operation_name=f"process.{launch_kind}",
            once=False,
        )
        try:
            with pytest.raises(RuntimePublicationPending):
                _invoke_public_launch(runtime, launch_kind, parent)

            publication = _single_new_launch_publication(
                runtime,
                before_publication_ids,
            )
            operation = _publication_operation(runtime, publication)
            failed_pid = str(publication["pid"])
            assert publication["state"] == "rollback_pending"
            assert publication["phase"] == "compensating"
            assert operation.state == OperationState.RUNNING
            assert operation.outcome == OperationOutcome.PENDING
            assert runtime.store.get_process(failed_pid) is None
            assert runtime.lifecycle.state == "close_failed"
            assert runtime.lifecycle.shutdown_reason == (
                f"runtime.recovery_required:{publication['publication_id']}"
            )
            with pytest.raises(
                RuntimeError,
                match="not accepting operations: state=close_failed",
            ):
                runtime.process.spawn(goal="must wait for launch recovery")
            publication_id = str(publication["publication_id"])
            operation_id = operation.operation_id
        finally:
            _release_fenced_runtime_or_close(runtime)

        first_snapshot: tuple[dict[str, Any], Any] | None = None
        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                operation = reopened.store.get_operation(operation_id)
                assert publication is not None
                assert publication["state"] == "rolled_back"
                assert publication["phase"] == "startup_compensated"
                assert operation is not None
                assert operation.state == OperationState.TERMINAL
                assert operation.outcome == OperationOutcome.FAILED
                assert operation.metadata["runtime_publication_id"] == publication_id
                assert reopened.store.get_process(failed_pid) is None
                assert reopened.lifecycle.state == "open"
                snapshot = (publication, operation)
                if first_snapshot is None:
                    first_snapshot = snapshot
                else:
                    assert snapshot == first_snapshot
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_reopen_rejects_fully_matching_unbound_operation_forgery(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    """A same-identity operation cannot self-authenticate a forged plan link."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            legitimate = runtime.store.get_operation(operation_id)
            assert legitimate is not None
            forged_operation = runtime.operations.start(
                kind=legitimate.kind,
                name=legitimate.name,
                actor=legitimate.actor,
                pid=legitimate.pid,
            )
            _forge_publication_operation_id(
                runtime,
                publication_id,
                forged_operation.operation_id,
            )
            legitimate_snapshot = runtime.store.get_operation(operation_id)
            forged_snapshot = runtime.store.get_operation(forged_operation.operation_id)
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="exact durable"):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert durable.get_operation(operation_id) == legitimate_snapshot
                assert durable.get_operation(forged_operation.operation_id) == forged_snapshot
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_reopen_rejects_missing_publication_operation(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    """A plan link whose operation row disappeared is never silently skipped."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        missing_operation_id = f"op-missing-{uuid4().hex}"
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            _forge_publication_operation_id(
                runtime,
                publication_id,
                missing_operation_id,
            )
            legitimate_snapshot = runtime.store.get_operation(operation_id)
            publication_snapshot = runtime.store.get_runtime_publication(publication_id)
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="missing operation"):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert durable.get_operation(missing_operation_id) is None
                assert durable.get_operation(operation_id) == legitimate_snapshot
                assert durable.get_runtime_publication(publication_id) == publication_snapshot
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_reopen_rejects_blank_publication_operation_binding(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    """Clearing one side of a current-version binding is not legacy data."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            _forge_publication_operation_id(runtime, publication_id, "")
            operation_snapshot = runtime.store.get_operation(operation_id)
            publication_snapshot = runtime.store.get_runtime_publication(publication_id)
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="lost its durable operation binding"):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert durable.get_operation(operation_id) == operation_snapshot
                assert durable.get_runtime_publication(publication_id) == publication_snapshot
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_normalized_reverse_binding_survives_reopen(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            assert runtime.operations.store.list_operation_ids_by_runtime_publication_id(
                publication_id
            ) == [operation_id]
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            assert reopened.operations.store.list_operation_ids_by_runtime_publication_id(
                publication_id
            ) == [operation_id]
            row = reopened.store._query(  # type: ignore[attr-defined]
                "SELECT runtime_publication_id FROM operations WHERE operation_id = ?",
                (operation_id,),
            )[0]
            assert row["runtime_publication_id"] == publication_id
            if kind == "postgres":
                indexes = reopened.store._query(  # type: ignore[attr-defined]
                    "SELECT indexdef FROM pg_indexes "
                    "WHERE schemaname = current_schema() AND indexname = ?",
                    ("idx_operations_runtime_publication",),
                )
                assert len(indexes) == 1
                index_definition = str(indexes[0]["indexdef"])
                assert "CREATE UNIQUE INDEX" in index_definition
                assert "(runtime_publication_id)" in index_definition
        finally:
            reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_repository_and_operation_api_reject_duplicate_reverse_binding(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            with pytest.raises(ValidationError, match="metadata is reserved"):
                runtime.operations.start(
                    kind="runtime",
                    name="forged.binding",
                    actor="test",
                    pid=None,
                    metadata={"runtime_publication_id": publication_id},
                )

            duplicate = runtime.operations.start(
                kind="runtime",
                name="repository.binding.conflict",
                actor="test",
                pid=None,
            )
            duplicate_with_binding = replace(
                duplicate,
                metadata={
                    "runtime_publication_id": publication_id,
                    "runtime_publication_kind": publication_kind,
                    "runtime_publication_bound": True,
                    "runtime_publication_binding_version": 1,
                },
            )
            with pytest.raises(ValidationError, match="already bound to another operation"):
                runtime.store.update_operation(
                    duplicate_with_binding,
                    expected_states=[duplicate.state.value],
                )

            assert runtime.operations.store.list_operation_ids_by_runtime_publication_id(
                publication_id
            ) == [operation_id]
            assert runtime.store.get_operation(duplicate.operation_id) == duplicate
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_concurrent_repository_binding_conflict_is_a_domain_error(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id = f"publication-repository-race-{uuid4().hex}"
            operations = [
                runtime.operations.start(
                    kind="runtime",
                    name=f"repository.binding.race.{index}",
                    actor="test",
                    pid=None,
                )
                for index in range(2)
            ]
            barrier = Barrier(2)

            def bind(operation: Any) -> bool | BaseException:
                candidate = replace(
                    operation,
                    metadata={"runtime_publication_id": publication_id},
                )
                barrier.wait(timeout=5)
                try:
                    return runtime.operations.store.update_operation(
                        candidate,
                        expected_states=[operation.state.value],
                    )
                except BaseException as exc:
                    return exc

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(bind, operations))

            assert sum(result is True for result in results) == 1
            failures = [result for result in results if isinstance(result, BaseException)]
            assert len(failures) == 1
            assert isinstance(failures[0], ValidationError)
            assert "already bound to another operation" in str(failures[0])
            assert len(
                runtime.operations.store.list_operation_ids_by_runtime_publication_id(
                    publication_id
                )
            ) == 1
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_bound_publication_operation_id_is_immutable(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            assert not runtime.store.update_runtime_publication_plan(
                publication_id,
                {"operation_id": f"op-replacement-{uuid4().hex}"},
                expected_states={"committed"},
            )
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["plan"]["operation_id"] == operation_id
            assert publication["plan"]["operation_binding_version"] == 1
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize(
    "publication_kind",
    ["process_launch", "process_exec", "checkpoint_restore"],
)
def test_terminal_publication_plans_reject_effective_updates(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, _operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            snapshot = runtime.store.get_runtime_publication(publication_id)
            assert snapshot is not None
            if publication_kind == "checkpoint_restore":
                with pytest.raises(ValidationError, match="internal writer"):
                    runtime.store.update_runtime_publication_plan(
                        publication_id,
                        {"forged_recovery_input": True},
                        expected_states={"committed"},
                    )
            else:
                assert not runtime.store.update_runtime_publication_plan(
                    publication_id,
                    {"forged_recovery_input": True},
                    expected_states={"committed"},
                )
            assert runtime.store.get_runtime_publication(publication_id) == snapshot
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_host_generic_checkpoint_publication_mutations_are_rejected(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "checkpoint_restore",
            )
            snapshot = runtime.store.get_runtime_publication(publication_id)
            assert snapshot is not None
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.advance_runtime_publication(
                    publication_id,
                    state="committed",
                    phase="reconciled",
                    expected_states={"committed"},
                )
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.record_runtime_publication_artifact(
                    publication_id,
                    {"artifact_id": "host-forged-checkpoint-artifact"},
                    expected_states={"committed"},
                )
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.update_runtime_publication_plan(
                    publication_id,
                    {},
                    expected_states={"committed"},
                )
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.claim_runtime_publication_recovery(
                    publication_id,
                    claimant_instance_id=runtime.instance_id,
                    expected_owner_instance_id=snapshot["owner_instance_id"],
                    expected_state="failed",
                    classification="reconcile_checkpoint_restore",
                    claimed_state="reconciliation_pending",
                )
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.mark_runtime_publication_operation_reconciled(
                    publication_id,
                    expected_kind="checkpoint_restore",
                    expected_state="committed",
                    expected_phase="reconciled",
                    expected_operation_id=operation_id,
                )
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.insert_runtime_publication(
                    publication_id=f"publication-host-forged-{uuid4().hex}",
                    kind="checkpoint_restore",
                    pid=str(snapshot["pid"]),
                    owner_instance_id=runtime.instance_id,
                    plan=snapshot["plan"],
                )
            assert runtime.store.get_runtime_publication(publication_id) == snapshot
        finally:
            runtime.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_reopen_rejects_multiple_reverse_operation_bindings(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    """One publication cannot converge one of multiple reverse-bound rows."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            duplicate = runtime.operations.start(
                kind=operation.kind,
                name=operation.name,
                actor=operation.actor,
                pid=operation.pid,
            )
            _forge_duplicate_reverse_binding(
                runtime,
                operation_id=duplicate.operation_id,
                publication_id=publication_id,
                publication_kind=publication_kind,
            )
            operation_rows = _operation_storage_rows_from_store(
                runtime,
                (operation_id, duplicate.operation_id),
            )
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(
                ValidationError,
                match=(
                    "exact durable runtime-publication binding index|"
                    "already bound to another operation"
                ),
            ):
                Runtime.open(target, config=config)
            # The schema itself is corrupt, so the low-level supported-store API
            # must reject it too.  Inspect raw rows read-only to prove both failed
            # opens were non-mutating without weakening the canonical-v5 gate.
            with pytest.raises(
                ValidationError,
                match="exact durable runtime-publication binding index",
            ):
                open_store(target, config=config)
            assert _raw_operation_storage_rows(
                target,
                (operation_id, duplicate.operation_id),
            ) == operation_rows


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize("publication_kind", ["process_exec", "process_launch"])
def test_reopen_rejects_duplicate_publication_operation_binding(
    kind: str,
    publication_kind: str,
    tmp_path: Path,
) -> None:
    """A second publication cannot take over an operation's durable binding."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            _reconcile_terminal_publications(runtime, publication_kind)
            operation_snapshot = runtime.store.get_operation(operation_id)
            assert operation_snapshot is not None
            duplicate_id = _insert_unbound_duplicate_publication(
                runtime,
                publication_id,
            )
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="exact durable"):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                operation = durable.get_operation(operation_id)
                assert operation == operation_snapshot
                assert operation is not None
                assert operation.metadata["runtime_publication_id"] == publication_id
                assert operation.metadata["runtime_publication_id"] != duplicate_id
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize(
    "publication_kind",
    ["process_exec", "process_launch", "checkpoint_restore"],
)
@pytest.mark.parametrize("tampered_field", ["kind", "name", "actor", "pid"])
def test_reopen_rejects_bound_operation_identity_tampering(
    kind: str,
    publication_kind: str,
    tampered_field: str,
    tmp_path: Path,
) -> None:
    """A matching binding ID cannot excuse a changed operation identity."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            _publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                publication_kind,
            )
            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            replacements: dict[str, object] = {
                "kind": OperationKind.PRIMITIVE,
                "name": "primitive.forged",
                "actor": "pid-forged-actor",
                "pid": "pid-forged-target",
            }
            tampered = replace(operation, **{tampered_field: replacements[tampered_field]})
            assert runtime.store.update_operation(
                tampered,
                expected_states=[operation.state.value],
            )
            tampered_snapshot = runtime.store.get_operation(operation_id)
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert durable.get_operation(operation_id) == tampered_snapshot
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_reconciles_dirtied_checkpoint_restore_operation_before_stale_interrupt(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "checkpoint_restore",
            )
            publication = runtime.store.get_runtime_publication(publication_id)
            operation = runtime.store.get_operation(operation_id)
            assert publication is not None
            assert operation is not None
            assert publication["operation_reconciled"] is True

            # Host-visible generic mutation is denied. This raw marker write
            # deliberately models corruption outside the application-integrity
            # boundary so startup repair still receives coverage.
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.update_runtime_publication_plan(
                    publication_id,
                    {"snapshot_sha256": publication["plan"]["snapshot_sha256"]},
                    expected_states={"committed"},
                )
            interrupted = replace(
                operation,
                state=OperationState.RUNNING,
                outcome=OperationOutcome.PENDING,
                completed_at=None,
            )
            assert runtime.store.update_operation(
                interrupted,
                expected_states=[OperationState.TERMINAL.value],
            )
            with runtime.store.transaction():
                dirtied_marker = runtime.store._execute(  # type: ignore[attr-defined]
                    "UPDATE runtime_publications SET operation_reconciled = 0 "
                    "WHERE publication_id = ?",
                    (publication_id,),
                )
                assert dirtied_marker.rowcount == 1
            dirtied = runtime.store.get_runtime_publication(publication_id)
            assert dirtied is not None
            assert dirtied["operation_reconciled"] is False
        finally:
            runtime.close()

        reopened = Runtime.open(target, config=config)
        try:
            publication = reopened.store.get_runtime_publication(publication_id)
            operation = reopened.store.get_operation(operation_id)
            assert publication is not None
            assert operation is not None
            assert publication["operation_reconciled"] is True
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.SUCCEEDED
            assert (
                operation_id
                not in reopened.recovered_stale_operations.sample_operation_ids
            )
            assert (
                reopened.checkpoint.reconcile_terminal_restore_publications()
                == []
            )
        finally:
            reopened.close()

        reopened_again = Runtime.open(target, config=config)
        try:
            assert (
                reopened_again.checkpoint.reconcile_terminal_restore_publications()
                == []
            )
        finally:
            reopened_again.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_rejects_committed_checkpoint_with_incomplete_phase_transcript(
    kind: str,
    tmp_path: Path,
) -> None:
    """Raw corruption outside the Host boundary cannot forge terminal success."""

    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "checkpoint_restore",
            )
            publication = runtime.store.get_runtime_publication(publication_id)
            operation = runtime.store.get_operation(operation_id)
            assert publication is not None
            assert operation is not None
            receipt = dict(publication["receipt"])
            receipt["phases"] = [
                marker
                for marker in receipt["phases"]
                if marker
                != {
                    "phase": "checkpoint_restore_phase_completed",
                    "name": "jit_source_reconciliation",
                }
            ]
            interrupted = replace(
                operation,
                state=OperationState.RUNNING,
                outcome=OperationOutcome.PENDING,
                completed_at=None,
            )
            assert runtime.store.update_operation(
                interrupted,
                expected_states=[OperationState.TERMINAL.value],
            )
            with runtime.store.transaction():
                corrupted = runtime.store._execute(  # type: ignore[attr-defined]
                    "UPDATE runtime_publications SET receipt_json = ?, "
                    "operation_reconciled = 0 WHERE publication_id = ?",
                    (dumps(receipt), publication_id),
                )
                assert corrupted.rowcount == 1
            publication_snapshot = runtime.store.get_runtime_publication(publication_id)
            operation_snapshot = runtime.store.get_operation(operation_id)
        finally:
            runtime.close()

        with pytest.raises(ValidationError, match="completion transcript"):
            Runtime.open(target, config=config)
        durable = open_store(target, config=config)
        try:
            assert durable.get_runtime_publication(publication_id) == publication_snapshot
            assert durable.get_operation(operation_id) == operation_snapshot
        finally:
            durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
@pytest.mark.parametrize(
    ("plan_field", "replacement"),
    [
        pytest.param("unexpected_contract_field", True, id="unknown-field"),
        pytest.param("pid", "pid-forged-restore", id="pid"),
        pytest.param("actor", "actor-forged-restore", id="actor"),
        pytest.param("snapshot_sha256", "0" * 64, id="snapshot-digest"),
        pytest.param("stale_tool_ids", ["tool-forged-restore"], id="stale-tools"),
    ],
)
def test_reopen_rejects_dirtied_checkpoint_restore_plan_tampering(
    kind: str,
    plan_field: str,
    replacement: object,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "checkpoint_restore",
            )
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            with pytest.raises(ValidationError, match="internal writer"):
                runtime.store.update_runtime_publication_plan(
                    publication_id,
                    {plan_field: replacement},
                    expected_states={"committed"},
                )
            plan = dict(publication["plan"])
            plan[plan_field] = replacement
            with runtime.store.transaction():
                updated = runtime.store._execute(  # type: ignore[attr-defined]
                    "UPDATE runtime_publications "
                    "SET plan_json = ?, operation_reconciled = 0 "
                    "WHERE publication_id = ?",
                    (dumps(plan), publication_id),
                )
                assert updated.rowcount == 1
            publication_snapshot = runtime.store.get_runtime_publication(publication_id)
            operation_snapshot = runtime.store.get_operation(operation_id)
            assert publication_snapshot is not None
            assert publication_snapshot["operation_reconciled"] is False
            assert operation_snapshot is not None
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert (
                    durable.get_runtime_publication(publication_id)
                    == publication_snapshot
                )
                assert durable.get_operation(operation_id) == operation_snapshot
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_rejects_dirtied_checkpoint_restore_exact_binding_tampering(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "checkpoint_restore",
            )
            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            tampered = replace(
                operation,
                metadata={
                    **operation.metadata,
                    "runtime_publication_kind": "process_exec",
                },
            )
            assert runtime.store.update_operation(
                tampered,
                expected_states=[OperationState.TERMINAL.value],
            )
            publication_snapshot = runtime.store.get_runtime_publication(publication_id)
            operation_snapshot = runtime.store.get_operation(operation_id)
            assert publication_snapshot is not None
            assert publication_snapshot["operation_reconciled"] is False
            assert operation_snapshot == tampered
        finally:
            runtime.close()

        for _attempt in range(2):
            with pytest.raises(ValidationError, match="exact durable"):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                assert (
                    durable.get_runtime_publication(publication_id)
                    == publication_snapshot
                )
                assert durable.get_operation(operation_id) == operation_snapshot
            finally:
                durable.close()


def test_corrupt_failed_checkpoint_plan_is_rejected_before_recovery_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint-plan-before-recovery-claim.sqlite"
    runtime = Runtime.open(target)
    publication_id = ""
    plan: dict[str, Any] = {}
    try:
        pid = runtime.process.spawn(goal="corrupt failed checkpoint plan")
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "corrupt plan before recovery claim",
            actor=pid,
            require_capability=False,
        )

        def fail_image_reconciliation(_snapshot: object) -> None:
            raise RuntimeError("injected checkpoint reconciliation failure")

        monkeypatch.setattr(
            runtime.checkpoint,
            "_restore_images",
            fail_image_reconciliation,
        )
        result = runtime.checkpoint.restore(
            "test",
            checkpoint_id,
            require_capability=False,
        )
        publication_id = str(result["publication_id"])
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "failed"
        plan = dict(publication["plan"])
        plan["stale_tool_ids"] = ["tool-forged-before-recovery"]
    finally:
        _release_fenced_runtime_or_close(runtime)

    durable = open_store(target)
    try:
        with durable.transaction():
            updated = durable._execute(  # type: ignore[attr-defined]
                "UPDATE runtime_publications "
                "SET plan_json = ?, operation_reconciled = 0 "
                "WHERE publication_id = ?",
                (dumps(plan), publication_id),
            )
            assert updated.rowcount == 1
        publication_snapshot = durable.get_runtime_publication(publication_id)
        assert publication_snapshot is not None
    finally:
        durable.close()

    for _attempt in range(2):
        with pytest.raises(ValidationError, match="plan anchor"):
            Runtime.open(target)
        durable = open_store(target)
        try:
            # Validation happens before the recovery claim, attempt increment,
            # phase callback, or operation rewrite.
            assert (
                durable.get_runtime_publication(publication_id)
                == publication_snapshot
            )
        finally:
            durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_changed_checkpoint_snapshot_is_rejected_before_recovery_claim(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        publication_id = ""
        checkpoint_id = ""
        changed_snapshot: dict[str, Any] = {}
        try:
            pid = runtime.process.spawn(
                goal="changed checkpoint snapshot before recovery claim"
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                "snapshot identity before corruption",
                actor=pid,
                require_capability=False,
            )

            def fail_image_reconciliation(_snapshot: object) -> None:
                raise RuntimeError("injected checkpoint reconciliation failure")

            monkeypatch.setattr(
                runtime.checkpoint,
                "_restore_images",
                fail_image_reconciliation,
            )
            result = runtime.checkpoint.restore(
                "test",
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result["publication_id"])
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            changed_snapshot = dict(found[1])
            changed_snapshot["reason"] = "valid but no longer hash-bound"
        finally:
            _release_fenced_runtime_or_close(runtime)

        durable = open_store(target, config=config)
        try:
            with durable.transaction():
                updated = durable._execute(  # type: ignore[attr-defined]
                    "UPDATE checkpoints SET snapshot_json = ? "
                    "WHERE checkpoint_id = ?",
                    (dumps(changed_snapshot), checkpoint_id),
                )
                assert updated.rowcount == 1
            publication_snapshot = durable.get_runtime_publication(publication_id)
            assert publication_snapshot is not None
        finally:
            durable.close()

        original_replay = (
            SnapshotCheckpointRepository.reconcile_checkpoint_object_payloads
        )
        replay_calls = 0

        def observe_replay(
            repository: SnapshotCheckpointRepository,
            snapshot: Any,
        ) -> tuple[str, ...]:
            nonlocal replay_calls
            replay_calls += 1
            return original_replay(repository, snapshot)

        monkeypatch.setattr(
            SnapshotCheckpointRepository,
            "reconcile_checkpoint_object_payloads",
            observe_replay,
        )
        expected_error = (
            "cannot validate persisted Windows checkpoint identities"
            if os.name == "nt"
            else "snapshot changed"
        )
        for _attempt in range(2):
            with pytest.raises(ValidationError, match=expected_error):
                Runtime.open(target, config=config)
            durable = open_store(target, config=config)
            try:
                # Snapshot validation happens before recovery attempt mutation
                # or the volatile-payload replay callback.
                assert (
                    durable.get_runtime_publication(publication_id)
                    == publication_snapshot
                )
                assert replay_calls == 0
            finally:
                durable.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_reconciliation_is_idempotent_for_bound_launch_publication(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "process_launch",
            )
        finally:
            runtime.close()

        first_snapshot: tuple[dict[str, Any], Any] | None = None
        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                operation = reopened.store.get_operation(operation_id)
                assert publication is not None
                assert operation is not None
                assert operation.state == OperationState.TERMINAL
                assert operation.outcome == OperationOutcome.SUCCEEDED
                assert operation.metadata["runtime_publication_id"] == publication_id
                snapshot = (publication, operation)
                if first_snapshot is None:
                    first_snapshot = snapshot
                else:
                    assert snapshot == first_snapshot
            finally:
                reopened.close()


@pytest.mark.parametrize("kind", PERSISTENT_BACKENDS)
def test_reopen_canonicalizes_pre_return_spawn_operation_pid(
    kind: str,
    tmp_path: Path,
) -> None:
    with _persistent_target(kind, tmp_path) as (target, config):
        runtime = Runtime.open(target, config=config)
        try:
            publication_id, operation_id = _create_bound_terminal_publication(
                runtime,
                "process_launch",
            )
            operation = runtime.store.get_operation(operation_id)
            assert operation is not None
            interrupted_return = replace(
                operation,
                pid=None,
                state=OperationState.RUNNING,
                outcome=OperationOutcome.PENDING,
                completed_at=None,
            )
            assert runtime.store.update_operation(
                interrupted_return,
                expected_states=[OperationState.TERMINAL.value],
            )
        finally:
            runtime.close()

        first_snapshot: Any | None = None
        for _attempt in range(2):
            reopened = Runtime.open(target, config=config)
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                operation = reopened.store.get_operation(operation_id)
                assert publication is not None
                assert operation is not None
                assert operation.pid == publication["pid"]
                assert operation.state == OperationState.TERMINAL
                assert operation.outcome == OperationOutcome.SUCCEEDED
                if first_snapshot is None:
                    first_snapshot = operation
                else:
                    assert operation == first_snapshot
            finally:
                reopened.close()


def test_unlinked_pending_exception_cannot_suppress_operation_terminalization() -> None:
    runtime = Runtime.open("local")
    operation_id = ""
    try:
        with pytest.raises(RuntimePublicationPending):
            with runtime.operations.scope(
                kind="runtime",
                name="test.unlinked-publication-pending",
                actor="test",
                pid=None,
            ) as operation:
                operation_id = operation.operation_id
                raise RuntimePublicationPending(
                    publication_id="publication-does-not-exist",
                    operation_id=operation.operation_id,
                    state="rollback_pending",
                    phase="compensating",
                )

        stored = runtime.store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.TERMINAL
        assert stored.outcome == OperationOutcome.FAILED
        assert stored.metadata["runtime_publication_mismatch"] is True
    finally:
        _release_fenced_runtime_or_close(runtime)


def test_grouped_unlinked_pending_cannot_suppress_operation_terminalization() -> None:
    runtime = Runtime.open("local")
    operation_id = ""
    try:
        with pytest.raises(BaseExceptionGroup):
            with runtime.operations.scope(
                kind="runtime",
                name="test.grouped-unlinked-publication-pending",
                actor="test",
                pid=None,
            ) as operation:
                operation_id = operation.operation_id
                raise BaseExceptionGroup(
                    "forged grouped pending signal",
                    [
                        KeyboardInterrupt("unrelated control-flow interruption"),
                        RuntimePublicationPending(
                            publication_id="publication-does-not-exist",
                            operation_id=operation.operation_id,
                            state="rollback_pending",
                            phase="compensating",
                        ),
                    ],
                )

        stored = runtime.store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.TERMINAL
        assert stored.outcome == OperationOutcome.FAILED
        assert stored.metadata["runtime_publication_mismatch"] is True
    finally:
        runtime.close()


def test_unlinked_recovery_required_cannot_suppress_operation_terminalization() -> None:
    runtime = Runtime.open("local")
    operation_id = ""
    try:
        with pytest.raises(RuntimeRecoveryRequired):
            with runtime.operations.scope(
                kind="runtime",
                name="test.unlinked-recovery-required",
                actor="test",
                pid="pid-test",
            ) as operation:
                operation_id = operation.operation_id
                raise RuntimeRecoveryRequired(
                    publication_id="publication-does-not-exist",
                    operation_id=operation.operation_id,
                    pid="pid-test",
                    state="rollback_pending",
                    phase="compensating",
                )

        stored = runtime.store.get_operation(operation_id)
        assert stored is not None
        assert stored.state == OperationState.TERMINAL
        assert stored.outcome == OperationOutcome.FAILED
        assert stored.metadata["runtime_publication_mismatch"] is True
    finally:
        runtime.close()


def test_fenced_exact_nested_signal_terminalizes_related_outer_operation() -> None:
    runtime = Runtime.open("local")
    outer_operation_id = ""
    bound_operation_id = ""
    publication_id = ""
    try:
        pid = runtime.process.spawn(goal="nested fenced publication signal")
        with pytest.raises(RuntimePublicationPending) as caught:
            with runtime.lifecycle.admit():
                with runtime.operations.scope(
                    kind="runtime",
                    name="test.related-outer-publication-scope",
                    actor=pid,
                    pid=pid,
                ) as outer:
                    outer_operation_id = outer.operation_id
                    bound = runtime.operations.start(
                        kind="runtime",
                        name="process.exec",
                        actor=pid,
                        pid=pid,
                    )
                    bound_operation_id = bound.operation_id
                    publication_id = _insert_exec_publication(
                        runtime,
                        pid=pid,
                        operation_id=bound_operation_id,
                        state="applying",
                    )
                    publication = runtime.store.get_runtime_publication(
                        publication_id
                    )
                    assert publication is not None
                    runtime.lifecycle.mark_recovery_required(
                        publication_id=publication_id,
                    )
                    raise RuntimePublicationPending(
                        publication_id=publication_id,
                        operation_id=bound_operation_id,
                        state=str(publication["state"]),
                        phase=str(publication["phase"]),
                    )

        assert caught.value.publication_id == publication_id
        outer = runtime.store.get_operation(outer_operation_id)
        bound = runtime.store.get_operation(bound_operation_id)
        assert outer is not None
        assert outer.state == OperationState.TERMINAL
        assert outer.outcome == OperationOutcome.FAILED
        assert outer.metadata["runtime_publication_mismatch"] is True
        assert bound is not None
        assert bound.state == OperationState.RUNNING
        assert bound.outcome == OperationOutcome.PENDING
    finally:
        _release_fenced_runtime_or_close(runtime)


@pytest.mark.parametrize("method_name", ["merge_metadata", "finish", "wait"])
def test_public_operation_metadata_mutators_reject_reserved_binding(
    method_name: str,
) -> None:
    runtime = Runtime.open("local")
    try:
        operation = runtime.operations.start(
            kind="runtime",
            name=f"reserved.metadata.{method_name}",
            actor="test",
            pid=None,
        )
        metadata = {"runtime_publication_id": "publication-forged"}
        with pytest.raises(ValidationError, match="metadata is reserved"):
            if method_name == "merge_metadata":
                runtime.operations.merge_metadata(
                    metadata,
                    operation_id=operation.operation_id,
                )
            elif method_name == "finish":
                runtime.operations.finish(
                    OperationOutcome.FAILED,
                    operation_id=operation.operation_id,
                    metadata=metadata,
                )
            else:
                runtime.operations.wait(
                    operation_id=operation.operation_id,
                    metadata=metadata,
                )

        assert runtime.store.get_operation(operation.operation_id) == operation
        assert runtime.operations.store.list_operation_ids_by_runtime_publication_id(
            "publication-forged"
        ) == []
    finally:
        runtime.close()


def test_sqlite_reverse_binding_lookup_uses_unique_covering_index() -> None:
    runtime = Runtime.open("local")
    try:
        publication_id, operation_id = _create_bound_terminal_publication(
            runtime,
            "process_launch",
        )
        plan = list(
            runtime.store.conn.execute(
                "EXPLAIN QUERY PLAN "
                "SELECT operation_id FROM operations "
                "WHERE runtime_publication_id = ? ORDER BY operation_id",
                (publication_id,),
            )
        )
        details = "\n".join(str(row[3]) for row in plan)
        assert "idx_operations_runtime_publication" in details
        indexes = {
            str(row["name"]): row
            for row in runtime.store.conn.execute("PRAGMA index_list(operations)")
        }
        binding_index = indexes["idx_operations_runtime_publication"]
        assert int(binding_index["unique"]) == 1
        assert int(binding_index["partial"]) == 1
        assert runtime.operations.store.list_operation_ids_by_runtime_publication_id(
            publication_id
        ) == [operation_id]
    finally:
        runtime.close()


def test_publication_binding_never_scans_operation_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        original_list = runtime.operations.store.list_operations

        def reject_history_scan(*args: object, **kwargs: object) -> object:
            if not args and not kwargs:
                raise AssertionError(
                    "publication binding must use the indexed exact lookup"
                )
            return original_list(*args, **kwargs)

        monkeypatch.setattr(runtime.operations.store, "list_operations", reject_history_scan)
        for index in range(12):
            runtime.process.spawn(goal=f"indexed publication binding {index}")
    finally:
        runtime.close()


def _create_bound_terminal_publication(
    runtime: Runtime,
    publication_kind: str,
) -> tuple[str, str]:
    pid = runtime.process.spawn(goal=f"{publication_kind} binding contract")
    if publication_kind == "process_launch":
        publications = [
            publication
            for publication in runtime.store.list_runtime_publications(pid=pid)
            if publication["kind"] == "process_launch"
        ]
        assert len(publications) == 1
        publication = publications[0]
        operation_id = str(publication["plan"].get("operation_id") or "")
        assert publication["state"] == "committed"
        assert operation_id
        return str(publication["publication_id"]), operation_id
    if publication_kind == "process_exec":
        operation = runtime.operations.start(
            kind="runtime",
            name="process.exec",
            actor=pid,
            pid=pid,
        )
        publication_id = _insert_exec_publication(
            runtime,
            pid=pid,
            operation_id=operation.operation_id,
            state="committed",
        )
        return publication_id, operation.operation_id
    if publication_kind == "checkpoint_restore":
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "checkpoint restore operation binding contract",
            actor=pid,
            require_capability=False,
        )
        result = runtime.checkpoint.restore(
            "test",
            checkpoint_id,
            require_capability=False,
        )
        publication_id = str(result["publication_id"])
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["kind"] == "checkpoint_restore"
        assert publication["state"] == "committed"
        operation_id = str(publication["plan"].get("operation_id") or "")
        assert operation_id
        return publication_id, operation_id
    raise AssertionError(f"unknown publication kind: {publication_kind}")


def _invoke_public_launch(
    runtime: Runtime,
    launch_kind: str,
    parent: str | None,
) -> str:
    if launch_kind == "spawn":
        return runtime.process.spawn(goal="launch operation publication contract")
    assert parent is not None
    if launch_kind == "fork":
        return runtime.process.fork(parent, "fork operation publication contract")
    if launch_kind == "spawn_child":
        return runtime.process.spawn_child(
            parent,
            "spawn child operation publication contract",
        )
    raise AssertionError(f"unknown launch kind: {launch_kind}")


def _single_new_launch_publication(
    runtime: Runtime,
    before_publication_ids: set[str],
) -> dict[str, Any]:
    publications = [
        publication
        for publication in runtime.store.list_runtime_publications()
        if publication["kind"] == "process_launch"
        and publication["publication_id"] not in before_publication_ids
    ]
    assert len(publications) == 1
    return publications[0]


def _publication_operation(runtime: Runtime, publication: dict[str, Any]) -> Any:
    operation_id = str(publication["plan"].get("operation_id") or "")
    assert operation_id
    operation = runtime.store.get_operation(operation_id)
    assert operation is not None
    return operation


def _raise_launch_body_failure(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("injected launch body failure")


def _fail_launch_operation_publication_update(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publication_state: str,
    operation_name: str,
    once: bool,
) -> None:
    original_update = runtime.store.update_operation
    matched = 0

    def fail_selected_update(
        record: Any,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        nonlocal matched
        if (
            record.name == operation_name
            and record.metadata.get("runtime_publication_state")
            == publication_state
            and (not once or matched == 0)
        ):
            matched += 1
            raise RuntimeError(
                f"injected {publication_state} launch operation update failure"
            )
        return original_update(record, expected_states=expected_states)

    monkeypatch.setattr(runtime.store, "update_operation", fail_selected_update)


def _insert_unbound_duplicate_publication(
    runtime: Runtime,
    source_publication_id: str,
) -> str:
    source = runtime.store.get_runtime_publication(source_publication_id)
    assert source is not None
    duplicate_id = f"publication-duplicate-{uuid4().hex}"
    plan = dict(source["plan"])
    plan["artifact_owner"] = f"publication:{duplicate_id}"
    runtime.store.insert_runtime_publication(
        publication_id=duplicate_id,
        kind=str(source["kind"]),
        pid=str(source["pid"]),
        owner_instance_id="forged-duplicate-publication",
        plan=plan,
    )
    assert runtime.store.advance_runtime_publication(
        duplicate_id,
        state="committed",
        phase="committed",
        expected_states={"planning"},
    )
    return duplicate_id


def _forge_publication_operation_id(
    runtime: Runtime,
    publication_id: str,
    operation_id: str,
) -> None:
    """Seed unreconciled durable corruption, bypassing the immutable API."""

    publication = runtime.store.get_runtime_publication(publication_id)
    assert publication is not None
    plan = dict(publication["plan"])
    plan["operation_id"] = operation_id
    with runtime.store.transaction():
        updated = runtime.store._execute(  # type: ignore[attr-defined]
            "UPDATE runtime_publications "
            "SET plan_json = ?, operation_reconciled = 0 "
            "WHERE publication_id = ?",
            (dumps(plan), publication_id),
        )
        assert updated.rowcount == 1


def _forge_duplicate_reverse_binding(
    runtime: Runtime,
    *,
    operation_id: str,
    publication_id: str,
    publication_kind: str,
) -> None:
    """Simulate an administrator corrupting both the binding index and row."""

    metadata = {
        "runtime_publication_id": publication_id,
        "runtime_publication_kind": publication_kind,
        "runtime_publication_bound": True,
        "runtime_publication_binding_version": 1,
    }
    with runtime.store.transaction():
        runtime.store._execute(  # type: ignore[attr-defined]
            "DROP INDEX idx_operations_runtime_publication"
        )
        runtime.store._execute(  # type: ignore[attr-defined]
            "CREATE INDEX idx_operations_runtime_publication "
            "ON operations(runtime_publication_id)"
        )
        updated = runtime.store._execute(  # type: ignore[attr-defined]
            "UPDATE operations SET metadata_json = ?, runtime_publication_id = ? "
            "WHERE operation_id = ?",
            (dumps(metadata), publication_id, operation_id),
        )
        assert updated.rowcount == 1
        invalidated = runtime.store._execute(  # type: ignore[attr-defined]
            "UPDATE runtime_publications SET operation_reconciled = 0 "
            "WHERE publication_id = ?",
            (publication_id,),
        )
        assert invalidated.rowcount == 1


_OPERATION_STORAGE_COLUMNS = (
    "operation_id",
    "root_operation_id",
    "parent_operation_id",
    "kind",
    "name",
    "actor",
    "pid",
    "state",
    "outcome",
    "expected_roles_json",
    "metadata_json",
    "runtime_publication_id",
    "started_at",
    "updated_at",
    "completed_at",
)


def _operation_storage_rows_from_store(
    runtime: Runtime,
    operation_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    placeholders = ", ".join("?" for _ in operation_ids)
    columns = ", ".join(_OPERATION_STORAGE_COLUMNS)
    rows = runtime.store._query(  # type: ignore[attr-defined]
        f"SELECT {columns} FROM operations "
        f"WHERE operation_id IN ({placeholders}) ORDER BY operation_id",
        operation_ids,
    )
    return tuple(dict(row) for row in rows)


def _raw_operation_storage_rows(
    target: str | Path,
    operation_ids: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    columns = ", ".join(_OPERATION_STORAGE_COLUMNS)
    if isinstance(target, Path):
        connection = sqlite3.connect(
            f"{target.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        try:
            placeholders = ", ".join("?" for _ in operation_ids)
            rows = connection.execute(
                f"SELECT {columns} FROM operations "
                f"WHERE operation_id IN ({placeholders}) ORDER BY operation_id",
                operation_ids,
            )
            return tuple(dict(row) for row in rows)
        finally:
            connection.close()

    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(str(target), row_factory=dict_row) as connection:
        placeholders = ", ".join("%s" for _ in operation_ids)
        rows = connection.execute(
            f"SELECT {columns} FROM operations "
            f"WHERE operation_id IN ({placeholders}) ORDER BY operation_id",
            operation_ids,
        )
        return tuple(dict(row) for row in rows)


def _reconcile_terminal_publications(runtime: Runtime, publication_kind: str) -> None:
    if publication_kind == "process_exec":
        runtime.image_boot.reconcile_terminal_publications()
        return
    if publication_kind == "process_launch":
        runtime.process.reconcile_terminal_publications()
        return
    if publication_kind == "checkpoint_restore":
        runtime.checkpoint.reconcile_terminal_restore_publications()
        return
    raise AssertionError(f"unknown publication kind: {publication_kind}")


def _fail_publication_binding_update(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_update = runtime.store.update_operation

    def fail_binding_update(
        record: Any,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        if (
            record.metadata.get("runtime_publication_bound") is True
            and "runtime_publication_state" not in record.metadata
        ):
            raise RuntimeError("injected publication binding failure")
        return original_update(record, expected_states=expected_states)

    monkeypatch.setattr(runtime.store, "update_operation", fail_binding_update)


def _fail_operation_publication_update(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    *,
    publication_state: str,
) -> None:
    original_update = runtime.store.update_operation

    def fail_selected_update(
        record: Any,
        *,
        expected_states: Iterable[str] | None = None,
    ) -> bool:
        if record.metadata.get("runtime_publication_state") == publication_state:
            raise RuntimeError(
                f"injected {publication_state} operation update failure"
            )
        return original_update(record, expected_states=expected_states)

    monkeypatch.setattr(runtime.store, "update_operation", fail_selected_update)


def _exec_publication_and_operation(
    runtime: Runtime,
    pid: str,
) -> tuple[dict[str, Any], Any]:
    publications = [
        publication
        for publication in runtime.store.list_runtime_publications(pid=pid)
        if publication["kind"] == "process_exec"
    ]
    assert len(publications) == 1
    publication = publications[0]
    operation_id = str(publication["plan"].get("operation_id") or "")
    assert operation_id
    operation = runtime.store.get_operation(operation_id)
    assert operation is not None
    return publication, operation


def _raise_late_exec_failure(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("injected late exec failure")


def _raise_exec_restore_failure(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("injected exec restore failure")


def _insert_exec_publication(
    runtime: Runtime,
    *,
    pid: str,
    operation_id: str,
    state: str,
) -> str:
    before = runtime.process_exec_state.capture(pid)
    publication_id = f"publication-operation-contract-{uuid4().hex}"
    admission_token = None
    with runtime.store.transaction():
        if state == "applying":
            process = runtime.process.get(pid)
            admission_token = runtime.store.claim_host_process_exec(
                pid,
                owner_id="runtime-that-crashed:process.exec",
                expected_revision=process.revision,
                expected_state_generation=process.state_generation,
                expected_execution_generation=process.execution_generation,
            )
            assert admission_token is not None
        runtime.store.insert_runtime_publication(
            publication_id=publication_id,
            kind="process_exec",
            pid=pid,
            owner_instance_id="runtime-that-crashed",
            plan={
                "pid": pid,
                "image_id": "review-agent:v0",
                "before_snapshot": before.snapshot.to_mapping(),
                "before_tool_ids": sorted(before.tool_ids),
                "operation_id": operation_id,
                "operation_binding_version": 1,
                **(
                    {
                        "admission_execution_generation": admission_token.generation,
                        "admission_execution_owner_id": admission_token.owner_id,
                        "admission_execution_lease_id": admission_token.lease_id,
                    }
                    if admission_token is not None
                    else {}
                ),
            },
        )
        runtime.operations.bind_runtime_publication(
            operation_id,
            publication_id=publication_id,
            publication_kind="process_exec",
            expected_kind="runtime",
            expected_name="process.exec",
            expected_actor=pid,
            expected_pid=pid,
        )
    if state == "applying":
        assert admission_token is not None
        process = runtime.process.get(pid)
        with bind_process_execution(admission_token):
            runtime.store.patch_process(
                pid,
                {"image_id": "review-agent:v0"},
                expected_revision=process.revision,
            )
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="applying",
            phase="process_exec_applied",
            expected_states={"planning"},
        )
    elif state == "committed":
        assert runtime.store.advance_runtime_publication(
            publication_id,
            state="committed",
            phase="committed",
            expected_states={"planning"},
        )
    else:  # pragma: no cover - helper contract
        raise AssertionError(f"unsupported publication state: {state}")
    return publication_id


@contextlib.contextmanager
def _persistent_target(
    kind: str,
    tmp_path: Path,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    if kind == "sqlite-file":
        yield tmp_path / "publication-operation.sqlite", AgentLibOSConfig()
        return
    if kind == "postgres":
        with _postgres_schema_dsn() as dsn:
            yield dsn, AgentLibOSConfig(
                runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
            )
        return
    raise AssertionError(f"unknown persistent backend: {kind}")


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_publication_operation_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield _dsn_with_search_path(dsn, schema)
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
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
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
