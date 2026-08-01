from __future__ import annotations

import ast
import inspect
import sqlite3
import textwrap
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    AgentProcess,
    Checkpoint,
    HumanRequest,
    HumanRequestStatus,
    ObjectNamespace,
    ObjectRefCursor,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    ResourceUsageReservation,
    RuntimeModule,
    RuntimeModuleRegistration,
    RuntimeModuleStatus,
)
from agent_libos.storage import (
    AuthorityRepository,
    EvidenceRepository,
    ExtensionRepository,
    ObjectRepository,
    PayloadRetentionRepository,
    ProcessRepository,
    ProcessStateRepository,
    ResourceRepository,
    SnapshotCheckpointRepository,
    SnapshotCheckpointBackendProtocol,
    RuntimePublicationRepository,
    RuntimeModuleRepository,
    SQLRuntimeStore,
    SqlEngine,
    SqlSession,
    SQLiteStore,
    UnitOfWork,
    UnitOfWorkBackendProtocol,
)
from agent_libos.storage.repositories import (
    CheckpointRestorePublicationWriter,
    _checkpoint_object_payload_was_superseded,
    unit_of_work_backend_conformance_errors,
)
from agent_libos.storage.contracts import (
    CheckpointPublicationWriterBackendProtocol,
    OperationEvidenceBackendProtocol,
    ProcessBackendProtocol,
    ResourceBackendProtocol,
    RuntimePublicationBackendProtocol,
)
from agent_libos.evidence.payload_retention import PayloadRetentionStore
from agent_libos.evidence.payload_retention import PayloadRetentionTier
from agent_libos.ports.authority import CapabilityStorePort
from agent_libos.storage.postgres import _PostgresConnection, _PostgresCursor, _PostgresDialect
from agent_libos.models.exceptions import ValidationError
from agent_libos.models.snapshot import ProcessSnapshot
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import dumps


class _RecordingStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    @contextmanager
    def locked(self):
        yield

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        self.calls.append(("transaction", (), {"include_object_payloads": include_object_payloads}))
        yield object()

    def __getattr__(self, name: str):
        def record(*args: Any, **kwargs: Any) -> str:
            self.calls.append((name, args, kwargs))
            return name

        return record


class _EmptyMutationResult:
    rowcount = 0

    def __iter__(self):
        return iter(())


class _InvalidCasBackend:
    """Minimal backend that proves facade validation remains transactional."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.mutation_visible = False
        self.rolled_back = False
        self.transaction_entries = 0
        self.writer_token = object()

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        del include_object_payloads
        self.transaction_entries += 1
        before = self.mutation_visible
        try:
            yield self
        except BaseException:
            self.mutation_visible = before
            self.rolled_back = True
            raise

    def execute(self, _sql: str, _params: Any = ()) -> _EmptyMutationResult:
        return _EmptyMutationResult()

    def complete_execution(self, *_args: Any, **_kwargs: Any) -> Any:
        self.mutation_visible = True
        return self.result

    ack_process_message = complete_execution
    release_execution = complete_execution
    commit_capability_use_reservation = complete_execution
    consume_capability_uses = complete_execution
    reserve_capability_uses = complete_execution

    def settle_resource_usage_reservation(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> Any:
        self.mutation_visible = True
        return self.result

    def get_process(self, pid: str) -> AgentProcess:
        return _process(pid)

    def select_table_rows(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    def restore_process_for_exec(self, *_args: Any, **_kwargs: Any) -> Any:
        self.mutation_visible = True
        return self.result

    delete_tool_if_unreferenced = restore_process_for_exec
    mark_runtime_publication_operation_reconciled = restore_process_for_exec
    advance_runtime_publication = restore_process_for_exec
    update_runtime_publication_plan = restore_process_for_exec
    record_runtime_publication_artifact = restore_process_for_exec
    transition_checkpoint_restore_payload_delivery = restore_process_for_exec
    begin_checkpoint_payload_delivery_attempt = restore_process_for_exec
    ack_checkpoint_payload_delivery_attempt = restore_process_for_exec
    abort_checkpoint_payload_delivery_attempt = restore_process_for_exec
    update_operation = restore_process_for_exec
    insert_operation_evidence = restore_process_for_exec
    update_llm_call_payload_retention = restore_process_for_exec
    update_external_effect_payload_retention = restore_process_for_exec

    def _issue_checkpoint_restore_writer_token(self) -> object:
        return self.writer_token


def _publication_record(publication_id: str) -> dict[str, Any]:
    return {
        "publication_id": publication_id,
        "kind": "checkpoint_restore",
        "pid": "pid_checkpoint_writer",
        "owner_instance_id": "runtime.test",
        "state": "planning",
        "phase": "planned",
        "plan": {"checkpoint_id": "checkpoint_writer"},
        "receipt": {"phases": [], "artifacts": []},
        "error": None,
        "operation_reconciled": False,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


class _CheckpointPublicationBackend:
    def __init__(
        self,
        *,
        insert_result: dict[str, Any],
        claim_result: dict[str, Any] | None,
    ) -> None:
        self.insert_result = insert_result
        self.claim_result = claim_result
        self.token = object()

    def _issue_checkpoint_restore_writer_token(self) -> object:
        return self.token

    def insert_runtime_publication(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["_checkpoint_restore_writer_token"] is self.token
        return self.insert_result

    def claim_runtime_publication_recovery(
        self,
        _publication_id: str,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        assert kwargs["_checkpoint_restore_writer_token"] is self.token
        return self.claim_result


class _MiswiredPublicationBackend:
    def __init__(self) -> None:
        self.records = {
            "publication_expected": _publication_record("publication_expected"),
            "publication_other": _publication_record("publication_other"),
        }
        self.rolled_back = False

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        del include_object_payloads
        before = deepcopy(self.records)
        try:
            yield self
        except BaseException:
            self.records = before
            self.rolled_back = True
            raise

    def get_runtime_publication(self, publication_id: str) -> dict[str, Any] | None:
        return deepcopy(self.records.get(publication_id))

    def insert_runtime_publication(self, **_kwargs: Any) -> dict[str, Any]:
        other = deepcopy(self.records["publication_other"])
        self.records["publication_other"] = other
        return deepcopy(other)

    def claim_runtime_publication_recovery(
        self,
        publication_id: str,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.records[publication_id]["owner_instance_id"] = "runtime.claimant"
        self.records[publication_id]["state"] = "rollback_pending"
        return deepcopy(self.records["publication_other"])


class _CheckpointBindingBackend:
    def __init__(self, found: tuple[Any, dict[str, Any]] | None) -> None:
        self.found = found
        self.insert_calls = 0

    def get_checkpoint_snapshot(
        self,
        _checkpoint_id: str,
    ) -> tuple[Any, dict[str, Any]] | None:
        return deepcopy(self.found)

    def insert_checkpoint(self, _checkpoint: Any, _snapshot: dict[str, Any]) -> None:
        self.insert_calls += 1


def _process(pid: str) -> AgentProcess:
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


def test_snapshot_repository_normalizes_capability_booleans_at_sql_boundary() -> None:
    backend = _RecordingStore()
    repository = SnapshotCheckpointRepository(backend)  # type: ignore[arg-type]
    row = {
        "cap_id": "cap_snapshot_bool",
        "delegable": True,
        "revocable": False,
    }
    observed: dict[str, Any] = {}

    repository._insert_backend_row(
        object(),
        "capabilities",
        row,
        before_insert=lambda _cursor, _table, item: observed.update(item),
    )

    assert observed == row
    assert type(observed["delegable"]) is bool
    assert type(observed["revocable"]) is bool
    name, args, kwargs = backend.calls[-1]
    assert name == "insert_table_row"
    assert kwargs == {}
    assert args == (
        "capabilities",
        {
            "cap_id": "cap_snapshot_bool",
            "delegable": 1,
            "revocable": 0,
        },
    )
    assert row == {
        "cap_id": "cap_snapshot_bool",
        "delegable": True,
        "revocable": False,
    }


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_process_repository_rejects_non_boolean_completion_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = ProcessRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="complete execution backend returned a non-boolean CAS result",
    ):
        repository.complete_execution(object())  # type: ignore[arg-type]

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_resource_repository_rejects_non_boolean_settlement_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = ResourceRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match=(
            "settle resource usage reservation backend returned "
            "a non-boolean CAS result"
        ),
    ):
        repository.settle_resource_usage_reservation(
            "reservation-invalid-cas",
            status="released",
            settled_usage=ResourceUsage(),
            updated_at=utc_now(),
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_snapshot_repository_rejects_non_boolean_restore_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = SnapshotCheckpointRepository(backend)  # type: ignore[arg-type]
    rows = SimpleNamespace(
        processes=({"pid": "pid-invalid-restore-cas"},),
        object_namespaces=(),
        objects=(),
        object_links=(),
        capabilities=(),
        llm_pending_actions=(),
        tool_candidates=(),
    )
    snapshot = SimpleNamespace(
        header=SimpleNamespace(root_pid="pid-invalid-restore-cas"),
        rows=rows,
        owned_object_oids=(),
        owned_namespaces=(),
        object_payloads={},
    )

    with pytest.raises(
        ValidationError,
        match="restore process for exec backend returned a non-boolean CAS result",
    ):
        repository.restore_process_exec_snapshot(
            snapshot,  # type: ignore[arg-type]
            process_namespace="process/pid-invalid-restore-cas",
            captured_tool_ids=(),
            capability_rollback_token=None,
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_authority_repository_rejects_non_boolean_commit_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = AuthorityRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match=(
            "commit capability use reservation backend returned "
            "a non-boolean CAS result"
        ),
    ):
        repository.commit_capability_use_reservation(
            "reservation-invalid-cas",
            updated_at=utc_now(),
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("operation", ["consume", "reserve"])
@pytest.mark.parametrize("invalid_count", [True, False, 0, -1, 1.0, "1", None])
def test_authority_repository_rejects_non_integer_or_non_positive_use_counts(
    operation: str,
    invalid_count: Any,
) -> None:
    backend = _InvalidCasBackend(None)
    repository = AuthorityRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="count must be a positive integer"):
        if operation == "consume":
            repository.consume_capability_uses(
                "cap-invalid-count",
                invalid_count,  # type: ignore[arg-type]
            )
        else:
            repository.reserve_capability_uses(
                "cap-invalid-count",
                "reservation-invalid-count",
                count=invalid_count,  # type: ignore[arg-type]
                reserved_by="test",
                reason="test invalid facade count",
                created_at=utc_now(),
            )

    assert backend.transaction_entries == 0
    assert backend.mutation_visible is False


@pytest.mark.parametrize("operation", ["consume", "reserve"])
def test_authority_repository_wraps_valid_use_counts_in_a_transaction(
    operation: str,
) -> None:
    backend = _InvalidCasBackend(None)
    repository = AuthorityRepository(backend)  # type: ignore[arg-type]

    if operation == "consume":
        result = repository.consume_capability_uses("cap-valid-count", 2)
    else:
        result = repository.reserve_capability_uses(
            "cap-valid-count",
            "reservation-valid-count",
            count=2,
            reserved_by="test",
            reason="test valid facade count",
            created_at=utc_now(),
        )

    assert result is None
    assert backend.transaction_entries == 1
    assert backend.mutation_visible is True


@pytest.mark.parametrize("operation", ["consume", "reserve"])
def test_authority_repository_rejects_malformed_capability_mutation_result(
    operation: str,
) -> None:
    backend = _InvalidCasBackend("not-a-capability")
    repository = AuthorityRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(ValidationError, match="backend returned a non-capability result"):
        if operation == "consume":
            repository.consume_capability_uses("cap-invalid-result", 1)
        else:
            repository.reserve_capability_uses(
                "cap-invalid-result",
                "reservation-invalid-result",
                count=1,
                reserved_by="test",
                reason="test invalid facade result",
                created_at=utc_now(),
            )

    assert backend.transaction_entries == 1
    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_publication_repository_rejects_non_boolean_transition_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = RuntimePublicationRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="advance runtime publication backend returned a non-boolean CAS result",
    ):
        repository.advance_runtime_publication(
            "publication-invalid-cas",
            state="applying",
            phase="test",
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_checkpoint_writer_rejects_non_boolean_transition_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    writer = CheckpointRestorePublicationWriter(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match=(
            "advance checkpoint restore publication backend returned "
            "a non-boolean CAS result"
        ),
    ):
        writer.advance_runtime_publication(
            "publication-invalid-cas",
            state="reconciliation_pending",
            phase="test",
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_evidence_repository_rejects_non_boolean_update_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = EvidenceRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="update operation backend returned a non-boolean CAS result",
    ):
        repository.update_operation(object())  # type: ignore[arg-type]

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


@pytest.mark.parametrize("invalid_result", ["false", 1, None])
def test_retention_repository_rejects_non_boolean_update_cas_and_rolls_back(
    invalid_result: Any,
) -> None:
    backend = _InvalidCasBackend(invalid_result)
    repository = PayloadRetentionRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match=(
            "update LLM call payload retention backend returned "
            "a non-boolean CAS result"
        ),
    ):
        repository.update_llm_call_payload_retention(
            object(),  # type: ignore[arg-type]
            expected_payload_sha256="0" * 64,
            expected_tier=PayloadRetentionTier.FULL,
        )

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


def test_truthy_text_backend_cas_result_cannot_escape_typed_facade() -> None:
    backend = _InvalidCasBackend("false")
    assert bool(backend.result) is True

    with pytest.raises(ValidationError, match="non-boolean CAS result"):
        ProcessRepository(backend).release_execution(object())  # type: ignore[arg-type]

    assert backend.rolled_back is True
    assert backend.mutation_visible is False


def test_backend_exception_preserves_commit_before_error_reconciliation_semantics() -> None:
    class CommitThenRaiseBackend(_InvalidCasBackend):
        def release_execution(self, *_args: Any, **_kwargs: Any) -> Any:
            self.mutation_visible = True
            raise RuntimeError("injected lost acknowledgement")

    backend = CommitThenRaiseBackend(False)

    with pytest.raises(RuntimeError, match="injected lost acknowledgement"):
        ProcessRepository(backend).release_execution(object())  # type: ignore[arg-type]

    assert backend.rolled_back is False
    assert backend.mutation_visible is True


@pytest.mark.parametrize(
    ("updates", "marker", "expected"),
    [
        pytest.param(
            {"version": 2, "owner_id": "pid-transferred", "name": "renamed"},
            {"storage": "runtime_memory", "present": True},
            True,
            id="newer-live-owner-transfer",
        ),
        pytest.param(
            {"lifecycle_state": "released"},
            {"storage": "runtime_memory", "present": False},
            True,
            id="explicit-release",
        ),
        pytest.param(
            {"lifecycle_state": "released"},
            {
                "storage": "runtime_memory",
                "present": False,
                "recovered_after_reopen": True,
            },
            True,
            id="recovered-release",
        ),
        pytest.param(
            {"lifecycle_state": "released"},
            {
                "storage": "runtime_memory",
                "present": False,
                "recovered_after_reopen": "forged",
            },
            False,
            id="malformed-release-marker",
        ),
        pytest.param(
            {"owner_id": "pid-transferred"},
            {"storage": "runtime_memory", "present": True},
            False,
            id="same-version-drift",
        ),
        pytest.param(
            {"version": 2, "type": "evidence"},
            {"storage": "runtime_memory", "present": True},
            False,
            id="immutable-origin-drift",
        ),
        pytest.param(
            {"version": 2},
            {"storage": "runtime_memory", "present": False},
            False,
            id="newer-live-missing-marker",
        ),
    ],
)
def test_checkpoint_payload_supersession_is_selective_and_fail_closed(
    updates: dict[str, Any],
    marker: dict[str, Any],
    expected: bool,
) -> None:
    snapshot_row: dict[str, Any] = {
        "oid": "oid-checkpoint-payload",
        "namespace": "process:pid-owner",
        "name": "checkpoint-payload",
        "type": "summary",
        "schema_version": 1,
        "metadata_json": "{}",
        "provenance_json": "{}",
        "version": 1,
        "immutable": 0,
        "created_by": "pid-owner",
        "owner_kind": "process",
        "owner_id": "pid-owner",
        "lifecycle_state": "live",
        "deleted_at": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    current = {**snapshot_row, **updates}
    assert (
        _checkpoint_object_payload_was_superseded(
            current,
            snapshot_row,
            current_marker=marker,
            present_marker={"storage": "runtime_memory", "present": True},
        )
        is expected
    )


def test_unit_of_work_composes_typed_repositories() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)

        assert isinstance(unit.processes, ProcessRepository)
        assert isinstance(unit.objects, ObjectRepository)
        assert isinstance(unit.authority, AuthorityRepository)
        assert isinstance(unit.resources, ResourceRepository)
        assert isinstance(unit.publications, RuntimePublicationRepository)
        assert isinstance(unit.snapshots, SnapshotCheckpointRepository)
        assert isinstance(unit.evidence, EvidenceRepository)
        assert isinstance(unit.extensions, ExtensionRepository)
        assert isinstance(unit.module_publications, RuntimeModuleRepository)
        with unit.transaction(include_object_payloads=True) as active:
            assert active is unit
        with pytest.raises(AttributeError, match="no repository operation"):
            unit.processes.get_object("oid")
    finally:
        store.close()


@pytest.mark.parametrize("operation", ["insert", "claim"])
def test_checkpoint_restore_writer_rejects_malformed_backend_records(
    operation: str,
) -> None:
    malformed = {"publication_id": "publication_expected"}
    writer = CheckpointRestorePublicationWriter(
        _CheckpointPublicationBackend(
            insert_result=malformed,
            claim_result=malformed,
        )
    )

    with pytest.raises(
        ValidationError,
        match="invalid checkpoint restore publication writer record",
    ):
        if operation == "insert":
            writer.insert_runtime_publication(
                publication_id="publication_expected",
                kind="checkpoint_restore",
                pid="pid_checkpoint_writer",
                owner_instance_id="runtime.test",
                plan={"checkpoint_id": "checkpoint_writer"},
            )
        else:
            writer.claim_runtime_publication_recovery(
                "publication_expected",
                claimant_instance_id="runtime.test",
                expected_owner_instance_id="runtime.previous",
                expected_state="reconciliation_pending",
                classification="restart",
            )


@pytest.mark.parametrize("operation", ["insert", "claim"])
def test_checkpoint_restore_writer_rejects_miswired_backend_record_ids(
    operation: str,
) -> None:
    wrong_record = _publication_record("publication_other")
    writer = CheckpointRestorePublicationWriter(
        _CheckpointPublicationBackend(
            insert_result=wrong_record,
            claim_result=wrong_record,
        )
    )

    with pytest.raises(
        ValidationError,
        match=(
            "returned publication 'publication_other' for request "
            "'publication_expected'"
        ),
    ):
        if operation == "insert":
            writer.insert_runtime_publication(
                publication_id="publication_expected",
                kind="checkpoint_restore",
                pid="pid_checkpoint_writer",
                owner_instance_id="runtime.test",
                plan={"checkpoint_id": "checkpoint_writer"},
            )
        else:
            writer.claim_runtime_publication_recovery(
                "publication_expected",
                claimant_instance_id="runtime.test",
                expected_owner_instance_id="runtime.previous",
                expected_state="reconciliation_pending",
                classification="restart",
            )


def test_publication_repository_rejects_miswired_insert_and_rolls_back() -> None:
    backend = _MiswiredPublicationBackend()
    repository = RuntimePublicationRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="returned publication 'publication_other' for request 'publication_expected'",
    ):
        repository.insert_runtime_publication(
            publication_id="publication_expected",
            kind="checkpoint_restore",
            pid="pid_checkpoint_writer",
            owner_instance_id="runtime.test",
            plan={"checkpoint_id": "checkpoint_writer"},
        )

    assert backend.rolled_back is True


def test_publication_repository_rejects_miswired_get() -> None:
    backend = _MiswiredPublicationBackend()
    backend.get_runtime_publication = lambda _publication_id: deepcopy(  # type: ignore[method-assign]
        backend.records["publication_other"]
    )
    repository = RuntimePublicationRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="returned publication 'publication_other' for request 'publication_expected'",
    ):
        repository.get_runtime_publication("publication_expected")


def test_publication_repository_rolls_back_miswired_claim_before_cleanup() -> None:
    backend = _MiswiredPublicationBackend()
    before = deepcopy(backend.records)
    repository = RuntimePublicationRepository(backend)  # type: ignore[arg-type]
    cleanup_calls = 0

    with pytest.raises(
        ValidationError,
        match="returned publication 'publication_other' for request 'publication_expected'",
    ):
        claimed = repository.claim_runtime_publication_recovery(
            "publication_expected",
            claimant_instance_id="runtime.claimant",
            expected_owner_instance_id="runtime.test",
            expected_state="planning",
            classification="compensate_process_launch",
        )
        if claimed is not None:  # pragma: no cover - explicit side-effect fence
            cleanup_calls += 1

    assert cleanup_calls == 0
    assert backend.rolled_back is True
    assert backend.records == before


def _persisted_checkpoint_pair(
    reason: str,
) -> tuple[Any, dict[str, Any]]:
    runtime = Runtime.open(":memory:")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal=reason)
        checkpoint_id = runtime.checkpoint.create(pid, reason, actor=pid)
        found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert found is not None
        return found
    finally:
        runtime.close()


def test_checkpoint_repository_rejects_backend_a_to_b_binding() -> None:
    checkpoint_b, snapshot_b = _persisted_checkpoint_pair("checkpoint B")
    backend = _CheckpointBindingBackend((checkpoint_b, snapshot_b))
    repository = SnapshotCheckpointRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="returned checkpoint .* for request 'checkpoint_A'",
    ):
        repository.get_checkpoint_snapshot("checkpoint_A")


def test_checkpoint_repository_rejects_same_id_header_drift() -> None:
    checkpoint, snapshot = _persisted_checkpoint_pair("original reason")
    snapshot["reason"] = "drifted reason"
    backend = _CheckpointBindingBackend((checkpoint, snapshot))
    repository = SnapshotCheckpointRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="checkpoint snapshot changed: .*reason",
    ):
        repository.get_checkpoint_snapshot(checkpoint.checkpoint_id)


def test_checkpoint_repository_rejects_mismatch_before_insert_side_effect() -> None:
    checkpoint, snapshot = _persisted_checkpoint_pair("insert binding")
    snapshot["checkpoint_id"] = "checkpoint_other"
    backend = _CheckpointBindingBackend(None)
    repository = SnapshotCheckpointRepository(backend)  # type: ignore[arg-type]

    with pytest.raises(
        ValidationError,
        match="checkpoint snapshot changed: .*checkpoint_id",
    ):
        repository.insert_checkpoint(
            checkpoint,
            ProcessSnapshot.from_mapping(snapshot),
        )

    assert backend.insert_calls == 0


def test_unit_of_work_rejects_boundary_only_dynamic_backend_fail_fast() -> None:
    store = _RecordingStore()

    with pytest.raises(
        TypeError,
        match=r"UnitOfWork backend contract violation: .*process\.insert_process",
    ):
        UnitOfWork(store)

    assert store.calls == []


def test_unit_of_work_rejects_backend_signature_drift_fail_fast() -> None:
    class SignatureDriftStore(SQLiteStore):
        def claim_execution(self, pid: str, *, claimant_id: str):
            return super().claim_execution(pid, owner_id=claimant_id)

    store = SignatureDriftStore(":memory:")
    try:
        with pytest.raises(
            TypeError,
            match=r"process\.claim_execution: backend .* rejects protocol",
        ):
            UnitOfWork(store)
    finally:
        store.close()


def test_unit_of_work_rejects_missing_payload_retention_method_fail_fast() -> None:
    def protocol_stub(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs

    methods = {
        name: protocol_stub
        for name, value in inspect.getmembers(
            UnitOfWorkBackendProtocol,
            predicate=inspect.isfunction,
        )
        if not name.startswith("__")
    }
    methods.pop("scan_llm_call_payloads_for_retention")
    backend_type = type("MissingPayloadRetentionBackend", (), methods)

    with pytest.raises(
        TypeError,
        match=(
            r"UnitOfWork backend contract violation: .*payload-retention\."
            r"scan_llm_call_payloads_for_retention: missing concrete method"
        ),
    ):
        UnitOfWork(backend_type())  # type: ignore[arg-type]


def test_unit_of_work_rejects_missing_effect_ledger_reader_fail_fast() -> None:
    def protocol_stub(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs

    methods = {
        name: protocol_stub
        for name, value in inspect.getmembers(
            UnitOfWorkBackendProtocol,
            predicate=inspect.isfunction,
        )
        if not name.startswith("__")
    }
    methods.pop("current_effect_ledger_seq")
    backend_type = type("MissingEffectLedgerReaderBackend", (), methods)

    with pytest.raises(
        TypeError,
        match=(
            r"UnitOfWork backend contract violation: .*operation-evidence\."
            r"current_effect_ledger_seq: missing concrete method"
        ),
    ):
        UnitOfWork(backend_type())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("method_name", "surface"),
    [
        pytest.param(
            "get_persisted_object_state",
            "object-recovery",
            id="persisted-object-state",
        ),
        pytest.param(
            "list_capabilities",
            "authority-recovery",
            id="list-capabilities",
        ),
        pytest.param("get_tool_spec", "tool-artifact", id="get-tool-spec"),
        pytest.param(
            "get_tool_candidate",
            "tool-artifact",
            id="get-tool-candidate",
        ),
    ],
)
def test_unit_of_work_rejects_missing_security_projection_method_fail_fast(
    method_name: str,
    surface: str,
) -> None:
    def protocol_stub(self: object, *args: object, **kwargs: object) -> None:
        del self, args, kwargs

    methods = {
        name: protocol_stub
        for name, value in inspect.getmembers(
            UnitOfWorkBackendProtocol,
            predicate=inspect.isfunction,
        )
        if not name.startswith("__")
    }
    methods.pop(method_name)
    backend_type = type("MissingSecurityProjectionBackend", (), methods)

    with pytest.raises(
        TypeError,
        match=(
            rf"UnitOfWork backend contract violation: .*{surface}\."
            rf"{method_name}: missing concrete method"
        ),
    ):
        UnitOfWork(backend_type())  # type: ignore[arg-type]


def test_sql_runtime_store_statically_conforms_to_typed_uow_backend() -> None:
    assert unit_of_work_backend_conformance_errors(SQLRuntimeStore) == ()


def test_process_resource_and_publication_methods_are_explicit_facades() -> None:
    process_methods = {
        "insert_process",
        "get_process",
        "get_human_request",
        "get_object_task",
        "list_object_tasks",
        "list_processes_by_status",
        "patch_process",
        "claim_execution",
        "delete_process_scaffold",
    }
    publication_methods = {
        "insert_runtime_publication",
        "get_runtime_publication",
        "list_runtime_publications",
        "claim_runtime_publication_recovery",
        "advance_runtime_publication",
        "update_runtime_publication_plan",
        "record_runtime_publication_artifact",
    }
    operation_methods = {
        "list_events",
        "insert_operation",
        "get_operation",
        "list_operations",
        "scan_stale_running_operations",
        "stale_operation_recovery_index",
        "operation_ids_with_unknown_external_effects",
        "operation_has_unknown_external_effect",
        "update_operation",
        "insert_operation_evidence",
        "list_operation_evidence",
        "list_operation_ids_by_runtime_publication_id",
        "get_external_effect",
        "current_effect_ledger_seq",
        "list_external_effects_changed_after",
    }
    authority_mutation_methods = {
        "consume_capability_uses",
        "reserve_capability_uses",
        "commit_capability_use_reservation",
    }

    assert process_methods.isdisjoint(ProcessRepository._METHODS)
    assert publication_methods.isdisjoint(ProcessRepository._METHODS)
    assert process_methods <= ProcessRepository.__dict__.keys()
    assert publication_methods <= RuntimePublicationRepository.__dict__.keys()
    assert {
        "get_resource_usage_reservation",
        "list_resource_usage_reservations",
        "settle_resource_usage_reservation",
        "list_resource_reservations",
    } <= ResourceRepository.__dict__.keys()
    assert operation_methods.isdisjoint(EvidenceRepository._METHODS)
    assert operation_methods <= EvidenceRepository.__dict__.keys()
    assert authority_mutation_methods.isdisjoint(AuthorityRepository._METHODS)
    assert authority_mutation_methods <= AuthorityRepository.__dict__.keys()


def test_mutating_backend_bool_protocol_inventory_is_exact_and_transactional() -> None:
    protocols = (
        ProcessBackendProtocol,
        ResourceBackendProtocol,
        RuntimePublicationBackendProtocol,
        CheckpointPublicationWriterBackendProtocol,
        SnapshotCheckpointBackendProtocol,
        OperationEvidenceBackendProtocol,
        PayloadRetentionStore,
        CapabilityStorePort,
    )
    declared_bool_methods = {
        name
        for protocol in protocols
        for name, method in inspect.getmembers(protocol, predicate=inspect.isfunction)
        if inspect.signature(method).return_annotation in {bool, "bool"}
    }
    read_only_bool_methods = {
        "external_effect_transition_matches",
        "has_object_payload",
        "namespace_exists",
        "operation_has_unknown_external_effect",
        "runtime_publication_exists_for_pid",
        "tool_id_referenced_outside_process",
        "tool_id_referenced_outside_scope",
    }
    mutating_bool_methods = {
        "abort_checkpoint_payload_delivery_attempt",
        "ack_checkpoint_payload_delivery_attempt",
        "ack_process_message",
        "advance_runtime_publication",
        "begin_checkpoint_payload_delivery_attempt",
        "commit_capability_use_reservation",
        "complete_execution",
        "delete_tool_if_unreferenced",
        "insert_operation_evidence",
        "mark_runtime_publication_operation_reconciled",
            "record_runtime_publication_artifact",
            "redact_root_spawn_initial_goal",
            "release_execution",
        "restore_process_for_exec",
        "settle_resource_usage_reservation",
        "transition_checkpoint_restore_payload_delivery",
        "update_external_effect_payload_retention",
        "update_llm_call_payload_retention",
        "update_operation",
        "update_runtime_publication_plan",
    }
    assert declared_bool_methods == read_only_bool_methods | mutating_bool_methods

    facade_inventory = (
        (ProcessRepository, "ack_process_message"),
        (ProcessRepository, "complete_execution"),
        (ProcessRepository, "release_execution"),
        (AuthorityRepository, "commit_capability_use_reservation"),
        (ResourceRepository, "settle_resource_usage_reservation"),
        (RuntimePublicationRepository, "mark_runtime_publication_operation_reconciled"),
        (RuntimePublicationRepository, "advance_runtime_publication"),
        (RuntimePublicationRepository, "update_runtime_publication_plan"),
        (RuntimePublicationRepository, "record_runtime_publication_artifact"),
        (CheckpointRestorePublicationWriter, "advance_runtime_publication"),
        (CheckpointRestorePublicationWriter, "record_runtime_publication_artifact"),
        (
            CheckpointRestorePublicationWriter,
            "mark_runtime_publication_operation_reconciled",
        ),
        (CheckpointRestorePublicationWriter, "transition_payload_delivery"),
        (
            CheckpointRestorePublicationWriter,
            "begin_checkpoint_payload_delivery_attempt",
        ),
        (
            CheckpointRestorePublicationWriter,
            "ack_checkpoint_payload_delivery_attempt",
        ),
        (
            CheckpointRestorePublicationWriter,
            "abort_checkpoint_payload_delivery_attempt",
        ),
        (SnapshotCheckpointRepository, "restore_process_exec_snapshot"),
        (SnapshotCheckpointRepository, "delete_tool_if_unreferenced"),
        (EvidenceRepository, "update_operation"),
        (EvidenceRepository, "insert_operation_evidence"),
        (PayloadRetentionRepository, "update_llm_call_payload_retention"),
        (PayloadRetentionRepository, "update_external_effect_payload_retention"),
    )
    for repository, method_name in facade_inventory:
        source = inspect.getsource(getattr(repository, method_name))
        if (
            repository is SnapshotCheckpointRepository
            and method_name == "restore_process_exec_snapshot"
        ):
            assert "_exact_backend_cas_result" in source
            assert "with self.transaction(" in source
        else:
            assert "_transactional_backend_cas_result" in source, (
                repository.__name__,
                method_name,
            )


def test_checkpoint_effect_ledger_reads_use_the_typed_evidence_facade() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        ledger_seq = unit.evidence.current_effect_ledger_seq()

        assert isinstance(ledger_seq, int)
        assert unit.evidence.list_external_effects_changed_after(ledger_seq) == []
    finally:
        store.close()


def test_checkpoint_human_cancellation_preserves_a_competing_terminal_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        requests = (
            HumanRequest(
                request_id="human-before-checkpoint",
                pid="pid-human-cas",
                human="operator",
                payload={"question": "old"},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-01-01T00:00:00Z",
            ),
            HumanRequest(
                request_id="human-competing-approval",
                pid="pid-human-cas",
                human="operator",
                payload={"question": "approve"},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at="2026-01-03T00:00:00Z",
                updated_at="2026-01-03T00:00:00Z",
            ),
            HumanRequest(
                request_id="human-still-pending",
                pid="pid-human-cas",
                human="operator",
                payload={"question": "cancel"},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at="2026-01-04T00:00:00Z",
                updated_at="2026-01-04T00:00:00Z",
            ),
        )
        for request in requests:
            store.insert_human_request(request)

        original_list = store.list_human_requests

        def list_then_approve(*args: Any, **kwargs: Any) -> list[HumanRequest]:
            selected = original_list(*args, **kwargs)
            pending = next(
                request
                for request in selected
                if request.request_id == "human-competing-approval"
            )
            store.update_human_request(
                HumanRequest(
                    request_id=pending.request_id,
                    pid=pending.pid,
                    human=pending.human,
                    payload=pending.payload,
                    status=HumanRequestStatus.APPROVED,
                    decision={"approved": True},
                    blocking=pending.blocking,
                    created_at=pending.created_at,
                    updated_at="2026-01-05T00:00:00Z",
                )
            )
            persisted = store.get_human_request(pending.request_id)
            assert persisted is not None
            assert persisted.status is HumanRequestStatus.APPROVED
            return selected

        monkeypatch.setattr(store, "list_human_requests", list_then_approve)
        cancelled = unit.snapshots.cancel_pending_human_requests_after_checkpoint(
            ("pid-human-cas",),
            Checkpoint(
                checkpoint_id="checkpoint-human-cas",
                pid="pid-human-cas",
                reason="test",
                created_at="2026-01-02T00:00:00Z",
            ),
        )

        assert cancelled == ["human-still-pending"]
        approved = store.get_human_request("human-competing-approval")
        assert approved is not None
        assert approved.status is HumanRequestStatus.APPROVED
        assert approved.decision == {"approved": True}
        cancelled_request = store.get_human_request("human-still-pending")
        assert cancelled_request is not None
        assert cancelled_request.status is HumanRequestStatus.CANCELLED
        assert cancelled_request.decision == {
            "cancelled_by": "checkpoint:checkpoint-human-cas"
        }
        before_checkpoint = store.get_human_request("human-before-checkpoint")
        assert before_checkpoint is not None
        assert before_checkpoint.status is HumanRequestStatus.PENDING
        assert before_checkpoint.decision is None
    finally:
        store.close()


def _insert_persisted_projection_object(
    store: SQLiteStore,
    *,
    oid: str,
    payload_json: str,
) -> None:
    now = utc_now()
    with store.transaction() as cursor:
        cursor.execute(
            "INSERT INTO objects ("
            "oid, namespace, name, type, schema_version, payload_json, "
            "metadata_json, provenance_json, version, immutable, created_by, "
            "owner_kind, owner_id, lifecycle_state, deleted_at, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                oid,
                "system",
                oid,
                "summary",
                "1",
                payload_json,
                "{}",
                "{}",
                1,
                0,
                "test",
                "process",
                "pid-projection",
                "live",
                None,
                now,
                now,
            ),
        )


def _insert_object_ref_projection(
    store: SQLiteStore,
    *,
    oid: str,
    namespace: str,
    created_at: str,
    updated_at: str,
    lifecycle_state: str = "live",
) -> None:
    with store.transaction() as cursor:
        cursor.execute(
            "INSERT INTO objects ("
            "oid, namespace, name, type, schema_version, payload_json, "
            "metadata_json, provenance_json, version, immutable, created_by, "
            "owner_kind, owner_id, lifecycle_state, deleted_at, created_at, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                oid,
                namespace,
                f"name-{oid}",
                "summary",
                "1",
                dumps({"storage": "runtime_memory", "present": True}),
                dumps(
                    {
                        "title": f"title-{oid}",
                        "summary": f"summary-{oid}",
                        "tags": ["page", oid],
                    }
                ),
                "{}",
                1,
                0,
                "test",
                "process",
                "pid-object-query",
                lifecycle_state,
                None,
                created_at,
                updated_at,
            ),
        )


def test_object_ref_pages_are_payload_free_stable_and_hard_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    namespace = "process:pid-object-query"
    try:
        rows = (
            ("oid-a", "2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            ("oid-b", "2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z"),
            ("oid-c", "2026-03-01T00:00:00Z", "2026-04-01T00:00:00Z"),
            ("oid-d", "2026-02-01T00:00:00Z", "2026-04-01T00:00:00Z"),
            ("oid-e", "2026-01-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        )
        for oid, created_at, updated_at in rows:
            _insert_object_ref_projection(
                store,
                oid=oid,
                namespace=namespace,
                created_at=created_at,
                updated_at=updated_at,
            )
        _insert_object_ref_projection(
            store,
            oid="oid-other-namespace",
            namespace="process:other",
            created_at="2026-06-01T00:00:00Z",
            updated_at="2026-06-01T00:00:00Z",
        )
        _insert_object_ref_projection(
            store,
            oid="oid-released",
            namespace=namespace,
            created_at="2026-07-01T00:00:00Z",
            updated_at="2026-07-01T00:00:00Z",
            lifecycle_state="released",
        )

        def reject_payload_access(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("Object reference page accessed a payload")

        monkeypatch.setattr(store, "object_payload", reject_payload_access)
        monkeypatch.setattr(store, "has_object_payload", reject_payload_access)
        unit = UnitOfWork(store)

        observed: list[str] = []
        cursor: ObjectRefCursor | None = None
        page_sizes: list[int] = []
        while True:
            page = unit.objects.query_object_refs_page(
                namespace,
                after=cursor,
                limit=2,
            )
            page_sizes.append(len(page.records))
            observed.extend(record.oid for record in page.records)
            assert all("storage" not in record.search_text for record in page.records)
            if page.exhausted:
                assert page.next_cursor is None
                break
            assert page.next_cursor == page.records[-1].cursor
            cursor = page.next_cursor

        assert observed == ["oid-a", "oid-b", "oid-c", "oid-d", "oid-e"]
        assert page_sizes == [2, 2, 1]
    finally:
        store.close()


@pytest.mark.parametrize(
    "limit",
    [True, 0, -1, 1.5],
)
def test_object_ref_page_rejects_invalid_limits(limit: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="limit"):
            store.query_object_refs_page(
                "system",
                after=None,
                limit=limit,  # type: ignore[arg-type]
            )
    finally:
        store.close()


def test_object_ref_page_rejects_limit_above_configured_hard_cap() -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="hard cap"):
            store.query_object_refs_page(
                "system",
                after=None,
                limit=store.config.memory.query_scan_page_size + 1,
            )
    finally:
        store.close()


def test_child_namespace_pages_are_stable_direct_and_hard_bounded() -> None:
    store = SQLiteStore(":memory:")
    try:
        unit = UnitOfWork(store)
        parent = "process:parent"
        now = utc_now()
        for namespace in (
            "process:parent/zeta",
            "process:parent/alpha",
            "process:parent/middle",
        ):
            unit.objects.insert_namespace(
                ObjectNamespace(
                    namespace=namespace,
                    parent_namespace=parent,
                    metadata={"namespace": namespace},
                    created_by="test",
                    created_at=now,
                    updated_at=now,
                )
            )
        unit.objects.insert_namespace(
            ObjectNamespace(
                namespace="process:other/child",
                parent_namespace="process:other",
                metadata={},
                created_by="test",
                created_at=now,
                updated_at=now,
            )
        )

        first = unit.objects.query_child_namespaces_page(
            parent,
            after=None,
            limit=2,
        )
        assert [record.namespace for record in first.records] == [
            "process:parent/alpha",
            "process:parent/middle",
        ]
        assert first.next_cursor is not None
        assert first.next_cursor.namespace == "process:parent/middle"
        assert not first.exhausted

        second = unit.objects.query_child_namespaces_page(
            parent,
            after=first.next_cursor,
            limit=2,
        )
        assert [record.namespace for record in second.records] == [
            "process:parent/zeta"
        ]
        assert second.exhausted
        assert second.next_cursor is None
    finally:
        store.close()


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_child_namespace_page_rejects_invalid_limits(limit: object) -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="limit"):
            store.query_child_namespaces_page(
                "system",
                after=None,
                limit=limit,  # type: ignore[arg-type]
            )
    finally:
        store.close()


def test_child_namespace_page_rejects_limit_above_scan_page_cap() -> None:
    store = SQLiteStore(":memory:")
    try:
        with pytest.raises(ValidationError, match="hard cap"):
            store.query_child_namespaces_page(
                "system",
                after=None,
                limit=store.config.memory.query_scan_page_size + 1,
            )
    finally:
        store.close()


def test_persisted_object_state_projection_does_not_fill_payload_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        oid = "oid-projection-legacy-payload"
        _insert_persisted_projection_object(
            store,
            oid=oid,
            payload_json=dumps({"secret": "durable legacy payload"}),
        )
        assert oid not in store._object_payloads

        def reject_cache_write(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("security projection populated Object payload cache")

        monkeypatch.setattr(store, "_set_cached_object_payload", reject_cache_write)
        state = store.get_persisted_object_state(oid)

        assert state is not None
        assert state.payload_present is True
        assert state.recovered_after_reopen is False
        assert oid not in store._object_payloads
    finally:
        store.close()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        pytest.param("version", 0, id="zero-version"),
        pytest.param("lifecycle_state", "corrupt", id="invalid-lifecycle"),
        pytest.param(
            "payload_json",
            dumps({"storage": "runtime_memory", "present": "yes"}),
            id="malformed-runtime-marker",
        ),
    ],
)
def test_persisted_object_state_projection_rejects_corrupt_rows(
    column: str,
    value: object,
) -> None:
    store = SQLiteStore(":memory:")
    try:
        oid = f"oid-projection-corrupt-{column}"
        _insert_persisted_projection_object(
            store,
            oid=oid,
            payload_json=dumps({"storage": "runtime_memory", "present": False}),
        )
        with store.transaction() as cursor:
            cursor.execute(
                f"UPDATE objects SET {column} = ? WHERE oid = ?",
                (value, oid),
            )

        with pytest.raises(ValidationError, match="invalid persisted Object state"):
            store.get_persisted_object_state(oid)
    finally:
        store.close()


def _first_cursor_execute_sql(method: object) -> str:
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "execute"
            and node.args
        ):
            value = ast.literal_eval(node.args[0])
            if isinstance(value, str):
                return value
    raise AssertionError("method has no static cursor.execute SQL")


def test_payload_delivery_control_cas_uses_exact_guard_index_plans() -> None:
    store = SQLiteStore(":memory:")
    try:
        ack_sql = _first_cursor_execute_sql(
            SQLRuntimeStore.ack_checkpoint_payload_delivery_attempt
        )
        ack_params = (
            "now",
            "now",
            "attempt",
            "owner",
            "started",
            "attempt",
            "owner",
            "attempt",
            "attempt",
            "owner",
            "attempt",
            "owner",
            "attempt",
            "owner",
        )
        ack_plan = store.conn.execute(
            f"EXPLAIN QUERY PLAN {ack_sql}",
            ack_params,
        ).fetchall()
        ack_details = "\n".join(str(row["detail"]) for row in ack_plan)
        assert ack_details.count(
            "idx_runtime_publications_payload_delivery_guard"
        ) == 5
        assert "SCAN runtime_publications" not in ack_details
        assert "USE TEMP B-TREE" not in ack_details

        abort_sql = _first_cursor_execute_sql(
            SQLRuntimeStore.abort_checkpoint_payload_delivery_attempt
        )
        abort_plan = store.conn.execute(
            f"EXPLAIN QUERY PLAN {abort_sql}",
            ("now", "attempt", "owner", "started", "attempt"),
        ).fetchall()
        abort_details = "\n".join(str(row["detail"]) for row in abort_plan)
        assert "idx_runtime_publications_payload_delivery_guard" in abort_details
        assert "SCAN runtime_publications" not in abort_details
        assert "USE TEMP B-TREE" not in abort_details

        page_plan = store.conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM runtime_publications "
            "INDEXED BY idx_runtime_publications_payload_delivery_attempt "
            "WHERE kind = 'checkpoint_restore' AND state = 'committed' "
            "AND phase = 'reconciled' "
            "AND payload_delivery_attempt_id IS NOT NULL "
            "AND payload_delivery_attempt_id = ? "
            "AND payload_delivery_state = ? "
            "ORDER BY created_at COLLATE BINARY, publication_id COLLATE BINARY "
            "LIMIT ?",
            ("attempt", "completed", 3),
        ).fetchall()
        page_details = "\n".join(str(row["detail"]) for row in page_plan)
        assert "idx_runtime_publications_payload_delivery_attempt" in page_details
        assert "USE TEMP B-TREE" not in page_details
    finally:
        store.close()


def test_postgres_payload_delivery_control_cas_preserves_exact_predicates() -> None:
    ack_sql = _first_cursor_execute_sql(
        SQLRuntimeStore.ack_checkpoint_payload_delivery_attempt
    )
    prepared_ack = _PostgresDialect().prepare(ack_sql, with_params=True)
    assert "INDEXED BY" not in prepared_ack
    assert prepared_ack.count("%s") == 14
    assert prepared_ack.count("payload_delivery_attempt_id = %s") == 5
    assert prepared_ack.count("kind = 'checkpoint_restore'") == 5
    assert prepared_ack.count("state = 'committed'") == 5
    assert prepared_ack.count("phase = 'reconciled'") == 5

    abort_sql = _first_cursor_execute_sql(
        SQLRuntimeStore.abort_checkpoint_payload_delivery_attempt
    )
    prepared_abort = _PostgresDialect().prepare(abort_sql, with_params=True)
    assert "INDEXED BY" not in prepared_abort
    assert prepared_abort.count("%s") == 5
    assert "payload_delivery_attempt_id = %s" in prepared_abort


def test_publication_artifact_cleanup_methods_are_explicit_facades() -> None:
    methods_by_repository = {
        ObjectRepository: {
            "get_object",
            "get_persisted_object_state",
            "get_namespace",
            "has_object_payload",
            "namespace_exists",
            "object_payload",
            "payload_marker",
            "get_object_by_name",
            "list_namespaces_created_by",
            "list_objects_owned_by",
        },
        AuthorityRepository: {
            "get_capability",
            "list_capabilities",
            "delete_publication_capability",
        },
        ExtensionRepository: {
            "delete_tool",
            "get_image_artifact",
            "get_existing_tool_ids",
            "insert_image_artifact",
            "list_tools",
            "get_tool_candidate",
            "get_tool_spec",
            "upsert_image",
        },
    }

    for repository, methods in methods_by_repository.items():
        assert methods.isdisjoint(repository._METHODS)
        assert methods <= repository.__dict__.keys()


def test_image_publication_services_do_not_retain_raw_host_store() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "agent_libos/runtime/image_boot.py",
        "agent_libos/runtime/checkpoint_image.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ]
        assert constructors
        assert all(
            argument.arg != "store"
            for constructor in constructors
            for argument in (*constructor.args.args, *constructor.args.kwonlyargs)
        )
        assert "self._store" not in source


def test_snapshot_checkpoint_artifact_methods_are_explicit_typed_facades() -> None:
    artifact_methods = {
        "insert_checkpoint",
        "get_checkpoint_snapshot",
        "list_checkpoints",
        "get_jit_rehydration_artifacts",
        "registered_jit_tool_ids_for_processes",
    }

    assert artifact_methods.isdisjoint(ExtensionRepository._METHODS)
    assert artifact_methods <= SnapshotCheckpointRepository.__dict__.keys()


def test_public_bulk_restore_epoch_contract_does_not_expose_backend_cursor() -> None:
    public_parameters = inspect.signature(
        ProcessStateRepository.reserve_process_restore_epochs
    ).parameters
    repository_parameters = inspect.signature(
        ProcessRepository.reserve_process_restore_epochs
    ).parameters
    backend_parameters = inspect.signature(
        SnapshotCheckpointBackendProtocol.reserve_process_restore_epochs
    ).parameters
    implementation_parameters = inspect.signature(
        SQLRuntimeStore.reserve_process_restore_epochs
    ).parameters

    assert tuple(public_parameters) == ("self", "floors")
    assert tuple(repository_parameters) == ("self", "floors")
    assert tuple(backend_parameters) == ("self", "floors", "cursor")
    assert tuple(implementation_parameters) == ("self", "floors", "cursor")
    assert backend_parameters["cursor"].kind is inspect.Parameter.KEYWORD_ONLY
    assert implementation_parameters["cursor"].kind is inspect.Parameter.KEYWORD_ONLY


def test_runtime_module_publication_methods_are_explicit_typed_facades() -> None:
    methods = {
        "upsert_runtime_module",
        "get_runtime_module",
        "list_runtime_modules",
    }

    assert methods.isdisjoint(ExtensionRepository._METHODS)
    assert methods.isdisjoint(RuntimeModuleRepository._METHODS)
    assert methods <= RuntimeModuleRepository.__dict__.keys()


def test_runtime_module_model_round_trips_the_public_dictionary_contract() -> None:
    module = RuntimeModule(
        module_id="typed-module:v0",
        name="Typed module",
        version="v0",
        entrypoint="typed.module:register",
        manifest_path="/modules/typed/module.yaml",
        manifest_sha256="1" * 64,
        source_path="/modules/typed/module.py",
        source_sha256="2" * 64,
        status=RuntimeModuleStatus.LOADED,
        loaded_at="2026-01-01T00:00:00Z",
        registered=RuntimeModuleRegistration(
            tools=("typed_tool",),
            provider_hooks={"demo": 1},
            startup_hooks=("start",),
        ),
        metadata={"owner": "test"},
        updated_at="2026-01-01T00:00:01Z",
    )

    restored = RuntimeModule.from_persisted(module.to_public_dict())

    assert restored == module
    assert restored.to_public_dict()["registered"] == {
        "tools": ["typed_tool"],
        "images": [],
        "syscalls": [],
        "provider_hooks": {"demo": 1},
        "startup_hooks": ["start"],
        "durable_object_release_finalizers": [],
    }


def test_runtime_module_model_rejects_invalid_publication_states() -> None:
    with pytest.raises(ValueError, match="requires loaded_at"):
        RuntimeModule(
            module_id="invalid-loaded:v0",
            name="Invalid",
            version="v0",
            entrypoint="invalid:register",
            manifest_path="/invalid/module.yaml",
            manifest_sha256="1" * 64,
            source_path="/invalid/module.py",
            source_sha256="2" * 64,
            status=RuntimeModuleStatus.LOADED,
            loaded_at=None,
        )

    with pytest.raises(ValueError, match="non-negative integer"):
        RuntimeModuleRegistration(provider_hooks={"demo": True})


def test_runtime_module_repository_rejects_invalid_persisted_records() -> None:
    class InvalidModuleStore(_RecordingStore):
        def get_runtime_module(self, module_id: str) -> dict[str, object]:
            return {
                "module_id": module_id,
                "name": "Invalid",
                "version": "v0",
                "entrypoint": "invalid:register",
                "manifest_path": "/invalid/module.yaml",
                "manifest_sha256": "1" * 64,
                "source_path": "/invalid/module.py",
                "source_sha256": "2" * 64,
                "status": "unknown",
                "loaded_at": None,
                "registered": {},
                "error": None,
                "metadata": {},
                "updated_at": "2026-01-01T00:00:00Z",
            }

    repository = RuntimeModuleRepository(InvalidModuleStore())

    with pytest.raises(ValidationError, match="invalid persisted runtime module"):
        repository.get_runtime_module("invalid:v0")


def test_runtime_module_registry_cannot_regress_to_extension_any_dispatch() -> None:
    registry_path = (
        Path(__file__).resolve().parents[2]
        / "agent_libos/modules/registry.py"
    )
    tree = ast.parse(
        registry_path.read_text(encoding="utf-8"),
        filename=str(registry_path),
    )
    migrated_methods = {
        "upsert_runtime_module",
        "get_runtime_module",
        "list_runtime_modules",
    }
    untyped_bypasses: list[int] = []
    typed_calls: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in migrated_methods
        ):
            owner = node.args[0]
            if not (
                isinstance(owner, ast.Attribute)
                and isinstance(owner.value, ast.Name)
                and owner.value.id == "self"
                and owner.attr == "_module_publications"
            ):
                untyped_bypasses.append(node.lineno)
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        typed_owner = (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
            and owner.attr == "_module_publications"
        )
        if node.func.attr in migrated_methods and not typed_owner:
            untyped_bypasses.append(node.lineno)
        if typed_owner:
            typed_calls.add(node.func.attr)

    assert untyped_bypasses == []
    assert migrated_methods <= typed_calls


def test_typed_resource_and_publication_contracts_use_one_sqlite_uow() -> None:
    store = SQLiteStore(":memory:")
    unit = UnitOfWork(store)
    try:
        now = utc_now()
        unit.processes.insert_process(_process("pid_vertical"))
        unit.resources.insert_resource_usage_reservation(
            reservation_id="reservation_vertical",
            pid="pid_vertical",
            usage=ResourceUsage(tool_calls=2),
            reserved_by="effect_vertical",
            reason="contract",
            created_at=now,
        )
        publication = unit.publications.insert_runtime_publication(
            publication_id="publication_vertical",
            kind="process_launch",
            pid="pid_vertical",
            owner_instance_id="runtime.test",
            plan={"pid": "pid_vertical"},
        )

        reservation = unit.resources.get_resource_usage_reservation(
            "reservation_vertical"
        )
        assert isinstance(reservation, ResourceUsageReservation)
        assert reservation.usage.tool_calls == 2
        assert publication["publication_id"] == "publication_vertical"
        assert unit.publications.get_runtime_publication(
            "publication_vertical"
        ) == publication

        cleanup = unit.processes.delete_process_scaffold(
            "pid_vertical",
            namespace="process/pid_vertical",
            namespace_resource="object_namespace:process/pid_vertical",
        )
        assert cleanup.deleted_by_table["processes"] == 1
        assert unit.processes.get_process("pid_vertical") is None
    finally:
        store.close()


def test_repositories_share_one_outer_transaction_atomically() -> None:
    store = SQLiteStore(":memory:")
    unit = UnitOfWork(store)
    namespace = ObjectNamespace(
        namespace="rolled-back",
        parent_namespace=None,
        metadata={},
        created_by="test",
        created_at="1",
        updated_at="1",
    )
    try:
        with pytest.raises(RuntimeError, match="abort unit"):
            with unit.transaction():
                unit.objects.insert_namespace(namespace)
                unit.processes.insert_process(_process("pid_rolled_back"))
                raise RuntimeError("abort unit")

        assert store.get_namespace("rolled-back") is None
        assert store.get_process("pid_rolled_back") is None

        with unit.transaction():
            unit.objects.insert_namespace(namespace)
            unit.processes.insert_process(_process("pid_committed"))

        assert store.get_namespace("rolled-back") == namespace
        assert store.get_process("pid_committed") is not None
    finally:
        store.close()


class _FakePostgresCursor:
    rowcount = 0

    def execute(self, *_args: Any) -> None:
        return None

    def executemany(self, *_args: Any) -> None:
        return None

    def fetchone(self) -> None:
        return None

    def __iter__(self):
        return iter(())


def test_sqlite_and_postgres_adapters_share_sql_contract_shape() -> None:
    sqlite_connection = sqlite3.connect(":memory:")
    sqlite_cursor = sqlite_connection.cursor()
    postgres_connection = _PostgresConnection.__new__(_PostgresConnection)
    postgres_cursor = _PostgresCursor(_FakePostgresCursor(), _PostgresDialect())
    try:
        assert isinstance(sqlite_connection, SqlEngine)
        assert isinstance(sqlite_cursor, SqlSession)
        assert isinstance(postgres_connection, SqlEngine)
        assert isinstance(postgres_cursor, SqlSession)
    finally:
        sqlite_cursor.close()
        sqlite_connection.close()
