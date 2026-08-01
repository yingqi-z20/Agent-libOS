from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.payload_retention import (
    PayloadRetentionTier,
    external_effect_payload_sha256,
    human_request_payload_sha256,
    redact_terminal_task_run_human_request,
    retain_external_effect_payload,
)
from agent_libos.models import (
    AgentProcess,
    DataFlowContext,
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequest,
    HumanRequestStatus,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    TaskRunCommand,
    TaskRunCursor,
    TaskRunLedgerCursor,
    TaskRunLedgerItem,
    TaskRunLedgerKind,
    TaskRunLink,
    TaskRunPayload,
    TaskRunPayloadRetention,
    TaskRunRecord,
    TaskRunRequirement,
    TaskRunRequirementKind,
    TaskRunRequirementStatus,
    TaskRunResumePoint,
    TaskRunSpecV1,
    TaskRunStatus,
)
from agent_libos.models.exceptions import (
    TaskRunCommandConflict,
    TaskRunRevisionConflict,
    ValidationError,
)
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import PostgresStore, SQLiteStore


BACKENDS = ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
NOW = "2030-01-01T00:00:00+00:00"
LATER = "2030-01-02T00:00:00+00:00"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _config(*, plaintext: bool) -> object:
    return replace(
        DEFAULT_CONFIG,
        task_runs=replace(
            DEFAULT_CONFIG.task_runs,
            plaintext_payloads_enabled=plaintext,
            payload_max_bytes=4_096,
            list_page_size=2,
            list_hard_limit=10,
            recovery_page_size=2,
            recovery_page_hard_limit=10,
            ledger_page_size=2,
            ledger_page_hard_limit=10,
        ),
    )


def _spec(title: str = "Durable work") -> TaskRunSpecV1:
    return TaskRunSpecV1(
        goal={"goal": title},
        display_title=title,
        image_id="base-agent:v0",
    )


def _run(
    run_id: str,
    *,
    epoch: int,
    status: TaskRunStatus = TaskRunStatus.QUEUED,
    revision: int = 0,
    root_pid: str | None = None,
    created_at: str = NOW,
) -> TaskRunRecord:
    terminal = status in {
        TaskRunStatus.SUCCEEDED,
        TaskRunStatus.FAILED,
        TaskRunStatus.CANCELLED,
    }
    return TaskRunRecord.from_spec(
        run_id,
        _spec(run_id),
        status=status,
        revision=revision,
        runtime_epoch=epoch,
        root_pid=root_pid,
        active_pid=root_pid,
        created_at=created_at,
        updated_at=created_at,
        completed_at=created_at if terminal else None,
    )


def _bound_process(pid: str, run_id: str, epoch: int) -> AgentProcess:
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
        task_run_id=run_id,
        task_run_epoch=epoch,
        task_run_role="root",
        created_at=NOW,
        updated_at=NOW,
    )


def _external_effect(effect_id: str, pid: str) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=None,
        event_id=None,
        pid=pid,
        provider="retention-test-provider",
        operation="write",
        target="opaque-target",
        rollback_class=ExternalEffectRollbackClass.UNKNOWN,
        rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
        state_mutation=True,
        information_flow=True,
        provider_metadata={"secret": f"metadata-{effect_id}"},
        provider_receipt={"secret": f"receipt-{effect_id}"},
        canonical_args_hash=_sha(f"args-{effect_id}"),
        idempotency_key=f"idempotency-{effect_id}",
        effect_state="finalized",
        transaction_state="committed",
        created_at=NOW,
        updated_at=NOW,
    )


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_task_run_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    parsed = urlsplit(dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    scoped = urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), parsed.fragment)
    )
    try:
        yield scoped
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


@contextlib.contextmanager
def _store(
    backend: str,
    tmp_path: Path,
    *,
    plaintext: bool = True,
) -> Iterator[SQLiteStore | PostgresStore]:
    config = _config(plaintext=plaintext)
    if backend == "sqlite":
        store = SQLiteStore(tmp_path / f"task-run-{uuid4().hex}.sqlite", config=config)
        try:
            yield store
        finally:
            store.close()
        return
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn, config=config)
        try:
            yield store
        finally:
            store.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_revision_and_runtime_epoch_fences_are_backend_equivalent(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch_one = store.claim_runtime_epoch("runtime-one")
        store.insert_task_run(
            _run("run-fenced", epoch=epoch_one, root_pid="pid-root")
        )

        running = store.update_task_run_cas(
            "run-fenced",
            0,
            expected_runtime_epoch=epoch_one,
            updates={"status": TaskRunStatus.RUNNING, "updated_at": LATER},
        )
        assert running.revision == 1
        assert running.status is TaskRunStatus.RUNNING
        with pytest.raises(TaskRunRevisionConflict):
            store.update_task_run_cas(
                "run-fenced",
                0,
                expected_runtime_epoch=epoch_one,
                updates={"status": TaskRunStatus.PAUSED},
            )

        epoch_two = store.claim_runtime_epoch("runtime-two")
        claimed = store.claim_task_run_epoch(
            "run-fenced",
            expected_revision=1,
            runtime_epoch=epoch_two,
        )
        assert claimed.revision == 2
        assert claimed.runtime_epoch == epoch_two > epoch_one
        with pytest.raises(TaskRunRevisionConflict):
            store.update_task_run_cas(
                "run-fenced",
                2,
                expected_runtime_epoch=epoch_one,
                updates={"status": TaskRunStatus.PAUSED},
            )
        paused = store.update_task_run_cas(
            "run-fenced",
            2,
            expected_runtime_epoch=epoch_two,
            updates={"status": TaskRunStatus.PAUSED},
        )
        assert paused.revision == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_terminal_task_run_epoch_claim_is_narrow_and_rebinds_the_tree(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch_one = store.claim_runtime_epoch("runtime-terminal-one")
        store.insert_task_run(
            replace(
                _run(
                    "run-terminal-claim",
                    epoch=epoch_one,
                    status=TaskRunStatus.CANCELLED,
                    root_pid="pid-terminal-root",
                ),
                payloads_purged_at=NOW,
            )
        )
        store.insert_process(
            _bound_process(
                "pid-terminal-root",
                "run-terminal-claim",
                epoch_one,
            )
        )
        store.insert_task_run(_run("run-active-claim", epoch=epoch_one))

        epoch_two = store.claim_runtime_epoch("runtime-terminal-two")
        with pytest.raises(TaskRunRevisionConflict):
            store.claim_task_run_epoch(
                "run-terminal-claim",
                expected_revision=0,
                runtime_epoch=epoch_two,
            )
        with pytest.raises(TaskRunRevisionConflict):
            store.claim_terminal_task_run_epoch(
                "run-active-claim",
                expected_revision=0,
                runtime_epoch=epoch_two,
            )
        with pytest.raises(TaskRunRevisionConflict, match="stale Runtime epoch"):
            store.claim_terminal_task_run_epoch(
                "run-terminal-claim",
                expected_revision=0,
                runtime_epoch=epoch_one,
            )

        claimed = store.claim_terminal_task_run_epoch(
            "run-terminal-claim",
            expected_revision=0,
            runtime_epoch=epoch_two,
        )

        assert claimed.revision == 1
        assert claimed.runtime_epoch == epoch_two
        rebound = store.get_process("pid-terminal-root")
        assert rebound is not None and rebound.task_run_epoch == epoch_two
        with pytest.raises(TaskRunRevisionConflict):
            store.update_task_run_cas(
                claimed.run_id,
                claimed.revision,
                updates={"updated_at": LATER},
                expected_runtime_epoch=epoch_one,
            )
        with pytest.raises(TaskRunRevisionConflict):
            store.claim_terminal_task_run_epoch(
                claimed.run_id,
                expected_revision=claimed.revision,
                runtime_epoch=epoch_two,
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_command_idempotency_keys_must_resolve_to_one_exact_request(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch = store.claim_runtime_epoch("runtime-command-collisions")
        store.insert_task_run(_run("run-command-collisions", epoch=epoch))
        first = TaskRunCommand(
            command_id="command-first",
            client_request_id="client-first",
            run_id="run-command-collisions",
            command_kind="follow_up",
            request_hash=_sha("request-first"),
            result={"revision": 1},
            result_revision=1,
            created_at=NOW,
        )
        second = replace(
            first,
            command_id="command-second",
            client_request_id="client-second",
            request_hash=_sha("request-second"),
            result={"revision": 2},
            result_revision=2,
            created_at=LATER,
        )
        assert store.insert_task_run_command(first) == first

        conflict = "idempotency key was reused with a different request"
        with pytest.raises(TaskRunCommandConflict, match=conflict):
            store.insert_task_run_command(
                replace(first, client_request_id="client-new")
            )
        with pytest.raises(TaskRunCommandConflict, match=conflict):
            store.insert_task_run_command(
                replace(first, command_id="command-new")
            )
        with pytest.raises(TaskRunCommandConflict, match=conflict):
            store.insert_task_run_command(
                replace(first, request_hash=_sha("request-new"))
            )

        assert store.insert_task_run_command(second) == second
        with pytest.raises(TaskRunCommandConflict, match=conflict):
            store.insert_task_run_command(
                replace(
                    first,
                    client_request_id=second.client_request_id,
                )
            )

        assert (
            store.get_task_run_command(first.run_id, first.command_id) == first
        )
        assert (
            store.get_task_run_command(second.run_id, second.command_id) == second
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_command_recovery_list_is_scoped_bounded_and_bytewise_ordered(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch = store.claim_runtime_epoch("runtime-command-recovery-list")
        store.insert_task_run(_run("run-command-list", epoch=epoch))
        store.insert_task_run(_run("run-command-list-other", epoch=epoch))

        def command(
            command_id: str,
            *,
            run_id: str = "run-command-list",
            created_at: str,
        ) -> TaskRunCommand:
            return TaskRunCommand(
                command_id=command_id,
                client_request_id=None,
                run_id=run_id,
                command_kind="follow_up",
                request_hash=_sha(command_id),
                result={"revision": 1},
                result_revision=1,
                created_at=created_at,
            )

        for item in (
            command("command-z-second", created_at="Z"),
            command("command-a", created_at="a"),
            command("command-z-first", created_at="Z"),
            command(
                "command-other",
                run_id="run-command-list-other",
                created_at="0",
            ),
        ):
            store.insert_task_run_command(item)

        assert [
            item.command_id
            for item in store.list_task_run_commands(
                "run-command-list",
                limit=10,
            )
        ] == ["command-z-first", "command-z-second", "command-a"]
        assert [
            item.command_id
            for item in store.list_task_run_commands(
                "run-command-list",
                limit=2,
            )
        ] == ["command-z-first", "command-z-second"]

        store.insert_task_run(_run("run-command-list-cap", epoch=epoch))
        for index in range(11):
            store.insert_task_run_command(
                command(
                    f"command-cap-{index:02d}",
                    run_id="run-command-list-cap",
                    created_at=f"{index:02d}",
                )
            )
        assert len(
            store.list_task_run_commands("run-command-list-cap", limit=11)
        ) == 11
        with pytest.raises(ValidationError, match="exceeds hard cap"):
            store.list_task_run_commands("run-command-list-cap", limit=12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_command_mutations_are_globally_runtime_epoch_fenced(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch_one = store.claim_runtime_epoch("runtime-command-fence-old")
        run_id = "run-command-global-epoch-fence"
        store.insert_task_run(_run(run_id, epoch=epoch_one))
        existing = TaskRunCommand(
            command_id="command-before-epoch-advance",
            client_request_id=None,
            run_id=run_id,
            command_kind="run",
            request_hash=_sha("command-before-epoch-advance"),
            result={"settlement_state": "pending"},
            result_revision=1,
            created_at=NOW,
        )
        assert store.insert_task_run_command(
            existing,
            expected_runtime_epoch=epoch_one,
        ) == existing

        epoch_two = store.claim_runtime_epoch("runtime-command-fence-new")
        stale_gap_insert = replace(
            existing,
            command_id="linked-gap-stale-insert",
            request_hash=_sha("linked-gap-stale-insert"),
        )
        with pytest.raises(TaskRunRevisionConflict, match="epoch is stale"):
            store.insert_task_run_command(
                stale_gap_insert,
                expected_runtime_epoch=epoch_one,
            )
        assert store.get_task_run_command(run_id, stale_gap_insert.command_id) is None

        with pytest.raises(TaskRunRevisionConflict, match="epoch is stale"):
            store.update_task_run_command_result(
                run_id,
                existing.command_id,
                expected_result_revision=existing.result_revision,
                result={"settlement_state": "complete"},
                result_revision=existing.result_revision,
                expected_runtime_epoch=epoch_one,
            )
        assert store.get_task_run_command(run_id, existing.command_id) == existing

        inserted = store.insert_task_run_command(
            stale_gap_insert,
            expected_runtime_epoch=epoch_two,
        )
        assert inserted == stale_gap_insert
        completed = store.update_task_run_command_result(
            run_id,
            existing.command_id,
            expected_result_revision=existing.result_revision,
            result={"settlement_state": "complete"},
            result_revision=existing.result_revision,
            expected_runtime_epoch=epoch_two,
        )
        assert completed.result == {"settlement_state": "complete"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_commands_and_ledger_are_idempotent_append_only_projections(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch = store.claim_runtime_epoch("runtime-ledger")
        store.insert_task_run(_run("run-ledger", epoch=epoch))
        store.insert_task_run(_run("run-other", epoch=epoch))
        command = TaskRunCommand(
            command_id="command-1",
            client_request_id="client-1",
            run_id="run-ledger",
            command_kind="create",
            request_hash=_sha("request-1"),
            result={"run_id": "run-ledger", "revision": 0},
            result_revision=0,
            created_at=NOW,
        )
        assert store.insert_task_run_command(command) == command
        assert store.insert_task_run_command(command) == command
        completed_command = store.update_task_run_command_result(
            "run-ledger",
            "command-1",
            expected_result_revision=0,
            result={"run_id": "run-ledger", "revision": 2, "status": "running"},
            result_revision=2,
        )
        assert completed_command.result_revision == 2
        assert completed_command.result == {
            "run_id": "run-ledger",
            "revision": 2,
            "status": "running",
        }
        assert store.update_task_run_command_result(
            "run-ledger",
            "command-1",
            expected_result_revision=0,
            result=completed_command.result,
            result_revision=2,
        ) == completed_command
        with pytest.raises(TaskRunRevisionConflict, match="result revision conflict"):
            store.update_task_run_command_result(
                "run-ledger",
                "command-1",
                expected_result_revision=0,
                result={"run_id": "run-ledger", "revision": 3},
                result_revision=3,
            )
        with pytest.raises(TaskRunCommandConflict):
            store.insert_task_run_command(
                replace(command, command_id="command-2", request_hash=_sha("different"))
            )
        with pytest.raises(TaskRunCommandConflict):
            store.insert_task_run_command(
                replace(
                    command,
                    command_id="command-other",
                    run_id="run-other",
                )
            )

        first = TaskRunLedgerItem(
            item_id="ledger-1",
            run_id="run-ledger",
            seq=0,
            kind=TaskRunLedgerKind.STATUS_TRANSITION,
            status="queued",
            label="created",
            occurred_at=NOW,
            metadata={"from": None, "to": "queued"},
        )
        second = replace(
            first,
            item_id="ledger-2",
            kind=TaskRunLedgerKind.REQUIREMENT,
            status="pending",
            label="initial requirement",
            occurred_at=LATER,
            metadata={},
        )
        persisted_first = store.append_task_run_ledger_item(first)
        persisted_second = store.append_task_run_ledger_item(second)
        assert persisted_first.seq > 0
        assert persisted_second.seq == persisted_first.seq + 1
        assert store.append_task_run_ledger_item(
            replace(first, seq=persisted_first.seq)
        ) == persisted_first
        with pytest.raises(ValidationError, match="identity collision"):
            store.append_task_run_ledger_item(
                replace(first, seq=persisted_first.seq, label="forged replacement")
            )
        page = store.list_task_run_ledger(
            "run-ledger",
            after=None,
            limit=1,
        )
        assert page.records == (persisted_first,)
        assert page.next_cursor == TaskRunLedgerCursor(
            persisted_first.seq,
            persisted_first.item_id,
        )
        final = store.list_task_run_ledger(
            "run-ledger",
            after=page.next_cursor,
            limit=1,
        )
        assert final.records == (persisted_second,)
        assert final.next_cursor is None
        assert not hasattr(store, "update_task_run_ledger_item")
        assert not hasattr(store, "delete_task_run_ledger_item")

        link = TaskRunLink(
            link_id="link-1",
            run_id="run-ledger",
            ledger_seq=persisted_first.seq,
            evidence_type="operation",
            evidence_id="operation-1",
            role="operation",
            created_at=NOW,
            metadata={"projection": "canonical"},
        )
        store.insert_task_run_link(link)
        store.insert_task_run_link(link)
        with pytest.raises(ValidationError, match="link identity collision"):
            store.insert_task_run_link(
                replace(
                    link,
                    link_id="link-conflict",
                    ledger_seq=persisted_second.seq,
                    metadata={"projection": "forged"},
                )
            )
        distinct_role = replace(
            link,
            link_id="link-distinct-role",
            ledger_seq=persisted_second.seq,
            role="result",
            metadata={"projection": "distinct-role"},
        )
        store.insert_task_run_link(distinct_role)
        assert store.list_task_run_links("run-ledger") == [link, distinct_role]
        assert store.list_task_run_links("run-ledger", limit=1) == [link]
        assert store.list_task_run_links("run-ledger", limit=11) == [
            link,
            distinct_role,
        ]
        with pytest.raises(ValidationError, match="link list limit"):
            store.list_task_run_links("run-ledger", limit=0)
        with pytest.raises(ValidationError, match="link list limit"):
            store.list_task_run_links("run-ledger", limit=12)


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_plaintext_requires_opt_in_and_purge_is_complete(
    backend: str,
    tmp_path: Path,
) -> None:
    payload = TaskRunPayload.plaintext(
        payload_id="payload-goal",
        run_id="run-payload",
        role="goal",
        label="initial goal",
        value={"secret": "TASK_RUN_PURGE_SENTINEL"},
        created_at=NOW,
    )
    with _store(backend, tmp_path, plaintext=False) as disabled:
        epoch = disabled.claim_runtime_epoch("runtime-disabled")
        disabled.insert_task_run(_run("run-payload", epoch=epoch))
        with pytest.raises(ValidationError, match="explicitly enable"):
            disabled.insert_task_run_payload(payload)
        assert disabled.list_task_run_payloads("run-payload") == []

    with _store(backend, tmp_path, plaintext=True) as enabled:
        epoch = enabled.claim_runtime_epoch("runtime-enabled")
        enabled.insert_task_run(_run("run-payload", epoch=epoch))
        enabled.insert_task_run_payload(payload)
        persisted = enabled.get_task_run_payload("payload-goal")
        assert persisted == payload

        assert enabled.purge_task_run_payloads("run-payload", purged_at=LATER) == 1
        purged = enabled.get_task_run_payload("payload-goal")
        assert purged is not None
        assert purged.retention_state is TaskRunPayloadRetention.HASH_ONLY
        assert purged.canonical_json is None
        assert purged.sha256 == payload.sha256
        assert purged.size_bytes == payload.size_bytes
        assert purged.purged_at == LATER
        assert "TASK_RUN_PURGE_SENTINEL" not in repr(purged)


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_keyset_pages_are_strict_bounded_and_stable(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch = store.claim_runtime_epoch("runtime-page")
        for index in range(5):
            store.insert_task_run(
                _run(
                    f"run-{index}",
                    epoch=epoch,
                    created_at=f"2030-01-01T00:00:0{index}+00:00",
                )
            )
        first = store.list_task_runs(limit=2)
        assert [record.run_id for record in first.records] == ["run-0", "run-1"]
        assert first.next_cursor == TaskRunCursor(
            "2030-01-01T00:00:01+00:00",
            "run-1",
        )
        second = store.list_task_runs(after=first.next_cursor, limit=2)
        assert [record.run_id for record in second.records] == ["run-2", "run-3"]
        assert second.next_cursor is not None
        third = store.list_task_runs(after=second.next_cursor, limit=2)
        assert [record.run_id for record in third.records] == ["run-4"]
        assert third.next_cursor is None
        with pytest.raises(ValidationError, match="hard cap"):
            store.list_task_runs(limit=11)


def test_resume_point_commit_is_bound_to_current_run_process_and_epoch(
    tmp_path: Path,
) -> None:
    config = _config(plaintext=True)
    runtime = Runtime.open(tmp_path / "resume.sqlite", config=config)
    try:
        store = runtime.store
        epoch = store.claim_runtime_epoch("runtime-resume")
        pid = runtime.process.spawn(
            image="base-agent:v0",
            goal="resume point binding",
        )
        store.insert_task_run(
            _run("run-resume", epoch=epoch, root_pid=pid)
        )
        # This is test setup for the lower-level store contract. Production
        # creation/spawn binds these fields in its own atomic transaction.
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET task_run_id = ?, task_run_epoch = ?, "
                "task_run_role = ? WHERE pid = ?",
                ("run-resume", epoch, "root", pid),
            )
        transcript = TaskRunPayload.plaintext(
            payload_id="payload-transcript",
            run_id="run-resume",
            role="transcript",
            label="validated local transcript",
            value={"messages": [{"role": "assistant", "content": "done"}]},
            created_at=NOW,
        )
        store.insert_task_run_payload(transcript)
        point = TaskRunResumePoint(
            run_id="run-resume",
            pid=pid,
            task_run_epoch=epoch,
            process_revision=runtime.process.get(pid).revision,
            context_generation="generation-1",
            safe_point_seq=1,
            binding_hash=_sha("all-bindings"),
            image_binding_hash=_sha("image"),
            tool_binding_hash=_sha("tools"),
            provider_binding_hash=_sha("provider"),
            transcript_payload_id=transcript.payload_id,
            integrity_sha256=_sha("integrity-envelope"),
            created_at=NOW,
            updated_at=NOW,
            complete=True,
        )

        store.upsert_task_run_resume_point(point)
        assert store.get_task_run_resume_point(pid) == point
        with pytest.raises(TaskRunRevisionConflict, match="epoch fence"):
            store.upsert_task_run_resume_point(
                replace(
                    point,
                    task_run_epoch=epoch + 1,
                    safe_point_seq=2,
                    updated_at=LATER,
                )
            )
        assert store.get_task_run_resume_point(pid) == point
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_requirement_keyset_query_is_bounded_and_stable(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        epoch = store.claim_runtime_epoch("runtime-requirements")
        store.insert_task_run(_run("run-requirements", epoch=epoch))
        for ordinal in range(4):
            payload = TaskRunPayload.plaintext(
                payload_id=f"requirement-payload-{ordinal}",
                run_id="run-requirements",
                role="requirement",
                label=f"requirement {ordinal}",
                value={"ordinal": ordinal},
                created_at=f"2030-01-01T00:00:0{ordinal}+00:00",
            )
            store.insert_task_run_payload(payload)
            store.insert_task_run_requirement(
                TaskRunRequirement(
                    requirement_id=f"requirement-{ordinal}",
                    run_id="run-requirements",
                    ordinal=ordinal,
                    kind=(
                        TaskRunRequirementKind.INITIAL
                        if ordinal == 0
                        else TaskRunRequirementKind.FOLLOW_UP
                    ),
                    status=TaskRunRequirementStatus.PENDING,
                    payload_id=payload.payload_id,
                    requirement_sha256=payload.sha256,
                    label=f"requirement {ordinal}",
                    created_by="host",
                    created_at=payload.created_at,
                    updated_at=payload.updated_at,
                )
            )

        first = store.list_task_run_requirements(
            "run-requirements",
            limit=2,
        )
        assert [item.requirement_id for item in first] == [
            "requirement-0",
            "requirement-1",
        ]
        second = store.list_task_run_requirements(
            "run-requirements",
            after=(first[-1].ordinal, first[-1].requirement_id),
            limit=2,
        )
        assert [item.requirement_id for item in second] == [
            "requirement-2",
            "requirement-3",
        ]
        with pytest.raises(ValidationError, match="cursor"):
            store.list_task_run_requirements(
                "run-requirements",
                after=(-1, "requirement-0"),
                limit=2,
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_human_request_page_uses_newest_first_opaque_keyset(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        for index in range(5):
            store.insert_human_request(
                HumanRequest(
                    request_id=f"human-{index}",
                    pid="pid-a" if index % 2 else "pid-b",
                    human="host",
                    payload={"index": index},
                    status=(
                        HumanRequestStatus.REJECTED
                        if index == 3
                        else HumanRequestStatus.PENDING
                    ),
                    decision=None,
                    blocking=True,
                    created_at=f"2030-01-01T00:00:0{index}+00:00",
                    updated_at=f"2030-01-01T00:00:0{index}+00:00",
                )
            )

        first = store.list_human_requests_for_pids(
            ("pid-a", "pid-b"),
            statuses=(HumanRequestStatus.PENDING,),
            limit=2,
        )
        assert [item.request_id for item in first["records"]] == [
            "human-4",
            "human-2",
        ]
        assert str(first["next_cursor"]).startswith("h1.")
        second = store.list_human_requests_for_pids(
            ("pid-a", "pid-b"),
            statuses=("pending",),
            limit=2,
            cursor=first["next_cursor"],
        )
        assert [item.request_id for item in second["records"]] == [
            "human-1",
            "human-0",
        ]
        assert second["next_cursor"] is None
        with pytest.raises(ValidationError, match="hard cap"):
            store.list_human_requests_for_pids(("pid-a",), limit=11)


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_human_redaction_is_linked_atomic_and_backend_equivalent(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        run_id = "run-human-retention"
        pid = "pid-human-retention"
        epoch = store.claim_runtime_epoch("runtime-human-retention")
        store.insert_task_run(
            _run(
                run_id,
                epoch=epoch,
                status=TaskRunStatus.RUNNING,
                root_pid=pid,
            )
        )
        store.insert_process(_bound_process(pid, run_id, epoch))

        linked = HumanRequest(
            request_id="human-linked",
            pid=pid,
            human="host",
            payload={"type": "approval", "secret": "HUMAN_PROMPT_SENTINEL"},
            status=HumanRequestStatus.APPROVED,
            decision={"approved": True, "secret": "HUMAN_DECISION_SENTINEL"},
            blocking=True,
            created_at=NOW,
            updated_at=LATER,
        )
        unlinked = replace(linked, request_id="human-unlinked")
        wrong_member = replace(
            linked,
            request_id="human-wrong-member",
            pid="pid-other-run",
        )
        for request in (linked, unlinked, wrong_member):
            store.insert_human_request(request)

        for index, request in enumerate((linked, wrong_member), start=1):
            ledger = store.append_task_run_ledger_item(
                TaskRunLedgerItem(
                    item_id=f"ledger-human-{index}",
                    run_id=run_id,
                    seq=0,
                    kind=TaskRunLedgerKind.HUMAN_WAIT,
                    status=request.status.value,
                    label="linked Human request",
                    occurred_at=f"2030-01-01T00:00:0{index}+00:00",
                    pid=request.pid,
                    human_request_id=request.request_id,
                )
            )
            store.insert_task_run_link(
                TaskRunLink(
                    link_id=f"link-human-{index}",
                    run_id=run_id,
                    ledger_seq=ledger.seq,
                    evidence_type="human_request",
                    evidence_id=request.request_id,
                    role="human_wait",
                    created_at=ledger.occurred_at,
                )
            )

        assert store.list_task_run_human_requests(run_id, (pid,)) == [linked]
        with pytest.raises(ValidationError, match="unowned PID"):
            store.list_task_run_human_requests(run_id, ("pid-other-run",))

        source_sha256 = human_request_payload_sha256(linked)
        redacted = redact_terminal_task_run_human_request(linked)
        assert not store.redact_task_run_human_request_payload(
            redacted,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_status=linked.status.value,
        )
        store.update_task_run_cas(
            run_id,
            0,
            expected_runtime_epoch=epoch,
            updates={
                "status": TaskRunStatus.FINALIZING,
                "updated_at": LATER,
            },
        )

        with pytest.raises(RuntimeError, match="rollback Human redaction"):
            with store.transaction():
                assert store.redact_task_run_human_request_payload(
                    redacted,
                    run_id=run_id,
                    expected_payload_sha256=source_sha256,
                    expected_status=linked.status.value,
                )
                raise RuntimeError("rollback Human redaction")
        assert store.get_human_request(linked.request_id) == linked

        assert store.redact_task_run_human_request_payload(
            redacted,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_status=linked.status.value,
        )
        assert store.get_human_request(linked.request_id) == redacted
        assert human_request_payload_sha256(redacted) == source_sha256
        assert "HUMAN_PROMPT_SENTINEL" not in repr(redacted)
        assert "HUMAN_DECISION_SENTINEL" not in repr(redacted)
        assert not store.redact_task_run_human_request_payload(
            redacted,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_status=linked.status.value,
        )

        for request in (unlinked, wrong_member):
            target = redact_terminal_task_run_human_request(request)
            assert not store.redact_task_run_human_request_payload(
                target,
                run_id=run_id,
                expected_payload_sha256=human_request_payload_sha256(request),
                expected_status=request.status.value,
            )
            assert store.get_human_request(request.request_id) == request
        assert {link.evidence_id for link in store.list_task_run_links(run_id)} == {
            linked.request_id,
            wrong_member.request_id,
        }


def test_terminal_run_scoped_pending_and_message_purge_is_atomic(
    tmp_path: Path,
) -> None:
    runtime = Runtime.open(tmp_path / "run-scoped-purge.sqlite", config=_config(plaintext=True))
    try:
        store = runtime.store
        epoch = store.claim_runtime_epoch("runtime-purge")
        owned_pid = runtime.process.spawn(image="base-agent:v0", goal="owned")
        other_pid = runtime.process.spawn(image="base-agent:v0", goal="unrelated")
        store.insert_task_run(
            _run("run-purge", epoch=epoch, root_pid=owned_pid)
        )
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET task_run_id = ?, task_run_epoch = ?, "
                "task_run_role = ? WHERE pid = ?",
                ("run-purge", epoch, "root", owned_pid),
            )

        def pending(token: str) -> dict[str, object]:
            return {
                "wait_type": "message",
                "resume_token": token,
                "action": {"action": "receive_process_messages"},
                "data_flow_context": DataFlowContext().to_dict(),
                "content_preview": "local only",
                "tool_call_count": 1,
                "status": "pending",
            }

        store.upsert_llm_pending_action(owned_pid, pending("owned-token"))
        store.upsert_llm_pending_action(other_pid, pending("other-token"))
        with pytest.raises(ValidationError, match="finalizing"):
            store.purge_task_run_llm_pending_actions(
                "run-purge", (owned_pid,), purged_at=LATER
            )
        assert store.get_llm_pending_action(owned_pid) is not None

        finalizing = store.update_task_run_cas(
            "run-purge",
            0,
            expected_runtime_epoch=epoch,
            updates={"status": TaskRunStatus.FINALIZING, "updated_at": LATER},
        )
        assert finalizing.status is TaskRunStatus.FINALIZING
        with pytest.raises(ValidationError, match="unowned PID"):
            store.purge_task_run_llm_pending_actions(
                "run-purge", (owned_pid, other_pid), purged_at=LATER
            )
        assert store.get_llm_pending_action(owned_pid) is not None

        related = runtime.messages.post(
            sender="host",
            recipient_pid=owned_pid,
            channel="task-run-follow-up",
            correlation_id="requirement-42",
            subject="durable follow-up",
            metadata={"task_run_id": "run-purge"},
        )
        unrelated = runtime.messages.post(
            sender="host",
            recipient_pid=owned_pid,
            channel="ordinary-user-message",
            correlation_id="other-correlation",
            subject="unrelated",
        )
        outside_run = runtime.messages.post(
            sender="host",
            recipient_pid=other_pid,
            channel="ordinary-user-message",
            subject="outside Run",
        )

        with pytest.raises(RuntimeError, match="rollback purge"):
            with store.transaction():
                assert store.purge_task_run_llm_pending_actions(
                    "run-purge", (owned_pid,), purged_at=LATER
                ) == 1
                raise RuntimeError("rollback purge")
        assert store.get_llm_pending_action(owned_pid) is not None

        assert store.purge_task_run_llm_pending_actions(
            "run-purge", (owned_pid,), purged_at=LATER
        ) == 1
        assert store.get_llm_pending_action(owned_pid) is None
        assert store.get_llm_pending_action(other_pid) is not None
        assert store.purge_task_run_messages(
            "run-purge", (owned_pid,), purged_at=LATER
        ) == 2
        assert store.get_process_message(related.message_id) is None
        assert store.get_process_message(unrelated.message_id) is None
        assert store.get_process_message(outside_run.message_id) is not None
    finally:
        runtime.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_task_run_external_effect_redaction_is_linked_atomic_and_backend_equivalent(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        run_id = "run-effect-retention"
        pid = "pid-effect-retention"
        epoch = store.claim_runtime_epoch("runtime-effect-retention")
        store.insert_task_run(
            _run(
                run_id,
                epoch=epoch,
                status=TaskRunStatus.RUNNING,
                root_pid=pid,
            )
        )
        store.insert_process(_bound_process(pid, run_id, epoch))

        linked = _external_effect("effect-linked", pid)
        unlinked = _external_effect("effect-unlinked", pid)
        wrong_member = _external_effect("effect-wrong-member", "pid-other-run")
        for effect in (linked, unlinked, wrong_member):
            store.insert_external_effect(effect)

        for index, effect in enumerate((linked, wrong_member), start=1):
            ledger = store.append_task_run_ledger_item(
                TaskRunLedgerItem(
                    item_id=f"ledger-effect-{index}",
                    run_id=run_id,
                    seq=0,
                    kind=TaskRunLedgerKind.EFFECT,
                    status="finalized:committed",
                    label="linked external effect",
                    occurred_at=f"2030-01-01T00:00:0{index}+00:00",
                    pid=effect.pid,
                    effect_id=effect.effect_id,
                )
            )
            store.insert_task_run_link(
                TaskRunLink(
                    link_id=f"link-effect-{index}",
                    run_id=run_id,
                    ledger_seq=ledger.seq,
                    evidence_type="external_effect",
                    evidence_id=effect.effect_id,
                    role="effect",
                    created_at=ledger.occurred_at,
                )
            )

        assert store.list_task_run_external_effects(run_id, (pid,)) == [linked]
        with pytest.raises(ValidationError, match="unowned PID"):
            store.list_task_run_external_effects(run_id, ("pid-other-run",))

        source_sha256 = external_effect_payload_sha256(linked)
        summary = retain_external_effect_payload(
            linked,
            PayloadRetentionTier.SUMMARY,
        )
        assert not store.redact_task_run_external_effect_payload(
            summary,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_tier="full",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        store.update_task_run_cas(
            run_id,
            0,
            expected_runtime_epoch=epoch,
            updates={
                "status": TaskRunStatus.FINALIZING,
                "updated_at": LATER,
            },
        )
        with pytest.raises(RuntimeError, match="rollback redaction"):
            with store.transaction():
                assert store.redact_task_run_external_effect_payload(
                    summary,
                    run_id=run_id,
                    expected_payload_sha256=source_sha256,
                    expected_tier="full",
                    expected_effect_state="finalized",
                    expected_transaction_state="committed",
                )
                raise RuntimeError("rollback redaction")
        assert store.get_external_effect(linked.effect_id) == linked

        assert store.redact_task_run_external_effect_payload(
            summary,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_tier="full",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        assert not store.redact_task_run_external_effect_payload(
            summary,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_tier="full",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )

        hash_only = retain_external_effect_payload(
            summary,
            PayloadRetentionTier.HASH_ONLY,
        )
        assert store.redact_task_run_external_effect_payload(
            hash_only,
            run_id=run_id,
            expected_payload_sha256=source_sha256,
            expected_tier="summary",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        persisted = store.get_external_effect(linked.effect_id)
        assert persisted == hash_only
        assert store.list_task_run_links(run_id) == [
            TaskRunLink(
                link_id="link-effect-1",
                run_id=run_id,
                ledger_seq=store.list_task_run_ledger(
                    run_id,
                    after=None,
                    limit=10,
                ).records[0].seq,
                evidence_type="external_effect",
                evidence_id=linked.effect_id,
                role="effect",
                created_at="2030-01-01T00:00:01+00:00",
            ),
            TaskRunLink(
                link_id="link-effect-2",
                run_id=run_id,
                ledger_seq=store.list_task_run_ledger(
                    run_id,
                    after=None,
                    limit=10,
                ).records[1].seq,
                evidence_type="external_effect",
                evidence_id=wrong_member.effect_id,
                role="effect",
                created_at="2030-01-01T00:00:02+00:00",
            ),
        ]
        unlinked_summary = retain_external_effect_payload(
            unlinked,
            PayloadRetentionTier.SUMMARY,
        )
        assert not store.redact_task_run_external_effect_payload(
            unlinked_summary,
            run_id=run_id,
            expected_payload_sha256=external_effect_payload_sha256(unlinked),
            expected_tier="full",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        assert store.get_external_effect(unlinked.effect_id) == unlinked
        wrong_member_summary = retain_external_effect_payload(
            wrong_member,
            PayloadRetentionTier.SUMMARY,
        )
        assert not store.redact_task_run_external_effect_payload(
            wrong_member_summary,
            run_id=run_id,
            expected_payload_sha256=(
                external_effect_payload_sha256(wrong_member)
            ),
            expected_tier="full",
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        assert store.get_external_effect(wrong_member.effect_id) == wrong_member


@pytest.mark.parametrize("backend", BACKENDS)
def test_active_capability_reservations_for_run_pids_are_backend_equivalent(
    backend: str,
    tmp_path: Path,
) -> None:
    with _store(backend, tmp_path) as store:
        with store.transaction() as cursor:
            for index, pid in enumerate(("pid-run-a", "pid-run-b", "pid-other")):
                cap_id = f"cap-{index}"
                cursor.execute(
                    "INSERT INTO capabilities ("
                    "cap_id, subject, resource, rights_json, constraints_json, "
                    "issued_by, issued_at, expires_at, delegable, revocable, effect, "
                    "issuer_cap_id, parent_cap_id, delegation_depth, "
                    "max_delegation_depth, uses_remaining, status, metadata_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 0, 1, ?, NULL, NULL, 0, "
                    "NULL, 1, ?, ?)",
                    (
                        cap_id,
                        pid,
                        "resource:test",
                        '["read"]',
                        "{}",
                        "host",
                        NOW,
                        "allow",
                        "active",
                        "{}",
                    ),
                )
                cursor.execute(
                    "INSERT INTO capability_use_reservations ("
                    "reservation_id, cap_id, count, status, reserved_by, reason, "
                    "created_at, updated_at) VALUES (?, ?, 1, ?, ?, ?, ?, ?)",
                    (
                        f"reservation-{index}",
                        cap_id,
                        "committed" if index == 1 else "reserved",
                        "test",
                        "terminal convergence test",
                        f"2030-01-01T00:00:0{index}+00:00",
                        NOW,
                    ),
                )

        records = store.list_active_capability_use_reservations_for_pids(
            ("pid-run-a", "pid-run-b")
        )
        assert records == [
            {
                "reservation_id": "reservation-0",
                "cap_id": "cap-0",
                "pid": "pid-run-a",
                "count": 1,
                "status": "reserved",
                "created_at": "2030-01-01T00:00:00+00:00",
            }
        ]
        assert store.list_active_capability_use_reservations_for_pids(()) == []
