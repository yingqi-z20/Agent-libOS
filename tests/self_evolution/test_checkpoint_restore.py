from __future__ import annotations

import asyncio
import contextlib
import tempfile
import threading
import hashlib
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import AgentLibOSConfig, CheckpointDefaults
from agent_libos.models import CapabilityEffect, CapabilityRight, CapabilityStatus, DataFlowContext, DataLabels, EventType, ExternalEffectRecord, ExternalEffectRollbackClass, ExternalEffectRollbackStatus, HumanRequestStatus, ObjectMetadata, ObjectOwnerKind, ObjectPatch, ObjectTask, ObjectTaskStatus, ObjectType, ProcessMessageStatus, ProcessStatus
from agent_libos.models import HumanProcessWait, MessageProcessWait, PausedProcessWait, ToolProcessWait
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    ProcessRevisionConflict,
    ProcessMessageWaitRequired,
    ProcessWaitRequired,
    RuntimePublicationPending,
    ValidationError,
)
from agent_libos.memory.object_memory import ObjectMemoryManager
from agent_libos.modules import RuntimeModuleRegistry
from agent_libos.modules.loader import ModuleLoader
from agent_libos.runtime.checkpoint_reconciliation import (
    CHECKPOINT_RESTORE_PHASES,
    CHECKPOINT_RESTORE_V1_PHASES,
    CheckpointRestoreReconciler,
)
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.runtime.lifecycle import RuntimeLifecycle
from agent_libos.substrate import LocalHumanProvider, LocalResourceProviderSubstrate
from agent_libos.storage import SQLiteStore
from agent_libos.storage.repositories import SnapshotCheckpointRepository
from agent_libos.tools.builtin.checkpoint import RestoreCheckpointOutput
from agent_libos.utils.serde import dumps, loads
from tests.support.checkpoints import ClassifiedShellProvider


def _write_durable_finalizer_module(
    root: Path,
    *,
    marker: Path,
    attempts: Path,
) -> tuple[Path, str]:
    module_id = 'durable-finalizer-module:v0'
    source = root / 'durable_finalizer_module.py'
    source.write_text(
        f"""
from pathlib import Path

MARKER = Path({str(marker)!r})
ATTEMPTS = Path({str(attempts)!r})


def prepare(obj, _actor, _reason, _work_id):
    return {{
        'object_oid': obj.oid,
        'provider_resource_id': obj.payload['provider_resource_id'],
    }}


def finalize(_intent, _actor, _reason, work_id):
    with ATTEMPTS.open('a', encoding='utf-8') as stream:
        stream.write(work_id + '\\n')
    if not MARKER.exists():
        MARKER.write_text(work_id, encoding='utf-8')
        raise RuntimeError('injected failure after durable provider cleanup')
    if MARKER.read_text(encoding='utf-8') != work_id:
        raise RuntimeError('durable finalizer idempotency key changed')


def register_module(ctx):
    ctx.bind_durable_object_release_finalizer(
        'test.module-provider-release:v1',
        prepare,
        finalize,
    )
""".lstrip(),
        encoding='utf-8',
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = root / 'durable-finalizer-module.yaml'
    manifest.write_text(
        f"""
schema_version: 1
module_id: {module_id}
name: Durable finalizer module
version: v0
entrypoint: ./durable_finalizer_module.py:register_module
provides:
  durable_object_release_finalizers: ['test.module-provider-release:v1']
sha256: {source_sha}
""".lstrip(),
        encoding='utf-8',
    )
    manifest_sha = hashlib.sha256(
        manifest.read_text(encoding='utf-8').encode('utf-8')
    ).hexdigest()
    return manifest, ModuleLoader.trust_key(module_id, manifest_sha, source_sha)


@contextlib.contextmanager
def _postgres_checkpoint_target() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_checkpoint_restore_{uuid4().hex}"
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


def _release_checkpoint_recovery_runtime(runtime: Runtime) -> None:
    result = runtime.release_recovery_diagnostics()
    assert result["ok"] is True, result
    assert result["recovery_diagnostics_released"] is True
    assert runtime.lifecycle.closed


def _assert_missing_checkpoint_phase_receipt_is_fail_closed(
    target: str | Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_phase: str,
) -> None:
    runtime = Runtime.open(target)
    original_advance = runtime.store.advance_runtime_publication
    try:
        pid = runtime.process.spawn(
            goal=f"checkpoint {missing_phase} receipt must be durable",
        )
        checkpoint_id = runtime.checkpoint.create(
            pid,
            f"before missing {missing_phase} receipt",
            actor=pid,
        )
        reconciler = runtime.checkpoint._restore_reconciler
        effect_calls: list[str] = []
        for attribute, phase in (
            ("_restore_images", CHECKPOINT_RESTORE_PHASES[1]),
            ("_restore_jit_sources", CHECKPOINT_RESTORE_PHASES[2]),
            ("_prune_jit_tools", CHECKPOINT_RESTORE_PHASES[3]),
            ("_run_finalizer_items", CHECKPOINT_RESTORE_PHASES[4]),
        ):
            original_effect = getattr(reconciler, attribute)

            def observe_effect(
                *args: object,
                _original=original_effect,
                _phase=phase,
                **kwargs: object,
            ):
                effect_calls.append(_phase)
                return _original(*args, **kwargs)

            monkeypatch.setattr(reconciler, attribute, observe_effect)

        def omit_phase_receipt(publication_id: str, **kwargs: object) -> bool:
            if kwargs.get("receipt") == {
                "phase": "checkpoint_restore_phase_completed",
                "name": missing_phase,
            }:
                return True
            return original_advance(publication_id, **kwargs)

        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            omit_phase_receipt,
        )
        with pytest.raises(RuntimePublicationPending) as caught:
            runtime.checkpoint.restore(
                "cli",
                checkpoint_id,
                require_capability=False,
            )

        missing_index = CHECKPOINT_RESTORE_PHASES.index(missing_phase)
        assert effect_calls == list(CHECKPOINT_RESTORE_PHASES[1 : missing_index + 1])
        publication_id = caught.value.publication_id
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "reconciliation_pending"
        assert publication["phase"] == (
            "main_state_committed"
            if missing_index == 0
            else f"{CHECKPOINT_RESTORE_PHASES[missing_index - 1]}_completed"
        )
        completed = [
            item["name"]
            for item in publication["receipt"]["phases"]
            if item.get("phase") == "checkpoint_restore_phase_completed"
        ]
        assert completed == list(CHECKPOINT_RESTORE_PHASES[:missing_index])
        operation = runtime.store.get_operation(publication["plan"]["operation_id"])
        assert operation is not None
        assert operation.state.value == "running"
        assert operation.outcome.value == "pending"
        assert runtime.lifecycle.state == "close_failed"

        monkeypatch.setattr(
            runtime.store,
            "advance_runtime_publication",
            original_advance,
        )
        _release_checkpoint_recovery_runtime(runtime)

        reopened = Runtime.open(target)
        try:
            assert reopened.recovered_checkpoint_restore_publications == [
                publication_id
            ]
            recovered = reopened.store.get_runtime_publication(publication_id)
            assert recovered is not None
            assert recovered["state"] == "committed"
            assert recovered["phase"] == "reconciled"
            assert recovered["operation_reconciled"] is True
            completed = [
                item["name"]
                for item in recovered["receipt"]["phases"]
                if item.get("phase") == "checkpoint_restore_phase_completed"
            ]
            assert completed == list(CHECKPOINT_RESTORE_PHASES)
            recovery_claims = [
                item
                for item in recovered["receipt"]["phases"]
                if item.get("phase") == "recovery_claimed"
            ]
            assert len(recovery_claims) == 1
            assert recovered["receipt"]["recovery"]["disposition"] == "terminal"
            operation = reopened.store.get_operation(
                recovered["plan"]["operation_id"]
            )
            assert operation is not None
            assert operation.state.value == "terminal"
            assert operation.outcome.value == "succeeded"
        finally:
            reopened.close()

        reopened_again = Runtime.open(target)
        try:
            assert reopened_again.recovered_checkpoint_restore_publications == []
            recovered = reopened_again.store.get_runtime_publication(publication_id)
            assert recovered is not None
            recovery_claims = [
                item
                for item in recovered["receipt"]["phases"]
                if item.get("phase") == "recovery_claimed"
            ]
            assert len(recovery_claims) == 1
        finally:
            reopened_again.close()
    finally:
        if not runtime.lifecycle.closed:
            runtime.close()


def _prepare_failed_checkpoint_restore(
    target: str | Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(goal="checkpoint recovery terminal probe")
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "before checkpoint recovery terminal probe",
            actor=pid,
        )

        def fail_image_reconciliation(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected checkpoint recovery prerequisite")

        monkeypatch.setattr(
            runtime.checkpoint,
            "_restore_images",
            fail_image_reconciliation,
        )
        result = runtime.checkpoint.restore(
            "cli",
            checkpoint_id,
            require_capability=False,
        )
        publication_id = str(result["publication_id"])
        assert result["status"] == "restored_with_warnings"
        publication = runtime.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "failed"
        assert runtime.lifecycle.state == "close_failed"
        _release_checkpoint_recovery_runtime(runtime)
        return publication_id
    finally:
        if not runtime.lifecycle.closed:
            runtime.close()


def _assert_recovery_finish_postcommit_exception_confirms_terminal_truth(
    target: str | Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_id = _prepare_failed_checkpoint_restore(target, monkeypatch)
    original_finish = CheckpointRestoreReconciler._finish
    injected = False

    def interrupt_recovery_finish(
        reconciler: CheckpointRestoreReconciler,
        selected_id: str,
        *,
        recovery_lease_id: str | None,
    ) -> None:
        nonlocal injected
        original_finish(
            reconciler,
            selected_id,
            recovery_lease_id=recovery_lease_id,
        )
        if (
            selected_id == publication_id
            and recovery_lease_id is not None
            and not injected
        ):
            injected = True
            raise RuntimeError("injected recovery finish postcommit exception")

    monkeypatch.setattr(
        CheckpointRestoreReconciler,
        "_finish",
        interrupt_recovery_finish,
    )
    reopened = Runtime.open(target)
    try:
        assert injected is True
        assert reopened.recovered_checkpoint_restore_publications == [publication_id]
        publication = reopened.store.get_runtime_publication(publication_id)
        assert publication is not None
        assert publication["state"] == "committed"
        assert publication["phase"] == "reconciled"
        assert publication["operation_reconciled"] is True
        operation = reopened.store.get_operation(publication["plan"]["operation_id"])
        assert operation is not None
        assert operation.state.value == "terminal"
        assert operation.outcome.value == "succeeded"
        assert reopened.lifecycle.state == "open"
    finally:
        reopened.close()

    monkeypatch.setattr(
        CheckpointRestoreReconciler,
        "_finish",
        original_finish,
    )
    reopened_again = Runtime.open(target)
    try:
        assert reopened_again.recovered_checkpoint_restore_publications == []
    finally:
        reopened_again.close()


def _assert_selective_payload_retry_after_mark_open_failure(
    target: str | Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newer startup mutation must not be overwritten on delivery retry."""

    runtime = Runtime.open(target)
    try:
        pid = runtime.process.spawn(goal="selective checkpoint payload retry")
        transferred_owner_pid = runtime.process.spawn(
            goal="own a superseded startup payload",
        )
        mutated = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"version": 1},
            ObjectMetadata(title="mutated delivery state"),
            immutable=False,
            name="payload.delivery.mutated",
        )
        unchanged = runtime.memory.create_object(
            pid,
            ObjectType.SUMMARY,
            {"version": 1},
            ObjectMetadata(title="unchanged delivery state"),
            immutable=False,
            name="payload.delivery.unchanged",
        )
        runtime.memory.link_objects(pid, mutated, "references", unchanged)
        checkpoint_id = runtime.checkpoint.create(
            pid,
            "before selective payload delivery retry",
            actor=pid,
        )
        runtime.memory.update_object(
            pid,
            mutated,
            ObjectPatch(payload={"version": 2}),
        )
        runtime.memory.update_object(
            pid,
            unchanged,
            ObjectPatch(payload={"version": 2}),
        )

        def fail_image_reconciliation(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected selective payload image failure")

        monkeypatch.setattr(
            runtime.checkpoint,
            "_restore_images",
            fail_image_reconciliation,
        )
        result = runtime.checkpoint.restore(
            "cli",
            checkpoint_id,
            require_capability=False,
        )
        publication_id = str(result["publication_id"])
        assert result["status"] == "restored_with_warnings"
        _release_checkpoint_recovery_runtime(runtime)

        original_hooks = RuntimeModuleRegistry._run_startup_hooks_locked
        hook_calls = 0
        mutation_rows: list[tuple[int, str | None]] = []

        def mutate_during_first_late_hook(
            registry: RuntimeModuleRegistry,
        ) -> None:
            nonlocal hook_calls
            hook_calls += 1
            original_hooks(registry)
            if hook_calls == 1:
                registry._hook_services.memory.update_object(
                    pid,
                    mutated,
                    ObjectPatch(payload={"version": 3}),
                )
                transferred = registry._hook_services.memory.transfer_owner(
                    ObjectOwnerKind.PROCESS,
                    pid,
                    ObjectOwnerKind.PROCESS,
                    transferred_owner_pid,
                    [mutated.oid],
                    actor="module:test-payload-delivery",
                    reason="startup_payload_supersession",
                )
                assert transferred == [mutated.oid]
                durable = registry._hook_services.store.get_object(mutated.oid)
                assert durable is not None
                mutation_rows.append((durable.version, durable.owner_id))

        monkeypatch.setattr(
            RuntimeModuleRegistry,
            "_run_startup_hooks_locked",
            mutate_during_first_late_hook,
        )
        original_mark_open = RuntimeLifecycle.mark_open
        mark_open_calls = 0

        def fail_first_mark_open(lifecycle: RuntimeLifecycle) -> None:
            nonlocal mark_open_calls
            mark_open_calls += 1
            if mark_open_calls == 1:
                raise RuntimeError("injected mark_open failure")
            original_mark_open(lifecycle)

        monkeypatch.setattr(RuntimeLifecycle, "mark_open", fail_first_mark_open)
        with pytest.raises(RuntimeError, match="injected mark_open failure"):
            Runtime.open(target)

        reopened = Runtime.open(target)
        try:
            assert hook_calls == 2
            assert mark_open_calls == 2
            assert mutation_rows == [(3, transferred_owner_pid)]

            # The newer v3 payload was deliberately not replayed from the v1
            # snapshot, so the ordinary missing-payload sweep releases it.
            assert reopened.store.get_object(mutated.oid) is None
            assert reopened.store.is_recovered_object_payload(mutated.oid)
            durable_rows = [
                row
                for row in reopened.store.select_table_rows("objects")
                if row["oid"] == mutated.oid
            ]
            assert len(durable_rows) == 1
            assert durable_rows[0]["owner_id"] == transferred_owner_pid
            mutated_capability = reopened.store.get_capability(
                mutated.capability_id
            )
            assert mutated_capability is not None
            assert mutated_capability.status == CapabilityStatus.REVOKED
            assert reopened.store.list_links(src=mutated.oid) == []
            assert reopened.store.list_links(dst=mutated.oid) == []

            assert reopened.store.object_payload(unchanged.oid) == {"version": 1}
            assert not reopened.store.is_recovered_object_payload(unchanged.oid)
            unchanged_capability = reopened.store.get_capability(
                unchanged.capability_id
            )
            assert unchanged_capability is not None
            assert unchanged_capability.status == CapabilityStatus.ACTIVE

            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "committed"
            assert publication["operation_reconciled"] is True
            assert publication["receipt"]["payload_delivery"] == {
                "state": "completed"
            }
        finally:
            reopened.close()
    finally:
        if not runtime.lifecycle.closed:
            runtime.close()


class TestCheckpointRestore:

    def test_restore_publishes_monotonic_revision_and_execution_epoch(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='restore must permanently fence stale process writers',
            )
            token = runtime.store.claim_execution(pid, owner_id='stale-worker')
            assert token is not None
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'capture an active execution lease',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            snapshot_row = found[1]['rows']['processes'][0]
            stale_revision = int(snapshot_row['revision'])
            stale_state_generation = int(snapshot_row['state_generation'])
            assert snapshot_row['execution_owner_id'] == token.owner_id
            assert snapshot_row['execution_lease_id'] == token.lease_id

            assert runtime.store.complete_execution(token) is True
            before_restore = runtime.process.get(pid)
            assert before_restore.execution_owner_id is None
            assert before_restore.execution_lease_id is None

            runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            first = runtime.process.get(pid)
            assert first.revision > before_restore.revision
            assert first.execution_generation > before_restore.execution_generation
            assert first.state_generation > before_restore.state_generation
            assert first.state_generation > stale_state_generation
            assert first.execution_owner_id is None
            assert first.execution_lease_id is None
            assert runtime.store.complete_execution(token) is False
            assert runtime.store.release_execution(token) is False
            with pytest.raises(ProcessRevisionConflict):
                runtime.store.patch_process(
                    pid,
                    {'status_message': 'stale writer must not commit'},
                    expected_revision=stale_revision,
                )

            runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            second = runtime.process.get(pid)
            assert second.revision > first.revision
            assert second.execution_generation > first.execution_generation
            assert second.state_generation > first.state_generation
            assert second.execution_owner_id is None
            assert second.execution_lease_id is None
        finally:
            runtime.close()

    def test_pre_03_checkpoint_is_rejected_before_restore_write(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='old checkpoint version')
            checkpoint_id = runtime.checkpoint.create(pid, 'old checkpoint version', actor=pid)
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            snapshot['version'] = 1
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )
            total_changes = runtime.store.conn.total_changes
            before = runtime.process.get(pid)

            with pytest.raises(ValidationError, match='unsupported snapshot version'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.store.conn.total_changes == total_changes
            assert runtime.process.get(pid) == before
        finally:
            runtime.close()

    def test_restore_rejects_snapshot_missing_process_table_without_mutation(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject a truncated process snapshot',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'truncated process snapshot',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            snapshot['rows'].pop('processes')
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )
            before = runtime.process.get(pid)

            with pytest.raises(ValidationError, match='missing tables'):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.process.get(pid) == before
        finally:
            runtime.close()

    def test_restore_rejects_missing_flow_carriers_without_mutation(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject a malformed checkpoint without trusted flow carriers',
            )
            message = runtime.messages.post(
                sender='malformed.sender',
                recipient_pid=pid,
                subject='malformed snapshot message',
                body='classification omitted from a malformed snapshot',
            )
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    'resume_token': 'malformed-checkpoint-token',
                    'wait_type': 'message',
                    'filters': {},
                    'action': {'action': 'receive_process_messages'},
                    'data_flow_context': DataFlowContext().to_dict(),
                    'content_preview': '',
                    'tool_call_count': 1,
                    'status': 'pending',
                },
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'malformed checkpoint flow state',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            snapshot['rows']['llm_pending_actions'][0].pop('data_flow_context_json')
            snapshot['rows']['process_messages'][0].pop('metadata_json')
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )
            before = runtime.process.get(pid)

            with pytest.raises(ValidationError, match='canonical'):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            after = runtime.process.get(pid)
            assert after.status == before.status == ProcessStatus.RUNNABLE
            assert runtime.store.get_process_message(message.message_id) is not None
            assert runtime.store.get_llm_pending_action(pid)['status'] == 'pending'
        finally:
            runtime.close()

    def test_restore_rejects_completed_pending_row_without_flow_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject completed malformed pending history',
            )
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    'resume_token': 'completed-malformed-checkpoint-token',
                    'wait_type': 'message',
                    'filters': {},
                    'action': {'action': 'receive_process_messages'},
                    'data_flow_context': DataFlowContext().to_dict(),
                    'content_preview': '',
                    'tool_call_count': 1,
                    'status': 'completed',
                },
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'completed malformed pending history',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            snapshot['rows']['llm_pending_actions'][0].pop('data_flow_context_json')
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            with pytest.raises(ValidationError, match='canonical'):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert runtime.store.get_llm_pending_action(pid)['status'] == 'completed'
        finally:
            runtime.close()

    def test_restore_rejects_incomplete_message_metadata_without_mutation(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject incomplete message metadata',
            )
            message = runtime.messages.post(
                sender='malformed.sender',
                recipient_pid=pid,
                metadata={'custom': 'preserved'},
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'incomplete message metadata',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            message_row = next(
                row
                for row in snapshot['rows']['process_messages']
                if row['message_id'] == message.message_id
            )
            message_row['metadata_json'] = dumps(
                {
                    'custom': 'preserved',
                    'data_labels': {
                        'sensitivity': 'normal',
                        'trust_level': 'trusted',
                        'integrity': 'verified',
                    },
                }
            )
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            before = runtime.process.get(pid)
            with pytest.raises(ValidationError, match='canonical'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.process.get(pid).status == before.status
            persisted = runtime.store.get_process_message(message.message_id)
            assert persisted is not None
            assert persisted.metadata['custom'] == 'preserved'
        finally:
            runtime.close()

    def test_restore_rejects_child_pending_row_without_flow_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(
                image='base-agent:v0',
                goal='malformed checkpoint parent',
            )
            child = runtime.process.spawn_child(parent, 'malformed checkpoint child')
            runtime.store.upsert_llm_pending_action(
                child,
                {
                    'resume_token': 'malformed-checkpoint-child-token',
                    'wait_type': 'message',
                    'filters': {},
                    'action': {'action': 'receive_process_messages'},
                    'data_flow_context': DataFlowContext().to_dict(),
                    'content_preview': '',
                    'tool_call_count': 1,
                    'status': 'pending',
                },
            )
            checkpoint_id = runtime.checkpoint.create(
                parent,
                'malformed child lifecycle',
                actor=parent,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            pending = next(
                row
                for row in snapshot['rows']['llm_pending_actions']
                if row['pid'] == child
            )
            pending.pop('data_flow_context_json')
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            with pytest.raises(ValidationError, match='canonical'):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
            assert runtime.store.get_llm_pending_action(child)['status'] == 'pending'
        finally:
            runtime.close()

    def test_restore_reserves_composite_one_shot_authority_until_main_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-composite-reservation:v0'
        try:
            runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name='checkpoint-composite-reservation',
                    system_prompt='captured prompt',
                ),
                actor='test',
            )
            owner = runtime.process.spawn(image=image_id, goal='checkpoint authority owner')
            controller = runtime.process.spawn(image='base-agent:v0', goal='checkpoint authority controller')
            checkpoint_id = runtime.checkpoint.create(owner, 'composite authority', actor=owner)
            runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name='checkpoint-composite-reservation',
                    system_prompt='current prompt',
                ),
                actor='test',
                replace=True,
            )
            checkpoint_cap = runtime.capability.grant_once(
                controller,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            image_cap = runtime.capability.grant_once(
                controller,
                f'image:{image_id}',
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            original_validate = runtime.checkpoint._validate_snapshot_restore_assets
            fail_once = True

            def injected_preflight_failure(snapshot: dict[str, object]) -> None:
                nonlocal fail_once
                if fail_once:
                    fail_once = False
                    raise ValidationError('injected restore preflight failure')
                original_validate(snapshot)

            monkeypatch.setattr(
                runtime.checkpoint,
                '_validate_snapshot_restore_assets',
                injected_preflight_failure,
            )

            with pytest.raises(ValidationError, match='injected restore preflight failure'):
                runtime.checkpoint.restore(controller, checkpoint_id)

            assert runtime.store.get_capability(checkpoint_cap.cap_id).uses_remaining == 1
            assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 1
            assert runtime.get_image(image_id).system_prompt == 'current prompt'

            restored = runtime.checkpoint.restore(controller, checkpoint_id)

            assert restored['main_state_committed'] is True
            assert runtime.store.get_capability(checkpoint_cap.cap_id).uses_remaining == 0
            assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 0
            assert runtime.get_image(image_id).system_prompt == 'captured prompt'
        finally:
            runtime.close()

    def test_checkpoint_create_audit_failure_rolls_back_row_head_capability_and_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='atomic checkpoint creation')
            before_events = list(runtime.events.list())
            before_audit = list(runtime.audit.trace())
            before_checkpoint_capabilities = {
                capability.cap_id
                for capability in runtime.capability.capabilities_for(pid)
                if capability.resource.startswith('checkpoint:')
            }
            original_record = runtime.audit.record

            def fail_create_audit(*args: object, **kwargs: object) -> object:
                if kwargs.get('action') == 'checkpoint.create':
                    raise RuntimeError('injected checkpoint create audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_create_audit)
            with pytest.raises(RuntimeError, match='injected checkpoint create audit failure'):
                runtime.checkpoint.create(pid, 'must be atomic', actor=pid, require_capability=False)

            assert runtime.process.get(pid).checkpoint_head is None
            assert runtime.checkpoint.list(pid, actor=pid, require_capability=False) == []
            assert runtime.events.list() == before_events
            assert runtime.audit.trace() == before_audit
            assert {
                capability.cap_id
                for capability in runtime.capability.capabilities_for(pid)
                if capability.resource.startswith('checkpoint:')
            } == before_checkpoint_capabilities
        finally:
            runtime.close()


    def test_restore_advances_llm_context_generation_to_break_provider_chain(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore breaks provider chain')
            runtime.store.set_llm_context_generation(pid, 'generation-before-checkpoint')
            checkpoint_id = runtime.checkpoint.create(pid, 'provider chain boundary', actor=pid)
            runtime.store.set_llm_context_generation(pid, 'generation-after-checkpoint')

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            generation = runtime.store.get_llm_context_generation(pid)
            assert generation not in {'generation-before-checkpoint', 'generation-after-checkpoint'}
        finally:
            runtime.close()

    def test_restore_reconciles_plural_human_wait_state_after_all_requests_resolve(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='plural human wait restore')
            first = runtime.human.query(
                pid,
                'owner',
                {'type': 'question', 'question': 'First?'},
                blocking=True,
            )
            second = runtime.human.query(
                pid,
                'owner',
                {'type': 'question', 'question': 'Second?'},
                blocking=True,
            )
            runtime.human.approve(first, {'approved': True, 'answer': 'one'})
            waiting = runtime.process.get(pid)
            assert waiting.status == ProcessStatus.WAITING_HUMAN
            assert waiting.status_message == f'waiting for human request {second}'
            assert waiting.wait_state == HumanProcessWait(request_ids=(second,))
            waiting_generation = waiting.state_generation
            checkpoint_id = runtime.checkpoint.create(pid, 'plural human wait', actor=pid)
            runtime.human.approve(second, {'approved': True, 'answer': 'two'})
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.process.get(pid)
            assert restored.status == ProcessStatus.RUNNABLE
            assert restored.status_message is None
            assert restored.wait_state is None
            assert restored.state_generation > waiting_generation
        finally:
            runtime.close()

    def test_restore_reconciles_message_wait_with_matching_unread_message(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='message wait restore')
            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(pid, block=True, channel='control')
            waiting = runtime.process.get(pid)
            assert isinstance(waiting.wait_state, MessageProcessWait)
            wait_state = waiting.wait_state
            wait_status_message = waiting.status_message
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                channel='control',
                subject='already available',
            )
            current = runtime.process.get(pid)
            runtime.process_transitions.transition(
                pid,
                ProcessStatus.WAITING_EVENT,
                expected_revision=current.revision,
                expected_status=ProcessStatus.RUNNABLE,
                expected_state_generation=current.state_generation,
                wait_state=wait_state,
                status_message=wait_status_message,
                control=True,
                allowed_statuses={ProcessStatus.RUNNABLE},
                reason='test reconstructs persisted message wait',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'persisted message wait with unread match',
                actor=pid,
            )
            current = runtime.process.get(pid)
            runtime.process_transitions.transition(
                pid,
                ProcessStatus.PAUSED,
                expected_revision=current.revision,
                expected_status=ProcessStatus.WAITING_EVENT,
                expected_state_generation=current.state_generation,
                wait_state=PausedProcessWait(),
                status_message='changed after checkpoint',
                control=True,
                allowed_statuses={ProcessStatus.WAITING_EVENT},
                reason='test changes process after checkpoint',
            )

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.process.get(pid)
            assert restored.status == ProcessStatus.RUNNABLE
            assert restored.status_message is None
            assert restored.wait_state is None
        finally:
            runtime.close()

    def test_legacy_full_table_snapshot_restore_are_disabled(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(RuntimeError, match='full-table SQLite snapshots are disabled'):
                runtime.store.snapshot_tables()
            with pytest.raises(RuntimeError, match='full-table SQLite restore is disabled'):
                runtime.store.restore_tables({})
        finally:
            runtime.close()

    def test_restore_invalidates_inflight_capability_reservations_before_replacing_scope(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore reservation boundary')
            cap = runtime.capability.issue_trusted(
                pid,
                'object:restore-reservation',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=2,
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before reservation', actor=pid)
            decision = runtime.capability.authorize(pid, cap.resource, CapabilityRight.READ)
            reservation_id = runtime.capability.reserve_decision_use(
                decision,
                used_by='test',
                reason='effect in flight while checkpoint restore begins',
            )
            assert reservation_id is not None
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.store.get_capability(cap.cap_id)
            assert restored is not None
            assert restored.uses_remaining == 1
            assert runtime.capability.restore_reserved_use(
                reservation_id,
                restored_by='test',
                reason='late provider cleanup after scope replacement',
            ) is None
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
        finally:
            runtime.close()

    def test_restore_recovers_process_subtree_objects_capabilities_and_cwd_only(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='root')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            child = runtime.spawn_child_process(pid, 'child')
            unrelated = runtime.process.spawn(image='base-agent:v0', goal='other')
            handle = runtime.memory.create_object(pid, ObjectType.SUMMARY, {'version': 1}, ObjectMetadata(title='state'), immutable=False, name='state')
            checkpoint_id = runtime.checkpoint.create(pid, 'before mutation', actor=pid)
            runtime.memory.update_object(pid, handle, ObjectPatch(payload={'version': 2}))
            runtime.memory.create_object(pid, ObjectType.SUMMARY, {'temp': True}, name='temp')
            runtime.process.set_working_directory(pid, 'src')
            runtime.process.signal(child, 'terminate', {'reason': 'bad branch'})
            runtime.capability.grant(pid, 'test:temporary', [CapabilityRight.READ], issued_by='test')
            other_handle = runtime.memory.create_object(unrelated, ObjectType.SUMMARY, {'keep': True}, name='other')
            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            restored = runtime.memory.get_object_by_name(pid, 'state')
            assert restored.payload == {'version': 1}
            with pytest.raises(Exception):
                runtime.memory.get_object_by_name(pid, 'temp')
            assert runtime.process.get(pid).working_directory == '.'
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert not runtime.capability.check(pid, 'test:temporary', CapabilityRight.READ)
            assert runtime.memory.get_object(unrelated, other_handle).payload == {'keep': True}
            assert result['restored_pids'] == [pid, child]
        finally:
            runtime.close()

    def test_restore_does_not_resurrect_revoked_or_currently_denied_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='restore authority')
                (Path(temp_dir) / 'secret.txt').write_text('secret', encoding='utf-8')
                secret = runtime.filesystem.resource_for_path('secret.txt')
                cap = runtime.filesystem.grant_path(pid, 'secret.txt', [CapabilityRight.READ], issued_by='test')
                checkpoint_id = runtime.checkpoint.create(pid, 'before revoke', actor=pid)
                runtime.capability.revoke(cap.cap_id, revoked_by=pid, reason='revoked before restore')
                runtime.capability.issue_trusted(pid, secret, [CapabilityRight.READ], issued_by='test', effect=CapabilityEffect.DENY)

                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

                assert not runtime.capability.check(pid, secret, CapabilityRight.READ)
                with pytest.raises(CapabilityDenied):
                    runtime.filesystem.read_text(pid, 'secret.txt')
            finally:
                runtime.close()

    def test_restore_concurrent_revoke_wins_over_snapshot_capability_reinsertion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        filter_reached = threading.Event()
        revoke_started = threading.Event()
        revoke_done = threading.Event()
        restore_errors: list[BaseException] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='concurrent restore revoke')
            cap = runtime.capability.grant(pid, 'test:restore-race', [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'before concurrent revoke', actor=pid)
            original_filter = runtime.checkpoint._filtered_restored_capability_rows

            def pause_after_filter(rows):
                filtered = original_filter(rows)
                filter_reached.set()
                assert revoke_started.wait(timeout=2)
                return filtered

            monkeypatch.setattr(runtime.checkpoint, '_filtered_restored_capability_rows', pause_after_filter)

            def restore() -> None:
                try:
                    runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                except BaseException as exc:  # pragma: no cover - asserted below
                    restore_errors.append(exc)

            restore_thread = threading.Thread(target=restore)
            restore_thread.start()
            assert filter_reached.wait(timeout=2)

            def revoke() -> None:
                revoke_started.set()
                runtime.capability.revoke(cap.cap_id, revoked_by=pid, reason='concurrent revoke wins')
                revoke_done.set()

            revoke_thread = threading.Thread(target=revoke)
            revoke_thread.start()
            restore_thread.join(timeout=2)
            revoke_thread.join(timeout=2)

            assert not restore_thread.is_alive()
            assert not revoke_thread.is_alive()
            assert revoke_done.is_set()
            assert restore_errors == []
            restored = runtime.store.get_capability(cap.cap_id)
            assert restored is not None
            assert not restored.active
            assert not runtime.capability.check(pid, cap.resource, CapabilityRight.READ)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'mutation_kind',
        ['spawn', 'capability', 'object', 'message', 'object_task'],
    )
    def test_restore_serializes_host_mutations_across_preflight_and_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mutation_kind: str,
    ) -> None:
        runtime = Runtime.open('local')
        preflight_reached = threading.Event()
        mutation_started = threading.Event()
        allow_commit = threading.Event()
        main_state_committed = threading.Event()
        mutation_finished = threading.Event()
        mutation_before_commit: list[bool] = []
        errors: list[BaseException] = []
        mutation_results: list[object] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore host mutation boundary')
            owner = None
            if mutation_kind in {'spawn', 'object_task'}:
                runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            if mutation_kind == 'object_task':
                owner = runtime.memory.create_object(
                    pid,
                    ObjectType.ARTIFACT,
                    {'owner': True},
                    name='restore.mutation.owner',
                    immutable=False,
                )
            checkpoint_id = runtime.checkpoint.create(pid, 'before host mutation race', actor=pid)
            original_validate = runtime.checkpoint._validate_snapshot_restore_assets
            original_restore_rows = runtime.checkpoint._restore_scoped_rows

            def pause_preflight(snapshot):
                original_validate(snapshot)
                preflight_reached.set()
                assert mutation_started.wait(timeout=2)
                assert allow_commit.wait(timeout=2)

            def mark_main_commit(*args, **kwargs):
                result = original_restore_rows(*args, **kwargs)
                main_state_committed.set()
                return result

            monkeypatch.setattr(runtime.checkpoint, '_validate_snapshot_restore_assets', pause_preflight)
            monkeypatch.setattr(runtime.checkpoint, '_restore_scoped_rows', mark_main_commit)

            def restore() -> None:
                try:
                    runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            restore_thread = threading.Thread(target=restore)
            restore_thread.start()
            assert preflight_reached.wait(timeout=2)

            def mutate() -> None:
                try:
                    mutation_started.set()
                    if mutation_kind == 'spawn':
                        mutation_results.append(runtime.process.spawn_child(pid, 'spawn racing restore'))
                    elif mutation_kind == 'capability':
                        mutation_results.append(
                            runtime.capability.grant(
                                pid,
                                'test:capability-racing-restore',
                                [CapabilityRight.READ],
                                issued_by='test',
                            )
                        )
                    elif mutation_kind == 'object':
                        mutation_results.append(
                            runtime.memory.create_object(
                                pid,
                                ObjectType.SUMMARY,
                                {'racing': True},
                                name='restore.mutation.object',
                            )
                        )
                    elif mutation_kind == 'message':
                        mutation_results.append(
                            runtime.messages.post(
                                sender='test',
                                recipient_pid=pid,
                                subject='message racing restore',
                            )
                        )
                    else:
                        assert owner is not None
                        mutation_results.append(
                            runtime.object_tasks.start(
                                pid,
                                owner,
                                'receive_process_messages',
                                {'channel': 'restore-mutation-never'},
                            )
                        )
                    mutation_before_commit.append(not main_state_committed.is_set())
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)
                finally:
                    mutation_finished.set()

            mutation_thread = threading.Thread(target=mutate)
            mutation_thread.start()
            assert mutation_started.wait(timeout=2)
            mutation_finished.wait(timeout=0.2)
            allow_commit.set()
            restore_thread.join(timeout=3)
            mutation_thread.join(timeout=3)

            assert not restore_thread.is_alive()
            assert not mutation_thread.is_alive()
            assert errors == []
            assert mutation_before_commit == [False]
            assert len(mutation_results) == 1
        finally:
            runtime.close()

    def test_restore_does_not_roll_back_borrowed_view_root_or_external_capabilities(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='shared object owner')
            borrower = runtime.process.spawn(image='base-agent:v0', goal='checkpoint borrower')
            shared = runtime.memory.create_object(
                owner,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='shared'),
                immutable=False,
                name='shared.state',
            )
            borrowed = runtime.capability.handle_for_object(
                borrower,
                shared.oid,
                [CapabilityRight.READ],
                issued_by='test.borrowed',
            )
            runtime._add_handle_to_process_view(borrower, borrowed)
            checkpoint_id = runtime.checkpoint.create(borrower, 'borrowed root checkpoint', actor=borrower)

            runtime.memory.update_object(owner, shared, ObjectPatch(payload={'version': 2}))
            owner_capability = runtime.store.get_capability(shared.capability_id)
            assert owner_capability is not None and owner_capability.active

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.store.get_object(shared.oid).payload == {'version': 2}
            owner_capability = runtime.store.get_capability(shared.capability_id)
            assert owner_capability is not None and owner_capability.active
            assert runtime.capability.check(owner, f'object:{shared.oid}', CapabilityRight.WRITE)
            restored_borrower = runtime.process.get(borrower)
            assert shared.oid in {handle.oid for handle in restored_borrower.memory_view.roots}
        finally:
            runtime.close()

    def test_replay_to_event_is_scoped_to_checkpoint_process_subtree(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='replay scoped')
            unrelated = runtime.process.spawn(image='base-agent:v0', goal='unrelated')
            checkpoint_id = runtime.checkpoint.create(pid, 'before events', actor=pid)
            unrelated_event = runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source=unrelated,
                target=unrelated,
                payload={'secret': 'unrelated'},
            )
            related_event = runtime.events.emit(EventType.PROCESS_SIGNAL, source=pid, target=pid, payload={'ok': True})

            with pytest.raises(NotFound):
                runtime.checkpoint.replay_to_event(checkpoint_id, unrelated_event.event_id, actor=pid)
            replayed = runtime.checkpoint.replay_to_event(checkpoint_id, related_event.event_id, actor=pid)

            replayed_ids = [event['event_id'] for event in replayed['events']]
            assert unrelated_event.event_id not in replayed_ids
            assert replayed_ids[-1] == related_event.event_id
        finally:
            runtime.close()

    def test_restore_preserves_append_only_history_and_reports_external_effects(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='write external file')
                runtime.filesystem.grant_path(pid, 'out.txt', [CapabilityRight.WRITE], issued_by='test')
                checkpoint_id = runtime.checkpoint.create(pid, 'before external write', actor=pid)
                before_audit_count = len(runtime.audit.trace())
                runtime.filesystem.write_text(pid, 'out.txt', 'side effect')
                after_write_audit_count = len(runtime.audit.trace())
                result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert len(runtime.audit.trace()) >= after_write_audit_count + 1
                assert after_write_audit_count > before_audit_count
                assert (Path(temp_dir) / 'out.txt').exists()
                assert result['external_effects_since_checkpoint']
                assert result['restore_external_policy'] == 'report_only'
                summary = result['external_effect_summary']['by_rollback_class']
                assert summary['irreversible'] == 1
                assert 'rollbackable' not in summary
                filesystem_effect = result['external_effects_since_checkpoint'][0]
                assert filesystem_effect['provider'] == 'filesystem'
                assert filesystem_effect['rollback_class'] == 'irreversible'
                assert filesystem_effect['rollback_status'] == 'not_supported'
                assert 'checkpoint.restore' in [record.action for record in runtime.audit.trace()]
            finally:
                runtime.close()

    def test_checkpoint_effect_diff_uses_ledger_not_timestamps(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(goal='effect ledger watermark')
            pending = ExternalEffectRecord(
                effect_id='effect-before-checkpoint',
                record_id=None,
                event_id=None,
                pid=pid,
                provider='test',
                operation='prepare',
                target='test:prepared',
                rollback_class=ExternalEffectRollbackClass.UNKNOWN,
                rollback_status=ExternalEffectRollbackStatus.UNKNOWN,
                state_mutation=True,
                information_flow=False,
                provider_metadata={},
                created_at='2099-01-01T00:00:00+00:00',
                effect_state='pending',
                transaction_state='prepared',
                updated_at='2099-01-01T00:00:00+00:00',
            )
            runtime.store.insert_external_effect(pending)
            checkpoint_id = runtime.checkpoint.create(pid, 'ledger watermark', actor=pid)
            checkpoint = runtime.store.get_checkpoint_snapshot(checkpoint_id)[0]  # type: ignore[index]

            assert runtime.store.transition_external_effect(
                pending.effect_id,
                expected_states={'prepared'},
                transaction_state='dispatched',
                updated_at='1900-01-01T00:00:00+00:00',
            )
            runtime.store.insert_external_effect(
                ExternalEffectRecord(
                    effect_id='effect-same-timestamp',
                    record_id='audit-test',
                    event_id='event-test',
                    pid=pid,
                    provider='test',
                    operation='commit',
                    target='test:committed',
                    rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
                    rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
                    state_mutation=True,
                    information_flow=False,
                    provider_metadata={},
                    created_at=checkpoint.created_at,
                    effect_state='finalized',
                    transaction_state='committed',
                    updated_at=checkpoint.created_at,
                )
            )

            diff = runtime.checkpoint.diff(checkpoint_id, actor=pid)
            effects = diff['external_effects_since_checkpoint']
            assert {effect['effect_id'] for effect in effects} == {
                'effect-before-checkpoint',
                'effect-same-timestamp',
            }
            assert diff['external_effect_summary']['total'] == len(effects) == 2
        finally:
            runtime.close()

    def test_restore_reports_provider_decided_external_effect_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            substrate = LocalResourceProviderSubstrate(temp_dir)
            substrate.shell = ClassifiedShellProvider()
            substrate.human = LocalHumanProvider(output_sink=lambda _message: None)
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='external effect classes')
                runtime.filesystem.grant_path(pid, 'out.txt', [CapabilityRight.WRITE], issued_by='test')
                runtime.capability.grant(pid, 'shell:git', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
                checkpoint_id = runtime.checkpoint.create(pid, 'before external effects', actor=pid)
                runtime.filesystem.write_text(pid, 'out.txt', 'side effect')
                runtime.shell.run(pid, ['git', 'status', '--short'])
                runtime.human.output(pid, 'visible once')
                result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                summary = result['external_effect_summary']['by_rollback_class']
                effects = result['external_effects_since_checkpoint']
                assert result['status'] == 'restored'
                assert result['restore_external_policy'] == 'report_only'
                assert 'rollbackable' not in summary
                assert summary['irreversible'] == 2
                assert summary['no_rollback_required'] == 1
                assert {(effect['provider'], effect['rollback_class']) for effect in effects} == {('filesystem', 'irreversible'), ('shell', 'irreversible'), ('human', 'no_rollback_required')}
                filesystem_effect = next(effect for effect in effects if effect['provider'] == 'filesystem')
                assert filesystem_effect['rollback_status'] == 'not_supported'
            finally:
                runtime.close()

    def test_checkpoint_external_effect_report_is_scoped_to_process_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
                runtime.capability.grant(owner, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                child = runtime.spawn_child_process(owner, 'child')
                unrelated = runtime.process.spawn(image='base-agent:v0', goal='other')
                runtime.filesystem.grant_path(owner, 'owner.txt', [CapabilityRight.WRITE], issued_by='test')
                runtime.filesystem.grant_path(child, 'child.txt', [CapabilityRight.WRITE], issued_by='test')
                runtime.filesystem.grant_path(unrelated, 'other.txt', [CapabilityRight.WRITE], issued_by='test')
                checkpoint_id = runtime.checkpoint.create(owner, 'before external writes', actor=owner)
                runtime.filesystem.write_text(owner, 'owner.txt', 'owner')
                runtime.filesystem.write_text(child, 'child.txt', 'child')
                runtime.filesystem.write_text(unrelated, 'other.txt', 'other')
                diff = runtime.checkpoint.diff(checkpoint_id, actor=owner)
                result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert diff['external_effect_summary']['total'] == 2
                assert result['external_effect_summary']['total'] == 2
                assert {effect['target'] for effect in result['external_effects_since_checkpoint']} == {'filesystem:workspace:owner.txt', 'filesystem:workspace:child.txt'}
            finally:
                runtime.close()

    def test_restore_supersedes_post_checkpoint_mailbox_and_cancels_human_requests(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='messages',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'human:owner', 'rights': [CapabilityRight.WRITE.value]}
                    ]
                },
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before messages', actor=pid)
            message = runtime.human.send_process_message(pid, 'late update')
            request_id = runtime.human.ask(pid, 'approve?', blocking=True)
            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            assert runtime.store.get_process_message(message.message_id).status == ProcessMessageStatus.SUPERSEDED_BY_RESTORE
            assert runtime.messages.unread(pid) == []
            assert runtime.human.get(request_id).status == HumanRequestStatus.CANCELLED
            assert result['superseded_messages'] == [message.message_id]
            assert result['cancelled_human_requests'] == [request_id]
        finally:
            runtime.close()

    def test_checkpoint_payload_capture_limit_is_enforced(self) -> None:
        config = AgentLibOSConfig(checkpoint=CheckpointDefaults(payload_capture_limit_bytes=8))
        runtime = Runtime.open('local', config=config)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='large payload')
            runtime.memory.create_object(pid, ObjectType.SUMMARY, {'data': 'too large'}, name='large')
            with pytest.raises(ValidationError):
                runtime.checkpoint.create(pid, 'too large', actor=pid)
        finally:
            runtime.close()

    def test_checkpoint_safe_point_normalizes_running_status(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='safe point')
            token = runtime.store.claim_execution(pid, owner_id="checkpoint-safe-point-test")
            assert token is not None
            checkpoint_id = runtime.checkpoint.create(pid, 'running safe point', actor=pid)
            inspected = runtime.checkpoint.inspect(checkpoint_id, actor=pid)
            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            assert inspected['processes'][0]['status'] == ProcessStatus.RUNNABLE.value
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_checkpoint_create_captures_object_row_and_payload_from_one_store_snapshot(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        snapshot_reached_payloads = threading.Event()
        writer_attempted = threading.Event()
        writer_acquired_store = threading.Event()
        allow_writer_commit = threading.Event()
        writer_errors: list[BaseException] = []
        writer: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='consistent checkpoint')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'generation': 1},
                ObjectMetadata(title='consistent state'),
                immutable=False,
                name='consistent.state',
            )
            before = runtime.store.get_object(handle.oid)
            assert before is not None
            original_payload_snapshot = runtime.checkpoint._object_payload_snapshot

            def coordinated_payload_snapshot(object_oids: list[str]) -> dict[str, object]:
                snapshot_reached_payloads.set()
                assert writer_attempted.wait(timeout=5)
                # Without a transaction spanning the checkpoint reads, the
                # writer acquires the store here and can split the object row
                # from its payload. With the transaction it remains blocked
                # until the checkpoint has been durably inserted.
                writer_acquired_store.wait(timeout=0.2)
                allow_writer_commit.set()
                return original_payload_snapshot(object_oids)

            monkeypatch.setattr(runtime.checkpoint, '_object_payload_snapshot', coordinated_payload_snapshot)

            def mutate_row_and_payload_atomically() -> None:
                try:
                    assert snapshot_reached_payloads.wait(timeout=5)
                    writer_attempted.set()
                    with runtime.store.transaction(include_object_payloads=True) as cur:
                        writer_acquired_store.set()
                        cur.execute(
                            'UPDATE objects SET version = ? WHERE oid = ?',
                            (before.version + 1, handle.oid),
                        )
                        runtime.store.set_object_payload(handle.oid, {'generation': 2})
                        assert allow_writer_commit.wait(timeout=5)
                except BaseException as exc:
                    writer_errors.append(exc)

            writer = threading.Thread(target=mutate_row_and_payload_atomically)
            writer.start()
            checkpoint_id = runtime.checkpoint.create(pid, 'consistent snapshot', actor=pid)
            writer.join(timeout=5)
            assert not writer.is_alive()
            assert writer_errors == []

            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _checkpoint, snapshot = found
            object_row = next(row for row in snapshot['rows']['objects'] if row['oid'] == handle.oid)
            captured_pair = (object_row['version'], snapshot['object_payloads'][handle.oid]['generation'])
            assert captured_pair in {(before.version, 1), (before.version + 1, 2)}
        finally:
            allow_writer_commit.set()
            if writer is not None:
                writer.join(timeout=5)
            runtime.close()

    def test_checkpoint_snapshot_canonicalizes_process_capability_index_during_concurrent_grant(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        capability_inserted = threading.Event()
        allow_process_attach = threading.Event()
        grant_errors: list[BaseException] = []
        grant_result: list[object] = []
        checkpoint_started = threading.Event()
        checkpoint_done = threading.Event()
        checkpoint_errors: list[BaseException] = []
        checkpoint_result: list[str] = []
        grant_thread: threading.Thread | None = None
        checkpoint_thread: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='capability snapshot')
            original_attach = runtime.capability._attach_to_process

            def pause_grant_attach(subject: str, cap_id: str) -> None:
                if threading.current_thread() is grant_thread:
                    capability_inserted.set()
                    assert allow_process_attach.wait(timeout=5)
                original_attach(subject, cap_id)

            monkeypatch.setattr(runtime.capability, '_attach_to_process', pause_grant_attach)

            def grant_capability() -> None:
                try:
                    grant_result.append(
                        runtime.capability.grant(pid, 'test:concurrent-checkpoint', [CapabilityRight.READ], issued_by='test')
                    )
                except BaseException as exc:
                    grant_errors.append(exc)

            grant_thread = threading.Thread(target=grant_capability)
            grant_thread.start()
            assert capability_inserted.wait(timeout=5)

            def create_checkpoint() -> None:
                checkpoint_started.set()
                try:
                    checkpoint_result.append(
                        runtime.checkpoint.create(pid, 'capability index snapshot', actor=pid)
                    )
                except BaseException as exc:
                    checkpoint_errors.append(exc)
                finally:
                    checkpoint_done.set()

            checkpoint_thread = threading.Thread(target=create_checkpoint)
            checkpoint_thread.start()
            assert checkpoint_started.wait(timeout=5)
            # Capability issue is now a single transaction.  A checkpoint must
            # wait instead of observing the inserted capability before its
            # process index attachment and evidence are committed.
            assert not checkpoint_done.wait(timeout=0.1)
            allow_process_attach.set()
            grant_thread.join(timeout=5)
            assert not grant_thread.is_alive()
            assert grant_errors == []
            assert len(grant_result) == 1
            checkpoint_thread.join(timeout=5)
            assert not checkpoint_thread.is_alive()
            assert checkpoint_errors == []
            assert len(checkpoint_result) == 1

            checkpoint_id = checkpoint_result[0]
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _checkpoint, snapshot = found
            captured_capability = next(
                row
                for row in snapshot['rows']['capabilities']
                if row['resource'] == 'test:concurrent-checkpoint'
            )
            process_row = next(row for row in snapshot['rows']['processes'] if row['pid'] == pid)
            assert captured_capability['cap_id'] in loads(process_row['capabilities_json'], [])
        finally:
            allow_process_attach.set()
            if grant_thread is not None:
                grant_thread.join(timeout=5)
            if checkpoint_thread is not None:
                checkpoint_thread.join(timeout=5)
            runtime.close()

    def test_restore_refuses_while_scheduler_quantum_is_active(self) -> None:
        runtime = Runtime.open('local')
        resume = threading.Event()
        entered = threading.Event()
        errors: list[BaseException] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='active restore')
            checkpoint_id = runtime.checkpoint.create(pid, 'before active quantum', actor=pid)

            def quantum(selected_pid: str) -> dict[str, object]:
                entered.set()
                assert selected_pid == pid
                assert resume.wait(timeout=5)
                return {'ok': True}

            def run_scheduler() -> None:
                try:
                    runtime.scheduler.run_pid_until_idle(pid, quantum, max_quanta=1)
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run_scheduler)
            thread.start()
            assert entered.wait(timeout=5)

            with pytest.raises(ValidationError, match='scheduler is running'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            resume.set()
            thread.join(timeout=5)
            assert not thread.is_alive()
            assert errors == []
        finally:
            resume.set()
            runtime.close()

    def test_restore_runs_release_finalizers_for_deleted_scoped_objects(self) -> None:
        runtime = Runtime.open('local')
        calls: list[tuple[str, str, str]] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore finalizer')
            checkpoint_id = runtime.checkpoint.create(pid, 'before temporary object', actor=pid)
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'temp': True},
                ObjectMetadata(title='temp'),
                immutable=False,
                name='temp',
            )
            runtime.memory.bind_durable_object_release_finalizer(
                'test.restore-finalizer:v1',
                lambda obj, _actor, _reason, _work_id: {'oid': obj.oid},
                lambda intent, actor, reason, _work_id: calls.append(
                    (str(intent['oid']), actor, reason)
                ),
            )

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert (handle.oid, 'checkpoint.restore', 'checkpoint_restore') in calls
            with pytest.raises(Exception):
                runtime.memory.get_object_by_name(pid, 'temp')
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        "nested_thread",
        [False, True],
        ids=["same-thread", "other-thread"],
    )
    def test_restore_rejects_nested_restore_from_durable_finalizer(
        self,
        nested_thread: bool,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="nested checkpoint restore must fail fast",
            )
            outer_checkpoint_id = runtime.checkpoint.create(
                pid,
                "before nested restore object",
                actor=pid,
            )
            temporary = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {"restored_by": "nested-checkpoint"},
                ObjectMetadata(title="nested restore object"),
                immutable=False,
                name="nested.restore.object",
            )
            nested_checkpoint_id = runtime.checkpoint.create(
                pid,
                "nested restore would resurrect object",
                actor=pid,
            )
            publications_before = {
                str(publication["publication_id"])
                for publication in runtime.store.list_runtime_publications()
                if publication["kind"] == "checkpoint_restore"
            }

            def invoke_nested_restore() -> None:
                runtime.checkpoint.restore(
                    "cli",
                    nested_checkpoint_id,
                    require_capability=False,
                )

            def nested_finalizer(
                _intent: object,
                _actor: str,
                _reason: str,
                _work_id: str,
            ) -> None:
                if not nested_thread:
                    invoke_nested_restore()
                    return
                errors: list[BaseException] = []

                def run() -> None:
                    try:
                        invoke_nested_restore()
                    except BaseException as exc:
                        errors.append(exc)

                thread = threading.Thread(target=run)
                thread.start()
                thread.join(timeout=5)
                if thread.is_alive():
                    raise RuntimeError("nested restore did not fail fast")
                if not errors:
                    raise RuntimeError("nested restore unexpectedly succeeded")
                raise errors[0]

            runtime.memory.bind_durable_object_release_finalizer(
                "test.nested-restore-finalizer:v1",
                lambda obj, _actor, _reason, _work_id: {"oid": obj.oid},
                nested_finalizer,
            )

            result = runtime.checkpoint.restore(
                "cli",
                outer_checkpoint_id,
                require_capability=False,
            )

            assert runtime.store.get_object(temporary.oid) is None
            assert result["status"] == "restored_with_warnings"
            assert result["main_state_committed"] is True
            assert result["post_commit_failures"] == [
                {
                    "phase": "object_release_finalizers",
                    "error_type": "ValidationError",
                    "message": "checkpoint restore or recovery is already in progress",
                }
            ]
            restore_publications = [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication["kind"] == "checkpoint_restore"
            ]
            assert {
                str(publication["publication_id"])
                for publication in restore_publications
            } == publications_before | {str(result["publication_id"])}
            publication = runtime.store.get_runtime_publication(
                str(result["publication_id"])
            )
            assert publication is not None
            assert publication["state"] == "failed"
            assert publication["phase"] == "object_release_finalizers_failed"
            assert publication["plan"]["checkpoint_id"] == outer_checkpoint_id
            assert not [
                receipt
                for receipt in publication["receipt"]["phases"]
                if receipt.get("phase")
                == "checkpoint_restore_finalizer_completed"
            ]
            operation = runtime.store.get_operation(
                publication["plan"]["operation_id"]
            )
            assert operation is not None
            assert operation.outcome.value == "unknown"
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    def test_restore_rejects_anonymous_finalizer_before_deleting_object(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject anonymous restore finalizer',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before anonymous-finalizer object',
                actor=pid,
            )
            temporary = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'temporary': True},
                ObjectMetadata(title='temporary'),
                immutable=False,
                name='anonymous.finalizer.temporary',
            )
            runtime.memory.bind_object_release_finalizer(
                lambda _obj, _actor, _reason: None
            )

            with pytest.raises(
                ValidationError,
                match='use bind_durable_object_release_finalizer',
            ):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.store.get_object(temporary.oid) is not None
            assert not [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            ]
        finally:
            runtime.close()

    def test_restore_does_not_release_finalizers_for_restored_live_objects(self) -> None:
        runtime = Runtime.open('local')
        calls: list[tuple[str, str, str]] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore keeps object')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'keep': True},
                ObjectMetadata(title='keep'),
                immutable=False,
                name='keep',
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'after persistent object', actor=pid)
            runtime.memory.bind_object_release_finalizer(
                lambda obj, actor, reason: calls.append((obj.oid, actor, reason))
            )

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert calls == []
            assert runtime.store.get_object(handle.oid) is not None
            restored = runtime.memory.get_object_by_name(pid, 'keep')
            assert restored.oid == handle.oid
        finally:
            runtime.close()

    def test_restore_reconciles_terminal_human_wait_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='human wait',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'human:owner', 'rights': [CapabilityRight.WRITE.value]}
                    ]
                },
            )
            request_id = runtime.human.ask(pid, 'continue?', blocking=True)
            checkpoint_id = runtime.checkpoint.create(pid, 'waiting for human', actor=pid)
            runtime.human.approve(request_id, {'approved': True, 'answer': 'yes'})

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.process.get(pid)
            assert restored.status == ProcessStatus.RUNNABLE
            assert restored.status_message is None
            assert runtime.human.get(request_id).status == HumanRequestStatus.APPROVED
        finally:
            runtime.close()

    def test_restore_refuses_scoped_active_object_task(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='object task restore')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before task', actor=pid)
            task = runtime.object_tasks.start(pid, owner, 'receive_process_messages', {'channel': 'never'})
            waiting = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
            restore_publications_before = {
                str(publication['publication_id'])
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            }

            with pytest.raises(ValidationError, match='ObjectTasks are active'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.object_tasks.get(task.task_id, actor_pid=pid).status == waiting.status
            assert {
                str(publication['publication_id'])
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            } == restore_publications_before
        finally:
            runtime.close()

    def test_restore_pauses_tool_wait_after_captured_object_task_completes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='captured ObjectTask terminal restore',
            )
            runtime.capability.grant(
                pid,
                'process:spawn',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            original_schedule = runtime.object_tasks._schedule_task_locked
            monkeypatch.setattr(
                runtime.object_tasks,
                '_schedule_task_locked',
                lambda _task_id: None,
            )
            task = runtime.object_tasks.start(
                pid,
                owner,
                'get_working_directory',
                {},
            )
            assert task.runner_pid is not None
            runner_pid = str(task.runner_pid)
            captured_runner = runtime.process.get(runner_pid)
            assert captured_runner.status == ProcessStatus.WAITING_TOOL
            assert captured_runner.wait_state == ToolProcessWait(
                operation_id=task.task_id
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'while ObjectTask runner is waiting for host execution',
                actor=pid,
            )

            monkeypatch.setattr(
                runtime.object_tasks,
                '_schedule_task_locked',
                original_schedule,
            )
            with runtime.object_tasks._lock:
                runtime.object_tasks._schedule_task_locked(task.task_id)
            completed = runtime.object_tasks.wait(
                task.task_id,
                actor_pid=pid,
                timeout=3,
            )
            assert completed.status == ObjectTaskStatus.SUCCEEDED

            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )

            restored_task = runtime.object_tasks.get(task.task_id, actor_pid=pid)
            assert restored_task.status == ObjectTaskStatus.SUPERSEDED_BY_RESTORE
            restored_runner = runtime.process.get(runner_pid)
            assert restored_runner.status == ProcessStatus.PAUSED
            assert restored_runner.wait_state == PausedProcessWait()
            assert restored_runner.status_message == (
                'restored tool wait ObjectTask is not active: '
                f'{task.task_id}/superseded_by_restore'
            )
            assert result['superseded_object_tasks'] == [task.task_id]
        finally:
            runtime.close()

    @pytest.mark.parametrize('task_binding', ['missing', 'runner-mismatch'])
    def test_restore_pauses_tool_wait_without_exact_active_object_task_binding(
        self,
        task_binding: str,
    ) -> None:
        runtime = Runtime.open('local')
        operation_id = f'otask_restore_{task_binding}'
        mismatched_task: ObjectTask | None = None
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal=f'{task_binding} ObjectTask restore binding',
            )
            runtime.capability.grant(
                pid,
                'process:spawn',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            runner_pid = runtime.process.spawn_child(
                pid,
                goal={'type': 'tool-wait-binding-test'},
                initial_status=ProcessStatus.WAITING_TOOL,
                initial_wait_state=ToolProcessWait(operation_id=operation_id),
            )
            if task_binding == 'runner-mismatch':
                mismatched_task = ObjectTask(
                    task_id=operation_id,
                    owner_oid='oid_outside_restore_scope',
                    creator_pid='pid_outside_restore_scope',
                    runner_pid='pid_different_runner',
                    tool='get_working_directory',
                    tool_id=None,
                    status=ObjectTaskStatus.QUEUED,
                    created_at='2026-01-01T00:00:00Z',
                    updated_at='2026-01-01T00:00:00Z',
                )
                runtime.store.insert_object_task(mismatched_task)
            checkpoint_id = runtime.checkpoint.create(
                pid,
                f'captured {task_binding} ObjectTask binding',
                actor=pid,
            )

            runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )

            restored_runner = runtime.process.get(runner_pid)
            assert restored_runner.status == ProcessStatus.PAUSED
            assert restored_runner.wait_state == PausedProcessWait()
            if task_binding == 'missing':
                assert restored_runner.status_message == (
                    f'restored tool wait ObjectTask is missing: {operation_id}'
                )
            else:
                assert restored_runner.status_message == (
                    'restored tool wait ObjectTask runner does not match: '
                    f'{operation_id}/expected={runner_pid}/actual=pid_different_runner'
                )
        finally:
            if mismatched_task is not None:
                latest = runtime.store.get_object_task(mismatched_task.task_id)
                if latest is not None and latest.status == ObjectTaskStatus.QUEUED:
                    latest.status = ObjectTaskStatus.FAILED
                    latest.error = 'test cleanup'
                    latest.completed_at = latest.updated_at
                    runtime.store.update_object_task(latest)
            runtime.close()

    def test_restore_supersedes_terminal_object_task_completed_after_checkpoint(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='terminal object task restore')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before terminal task', actor=pid)
            task = runtime.object_tasks.start(pid, owner, 'get_working_directory', {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=3)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.runner_pid is not None
            old_runner_pid = str(completed.runner_pid)
            old_result_oid = str(completed.result_oid) if completed.result_oid is not None else None

            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored_task = runtime.object_tasks.get(task.task_id, actor_pid=pid)
            assert restored_task.status == ObjectTaskStatus.SUPERSEDED_BY_RESTORE
            assert restored_task.runner_pid is None
            assert restored_task.result_oid is None
            assert restored_task.wait['superseded_by_restore'] == checkpoint_id
            assert restored_task.wait['previous_status'] == ObjectTaskStatus.SUCCEEDED.value
            assert restored_task.wait['previous_runner_pid'] == old_runner_pid
            assert restored_task.wait['previous_result_oid'] == old_result_oid
            assert runtime.store.get_process(old_runner_pid) is None
            if old_result_oid is not None:
                assert runtime.store.get_object(old_result_oid) is None
            assert result['superseded_object_tasks'] == [task.task_id]
            validated_output = RestoreCheckpointOutput.model_validate(result)
            assert validated_output.superseded_object_tasks == [task.task_id]
        finally:
            runtime.close()

    def test_reopen_then_restore_supersedes_task_completed_after_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database = tmp_path / 'checkpoint-object-task-post-success.sqlite'
        runtime = Runtime.open(database)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='reopen post-checkpoint task')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before terminal task', actor=pid)
            task = runtime.object_tasks.start(pid, owner, 'get_working_directory', {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=3)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.runner_pid is not None
            assert completed.result_oid is not None
            old_runner_pid = str(completed.runner_pid)
            old_result_oid = str(completed.result_oid)
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            degraded = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert degraded.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert degraded.runner_pid == old_runner_pid
            assert degraded.result_oid is None
            assert degraded.wait['previous_result_oid'] == old_result_oid

            original_reconcile = reopened.checkpoint._reconcile_restored_wait_states

            def fail_restore_after_task_supersede(cur: object, pids: list[str]) -> None:
                original_reconcile(cur, pids)
                raise RuntimeError('injected failure after ObjectTask supersede')

            monkeypatch.setattr(
                reopened.checkpoint,
                '_reconcile_restored_wait_states',
                fail_restore_after_task_supersede,
            )
            with pytest.raises(RuntimeError, match='injected failure after ObjectTask supersede'):
                reopened.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            rolled_back_task = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert rolled_back_task.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert rolled_back_task.runner_pid == old_runner_pid
            assert reopened.store.get_process(old_runner_pid) is not None
            monkeypatch.setattr(
                reopened.checkpoint,
                '_reconcile_restored_wait_states',
                original_reconcile,
            )

            result = reopened.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored_task = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert restored_task.status == ObjectTaskStatus.SUPERSEDED_BY_RESTORE
            assert restored_task.runner_pid is None
            assert restored_task.result_oid is None
            assert restored_task.wait['previous_status'] == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN.value
            assert restored_task.wait['previous_runner_pid'] == old_runner_pid
            assert restored_task.wait['previous_result_oid'] == old_result_oid
            assert reopened.store.get_process(old_runner_pid) is None
            assert reopened.store.get_object(old_result_oid) is None
            assert result['superseded_object_tasks'] == [task.task_id]
        finally:
            reopened.close()

    def test_reopen_then_restore_repairs_task_result_captured_by_checkpoint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        database = tmp_path / 'checkpoint-object-task-pre-success.sqlite'
        runtime = Runtime.open(database)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='reopen pre-checkpoint task')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            task = runtime.object_tasks.start(pid, owner, 'get_working_directory', {})
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=3)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.runner_pid is not None
            assert completed.result_oid is not None
            old_runner_pid = str(completed.runner_pid)
            old_result_oid = str(completed.result_oid)
            checkpoint_id = runtime.checkpoint.create(pid, 'after terminal task', actor=pid)
        finally:
            runtime.close()

        reopened = Runtime.open(database)
        try:
            degraded = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert degraded.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert degraded.result_oid is None
            assert reopened.store.get_object(old_result_oid) is None

            original_reconcile = reopened.checkpoint._reconcile_restored_object_task_results

            def fail_restore_after_result_repair(
                cur: object,
                snapshot: dict[str, object],
                checkpoint: object,
            ) -> list[str]:
                original_reconcile(cur, snapshot, checkpoint)  # type: ignore[arg-type]
                raise RuntimeError('injected failure after ObjectTask result repair')

            monkeypatch.setattr(
                reopened.checkpoint,
                '_reconcile_restored_object_task_results',
                fail_restore_after_result_repair,
            )
            with pytest.raises(RuntimeError, match='injected failure after ObjectTask result repair'):
                reopened.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            rolled_back_task = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert rolled_back_task.status == ObjectTaskStatus.RESULT_UNAVAILABLE_AFTER_REOPEN
            assert rolled_back_task.result_oid is None
            assert reopened.store.get_object(old_result_oid) is None
            monkeypatch.setattr(
                reopened.checkpoint,
                '_reconcile_restored_object_task_results',
                original_reconcile,
            )

            result = reopened.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored_task = reopened.object_tasks.get(task.task_id, actor_pid=pid)
            assert restored_task.status == ObjectTaskStatus.SUCCEEDED
            assert restored_task.runner_pid == old_runner_pid
            assert restored_task.result_oid == old_result_oid
            assert restored_task.error is None
            assert restored_task.wait['result_restored_by_checkpoint'] == checkpoint_id
            assert reopened.store.get_process(old_runner_pid) is not None
            assert reopened.store.get_object(old_result_oid) is not None
            assert result['superseded_object_tasks'] == []
        finally:
            reopened.close()

    def test_restore_keeps_task_completed_before_checkpoint_when_notification_finishes_afterward(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        notification_started = threading.Event()
        release_notification = threading.Event()
        notifications = runtime.object_tasks._notifications
        original_notify = notifications.notify

        def delayed_notify(task: object, *, phase: str) -> object:
            if phase == 'completed':
                notification_started.set()
                if not release_notification.wait(timeout=5):
                    raise TimeoutError('timed out waiting to release ObjectTask notification')
            return original_notify(task, phase=phase)  # type: ignore[arg-type]

        monkeypatch.setattr(notifications, 'notify', delayed_notify)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='terminal timestamp restore race')
            runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            owner = runtime.memory.create_object(
                pid,
                ObjectType.ARTIFACT,
                {'name': 'owner'},
                metadata=ObjectMetadata(title='owner'),
                immutable=False,
            )
            task = runtime.object_tasks.start(pid, owner, 'get_working_directory', {})
            assert notification_started.wait(timeout=3)

            terminal_before_checkpoint = runtime.store.get_object_task(task.task_id)
            assert terminal_before_checkpoint is not None
            assert terminal_before_checkpoint.status == ObjectTaskStatus.SUCCEEDED
            assert terminal_before_checkpoint.completed_at is not None
            result_oid = terminal_before_checkpoint.result_oid
            assert result_oid is not None
            checkpoint_id = runtime.checkpoint.create(pid, 'after terminal transition', actor=pid)
            checkpoint = next(
                item
                for item in runtime.store.list_checkpoints(pid=pid)
                if item.checkpoint_id == checkpoint_id
            )

            release_notification.set()
            completed = runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=3)
            assert completed.status == ObjectTaskStatus.SUCCEEDED
            assert completed.updated_at >= checkpoint.created_at

            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored_task = runtime.object_tasks.get(task.task_id, actor_pid=pid)
            assert restored_task.status == ObjectTaskStatus.SUCCEEDED
            assert restored_task.result_oid == result_oid
            assert runtime.store.get_object(result_oid) is not None
            assert result['superseded_object_tasks'] == []
        finally:
            release_notification.set()
            runtime.close()

    def test_restore_requires_image_admin_to_replace_current_image(self) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-restore-image:v0'
        try:
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-restore-image', system_prompt='snapshot prompt'),
                actor='test',
            )
            pid = runtime.process.spawn(image=image_id, goal='checkpoint image source')
            checkpoint_id = runtime.checkpoint.create(pid, 'image restore point', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.ADMIN], issued_by='test')
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-restore-image', system_prompt='current prompt'),
                actor='test',
                replace=True,
            )

            with pytest.raises(CapabilityDenied, match=f'image:{image_id}'):
                runtime.checkpoint.restore(pid, checkpoint_id)

            assert runtime.get_image(image_id).system_prompt == 'current prompt'
            runtime.capability.grant(pid, runtime.image_registry.resource_for(image_id), [CapabilityRight.WRITE], issued_by='test')
            with pytest.raises(CapabilityDenied, match=f'image:{image_id}'):
                runtime.checkpoint.restore(pid, checkpoint_id)
            assert runtime.get_image(image_id).system_prompt == 'current prompt'
            runtime.capability.grant(pid, runtime.image_registry.resource_for(image_id), [CapabilityRight.ADMIN], issued_by='test')
            restored = runtime.checkpoint.restore(pid, checkpoint_id)
            assert restored['status'] == 'restored'
            assert runtime.get_image(image_id).system_prompt == 'snapshot prompt'
        finally:
            runtime.close()

    def test_restore_holds_registry_lifecycle_lock_through_registry_reconciliation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        load_entered = threading.Event()
        allow_load = threading.Event()
        images_entered = threading.Event()
        allow_images = threading.Event()
        probe_acquired = threading.Event()
        restore_result: list[dict[str, object]] = []
        failures: list[BaseException] = []
        restore_thread: threading.Thread | None = None
        probe_thread: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='registry-ordered restore')
            checkpoint_id = runtime.checkpoint.create(pid, 'registry lock boundary', actor=pid)
            original_load = runtime.checkpoint._load_checkpoint_typed
            original_restore_images = runtime.checkpoint._restore_images

            def gated_load(selected_checkpoint_id: str):
                load_entered.set()
                if not allow_load.wait(timeout=3):
                    raise RuntimeError('timed out waiting to continue checkpoint load')
                return original_load(selected_checkpoint_id)

            def gated_restore_images(snapshot: dict[str, object], *, overwrite_existing: bool = True) -> None:
                images_entered.set()
                if not allow_images.wait(timeout=3):
                    raise RuntimeError('timed out waiting to continue image reconciliation')
                original_restore_images(snapshot, overwrite_existing=overwrite_existing)

            monkeypatch.setattr(runtime.checkpoint, '_load_checkpoint_typed', gated_load)
            monkeypatch.setattr(runtime.checkpoint, '_restore_images', gated_restore_images)

            def restore() -> None:
                try:
                    restore_result.append(
                        runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                    )
                except BaseException as exc:
                    failures.append(exc)

            def probe_registry() -> None:
                with runtime._registry_lifecycle_lock:
                    probe_acquired.set()

            restore_thread = threading.Thread(target=restore)
            restore_thread.start()
            assert load_entered.wait(timeout=3)

            probe_thread = threading.Thread(target=probe_registry)
            probe_thread.start()
            assert not probe_acquired.wait(timeout=0.2)

            allow_load.set()
            assert images_entered.wait(timeout=3)
            assert not probe_acquired.wait(timeout=0.2)

            allow_images.set()
            restore_thread.join(timeout=3)
            probe_thread.join(timeout=3)

            assert not restore_thread.is_alive()
            assert not probe_thread.is_alive()
            assert failures == []
            assert restore_result[0]['status'] == 'restored'
            assert probe_acquired.is_set()
        finally:
            allow_load.set()
            allow_images.set()
            if restore_thread is not None:
                restore_thread.join(timeout=3)
            if probe_thread is not None:
                probe_thread.join(timeout=3)
            runtime.close()

    def test_restore_image_write_failure_keeps_cache_and_store_in_sync(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'image-reconciliation.sqlite3'
        runtime = Runtime.open(target)
        image_id = 'checkpoint-restore-atomic-image:v0'
        publication_id = ''
        try:
            runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name='checkpoint-restore-atomic-image',
                    system_prompt='captured prompt',
                ),
                actor='test',
            )
            pid = runtime.process.spawn(image=image_id, goal='atomic image reconciliation')
            checkpoint_id = runtime.checkpoint.create(pid, 'captured image', actor=pid)
            runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name='checkpoint-restore-atomic-image',
                    system_prompt='current prompt',
                ),
                actor='test',
                replace=True,
            )
            original_upsert = runtime.store.upsert_image

            def fail_captured_image_write(image: AgentImage, *args: object, **kwargs: object) -> None:
                if image.image_id == image_id and image.system_prompt == 'captured prompt':
                    raise RuntimeError('injected restored image write failure')
                original_upsert(image, *args, **kwargs)

            monkeypatch.setattr(runtime.store, 'upsert_image', fail_captured_image_write)

            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            publication_id = str(result['publication_id'])

            persisted = runtime.store.get_image(image_id)
            assert persisted is not None
            assert result['status'] == 'restored_with_warnings'
            assert [failure['phase'] for failure in result['post_commit_failures']] == ['image_reconciliation']
            assert runtime.get_image(image_id).system_prompt == 'current prompt'
            assert persisted[0].system_prompt == 'current prompt'
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['state'] == 'failed'
            assert runtime.lifecycle.state == 'close_failed'

            monkeypatch.setattr(runtime.store, 'upsert_image', original_upsert)
            _release_checkpoint_recovery_runtime(runtime)
            reopened = Runtime.open(target)
            try:
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                assert reopened.get_image(image_id).system_prompt == 'captured prompt'
                persisted = reopened.store.get_image(image_id)
                assert persisted is not None
                assert persisted[0].system_prompt == 'captured prompt'
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
            finally:
                reopened.close()

            reopened_again = Runtime.open(target)
            try:
                assert reopened_again.recovered_checkpoint_restore_publications == []
                assert reopened_again.get_image(image_id).system_prompt == 'captured prompt'
            finally:
                reopened_again.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    @pytest.mark.postgres
    def test_postgres_reopen_reconciles_checkpoint_restore_image_phase_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_checkpoint_target() as target:
            runtime = Runtime.open(target)
            image_id = "checkpoint-postgres-reconciliation:v0"
            publication_id = ""
            try:
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name="checkpoint-postgres-reconciliation",
                        system_prompt="captured prompt",
                    ),
                    actor="test",
                )
                pid = runtime.process.spawn(
                    image=image_id,
                    goal="postgres checkpoint reconciliation",
                )
                handle = runtime.memory.create_object(
                    pid,
                    ObjectType.SUMMARY,
                    {"version": 1},
                    ObjectMetadata(title="postgres restored payload"),
                    immutable=False,
                    name="postgres.restore.payload",
                )
                checkpoint_id = runtime.checkpoint.create(
                    pid,
                    "captured postgres image",
                    actor=pid,
                )
                runtime.memory.update_object(
                    pid,
                    handle,
                    ObjectPatch(payload={"version": 2}),
                )
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name="checkpoint-postgres-reconciliation",
                        system_prompt="current prompt",
                    ),
                    actor="test",
                    replace=True,
                )
                original_upsert = runtime.store.upsert_image

                def fail_captured_image_write(
                    image: AgentImage,
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    if (
                        image.image_id == image_id
                        and image.system_prompt == "captured prompt"
                    ):
                        raise RuntimeError("injected postgres image phase failure")
                    original_upsert(image, *args, **kwargs)

                monkeypatch.setattr(
                    runtime.store,
                    "upsert_image",
                    fail_captured_image_write,
                )
                result = runtime.checkpoint.restore(
                    "cli",
                    checkpoint_id,
                    require_capability=False,
                )
                publication_id = str(result["publication_id"])
                assert result["main_state_committed"] is True
                assert result["reconciliation_pending"] is True
                assert runtime.lifecycle.state == "close_failed"
                publication = runtime.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication["state"] == "failed"

                monkeypatch.setattr(runtime.store, "upsert_image", original_upsert)
                _release_checkpoint_recovery_runtime(runtime)
                reopened = Runtime.open(target)
                try:
                    assert reopened.recovered_checkpoint_restore_publications == [
                        publication_id
                    ]
                    publication = reopened.store.get_runtime_publication(
                        publication_id
                    )
                    assert publication is not None
                    assert publication["state"] == "committed"
                    assert publication["phase"] == "reconciled"
                    assert publication["receipt"]["payload_delivery"] == {
                        "state": "completed"
                    }
                    assert reopened.store.object_payload(handle.oid) == {
                        "version": 1
                    }
                    assert (
                        reopened.get_image(image_id).system_prompt
                        == "captured prompt"
                    )
                finally:
                    reopened.close()

                reopened_again = Runtime.open(target)
                try:
                    assert (
                        reopened_again.recovered_checkpoint_restore_publications
                        == []
                    )
                    publication = reopened_again.store.get_runtime_publication(
                        publication_id
                    )
                    assert publication is not None
                    recovery_claims = [
                        item
                        for item in publication["receipt"]["phases"]
                        if item.get("phase") == "recovery_claimed"
                    ]
                    assert len(recovery_claims) == 1
                finally:
                    reopened_again.close()
            finally:
                if not runtime.lifecycle.closed:
                    runtime.close()

    def test_restore_rolls_back_rows_and_payloads_when_insert_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        finalizer_calls: list[tuple[str, str, str]] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='rollback restore')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='state'),
                immutable=False,
                name='state',
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'before failed restore', actor=pid)
            runtime.memory.update_object(pid, handle, ObjectPatch(payload={'version': 2}))
            temporary = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'temporary': True},
                ObjectMetadata(title='temporary'),
                immutable=False,
                name='temporary.after.checkpoint',
            )
            runtime.memory.bind_durable_object_release_finalizer(
                'test.rollback-finalizer:v1',
                lambda obj, _actor, _reason, _work_id: {'oid': obj.oid},
                lambda intent, actor, reason, _work_id: finalizer_calls.append(
                    (str(intent['oid']), actor, reason)
                ),
            )
            message = runtime.human.send_process_message(pid, 'late message')
            request_id = runtime.human.query(
                pid=pid,
                human=runtime.config.runtime.default_human,
                request={'type': 'question', 'question': 'still pending?'},
                blocking=False,
            )
            original_insert = runtime.checkpoint._insert_row

            def fail_on_process_insert(cur, table, row):
                if table == 'processes':
                    raise RuntimeError('injected restore failure')
                return original_insert(cur, table, row)

            monkeypatch.setattr(runtime.checkpoint, '_insert_row', fail_on_process_insert)
            with pytest.raises(RuntimeError, match='injected restore failure'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.memory.get_object_by_name(pid, 'state')
            assert restored.payload == {'version': 2}
            assert runtime.store.get_object(temporary.oid) is not None
            assert finalizer_calls == []
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert runtime.store.get_process_message(message.message_id).status == ProcessMessageStatus.UNREAD
            assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
            assert not [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            ]
        finally:
            runtime.close()

    def test_restore_fences_before_secondary_checkpoint_reload_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'secondary-checkpoint-reload.sqlite3'
        runtime = Runtime.open(target)
        publication_id = ''
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='secondary checkpoint reload must fail closed',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before secondary reload failure',
                actor=pid,
            )
            original_load = runtime.checkpoint._restore_reconciler._load_checkpoint
            load_calls = 0

            def fail_reconciliation_and_diagnostic_reload(selected_id: str):
                nonlocal load_calls
                load_calls += 1
                if load_calls == 1:
                    raise RuntimeError('injected reconciliation checkpoint reload failure')
                raise RuntimeError('injected secondary diagnostic checkpoint reload failure')

            monkeypatch.setattr(
                runtime.checkpoint._restore_reconciler,
                '_load_checkpoint',
                fail_reconciliation_and_diagnostic_reload,
            )
            with pytest.raises(RuntimePublicationPending) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )
            publication_id = caught.value.publication_id

            assert load_calls == 2
            assert runtime.lifecycle.state == 'close_failed'
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['state'] == 'reconciliation_pending'
            assert publication['phase'] == 'main_state_committed'
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.state.value == 'running'
            assert operation.outcome.value == 'pending'
            with pytest.raises(
                RuntimeError,
                match='runtime is not accepting operations: state=close_failed',
            ):
                runtime.process.spawn(goal='must not cross checkpoint recovery fence')

            monkeypatch.setattr(
                runtime.checkpoint._restore_reconciler,
                '_load_checkpoint',
                original_load,
            )
            _release_checkpoint_recovery_runtime(runtime)
            reopened = Runtime.open(target)
            try:
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                recovered = reopened.store.get_runtime_publication(publication_id)
                assert recovered is not None
                assert recovered['state'] == 'committed'
                operation = reopened.store.get_operation(
                    recovered['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'succeeded'
            finally:
                reopened.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_restore_post_commit_base_exception_fences_before_propagation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='base exception after checkpoint commit',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before base exception',
                actor=pid,
            )

            def interrupt_reconciliation(*_args, **_kwargs):
                raise fault_type('injected post-commit interruption')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_images',
                interrupt_reconciliation,
            )
            with pytest.raises(fault_type):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.lifecycle.state == 'close_failed'
            publications = [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            ]
            publication = publications[-1]
            assert publication['state'] == 'failed'
            assert publication['phase'] == 'image_reconciliation_failed'
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.outcome.value == 'unknown'
            assert any(
                record.action == 'checkpoint.restore.post_commit_failure'
                and record.decision['error_type'] == fault_type.__name__
                for record in runtime.audit.trace()
            )
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_restore_authority_postcommit_base_exception_confirms_main_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(goal='confirm restore authority postcommit')
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before authority postcommit interruption',
                actor=pid,
            )
            original_scope = runtime.checkpoint._restore_authority_transaction

            @contextlib.contextmanager
            def interrupt_after_commit(*args, **kwargs):
                with original_scope(*args, **kwargs):
                    yield
                raise fault_type('injected authority postcommit interruption')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_authority_transaction',
                interrupt_after_commit,
            )
            with pytest.raises(fault_type):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'checkpoint_restore'
            ][-1]
            assert publication['kind'] == 'checkpoint_restore'
            assert publication['state'] == 'failed'
            assert publication['phase'] == 'object_payload_reconciliation_failed'
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.outcome.value == 'unknown'
            assert runtime.lifecycle.state == 'close_failed'
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    def test_restore_authority_postcommit_confirmation_failure_stays_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(goal='fail restore main commit confirmation')
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before main commit confirmation failure',
                actor=pid,
            )
            original_scope = runtime.checkpoint._restore_authority_transaction
            original_get = runtime.store.get_runtime_publication
            after_commit = False
            confirmation_failed = False

            @contextlib.contextmanager
            def interrupt_after_commit(*args, **kwargs):
                nonlocal after_commit
                with original_scope(*args, **kwargs):
                    yield
                after_commit = True
                raise KeyboardInterrupt('injected authority postcommit interruption')

            def fail_confirmation_once(publication_id: str):
                nonlocal confirmation_failed
                if after_commit and not confirmation_failed:
                    confirmation_failed = True
                    raise RuntimeError('injected durable confirmation read failure')
                return original_get(publication_id)

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_authority_transaction',
                interrupt_after_commit,
            )
            monkeypatch.setattr(
                runtime.store,
                'get_runtime_publication',
                fail_confirmation_once,
            )
            with pytest.raises(BaseExceptionGroup) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            leaves = list(caught.value.exceptions)
            assert any(isinstance(item, KeyboardInterrupt) for item in leaves)
            assert any(
                isinstance(item, RuntimeError)
                and str(item) == 'injected durable confirmation read failure'
                for item in leaves
            )
            assert any(isinstance(item, RuntimePublicationPending) for item in leaves)
            publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'checkpoint_restore'
            ][-1]
            assert publication['state'] == 'reconciliation_pending'
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.state.value == 'running'
            assert operation.outcome.value == 'pending'
            assert runtime.lifecycle.state == 'close_failed'
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_restore_base_exception_and_diagnostic_failure_remain_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(goal='preserve checkpoint diagnostic secondary')
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before diagnostic secondary',
                actor=pid,
            )

            def interrupt_phase(*_args, **_kwargs):
                raise fault_type('injected post-commit phase interruption')

            def fail_diagnostic(*_args, **_kwargs):
                raise RuntimeError('injected checkpoint diagnostic failure')

            monkeypatch.setattr(runtime.checkpoint, '_restore_images', interrupt_phase)
            monkeypatch.setattr(
                runtime.checkpoint._restore_reconciler,
                '_record_failure',
                fail_diagnostic,
            )
            with pytest.raises(BaseExceptionGroup) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            leaves = list(caught.value.exceptions)
            assert any(isinstance(item, fault_type) for item in leaves)
            assert any(
                isinstance(item, RuntimeError)
                and str(item) == 'injected checkpoint diagnostic failure'
                for item in leaves
            )
            assert any(isinstance(item, RuntimePublicationPending) for item in leaves)
            publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'checkpoint_restore'
            ][-1]
            assert publication['state'] == 'reconciliation_pending'
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.state.value == 'running'
            assert operation.outcome.value == 'pending'
            assert runtime.lifecycle.state == 'close_failed'
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.parametrize(
        'fault_type',
        [RuntimeError, KeyboardInterrupt, asyncio.CancelledError],
        ids=['runtime_error', 'keyboard_interrupt', 'cancelled_error'],
    )
    def test_restore_finish_postcommit_exception_preserves_terminal_truth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(goal='confirm checkpoint finish postcommit')
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before finish postcommit interruption',
                actor=pid,
            )
            original_transaction = runtime.store.transaction
            depth = 0
            injected = False

            @contextlib.contextmanager
            def interrupt_terminal_commit(*args, **kwargs):
                nonlocal depth, injected
                depth += 1
                try:
                    with original_transaction(*args, **kwargs) as cursor:
                        yield cursor
                finally:
                    depth -= 1
                if depth == 0 and not injected:
                    publications = [
                        item
                        for item in runtime.store.list_runtime_publications()
                        if item['kind'] == 'checkpoint_restore'
                    ]
                    if publications and publications[-1]['state'] == 'committed':
                        injected = True
                        raise fault_type('injected finish postcommit interruption')

            monkeypatch.setattr(runtime.store, 'transaction', interrupt_terminal_commit)
            with pytest.raises(fault_type):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'checkpoint_restore'
            ][-1]
            assert publication['state'] == 'committed'
            assert publication['phase'] == 'reconciled'
            assert publication['operation_reconciled'] is True
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.state.value == 'terminal'
            assert operation.outcome.value == 'succeeded'
            assert runtime.lifecycle.state == 'open'
        finally:
            runtime.close()

    def test_checkpoint_restore_recovery_is_single_flight_per_runtime(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            reconciler = runtime.checkpoint._restore_reconciler
            monkeypatch.setattr(
                reconciler,
                '_require_recovery_lease',
                lambda: None,
            )
            callers_ready = threading.Barrier(3)
            first_entered = threading.Event()
            release_first = threading.Event()
            overlap = threading.Event()
            counter_lock = threading.Lock()
            active = 0
            calls = 0
            results: list[list[str]] = []
            failures: list[BaseException] = []

            def probe_state(*_args, **_kwargs):
                nonlocal active, calls
                with counter_lock:
                    active += 1
                    calls += 1
                    if active > 1:
                        overlap.set()
                    first_call = calls == 1
                try:
                    if first_call:
                        first_entered.set()
                        assert release_first.wait(timeout=3)
                finally:
                    with counter_lock:
                        active -= 1

            monkeypatch.setattr(
                reconciler,
                '_recover_publication_state',
                probe_state,
            )

            def recover() -> None:
                try:
                    callers_ready.wait(timeout=3)
                    results.append(reconciler.recover_incomplete())
                except BaseException as exc:
                    failures.append(exc)

            threads = [threading.Thread(target=recover) for _ in range(2)]
            for thread in threads:
                thread.start()
            callers_ready.wait(timeout=3)
            assert first_entered.wait(timeout=3)
            assert not overlap.wait(timeout=0.2)
            release_first.set()
            for thread in threads:
                thread.join(timeout=3)

            assert not failures
            assert all(not thread.is_alive() for thread in threads)
            assert results == [[], []]
            assert calls == 20
            assert not overlap.is_set()
        finally:
            runtime.close()

    def test_checkpoint_restore_recovery_rejects_open_runtime_before_read_or_claim(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open("local")
        release_phase = threading.Event()
        restore_thread: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(goal="online restore recovery admission")
            checkpoint_id = runtime.checkpoint.create(
                pid,
                "before online recovery admission probe",
                actor=pid,
            )
            reconciler = runtime.checkpoint._restore_reconciler
            original_restore_images = reconciler._restore_images
            phase_entered = threading.Event()
            effect_calls = 0

            def pause_image_phase(snapshot: dict[str, object]) -> None:
                nonlocal effect_calls
                effect_calls += 1
                phase_entered.set()
                assert release_phase.wait(timeout=3)
                original_restore_images(snapshot)

            monkeypatch.setattr(
                reconciler,
                "_restore_images",
                pause_image_phase,
            )
            restore_results: list[dict[str, object]] = []
            restore_errors: list[BaseException] = []

            def restore() -> None:
                try:
                    restore_results.append(
                        runtime.checkpoint.restore(
                            "cli",
                            checkpoint_id,
                            require_capability=False,
                        )
                    )
                except BaseException as exc:
                    restore_errors.append(exc)

            restore_thread = threading.Thread(target=restore)
            restore_thread.start()
            assert phase_entered.wait(timeout=3)
            publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item["kind"] == "checkpoint_restore"
            ][-1]
            publication_id = str(publication["publication_id"])

            def unexpected_recovery_store_access(*_args: object, **_kwargs: object):
                raise AssertionError(
                    "startup recovery touched the store without its lifecycle lease"
                )

            monkeypatch.setattr(
                runtime.store,
                "query_runtime_publication_recovery",
                unexpected_recovery_store_access,
            )
            monkeypatch.setattr(
                runtime.store,
                "claim_runtime_publication_recovery",
                unexpected_recovery_store_access,
            )
            with pytest.raises(
                RuntimeError,
                match="active startup recovery lease",
            ):
                runtime.checkpoint.recover_incomplete_restore_publications()

            unchanged = runtime.store.get_runtime_publication(publication_id)
            assert unchanged == publication
            assert unchanged is not None
            assert unchanged["receipt"].get("recovery") is None
            assert effect_calls == 1

            release_phase.set()
            restore_thread.join(timeout=3)
            assert not restore_thread.is_alive()
            assert restore_errors == []
            assert len(restore_results) == 1
            assert restore_results[0]["status"] == "restored"
            committed = runtime.store.get_runtime_publication(publication_id)
            assert committed is not None
            assert committed["state"] == "committed"
            assert committed["operation_reconciled"] is True
            assert runtime.lifecycle.state == "open"
        finally:
            release_phase.set()
            if restore_thread is not None:
                restore_thread.join(timeout=3)
            runtime.close()

    def test_recovery_finish_postcommit_exception_confirms_terminal_truth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _assert_recovery_finish_postcommit_exception_confirms_terminal_truth(
            tmp_path / "recovery-finish-postcommit.sqlite3",
            monkeypatch,
        )

    @pytest.mark.postgres
    def test_postgres_recovery_finish_postcommit_exception_confirms_terminal_truth(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_checkpoint_target() as target:
            _assert_recovery_finish_postcommit_exception_confirms_terminal_truth(
                target,
                monkeypatch,
            )

    def test_recovery_finish_postcommit_control_flow_preserves_terminal_truth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "recovery-finish-control-flow.sqlite3"
        publication_id = _prepare_failed_checkpoint_restore(target, monkeypatch)
        original_finish = CheckpointRestoreReconciler._finish
        injected = False

        def interrupt_recovery_finish(
            reconciler: CheckpointRestoreReconciler,
            selected_id: str,
            *,
            recovery_lease_id: str | None,
        ) -> None:
            nonlocal injected
            original_finish(
                reconciler,
                selected_id,
                recovery_lease_id=recovery_lease_id,
            )
            if (
                selected_id == publication_id
                and recovery_lease_id is not None
                and not injected
            ):
                injected = True
                raise KeyboardInterrupt(
                    "injected recovery finish postcommit control flow"
                )

        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_finish",
            interrupt_recovery_finish,
        )
        with pytest.raises(KeyboardInterrupt):
            Runtime.open(target)
        assert injected is True

        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_finish",
            original_finish,
        )
        reopened = Runtime.open(target)
        try:
            assert reopened.recovered_checkpoint_restore_publications == [
                publication_id
            ]
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "committed"
            assert publication["phase"] == "reconciled"
            assert publication["operation_reconciled"] is True
            assert publication["receipt"]["payload_delivery"] == {
                "state": "completed"
            }
            operation = reopened.store.get_operation(
                publication["plan"]["operation_id"]
            )
            assert operation is not None
            assert operation.state.value == "terminal"
            assert operation.outcome.value == "succeeded"
        finally:
            reopened.close()

    def test_recovery_finish_confirmation_failure_preserves_both_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / "recovery-finish-confirmation-failure.sqlite3"
        publication_id = _prepare_failed_checkpoint_restore(target, monkeypatch)
        original_finish = CheckpointRestoreReconciler._finish
        original_confirmation = CheckpointRestoreReconciler._terminal_commit_confirmed
        injected = False

        def interrupt_recovery_finish(
            reconciler: CheckpointRestoreReconciler,
            selected_id: str,
            *,
            recovery_lease_id: str | None,
        ) -> None:
            nonlocal injected
            original_finish(
                reconciler,
                selected_id,
                recovery_lease_id=recovery_lease_id,
            )
            if (
                selected_id == publication_id
                and recovery_lease_id is not None
                and not injected
            ):
                injected = True
                raise RuntimeError("injected recovery finish postcommit exception")

        def fail_terminal_confirmation(
            reconciler: CheckpointRestoreReconciler,
            selected_id: str,
            *,
            publication=None,
        ) -> bool:
            if selected_id == publication_id and injected:
                raise OSError("injected recovery terminal confirmation failure")
            return original_confirmation(
                reconciler,
                selected_id,
                publication=publication,
            )

        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_finish",
            interrupt_recovery_finish,
        )
        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_terminal_commit_confirmed",
            fail_terminal_confirmation,
        )
        with pytest.raises(BaseExceptionGroup) as caught:
            Runtime.open(target)
        assert injected is True
        assert any(
            isinstance(error, RuntimeError)
            and str(error) == "injected recovery finish postcommit exception"
            for error in caught.value.exceptions
        )
        assert any(
            isinstance(error, OSError)
            and str(error) == "injected recovery terminal confirmation failure"
            for error in caught.value.exceptions
        )

        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_finish",
            original_finish,
        )
        monkeypatch.setattr(
            CheckpointRestoreReconciler,
            "_terminal_commit_confirmed",
            original_confirmation,
        )
        reopened = Runtime.open(target)
        try:
            assert reopened.recovered_checkpoint_restore_publications == [
                publication_id
            ]
            publication = reopened.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication["state"] == "committed"
            assert publication["phase"] == "reconciled"
            assert publication["operation_reconciled"] is True
            assert publication["receipt"]["payload_delivery"] == {
                "state": "completed"
            }
            operation = reopened.store.get_operation(
                publication["plan"]["operation_id"]
            )
            assert operation is not None
            assert operation.outcome.value == "succeeded"
        finally:
            reopened.close()

    @pytest.mark.postgres
    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_postgres_restore_authority_postcommit_base_exception_confirms_main_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        with _postgres_checkpoint_target() as target:
            runtime = Runtime.open(target)
            try:
                pid = runtime.process.spawn(goal='postgres authority postcommit')
                checkpoint_id = runtime.checkpoint.create(
                    pid,
                    'before postgres authority postcommit interruption',
                    actor=pid,
                )
                original_scope = runtime.checkpoint._restore_authority_transaction

                @contextlib.contextmanager
                def interrupt_after_commit(*args, **kwargs):
                    with original_scope(*args, **kwargs):
                        yield
                    raise fault_type('injected authority postcommit interruption')

                monkeypatch.setattr(
                    runtime.checkpoint,
                    '_restore_authority_transaction',
                    interrupt_after_commit,
                )
                with pytest.raises(fault_type):
                    runtime.checkpoint.restore(
                        'cli',
                        checkpoint_id,
                        require_capability=False,
                    )

                publication = [
                    item
                    for item in runtime.store.list_runtime_publications()
                    if item['kind'] == 'checkpoint_restore'
                ][-1]
                assert publication['state'] == 'failed'
                assert publication['phase'] == 'object_payload_reconciliation_failed'
                operation = runtime.store.get_operation(
                    publication['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'unknown'
                assert runtime.lifecycle.state == 'close_failed'
            finally:
                _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.postgres
    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_postgres_restore_base_exception_and_diagnostic_failure_remain_pending(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        with _postgres_checkpoint_target() as target:
            runtime = Runtime.open(target)
            try:
                pid = runtime.process.spawn(goal='postgres diagnostic secondary')
                checkpoint_id = runtime.checkpoint.create(
                    pid,
                    'before postgres diagnostic secondary',
                    actor=pid,
                )

                def interrupt_phase(*_args, **_kwargs):
                    raise fault_type('injected post-commit phase interruption')

                def fail_diagnostic(*_args, **_kwargs):
                    raise RuntimeError('injected checkpoint diagnostic failure')

                monkeypatch.setattr(
                    runtime.checkpoint,
                    '_restore_images',
                    interrupt_phase,
                )
                monkeypatch.setattr(
                    runtime.checkpoint._restore_reconciler,
                    '_record_failure',
                    fail_diagnostic,
                )
                with pytest.raises(BaseExceptionGroup) as caught:
                    runtime.checkpoint.restore(
                        'cli',
                        checkpoint_id,
                        require_capability=False,
                    )

                leaves = list(caught.value.exceptions)
                assert any(isinstance(item, fault_type) for item in leaves)
                assert any(isinstance(item, RuntimeError) for item in leaves)
                assert any(
                    isinstance(item, RuntimePublicationPending) for item in leaves
                )
                publication = [
                    item
                    for item in runtime.store.list_runtime_publications()
                    if item['kind'] == 'checkpoint_restore'
                ][-1]
                assert publication['state'] == 'reconciliation_pending'
                operation = runtime.store.get_operation(
                    publication['plan']['operation_id']
                )
                assert operation is not None
                assert operation.state.value == 'running'
                assert operation.outcome.value == 'pending'
                assert runtime.lifecycle.state == 'close_failed'
            finally:
                _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.postgres
    @pytest.mark.parametrize(
        'fault_type',
        [RuntimeError, KeyboardInterrupt],
        ids=['runtime_error', 'keyboard_interrupt'],
    )
    def test_postgres_restore_finish_postcommit_exception_preserves_terminal_truth(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        with _postgres_checkpoint_target() as target:
            runtime = Runtime.open(target)
            try:
                pid = runtime.process.spawn(goal='postgres finish postcommit')
                checkpoint_id = runtime.checkpoint.create(
                    pid,
                    'before postgres finish postcommit interruption',
                    actor=pid,
                )
                original_transaction = runtime.store.transaction
                depth = 0
                injected = False

                @contextlib.contextmanager
                def interrupt_terminal_commit(*args, **kwargs):
                    nonlocal depth, injected
                    depth += 1
                    try:
                        with original_transaction(*args, **kwargs) as cursor:
                            yield cursor
                    finally:
                        depth -= 1
                    if depth == 0 and not injected:
                        publications = [
                            item
                            for item in runtime.store.list_runtime_publications()
                            if item['kind'] == 'checkpoint_restore'
                        ]
                        if publications and publications[-1]['state'] == 'committed':
                            injected = True
                            raise fault_type(
                                'injected finish postcommit interruption'
                            )

                monkeypatch.setattr(
                    runtime.store,
                    'transaction',
                    interrupt_terminal_commit,
                )
                with pytest.raises(fault_type):
                    runtime.checkpoint.restore(
                        'cli',
                        checkpoint_id,
                        require_capability=False,
                    )

                publication = [
                    item
                    for item in runtime.store.list_runtime_publications()
                    if item['kind'] == 'checkpoint_restore'
                ][-1]
                assert publication['state'] == 'committed'
                assert publication['phase'] == 'reconciled'
                assert publication['operation_reconciled'] is True
                operation = runtime.store.get_operation(
                    publication['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'succeeded'
                assert runtime.lifecycle.state == 'open'
            finally:
                runtime.close()

    @pytest.mark.parametrize(
        'fault_site',
        ['phase', 'finish'],
    )
    def test_restore_preserves_primary_when_recovery_fence_reports_secondary_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_site: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal=f'preserve primary across {fault_site} fence failure',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                f'before {fault_site} fence failure',
                actor=pid,
            )
            reconciler = runtime.checkpoint._restore_reconciler
            original_fence = reconciler._recovery_required

            def fence_then_report_failure(**kwargs):
                assert original_fence is not None
                original_fence(**kwargs)
                raise RuntimeError('injected secondary recovery fence failure')

            monkeypatch.setattr(
                reconciler,
                '_recovery_required',
                fence_then_report_failure,
            )
            if fault_site == 'phase':
                def interrupt_phase(*_args, **_kwargs):
                    raise KeyboardInterrupt('injected primary phase interruption')

                monkeypatch.setattr(
                    runtime.checkpoint,
                    '_restore_images',
                    interrupt_phase,
                )
            else:
                original_advance = runtime.store.advance_runtime_publication

                def interrupt_finish(selected_id: str, **kwargs):
                    publication = runtime.store.get_runtime_publication(selected_id)
                    if (
                        kwargs.get('state') == 'committed'
                        and publication is not None
                        and publication['kind'] == 'checkpoint_restore'
                    ):
                        raise KeyboardInterrupt(
                            'injected primary finish interruption'
                        )
                    return original_advance(selected_id, **kwargs)

                monkeypatch.setattr(
                    runtime.store,
                    'advance_runtime_publication',
                    interrupt_finish,
                )

            with pytest.raises(BaseExceptionGroup) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            leaves: list[BaseException] = []
            stack: list[BaseException] = [caught.value]
            while stack:
                current = stack.pop()
                if isinstance(current, BaseExceptionGroup):
                    stack.extend(current.exceptions)
                else:
                    leaves.append(current)
            assert any(
                isinstance(error, KeyboardInterrupt)
                and 'injected primary' in str(error)
                for error in leaves
            )
            assert any(
                isinstance(error, RuntimeError)
                and str(error) == 'injected secondary recovery fence failure'
                for error in leaves
            )
            assert runtime.lifecycle.state == 'close_failed'
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    @pytest.mark.parametrize(
        ('fault_type', 'expected_type'),
        [
            (RuntimeError, RuntimePublicationPending),
            (KeyboardInterrupt, KeyboardInterrupt),
        ],
        ids=['exception', 'base_exception'],
    )
    def test_restore_finish_failure_fences_and_recovers_operation_outcome(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        fault_type: type[BaseException],
        expected_type: type[BaseException],
    ) -> None:
        target = tmp_path / f'finish-{fault_type.__name__}.sqlite3'
        runtime = Runtime.open(target)
        publication_id = ''
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='checkpoint terminalization must fail closed',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before terminalization failure',
                actor=pid,
            )
            original_advance = runtime.store.advance_runtime_publication

            def interrupt_checkpoint_finish(selected_id: str, **kwargs):
                publication = runtime.store.get_runtime_publication(selected_id)
                if (
                    kwargs.get('state') == 'committed'
                    and publication is not None
                    and publication['kind'] == 'checkpoint_restore'
                ):
                    raise fault_type('injected checkpoint terminalization failure')
                return original_advance(selected_id, **kwargs)

            monkeypatch.setattr(
                runtime.store,
                'advance_runtime_publication',
                interrupt_checkpoint_finish,
            )
            with pytest.raises(expected_type) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )
            if isinstance(caught.value, RuntimePublicationPending):
                publication_id = caught.value.publication_id
            else:
                publication_id = str(
                    runtime.store.list_runtime_publications()[-1]['publication_id']
                )

            assert runtime.lifecycle.state == 'close_failed'
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['state'] == 'reconciliation_pending'
            assert publication['phase'] == 'object_release_finalizers_completed'

            _release_checkpoint_recovery_runtime(runtime)
            reopened = Runtime.open(target)
            try:
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                recovered = reopened.store.get_runtime_publication(publication_id)
                assert recovered is not None
                assert recovered['state'] == 'committed'
                operation = reopened.store.get_operation(
                    recovered['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'succeeded'
            finally:
                reopened.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_restore_finish_rejects_missing_finalizer_receipt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'missing-finalizer-receipt.sqlite3'
        runtime = Runtime.open(target)
        finalizer_calls: list[str] = []
        publication_id = ''
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='finalizer receipt is required before commit',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before finalizer receipt omission',
                actor=pid,
            )
            runtime.memory.bind_durable_object_release_finalizer(
                'test.missing-receipt-finalizer:v1',
                lambda obj, _actor, _reason, _work_id: {'oid': obj.oid},
                lambda _intent, _actor, _reason, work_id: finalizer_calls.append(
                    work_id
                ),
            )
            runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'temporary': True},
                ObjectMetadata(title='temporary finalizer receipt object'),
                immutable=False,
                name='temporary.finalizer.receipt',
            )
            original_advance = runtime.store.advance_runtime_publication

            def omit_finalizer_receipt(selected_id: str, **kwargs):
                receipt = kwargs.get('receipt') or {}
                if receipt.get('phase') == 'checkpoint_restore_finalizer_completed':
                    return True
                return original_advance(selected_id, **kwargs)

            monkeypatch.setattr(
                runtime.store,
                'advance_runtime_publication',
                omit_finalizer_receipt,
            )
            with pytest.raises(RuntimePublicationPending) as caught:
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            publication = runtime.store.get_runtime_publication(
                caught.value.publication_id
            )
            publication_id = caught.value.publication_id
            assert publication is not None
            assert publication['state'] == 'reconciliation_pending'
            assert publication['phase'] == 'jit_pruning_completed'
            assert len(finalizer_calls) == 1
            receipts = publication['receipt']['phases']
            assert not [
                item
                for item in receipts
                if item.get('phase') == 'checkpoint_restore_finalizer_completed'
            ]
            assert [
                item['name']
                for item in receipts
                if item.get('phase') == 'checkpoint_restore_phase_completed'
            ] == list(CHECKPOINT_RESTORE_PHASES[:-1])
            assert runtime.lifecycle.state == 'close_failed'
            _release_checkpoint_recovery_runtime(runtime)

            with pytest.raises(
                ValidationError,
                match='cannot reconcile checkpoint restore publication',
            ):
                Runtime.open(target)
            store = SQLiteStore(target)
            try:
                publication = store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'manual'
                assert (
                    publication['phase']
                    == 'durable_finalizer_handler_unavailable'
                )
                assert publication['state'] != 'committed'
                assert not [
                    item
                    for item in publication['receipt']['phases']
                    if item.get('phase')
                    == 'checkpoint_restore_finalizer_completed'
                ]
                operation = store.get_operation(
                    publication['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'unknown'
            finally:
                store.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    @pytest.mark.parametrize(
        "missing_phase",
        CHECKPOINT_RESTORE_PHASES[:-1],
    )
    def test_restore_phase_receipt_ack_without_persistence_stops_pipeline_and_recovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        missing_phase: str,
    ) -> None:
        _assert_missing_checkpoint_phase_receipt_is_fail_closed(
            tmp_path / f"missing-{missing_phase}-receipt.sqlite3",
            monkeypatch,
            missing_phase,
        )

    @pytest.mark.postgres
    @pytest.mark.parametrize(
        "missing_phase",
        CHECKPOINT_RESTORE_PHASES[:-1],
    )
    def test_postgres_restore_phase_receipt_ack_without_persistence_stops_pipeline_and_recovers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        missing_phase: str,
    ) -> None:
        with _postgres_checkpoint_target() as target:
            _assert_missing_checkpoint_phase_receipt_is_fail_closed(
                target,
                monkeypatch,
                missing_phase,
            )

    @pytest.mark.parametrize(
        ('method_name', 'phase'),
        [
            ('_restore_images', 'image_reconciliation'),
            ('_restore_jit_sources', 'jit_source_reconciliation'),
            ('_prune_stale_ephemeral_jit_tools', 'jit_pruning'),
        ],
    )
    def test_restore_durably_recovers_post_commit_reconciliation_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        method_name: str,
        phase: str,
    ) -> None:
        target = tmp_path / f'{phase}.sqlite3'
        runtime = Runtime.open(target)
        publication_id = ''
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='post-commit restore failure')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='state'),
                immutable=False,
                name='post.commit.state',
            )
            linked = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'linked': True},
                ObjectMetadata(title='linked state'),
                immutable=False,
                name='post.commit.linked',
            )
            runtime.memory.link_objects(pid, handle, 'references', linked)
            checkpoint_id = runtime.checkpoint.create(pid, 'before post-commit failure', actor=pid)
            runtime.memory.update_object(pid, handle, ObjectPatch(payload={'version': 2}))
            unrelated_pid = runtime.process.spawn(goal='ordinary missing payload cleanup')
            unrelated = runtime.memory.create_object(
                unrelated_pid,
                ObjectType.SUMMARY,
                {'unrelated': True},
                ObjectMetadata(title='unrelated state'),
                immutable=False,
                name='ordinary.missing.payload',
            )

            def fail_reconciliation(*_args, **_kwargs):
                raise RuntimeError(f'injected {phase} failure')

            monkeypatch.setattr(runtime.checkpoint, method_name, fail_reconciliation)
            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            publication_id = str(result['publication_id'])

            assert runtime.store.object_payload(handle.oid) == {'version': 1}
            assert result['status'] == 'restored_with_warnings'
            assert result['main_state_committed'] is True
            assert result['reconciliation_pending'] is True
            assert [failure['phase'] for failure in result['post_commit_failures']] == [phase]
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['kind'] == 'checkpoint_restore'
            assert publication['state'] == 'failed'
            completed_before_reopen = [
                item['name']
                for item in publication['receipt']['phases']
                if item.get('phase') == 'checkpoint_restore_phase_completed'
            ]
            expected_prefix = {
                'image_reconciliation': ['object_payload_reconciliation'],
                'jit_source_reconciliation': [
                    'object_payload_reconciliation',
                    'image_reconciliation',
                ],
                'jit_pruning': [
                    'object_payload_reconciliation',
                    'image_reconciliation',
                    'jit_source_reconciliation',
                ],
            }
            assert completed_before_reopen == expected_prefix[phase]
            operation = runtime.store.get_operation(publication['plan']['operation_id'])
            assert operation is not None
            assert operation.outcome.value == 'unknown'
            assert runtime.lifecycle.state == 'close_failed'
            failure_audits = [
                record
                for record in runtime.audit.trace()
                if record.action == 'checkpoint.restore.post_commit_failure'
            ]
            assert failure_audits[-1].decision['phase'] == phase
            assert failure_audits[-1].decision['main_state_committed'] is True
            _release_checkpoint_recovery_runtime(runtime)

            original_payload_reconcile = (
                SnapshotCheckpointRepository.reconcile_checkpoint_object_payloads
            )
            payload_reconcile_calls = 0

            def observe_payload_reconcile(
                repository: SnapshotCheckpointRepository,
                snapshot: object,
            ) -> tuple[str, ...]:
                nonlocal payload_reconcile_calls
                payload_reconcile_calls += 1
                return original_payload_reconcile(repository, snapshot)

            monkeypatch.setattr(
                SnapshotCheckpointRepository,
                'reconcile_checkpoint_object_payloads',
                observe_payload_reconcile,
            )

            reopened = Runtime.open(target)
            try:
                assert payload_reconcile_calls == 1
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
                assert publication['phase'] == 'reconciled'
                assert publication['error'] is None
                recovery_claims = [
                    item
                    for item in publication['receipt']['phases']
                    if item.get('phase') == 'recovery_claimed'
                ]
                assert len(recovery_claims) == 1
                assert recovery_claims[0]['attempt'] == 1
                assert recovery_claims[0]['claimant_instance_id'] == reopened.instance_id
                assert recovery_claims[0]['lease_id']
                assert publication['receipt']['recovery']['disposition'] == 'terminal'
                completed = [
                    item['name']
                    for item in publication['receipt']['phases']
                    if item.get('phase') == 'checkpoint_restore_phase_completed'
                ]
                assert completed == list(CHECKPOINT_RESTORE_PHASES)
                operation = reopened.store.get_operation(
                    publication['plan']['operation_id']
                )
                assert operation is not None
                assert operation.outcome.value == 'succeeded'
                assert reopened.store.object_payload(handle.oid) == {'version': 1}
                assert not reopened.store.is_recovered_object_payload(handle.oid)
                assert [
                    (link.src, link.relation.value, link.dst)
                    for link in reopened.store.list_links(src=handle.oid)
                ] == [(handle.oid, 'references', linked.oid)]
                restored_capability = reopened.store.get_capability(
                    handle.capability_id
                )
                assert restored_capability is not None
                assert restored_capability.status == CapabilityStatus.ACTIVE
                assert reopened.store.get_object(unrelated.oid) is None
                assert reopened.store.is_recovered_object_payload(unrelated.oid)
                unrelated_capability = reopened.store.get_capability(
                    unrelated.capability_id
                )
                assert unrelated_capability is not None
                assert unrelated_capability.status == CapabilityStatus.REVOKED
            finally:
                reopened.close()

            reopened_again = Runtime.open(target)
            try:
                assert reopened_again.recovered_checkpoint_restore_publications == []
                publication = reopened_again.store.get_runtime_publication(
                    publication_id
                )
                assert publication is not None
                assert publication['state'] == 'committed'
                assert len(
                    [
                        item
                        for item in publication['receipt']['phases']
                        if item.get('phase') == 'recovery_claimed'
                    ]
                ) == 1
            finally:
                reopened_again.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_recovered_payload_delivery_survives_late_startup_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'payload-delivery-late-startup-failure.sqlite3'
        runtime = Runtime.open(target)
        try:
            pid = runtime.process.spawn(goal='payload delivery startup retry')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='delivery state'),
                immutable=False,
                name='payload.delivery.state',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before payload delivery retry',
                actor=pid,
            )
            runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(payload={'version': 2}),
            )

            def fail_image_reconciliation(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError('injected image reconciliation failure')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_images',
                fail_image_reconciliation,
            )
            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result['publication_id'])
            assert result['status'] == 'restored_with_warnings'
            _release_checkpoint_recovery_runtime(runtime)

            original_hooks = RuntimeModuleRegistry.run_startup_hooks
            hook_calls = 0

            def fail_first_late_hook(registry: RuntimeModuleRegistry) -> None:
                nonlocal hook_calls
                hook_calls += 1
                if hook_calls == 1:
                    raise RuntimeError('injected late startup hook failure')
                original_hooks(registry)

            monkeypatch.setattr(
                RuntimeModuleRegistry,
                'run_startup_hooks',
                fail_first_late_hook,
            )
            with pytest.raises(RuntimeError, match='late startup hook failure'):
                Runtime.open(target)

            reopened = Runtime.open(target)
            try:
                assert hook_calls == 2
                assert reopened.store.object_payload(handle.oid) == {'version': 1}
                assert not reopened.store.is_recovered_object_payload(handle.oid)
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
                assert publication['operation_reconciled'] is True
                assert publication['receipt']['payload_delivery'] == {
                    'state': 'completed'
                }
            finally:
                reopened.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_selective_payload_retry_after_mark_open_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _assert_selective_payload_retry_after_mark_open_failure(
            tmp_path / "selective-payload-mark-open-failure.sqlite3",
            monkeypatch,
        )

    @pytest.mark.postgres
    def test_postgres_selective_payload_retry_after_mark_open_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _postgres_checkpoint_target() as target:
            _assert_selective_payload_retry_after_mark_open_failure(
                target,
                monkeypatch,
            )

    def test_reopen_recovers_exact_v1_restore_plan_without_rewriting_anchor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'checkpoint-restore-v1-plan.sqlite3'
        runtime = Runtime.open(target)
        try:
            pid = runtime.process.spawn(goal='recover v1 restore publication')
            handle = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='v1 state'),
                immutable=False,
                name='v1.restore.state',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before v1 recovery fixture',
                actor=pid,
            )
            runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(payload={'version': 2}),
            )

            def fail_image_reconciliation(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError('injected v1 image failure')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_images',
                fail_image_reconciliation,
            )
            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result['publication_id'])
            _release_checkpoint_recovery_runtime(runtime)

            store = SQLiteStore(target)
            try:
                publication = store.get_runtime_publication(publication_id)
                assert publication is not None
                v1_plan = dict(publication['plan'])
                v1_plan['plan_version'] = 1
                v1_plan['phase_order'] = list(CHECKPOINT_RESTORE_V1_PHASES)
                v1_receipt = loads(dumps(publication['receipt']), {})
                v1_receipt['phases'] = [
                    item
                    for item in v1_receipt['phases']
                    if item
                    != {
                        'phase': 'checkpoint_restore_phase_completed',
                        'name': 'object_payload_reconciliation',
                    }
                ]
                v1_anchor = {
                    'artifact_id': (
                        f'{publication_id}:checkpoint_restore_plan:v1'
                    ),
                    'artifact_type': 'checkpoint_restore_plan_anchor',
                    'anchor_version': 1,
                    'plan_sha256': hashlib.sha256(
                        dumps(v1_plan).encode('utf-8')
                    ).hexdigest(),
                }
                v1_receipt['artifacts'] = [v1_anchor]
                with store.transaction() as cursor:
                    cursor.execute(
                        'UPDATE runtime_publications '
                        'SET plan_json = ?, receipt_json = ? '
                        'WHERE publication_id = ?',
                        (dumps(v1_plan), dumps(v1_receipt), publication_id),
                    )
            finally:
                store.close()

            reopened = Runtime.open(target)
            try:
                assert reopened.store.object_payload(handle.oid) == {'version': 1}
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
                assert publication['plan'] == v1_plan
                assert publication['receipt']['artifacts'] == [v1_anchor]
                completed = [
                    item['name']
                    for item in publication['receipt']['phases']
                    if item.get('phase') == 'checkpoint_restore_phase_completed'
                ]
                assert completed == list(CHECKPOINT_RESTORE_V1_PHASES)
            finally:
                reopened.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_restore_reports_release_finalizer_failure_after_main_state_commit(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='finalizer failure')
            checkpoint_id = runtime.checkpoint.create(pid, 'before temporary object', actor=pid)
            temporary = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'temporary': True},
                ObjectMetadata(title='temporary'),
                immutable=False,
                name='post.commit.temporary',
            )

            def fail_finalizer(_intent, _actor, _reason, _work_id):
                raise RuntimeError('injected release finalizer failure')

            runtime.memory.bind_durable_object_release_finalizer(
                'test.failing-finalizer:v1',
                lambda obj, _actor, _reason, _work_id: {'oid': obj.oid},
                fail_finalizer,
            )
            result = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.store.get_object(temporary.oid) is None
            assert result['status'] == 'restored_with_warnings'
            assert result['main_state_committed'] is True
            assert result['post_commit_failures'][0]['phase'] == 'object_release_finalizers'
            assert any(
                record.action == 'checkpoint.restore.post_commit_failure'
                and record.decision['phase'] == 'object_release_finalizers'
                for record in runtime.audit.trace()
            )
        finally:
            _release_checkpoint_recovery_runtime(runtime)

    def test_missing_durable_finalizer_handler_becomes_manual_without_losing_work(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'missing-finalizer-handler.sqlite3'
        runtime = Runtime.open(target)
        publication_id = ''
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='durable finalizer manual recovery',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before temporary durable resource',
                actor=pid,
            )
            temporary = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'provider_resource_id': 'resource-123'},
                ObjectMetadata(title='temporary durable resource'),
                immutable=False,
                name='temporary.durable.resource',
            )

            def fail_online(_intent, _actor, _reason, _work_id):
                raise RuntimeError('injected durable finalizer outage')

            runtime.memory.bind_durable_object_release_finalizer(
                'test.provider-resource-release:v1',
                lambda obj, _actor, _reason, _work_id: {
                    'object_oid': obj.oid,
                    'provider_resource_id': obj.payload['provider_resource_id'],
                },
                fail_online,
            )
            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result['publication_id'])
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['state'] == 'failed'
            work = publication['plan']['finalizer_work_items']
            assert len(work) == 1
            assert work[0]['finalizer_id'] == 'test.provider-resource-release:v1'
            assert work[0]['object_oid'] == temporary.oid
            assert work[0]['intent'] == {
                'object_oid': temporary.oid,
                'provider_resource_id': 'resource-123',
            }
            stable_work_id = work[0]['work_id']
            assert stable_work_id.startswith('checkpoint_finalizer:')
            _release_checkpoint_recovery_runtime(runtime)

            with pytest.raises(
                ValidationError,
                match='cannot reconcile checkpoint restore publication',
            ):
                Runtime.open(target)

            store = SQLiteStore(target)
            try:
                publication = store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'manual'
                assert publication['phase'] == 'durable_finalizer_handler_unavailable'
                recovery_claims = [
                    item
                    for item in publication['receipt']['phases']
                    if item.get('phase') == 'recovery_claimed'
                ]
                assert len(recovery_claims) == 1
                assert recovery_claims[0]['attempt'] == 1
                assert recovery_claims[0]['lease_id']
                assert publication['receipt']['recovery']['disposition'] == 'manual'
                work_after_reopen = publication['plan']['finalizer_work_items']
                assert work_after_reopen == work
                assert work_after_reopen[0]['work_id'] == stable_work_id
                operation = store.get_operation(publication['plan']['operation_id'])
                assert operation is not None
                assert operation.outcome.value == 'unknown'
            finally:
                store.close()

            with pytest.raises(
                ValidationError,
                match='requires manual recovery',
            ):
                Runtime.open(target)

            store = SQLiteStore(target)
            try:
                publication = store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'manual'
                assert publication['plan']['finalizer_work_items'] == work
                assert len(
                    [
                        item
                        for item in publication['receipt']['phases']
                        if item.get('phase') == 'recovery_claimed'
                    ]
                ) == 1
            finally:
                store.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    def test_startup_module_reconstitutes_durable_finalizer_with_stable_work_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'module-finalizer-recovery.sqlite3'
        marker = tmp_path / 'module-finalizer-marker.txt'
        attempts = tmp_path / 'module-finalizer-attempts.txt'
        manifest, trust_key = _write_durable_finalizer_module(
            tmp_path,
            marker=marker,
            attempts=attempts,
        )
        open_kwargs = {
            'module_manifests': (str(manifest),),
            'trusted_modules': (trust_key,),
        }
        runtime = Runtime.open(target, **open_kwargs)
        publication_id = ''
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='module durable finalizer recovery',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before module-owned resource',
                actor=pid,
            )
            runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'provider_resource_id': 'module-resource-123'},
                ObjectMetadata(title='module-owned resource'),
                immutable=False,
                name='module.owned.resource',
            )
            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result['publication_id'])
            assert result['status'] == 'restored_with_warnings'
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            work_id = publication['plan']['finalizer_work_items'][0]['work_id']
            assert attempts.read_text(encoding='utf-8').splitlines() == [work_id]
            assert marker.read_text(encoding='utf-8') == work_id
            operation_id = str(publication['plan']['operation_id'])
            _release_checkpoint_recovery_runtime(runtime)

            recovery_operation_ids: list[str | None] = []
            original_run_finalizer = (
                ObjectMemoryManager.run_checkpoint_restore_finalizer
            )

            def observe_recovery_operation(
                manager: ObjectMemoryManager,
                work_item,
                *,
                actor: str,
                reason: str,
            ) -> None:
                recovery_operation_ids.append(
                    manager.operations.current_id()
                    if manager.operations is not None
                    else None
                )
                original_run_finalizer(
                    manager,
                    work_item,
                    actor=actor,
                    reason=reason,
                )

            monkeypatch.setattr(
                ObjectMemoryManager,
                'run_checkpoint_restore_finalizer',
                observe_recovery_operation,
            )
            reopened = Runtime.open(target, **open_kwargs)
            try:
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
                recovery_claims = [
                    item
                    for item in publication['receipt']['phases']
                    if item.get('phase') == 'recovery_claimed'
                ]
                assert len(recovery_claims) == 1
                assert recovery_claims[0]['attempt'] == 1
                assert recovery_claims[0]['claimant_instance_id'] == reopened.instance_id
                assert publication['receipt']['recovery']['disposition'] == 'terminal'
                assert attempts.read_text(encoding='utf-8').splitlines() == [
                    work_id,
                    work_id,
                ]
                assert recovery_operation_ids == [operation_id]
            finally:
                reopened.close()

            reopened_again = Runtime.open(target, **open_kwargs)
            try:
                assert reopened_again.recovered_checkpoint_restore_publications == []
                assert attempts.read_text(encoding='utf-8').splitlines() == [
                    work_id,
                    work_id,
                ]
            finally:
                reopened_again.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()

    @pytest.mark.parametrize('sink', ['event', 'audit'])
    def test_restore_core_event_and_audit_failures_roll_back_main_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sink: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'{sink} failure after restore')
            state = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'version': 1},
                ObjectMetadata(title='state'),
                immutable=False,
                name=f'{sink}.failure.state',
            )
            checkpoint_id = runtime.checkpoint.create(pid, f'before {sink} failure', actor=pid)
            runtime.memory.update_object(pid, state, ObjectPatch(payload={'version': 2}))

            if sink == 'event':
                original_emit = runtime.events.emit

                def fail_restore_event(event_type, *args, **kwargs):
                    if event_type == EventType.ROLLBACK:
                        raise RuntimeError('injected restore event failure')
                    return original_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', fail_restore_event)
            else:
                original_record = runtime.audit.record

                def fail_restore_audit(*args, **kwargs):
                    if kwargs.get('action') == 'checkpoint.restore':
                        raise RuntimeError('injected restore audit failure')
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', fail_restore_audit)

            before_event_ids = {event.event_id for event in runtime.events.list()}
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}

            with pytest.raises(RuntimeError, match=f'injected restore {sink} failure'):
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert runtime.memory.get_object_by_name(pid, f'{sink}.failure.state').payload == {'version': 2}
            assert not [
                event
                for event in runtime.events.list()
                if event.event_id not in before_event_ids
                and event.type == EventType.ROLLBACK
            ]
            assert not [
                record
                for record in runtime.audit.trace()
                if record.record_id not in before_audit_ids
                and record.action == 'checkpoint.restore'
            ]
            assert not [
                publication
                for publication in runtime.store.list_runtime_publications()
                if publication['kind'] == 'checkpoint_restore'
            ]
        finally:
            runtime.close()
