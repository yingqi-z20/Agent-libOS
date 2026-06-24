from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, CheckpointDefaults
from agent_libos.models import CapabilityEffect, CapabilityRight, EventType, HumanRequestStatus, ObjectMetadata, ObjectPatch, ObjectType, ProcessMessageStatus, ProcessStatus
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.substrate import LocalHumanProvider, LocalResourceProviderSubstrate
from tests.support.checkpoints import ClassifiedShellProvider


class TestCheckpointRestore:

    def test_legacy_full_table_snapshot_restore_are_disabled(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(RuntimeError, match='full-table SQLite snapshots are disabled'):
                runtime.store.snapshot_tables()
            with pytest.raises(RuntimeError, match='full-table SQLite restore is disabled'):
                runtime.store.restore_tables({})
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
                assert result['external_effect_summary']['by_rollback_class']['rollbackable'] == 1
                assert result['external_effects_since_checkpoint'][0]['provider'] == 'filesystem'
                assert result['external_effects_since_checkpoint'][0]['rollback_class'] == 'rollbackable'
                assert 'checkpoint.restore' in [record.action for record in runtime.audit.trace()]
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
                assert summary['rollbackable'] == 1
                assert summary['irreversible'] == 1
                assert summary['no_rollback_required'] == 1
                assert {(effect['provider'], effect['rollback_class']) for effect in effects} == {('filesystem', 'rollbackable'), ('shell', 'irreversible'), ('human', 'no_rollback_required')}
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
            pid = runtime.process.spawn(image='base-agent:v0', goal='messages')
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
            process = runtime.process.get(pid)
            process.status = ProcessStatus.RUNNING
            runtime.store.update_process(process)
            checkpoint_id = runtime.checkpoint.create(pid, 'running safe point', actor=pid)
            inspected = runtime.checkpoint.inspect(checkpoint_id, actor=pid)
            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            assert inspected['processes'][0]['status'] == ProcessStatus.RUNNABLE.value
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_restore_reconciles_terminal_human_wait_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='human wait')
            request_id = runtime.human.ask(pid, 'continue?', blocking=True)
            checkpoint_id = runtime.checkpoint.create(pid, 'waiting for human', actor=pid)
            runtime.human.approve(request_id, {'approved': True})

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            restored = runtime.process.get(pid)
            assert restored.status == ProcessStatus.RUNNABLE
            assert restored.status_message is None
            assert runtime.human.get(request_id).status == HumanRequestStatus.APPROVED
        finally:
            runtime.close()

    def test_restore_rolls_back_rows_and_payloads_when_insert_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
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
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert runtime.store.get_process_message(message.message_id).status == ProcessMessageStatus.UNREAD
            assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        finally:
            runtime.close()
