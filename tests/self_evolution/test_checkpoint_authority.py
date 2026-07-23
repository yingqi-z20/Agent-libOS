from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, EventType
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.runtime.syscalls import LibOSSyscallSession
from tests.support.checkpoints import checkpoint_cli_json


class TestCheckpointAuthority:

    def test_checkpoint_list_rejects_unbounded_limits(self) -> None:
        runtime = Runtime.open('local')
        try:
            for limit in (0, -1, True, runtime.config.checkpoint.list_limit + 1):
                with pytest.raises(ValidationError, match='limit'):
                    runtime.checkpoint.list(require_capability=False, limit=limit)  # type: ignore[arg-type]
        finally:
            runtime.close()

    def test_one_shot_checkpoint_read_is_consumed_by_inspect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            reader = runtime.process.spawn(image='base-agent:v0', goal='reader')
            checkpoint_id = runtime.checkpoint.create(owner, 'one-shot read', actor=owner)
            cap = runtime.capability.issue_trusted(
                reader,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
            )
            original_counts = runtime.checkpoint._snapshot_counts

            def fail_inspect(_snapshot: dict[str, Any]) -> dict[str, int]:
                raise RuntimeError('injected checkpoint inspect failure')

            monkeypatch.setattr(runtime.checkpoint, '_snapshot_counts', fail_inspect)
            with pytest.raises(RuntimeError, match='injected checkpoint inspect failure'):
                runtime.checkpoint.inspect(checkpoint_id, actor=reader)
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            monkeypatch.setattr(runtime.checkpoint, '_snapshot_counts', original_counts)

            inspected = runtime.checkpoint.inspect(checkpoint_id, actor=reader)

            assert inspected['checkpoint']['checkpoint_id'] == checkpoint_id
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.inspect(checkpoint_id, actor=reader)
        finally:
            runtime.close()

    def test_one_shot_checkpoint_read_is_transactional_for_diff(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='diff owner')
            reader = runtime.process.spawn(image='base-agent:v0', goal='diff reader')
            checkpoint_id = runtime.checkpoint.create(owner, 'one-shot diff', actor=owner)
            cap = runtime.capability.issue_trusted(
                reader,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
            )
            original_build = runtime.checkpoint._build_current_state_for_diff

            def fail_diff(_snapshot: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError('injected checkpoint diff failure')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_build_current_state_for_diff',
                fail_diff,
            )
            with pytest.raises(RuntimeError, match='injected checkpoint diff failure'):
                runtime.checkpoint.diff(checkpoint_id, actor=reader)
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1

            monkeypatch.setattr(
                runtime.checkpoint,
                '_build_current_state_for_diff',
                original_build,
            )
            diff = runtime.checkpoint.diff(checkpoint_id, actor=reader)

            assert diff['checkpoint_id'] == checkpoint_id
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.diff(checkpoint_id, actor=reader)
        finally:
            runtime.close()

    def test_one_shot_checkpoint_read_is_transactional_for_replay(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='replay owner')
            reader = runtime.process.spawn(image='base-agent:v0', goal='replay reader')
            checkpoint_id = runtime.checkpoint.create(owner, 'one-shot replay', actor=owner)
            event = runtime.events.emit(
                EventType.PROCESS_SIGNAL,
                source=owner,
                target=owner,
                payload={'reason': 'replay target'},
            )
            cap = runtime.capability.issue_trusted(
                reader,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.READ],
                issued_by='test',
                uses_remaining=1,
            )

            with pytest.raises(NotFound, match='event not found after checkpoint'):
                runtime.checkpoint.replay_to_event(
                    checkpoint_id,
                    'event_missing',
                    actor=reader,
                )
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1

            replay = runtime.checkpoint.replay_to_event(
                checkpoint_id,
                event.event_id,
                actor=reader,
            )

            assert replay['event_id'] == event.event_id
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            assert any(
                record.actor == reader
                and record.action == 'checkpoint.replay_to_event'
                and record.target == f'checkpoint:{checkpoint_id}'
                for record in runtime.audit.trace(actor=reader)
            )
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.replay_to_event(
                    checkpoint_id,
                    event.event_id,
                    actor=reader,
                )
        finally:
            runtime.close()

    def test_checkpoint_capabilities_gate_inspect_restore_and_fork(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            other = runtime.process.spawn(image='base-agent:v0', goal='other')
            checkpoint_id = runtime.checkpoint.create(owner, 'owned', actor=owner)
            assert runtime.checkpoint.inspect(checkpoint_id, actor=owner)['checkpoint']['pid'] == owner
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.inspect(checkpoint_id, actor=other)
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.restore(owner, checkpoint_id)
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id)
            runtime.capability.grant(owner, f'checkpoint:{checkpoint_id}', [CapabilityRight.ADMIN], issued_by='test')
            assert runtime.checkpoint.restore(owner, checkpoint_id)['status'] == 'restored'
            runtime.capability.grant(owner, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id)
            assert forked['fork_root_pid'] != owner
        finally:
            runtime.close()

    def test_checkpoint_control_authority_is_not_snapshotted(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            first = runtime.checkpoint.create(owner, 'first', actor=owner)
            runtime.capability.grant(
                owner,
                f'checkpoint:{first}',
                [CapabilityRight.ADMIN, CapabilityRight.EXECUTE],
                issued_by='test',
            )

            second = runtime.checkpoint.create(owner, 'second', actor=owner)
            found = runtime.store.get_checkpoint_snapshot(second)

            assert found is not None
            _checkpoint, snapshot = found
            assert all(
                not str(row['resource']).startswith('checkpoint:')
                for row in snapshot['rows']['capabilities']
            )
            process_row = next(
                row for row in snapshot['rows']['processes'] if row['pid'] == owner
            )
            captured_ids = {
                str(row['cap_id']) for row in snapshot['rows']['capabilities']
            }
            assert set(json.loads(process_row['capabilities_json'])) == captured_ids
        finally:
            runtime.close()

    def test_restore_and_fork_filter_legacy_checkpoint_control_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='legacy control owner')
            ordinary = runtime.capability.grant(
                owner,
                'test:ordinary-checkpoint-state',
                [CapabilityRight.READ],
                issued_by='test',
            )
            checkpoint_id = runtime.checkpoint.create(
                owner,
                'legacy checkpoint control row',
                actor=owner,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _checkpoint, snapshot = found
            ordinary_row = next(
                row
                for row in snapshot['rows']['capabilities']
                if row['cap_id'] == ordinary.cap_id
            )
            legacy_cap_id = 'cap_legacy_checkpoint_control'
            snapshot['rows']['capabilities'].append(
                {
                    **ordinary_row,
                    'cap_id': legacy_cap_id,
                    'resource': 'checkpoint:legacy-artifact',
                    'issuer_cap_id': None,
                    'parent_cap_id': None,
                }
            )
            process_row = next(
                row for row in snapshot['rows']['processes'] if row['pid'] == owner
            )
            capability_ids = json.loads(process_row['capabilities_json'])
            capability_ids.append(legacy_cap_id)
            process_row['capabilities_json'] = json.dumps(sorted(capability_ids))
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (json.dumps(snapshot), checkpoint_id),
            )

            runtime.capability.grant(
                owner,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            forked = runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id)
            fork_resources = {
                capability.resource
                for capability in runtime.capability.list_subject(forked['fork_root_pid'])
            }
            assert 'test:ordinary-checkpoint-state' in fork_resources
            assert not any(resource.startswith('checkpoint:') for resource in fork_resources)

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            restored_resources = {
                capability.resource
                for capability in runtime.capability.list_subject(owner)
            }
            assert 'test:ordinary-checkpoint-state' in restored_resources
            assert 'checkpoint:legacy-artifact' not in restored_resources
            assert runtime.store.get_capability(legacy_cap_id) is None
        finally:
            runtime.close()

    def test_checkpoint_syscalls_use_primitive_capabilities(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='syscall')
            other = runtime.process.spawn(image='base-agent:v0', goal='other')
            session = LibOSSyscallSession(runtime, pid)
            other_session = LibOSSyscallSession(runtime, other)
            checkpoint = self._run(session.handle('checkpoint.create', {'reason': 'syscall'}))
            inspected = self._run(session.handle('checkpoint.inspect', {'checkpoint_id': checkpoint['checkpoint_id']}))
            assert inspected['checkpoint']['pid'] == pid
            with pytest.raises(CapabilityDenied):
                self._run(other_session.handle('checkpoint.inspect', {'checkpoint_id': checkpoint['checkpoint_id']}))
        finally:
            runtime.close()

    def test_default_images_expose_only_low_risk_checkpoint_tools(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='tool table')
            assert runtime.tools.call(pid, 'create_checkpoint', {'reason': 'tool'}).ok
            assert 'create_checkpoint' in runtime.process.get(pid).tool_table
            assert 'inspect_checkpoint' in runtime.process.get(pid).tool_table
            assert 'diff_checkpoint' in runtime.process.get(pid).tool_table
            assert 'list_checkpoints' in runtime.process.get(pid).tool_table
            assert 'restore_checkpoint' not in runtime.process.get(pid).tool_table
            assert 'fork_checkpoint' not in runtime.process.get(pid).tool_table
        finally:
            runtime.close()

    def test_checkpoint_cli_outputs_stable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / 'runtime.sqlite')
            spawned = checkpoint_cli_json(['--db', db_path, 'spawn', '--goal', 'cli checkpoint'])
            created = checkpoint_cli_json(['--db', db_path, 'checkpoint', 'create', spawned['pid'], 'cli reason'])
            listed = checkpoint_cli_json(['--db', db_path, 'checkpoint', 'list', '--pid', spawned['pid']])
            inspected = checkpoint_cli_json(['--db', db_path, 'checkpoint', 'inspect', created['checkpoint_id']])
            assert created['checkpoint_id'].startswith('ckpt_')
            assert listed[0]['checkpoint_id'] == created['checkpoint_id']
            assert inspected['checkpoint']['pid'] == spawned['pid']

    def _run(self, awaitable: Any) -> Any:
        return asyncio.run(awaitable)
