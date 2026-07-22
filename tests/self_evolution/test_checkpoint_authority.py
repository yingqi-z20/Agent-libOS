from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
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

    def test_one_shot_checkpoint_read_is_consumed_by_inspect(self) -> None:
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

            inspected = runtime.checkpoint.inspect(checkpoint_id, actor=reader)

            assert inspected['checkpoint']['checkpoint_id'] == checkpoint_id
            assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.inspect(checkpoint_id, actor=reader)
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
