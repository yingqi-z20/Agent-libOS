from __future__ import annotations

import threading
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ObjectOwnerKind, ObjectType, ToolCandidateStatus
from agent_libos.models.exceptions import NotFound, ValidationError


class TestCheckpointJit:

    def test_process_tool_configuration_rejects_cross_owner_process_local_jit(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(
                image='toolmaker-agent:v0',
                goal='own private jit',
            )
            other = runtime.process.spawn(
                image='base-agent:v0',
                goal='must not borrow private jit',
            )
            candidate_id = runtime.tools.propose(
                owner,
                {
                    'name': 'owner_only_echo',
                    'description': 'Owner-local echo.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args, libos) { return args; }',
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            handle = runtime.tools.register(owner, candidate_id)
            assert runtime.tools.configure_process_tools(
                owner,
                [handle],
                assigned_by='test.same-owner',
            ) == {handle.name: handle.tool_id}
            runtime.tools.configure_process_tools(
                other,
                ['get_current_time'],
                assigned_by='test.shared-builtin',
            )
            before = runtime.process.get(other)

            with pytest.raises(ValidationError, match='process-local JIT'):
                runtime.tools.configure_process_tools(
                    other,
                    [handle],
                    assigned_by='test.cross-owner',
                )

            after = runtime.process.get(other)
            assert after.tool_table == before.tool_table
            assert after.model_tool_table == before.model_tool_table
            assert after.revision == before.revision
            denial = [
                record
                for record in runtime.audit.trace()
                if record.action == 'process.tools.configure_denied'
            ][-1]
            assert denial.actor == 'test.cross-owner'
            assert denial.target == f'process:{other}'
            assert denial.decision == {
                'reason': 'process_local_jit_owner_mismatch',
                'tools': [
                    {
                        'name': handle.name,
                        'tool_id': handle.tool_id,
                    }
                ],
            }
            runtime.tools.forget_loaded_jit(handle.tool_id)
            with pytest.raises(ValidationError, match='process-local JIT'):
                runtime.tools.configure_process_tools(
                    other,
                    [handle],
                    assigned_by='test.cross-owner.unloaded',
                )
            assert runtime.process.get(other).tool_table == before.tool_table
            assert [
                record.actor
                for record in runtime.audit.trace()
                if record.action == 'process.tools.configure_denied'
            ][-1] == 'test.cross-owner.unloaded'
        finally:
            runtime.close()

    def test_checkpoint_restore_rejects_legacy_owner_unbound_cross_owner_jit(
        self,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'legacy-cross-owner-jit.sqlite3'
        runtime = Runtime.open(target)
        owner = ''
        other = ''
        candidate_id = ''
        handle = None
        other_before: dict[str, str] = {}
        try:
            owner = runtime.process.spawn(
                image='toolmaker-agent:v0',
                goal='legacy jit owner',
            )
            checkpoint_id = runtime.checkpoint.create(
                owner,
                'before private jit',
                actor=owner,
            )
            other = runtime.process.spawn(
                image='base-agent:v0',
                goal='legacy invalid borrower',
            )
            candidate_id = runtime.tools.propose(
                owner,
                {
                    'name': 'legacy_owner_echo',
                    'description': 'Legacy owner-local echo.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args, libos) { return args; }',
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            handle = runtime.tools.register(owner, candidate_id)
            runtime.tools.configure_process_tools(
                owner,
                [],
                assigned_by='test.legacy-owner-unbind',
            )

            # Simulate a binding persisted by a pre-fix Runtime. Public
            # configure_process_tools now rejects this state before it exists.
            other_process = runtime.store.get_process(other)
            other_before = dict(other_process.tool_table)
            borrowed = {**other_process.tool_table, handle.name: handle.tool_id}
            runtime.store.patch_process(
                other,
                {
                    'tool_table': borrowed,
                    'model_tool_table': borrowed,
                },
                expected_revision=other_process.revision,
            )

            with pytest.raises(
                ValidationError,
                match='outside the durable owner scope',
            ):
                runtime.checkpoint.restore(
                    'cli',
                    checkpoint_id,
                    require_capability=False,
                )

            assert runtime.store.get_tool_candidate(candidate_id) is not None
            assert handle.tool_id not in runtime.process.get(owner).tool_table.values()
            assert runtime.process.get(other).tool_table[handle.name] == handle.tool_id
        finally:
            runtime.close()

        reopened = Runtime.open(target)
        try:
            assert handle is not None
            assert handle.tool_id not in reopened.process.get(owner).tool_table.values()
            assert reopened.process.get(other).tool_table == other_before
            assert reopened.store.get_tool_candidate(candidate_id) is not None
            assert handle.tool_id in {
                row['tool_id'] for row in reopened.store.list_tools()
            }
            assert reopened.tools.jit_source(handle.tool_id) is None
        finally:
            reopened.close()

    def test_checkpoint_fork_does_not_publish_process_before_jit_assets_are_ready(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='atomic jit fork')
            source = 'export function run(args, libos) { return { ok: true }; }'
            candidate_id = runtime.tools.propose(
                pid,
                {
                    'name': 'atomic_echo',
                    'description': 'Atomic fork tool.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code=source,
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            runtime.tools.register(pid, candidate_id)
            checkpoint_id = runtime.checkpoint.create(pid, 'atomic jit fork', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}
            restore_entered = threading.Event()
            allow_restore = threading.Event()
            original_restore = runtime.checkpoint._restore_jit_sources
            outcome: dict[str, object] = {}

            def blocked_restore(snapshot):
                restore_entered.set()
                assert allow_restore.wait(timeout=2.0)
                return original_restore(snapshot)

            def fork() -> None:
                try:
                    outcome['result'] = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
                except BaseException as exc:  # pragma: no cover - asserted below
                    outcome['error'] = exc

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', blocked_restore)
            worker = threading.Thread(target=fork)
            worker.start()
            assert restore_entered.wait(timeout=2.0)
            try:
                assert {process.pid for process in runtime.process.list()} == before_pids
            finally:
                allow_restore.set()
                worker.join(timeout=2.0)

            assert not worker.is_alive()
            assert 'error' not in outcome
            result = outcome['result']
            assert isinstance(result, dict)
            assert runtime.store.get_process(result['fork_root_pid']) is not None
        finally:
            runtime.close()

    def test_checkpoint_fork_clones_registered_jit_tool_and_candidate_identity(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit checkpoint isolation')
            source = 'export function run(args, libos) { return { value: args.value }; }'
            candidate_id = runtime.tools.propose(
                pid,
                {
                    'name': 'isolated_echo',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object', 'properties': {'value': {'type': 'string'}}},
                    'output_schema': {'type': 'object'},
                },
                source_code=source,
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            source_handle = runtime.tools.register(pid, candidate_id)
            process = runtime.store.get_process(pid)
            process.loaded_skills['skill:test-jit'] = {
                'tool_ids': {},
                'jit_tool_ids': {'isolated_echo': source_handle.tool_id},
                'base_tool_ids': {},
                'base_model_tool_ids': {},
            }
            runtime.store.update_process(process)
            checkpoint_id = runtime.checkpoint.create(pid, 'jit registered', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_pid = forked['fork_root_pid']
            fork_handle = runtime.tools.resolve('isolated_echo', pid=fork_pid)

            assert fork_handle.tool_id != source_handle.tool_id
            assert runtime.tools.jit_source(fork_handle.tool_id) == source
            fork_process = runtime.store.get_process(fork_pid)
            assert (
                fork_process.loaded_skills['skill:test-jit']['jit_tool_ids']['isolated_echo']
                == fork_handle.tool_id
            )
            assert fork_process.tool_table['isolated_echo'] == fork_handle.tool_id
            assert fork_process.model_tool_table['isolated_echo'] == fork_handle.tool_id
            assert source_handle.tool_id not in fork_process.model_tool_table.values()
            fork_candidates = runtime.store.select_table_rows(
                'tool_candidates',
                'pid = ?',
                (fork_pid,),
            )
            assert len(fork_candidates) == 1
            assert fork_candidates[0]['candidate_id'] != candidate_id
            assert fork_candidates[0]['registered_tool_id'] == fork_handle.tool_id
            candidate_objects = [
                obj
                for obj in runtime.store.list_objects_owned_by(ObjectOwnerKind.PROCESS, fork_pid)
                if obj.type == ObjectType.TOOL_CANDIDATE
            ]
            assert len(candidate_objects) == 1
            assert candidate_objects[0].payload['candidate_id'] == fork_candidates[0]['candidate_id']

            unloaded = runtime.skills.unload_skill(
                fork_pid,
                'skill:test-jit',
                require_capability=False,
            )
            assert unloaded['removed_tools'] == ['isolated_echo']
            with pytest.raises(NotFound):
                runtime.tools.resolve('isolated_echo', pid=fork_pid)
            assert runtime.tools.resolve('isolated_echo', pid=pid).tool_id == source_handle.tool_id
            assert runtime.tools.jit_source(source_handle.tool_id) == source
        finally:
            runtime.close()

    def test_checkpoint_restores_registered_jit_tool_source(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit checkpoint')
            source = 'export function run(args, libos) { return { value: args.value }; }'
            candidate_id = runtime.tools.propose(pid, {'name': 'echo_value', 'description': 'Echo a value.', 'input_schema': {'type': 'object', 'properties': {'value': {'type': 'string'}}}, 'output_schema': {'type': 'object'}}, source_code=source)
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            handle = runtime.tools.register(pid, candidate_id)
            checkpoint_id = runtime.checkpoint.create(pid, 'jit registered', actor=pid)
            runtime.tools.forget_loaded_jit(handle.tool_id)
            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            assert runtime.tools.jit_source(handle.tool_id) == source
            assert runtime.tools.resolve('echo_value', pid=pid).tool_id == handle.tool_id
            with pytest.raises(NotFound):
                runtime.tools.resolve('echo_value')
            other = runtime.process.spawn(image='base-agent:v0', goal='cannot import restored JIT')
            with pytest.raises(NotFound):
                runtime.tools.configure_process_tools(other, ['echo_value'], assigned_by='test')
            runtime.tools.forget_loaded_jit(handle.tool_id)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_handle = runtime.tools.resolve('echo_value', pid=forked['fork_root_pid'])
            assert fork_handle.tool_id != handle.tool_id
            assert runtime.tools.jit_source(fork_handle.tool_id) == source
        finally:
            runtime.close()

    def test_checkpoint_restore_prunes_post_checkpoint_ephemeral_jit_tool(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit checkpoint prune')
            checkpoint_id = runtime.checkpoint.create(pid, 'before jit registration', actor=pid)
            source = 'export function run(args, libos) { return { value: args.value }; }'
            candidate_id = runtime.tools.propose(
                pid,
                {
                    'name': 'late_echo_value',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object', 'properties': {'value': {'type': 'string'}}},
                    'output_schema': {'type': 'object'},
                },
                source_code=source,
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            handle = runtime.tools.register(pid, candidate_id)

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)

            assert not runtime.tools.is_jit_tool_id(handle.tool_id)
            assert runtime.tools.loaded_tool_handle(handle.tool_id) is None
            assert handle.tool_id not in {row['tool_id'] for row in runtime.store.list_tools()}
            with pytest.raises(NotFound):
                runtime.tools.resolve('late_echo_value', pid=pid)
        finally:
            runtime.close()

    def test_checkpoint_restore_prune_failure_recovers_before_rehydrate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        target = tmp_path / 'jit-prune-recovery.sqlite3'
        runtime = Runtime.open(target)
        publication_id = ''
        handle = None
        try:
            pid = runtime.process.spawn(
                image='toolmaker-agent:v0',
                goal='durable jit prune recovery',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'before durable jit registration',
                actor=pid,
            )
            source = 'export function run(args, libos) { return { value: args.value }; }'
            candidate_id = runtime.tools.propose(
                pid,
                {
                    'name': 'late_durable_echo',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code=source,
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            handle = runtime.tools.register(pid, candidate_id)
            original_delete = runtime.store.delete_tool
            failed = False

            def fail_first_delete(tool_id: str) -> None:
                nonlocal failed
                if tool_id == handle.tool_id and not failed:
                    failed = True
                    raise RuntimeError('injected durable JIT prune failure')
                original_delete(tool_id)

            monkeypatch.setattr(runtime.store, 'delete_tool', fail_first_delete)
            result = runtime.checkpoint.restore(
                'cli',
                checkpoint_id,
                require_capability=False,
            )
            publication_id = str(result['publication_id'])

            assert result['status'] == 'restored_with_warnings'
            assert result['post_commit_failures'][0]['phase'] == 'jit_pruning'
            publication = runtime.store.get_runtime_publication(publication_id)
            assert publication is not None
            assert publication['state'] == 'failed'
            assert publication['plan']['stale_tool_ids'] == [handle.tool_id]
            assert runtime.tools.is_jit_tool_id(handle.tool_id)
            assert handle.tool_id in {row['tool_id'] for row in runtime.store.list_tools()}
            assert runtime.lifecycle.state == 'close_failed'

            monkeypatch.setattr(runtime.store, 'delete_tool', original_delete)
            release = runtime.release_recovery_diagnostics()
            assert release['ok'] is True, release
            assert release['recovery_diagnostics_released'] is True
            assert runtime.lifecycle.closed
            reopened = Runtime.open(target)
            try:
                assert reopened.recovered_checkpoint_restore_publications == [
                    publication_id
                ]
                assert handle.tool_id not in {
                    row['tool_id'] for row in reopened.store.list_tools()
                }
                assert not reopened.tools.is_jit_tool_id(handle.tool_id)
                assert reopened.tools.loaded_tool_handle(handle.tool_id) is None
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'committed'
            finally:
                reopened.close()

            reopened_again = Runtime.open(target)
            try:
                assert reopened_again.recovered_checkpoint_restore_publications == []
                assert handle.tool_id not in {
                    row['tool_id'] for row in reopened_again.store.list_tools()
                }
            finally:
                reopened_again.close()
        finally:
            if not runtime.lifecycle.closed:
                runtime.close()
