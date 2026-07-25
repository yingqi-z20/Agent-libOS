from __future__ import annotations
import pytest
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ObjectMetadata, ObjectType, ResourceBudget, ResourceUsage, ToolCandidateStatus
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ProcessError, ValidationError
from tests.support.deno import COUNT_CHARS_SOURCE
from tests.support.skills import write_skill_package


def _identity_manifest(allowed_tenants: list[str]) -> dict[str, object]:
    return {
        'data_flow_policy': {
            'schema_version': 1,
            'allowed_tenants': allowed_tenants,
            'allowed_principals': [],
        }
    }


def _commit_tenant_image(runtime: Runtime) -> str:
    source = runtime.process.spawn(
        image='base-agent:v0',
        goal='capture tenant-a state',
        authority_manifest=_identity_manifest(['tenant-a']),
    )
    runtime.memory.create_object(
        source,
        ObjectType.ARTIFACT,
        {'marker': 'CHECKPOINT_IMAGE_TENANT_A'},
        metadata=ObjectMetadata(
            title='tenant-a baked state',
            sensitivity='secret',
            tenant='tenant-a',
        ),
        name='tenant-a-baked-state',
        immutable=True,
    )
    checkpoint_id = runtime.checkpoint.create(source, 'tenant-a image state', actor=source)
    runtime.image_registry.grant_register(source, 'tenant-domain-image:v0', issued_by='test')
    runtime.image_registry.commit_from_checkpoint(
        actor=source,
        checkpoint_id=checkpoint_id,
        image_id='tenant-domain-image:v0',
        name='tenant-domain-image',
    )
    return 'tenant-domain-image:v0'

class TestImageCommit:

    def test_commit_preserves_source_image_projection_policy(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='coding-agent:v0', goal='commit coding projection')
            checkpoint_id = runtime.checkpoint.create(source, 'coding projection', actor=source)
            runtime.image_registry.grant_register(source, 'coding-projection:v0', issued_by='test')

            result = runtime.image_registry.commit_from_checkpoint(
                actor=source,
                checkpoint_id=checkpoint_id,
                image_id='coding-projection:v0',
                name='coding-projection',
            )

            assert result.image.metadata['tool_projection'] == 'skills'
            assert result.image.default_skills == []
            assert 'lazy_tool_groups' not in result.image.metadata
            assert 'initial_tool_groups' not in result.image.metadata

    def test_pre_03_checkpoint_is_rejected_before_image_commit_write(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='base-agent:v0', goal='old commit source')
            checkpoint_id = runtime.checkpoint.create(source, 'old commit checkpoint', actor=source)
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            snapshot['version'] = 1
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (json.dumps(snapshot, sort_keys=True), checkpoint_id),
            )
            total_changes = runtime.store.conn.total_changes

            with pytest.raises(ValidationError, match='unsupported snapshot version'):
                runtime.image_registry.commit_from_checkpoint(
                    actor=source,
                    checkpoint_id=checkpoint_id,
                    image_id='must-not-commit:v0',
                    name='must-not-commit',
                    require_capability=False,
                )

            assert runtime.store.conn.total_changes == total_changes
            assert 'must-not-commit:v0' not in runtime.images
            assert runtime.store.get_image('must-not-commit:v0') is None

    def test_pre_03_checkpoint_artifact_is_rejected_before_process_write(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='base-agent:v0', goal='old artifact source')
            checkpoint_id = runtime.checkpoint.create(source, 'old artifact', actor=source)
            runtime.image_registry.grant_register(source, 'old-artifact:v0', issued_by='test')
            committed = runtime.image_registry.commit_from_checkpoint(
                actor=source,
                checkpoint_id=checkpoint_id,
                image_id='old-artifact:v0',
                name='old-artifact',
            )
            artifact_id = committed.image.boot['artifact_id']
            found = runtime.store.get_image_artifact(artifact_id)
            assert found is not None
            artifact, _metadata = found
            artifact['artifact_version'] = 1
            runtime.store._execute(
                'UPDATE image_artifacts SET artifact_json = ? WHERE artifact_id = ?',
                (json.dumps(artifact, sort_keys=True), artifact_id),
            )
            process_ids = {process.pid for process in runtime.store.list_processes()}
            total_changes = runtime.store.conn.total_changes

            with pytest.raises(RuntimeError, match='artifact version mismatch'):
                runtime.process.spawn(image='old-artifact:v0', goal='must not start')

            assert runtime.store.conn.total_changes == total_changes
            assert {process.pid for process in runtime.store.list_processes()} == process_ids
            persisted = runtime.store.get_image_artifact(artifact_id)
            assert persisted is not None
            assert persisted[0]['artifact_version'] == 1

    def test_commit_settles_checkpoint_read_and_image_write_only_with_image_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='base-agent:v0', goal='commit source')
            controller = runtime.process.spawn(image='base-agent:v0', goal='commit controller')
            checkpoint_id = runtime.checkpoint.create(source, 'one-shot commit point', actor=source)
            checkpoint_cap = runtime.capability.grant_once(
                controller,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.READ],
                issued_by='test',
            )
            image_cap = runtime.capability.grant_once(
                controller,
                'image:one-shot-commit:v0',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            original_record = runtime.audit.record
            fail_commit_audit = True

            def fail_once(*args: Any, **kwargs: Any) -> Any:
                if fail_commit_audit and kwargs.get('action') == 'image.commit':
                    raise RuntimeError('injected image commit audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_once)
            with pytest.raises(RuntimeError, match='image commit audit failure'):
                runtime.image_registry.commit_from_checkpoint(
                    actor=controller,
                    checkpoint_id=checkpoint_id,
                    image_id='one-shot-commit:v0',
                    name='one-shot-commit',
                )

            assert runtime.store.get_capability(checkpoint_cap.cap_id).uses_remaining == 1
            assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 1
            assert runtime.store.get_image('one-shot-commit:v0') is None
            assert 'one-shot-commit:v0' not in runtime.images

            fail_commit_audit = False
            result = runtime.image_registry.commit_from_checkpoint(
                actor=controller,
                checkpoint_id=checkpoint_id,
                image_id='one-shot-commit:v0',
                name='one-shot-commit',
            )

            assert result.image.image_id == 'one-shot-commit:v0'
            assert runtime.store.get_capability(checkpoint_cap.cap_id).uses_remaining == 0
            assert runtime.store.get_capability(image_cap.cap_id).uses_remaining == 0

    def test_commit_requires_checkpoint_read_and_image_write(self) -> None:
        with _runtime() as runtime:
            pid = runtime.process.spawn(image='base-agent:v0', goal='commit source')
            checkpoint_id = runtime.checkpoint.create(pid, 'commit point', actor=pid)
            with pytest.raises(CapabilityDenied):
                runtime.image_registry.commit_from_checkpoint(actor=pid, checkpoint_id=checkpoint_id, image_id='committed-no-write:v0', name='committed-no-write')
            runtime.image_registry.grant_register(pid, 'committed-no-read:v0', issued_by='test')
            other = runtime.process.spawn(image='base-agent:v0', goal='other')
            with pytest.raises(CapabilityDenied):
                runtime.image_registry.commit_from_checkpoint(actor=other, checkpoint_id=checkpoint_id, image_id='committed-no-read:v0', name='committed-no-read')

    def test_committed_image_spawns_baked_memory_without_external_authority(self) -> None:
        with _runtime() as runtime:
            pid = runtime.process.spawn(image='base-agent:v0', goal='learn state')
            runtime.memory.create_object(pid=pid, object_type=ObjectType.ARTIFACT, payload={'learned': 'state'}, metadata=ObjectMetadata(title='Baked state'), name='baked-state', immutable=True)
            runtime.filesystem.grant_path(pid, 'README.md', [CapabilityRight.READ], issued_by='test')
            runtime.capability.grant(pid, 'custom_provider:remote-state', [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'state ready', actor=pid)
            runtime.image_registry.grant_register(pid, 'stateful-agent:v0', issued_by='test')
            result = runtime.image_registry.commit_from_checkpoint(actor=pid, checkpoint_id=checkpoint_id, image_id='stateful-agent:v0', name='stateful-agent')
            assert result.image.boot['kind'] == 'checkpoint_commit'
            assert result.image.required_capabilities
            required_resources = {item['resource'] for item in result.image.required_capabilities}
            assert 'filesystem:workspace:README.md' in required_resources
            assert 'custom_provider:remote-state' in required_resources
            child = runtime.process.spawn(image='stateful-agent:v0', goal='use baked state')
            baked = runtime.memory.get_object_by_name(child, 'baked-state')
            assert baked.payload == {'learned': 'state'}
            assert not runtime.capability.check(child, 'filesystem:workspace:README.md', CapabilityRight.READ)
            assert not runtime.capability.check(child, 'custom_provider:remote-state', CapabilityRight.READ)
            assert 'image.required_capabilities_declared_only' in [record.action for record in runtime.audit.trace()]

    def test_exec_into_committed_image_restores_baked_memory_without_granting_required_caps(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='base-agent:v0', goal='source')
            runtime.memory.create_object(pid=source, object_type=ObjectType.ARTIFACT, payload={'role': 'committed'}, metadata=ObjectMetadata(title='Role'), name='role', immutable=True)
            runtime.capability.grant(source, 'shell:python', [CapabilityRight.EXECUTE], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(source, 'before commit', actor=source)
            runtime.image_registry.grant_register(source, 'exec-state:v0', issued_by='test')
            runtime.image_registry.commit_from_checkpoint(actor=source, checkpoint_id=checkpoint_id, image_id='exec-state:v0', name='exec-state')
            target = runtime.process.spawn(image='base-agent:v0', goal='target')
            runtime.capability.grant(target, runtime.image_registry.resource_for('exec-state:v0'), [CapabilityRight.READ], issued_by='test')
            runtime.exec_process(target, 'exec-state:v0', goal='new goal', preserve_capabilities=False)
            assert runtime.memory.get_object_by_name(target, 'role').payload == {'role': 'committed'}
            assert not runtime.capability.check(target, 'shell:python', CapabilityRight.EXECUTE)

    def test_committed_image_spawn_rejects_baked_object_outside_identity_domain(self) -> None:
        with _runtime() as runtime:
            image_id = _commit_tenant_image(runtime)
            processes_before = {process.pid for process in runtime.store.list_processes()}

            with pytest.raises(CapabilityDenied, match='data_flow_policy'):
                runtime.process.spawn(
                    image=image_id,
                    goal='restricted root boot',
                    authority_manifest=_identity_manifest([]),
                )

            assert {process.pid for process in runtime.store.list_processes()} == processes_before
            allowed = runtime.process.spawn(
                image=image_id,
                goal='same-domain root boot',
                authority_manifest=_identity_manifest(['tenant-a']),
            )
            assert runtime.memory.get_object_by_name(
                allowed,
                'tenant-a-baked-state',
            ).payload == {'marker': 'CHECKPOINT_IMAGE_TENANT_A'}

    def test_committed_image_exec_rejects_baked_object_outside_identity_domain(self) -> None:
        with _runtime() as runtime:
            image_id = _commit_tenant_image(runtime)
            target = runtime.process.spawn(
                image='base-agent:v0',
                goal='restricted exec target',
                authority_manifest=_identity_manifest([]),
            )
            before = runtime.process.get(target)
            runtime.capability.grant(
                target,
                runtime.image_registry.resource_for(image_id),
                [CapabilityRight.READ],
                issued_by='test',
            )

            with pytest.raises(CapabilityDenied, match='data_flow_policy'):
                runtime.exec_process(
                    target,
                    image_id,
                    goal='boot committed image',
                    preserve_capabilities=False,
                )

            after = runtime.process.get(target)
            assert after.image_id == before.image_id == 'base-agent:v0'
            assert after.goal_oid == before.goal_oid
            with pytest.raises(NotFound):
                runtime.memory.get_object_by_name(target, 'tenant-a-baked-state')
            allowed = runtime.process.spawn(
                image='base-agent:v0',
                goal='same-domain exec target',
                authority_manifest=_identity_manifest(['tenant-a']),
            )
            runtime.capability.grant(
                allowed,
                runtime.image_registry.resource_for(image_id),
                [CapabilityRight.READ],
                issued_by='test',
            )
            runtime.exec_process(
                allowed,
                image_id,
                goal='same-domain committed image',
                preserve_capabilities=False,
            )
            assert runtime.memory.get_object_by_name(
                allowed,
                'tenant-a-baked-state',
            ).payload == {'marker': 'CHECKPOINT_IMAGE_TENANT_A'}

    def test_committed_image_boot_requires_image_read_authority(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='base-agent:v0', goal='source')
            runtime.memory.create_object(pid=source, object_type=ObjectType.ARTIFACT, payload={'secret': 'baked'}, metadata=ObjectMetadata(title='Secret'), name='baked-secret', immutable=True)
            checkpoint_id = runtime.checkpoint.create(source, 'before commit', actor=source)
            runtime.image_registry.grant_register(source, 'secret-state:v0', issued_by='test')
            runtime.image_registry.commit_from_checkpoint(actor=source, checkpoint_id=checkpoint_id, image_id='secret-state:v0', name='secret-state')

            attacker = runtime.process.spawn(image='base-agent:v0', goal='attacker')
            runtime.capability.grant(attacker, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            with pytest.raises(CapabilityDenied, match='image:secret-state:v0'):
                runtime.spawn_child_process(attacker, 'steal baked state', image='secret-state:v0')
            assert runtime.process.list_children(attacker) == []

            exec_target = runtime.process.spawn(image='base-agent:v0', goal='exec target')
            with pytest.raises(CapabilityDenied, match='image:secret-state:v0'):
                runtime.exec_process(exec_target, 'secret-state:v0', goal='steal via exec', preserve_capabilities=False)
            assert runtime.process.get(exec_target).image_id == 'base-agent:v0'

            runtime.capability.grant(attacker, runtime.image_registry.resource_for('secret-state:v0'), [CapabilityRight.READ], issued_by='test')
            child = runtime.spawn_child_process(attacker, 'authorized boot', image='secret-state:v0')
            assert runtime.memory.get_object_by_name(child, 'baked-secret').payload == {'secret': 'baked'}

    def test_committed_image_does_not_save_or_restore_resource_state(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(
                image='base-agent:v0',
                goal='source',
                resource_budget=ResourceBudget(max_tool_calls=5, max_llm_total_tokens=20),
            )
            runtime.resources.charge(source, ResourceUsage(tool_calls=2, llm_total_tokens=7), source='test')
            checkpoint_id = runtime.checkpoint.create(source, 'resource state', actor=source)
            runtime.image_registry.grant_register(source, 'resource-state:v0', issued_by='test')
            result = runtime.image_registry.commit_from_checkpoint(actor=source, checkpoint_id=checkpoint_id, image_id='resource-state:v0', name='resource-state')

            found = runtime.store.get_image_artifact(result.image.boot['artifact_id'])
            assert found is not None
            artifact, _metadata = found
            assert 'resource_budget_json' not in artifact['source_process']
            assert 'resource_usage_json' not in artifact['source_process']

            spawned = runtime.process.spawn(
                image='resource-state:v0',
                goal='spawned',
                resource_budget=ResourceBudget(max_tool_calls=3, max_llm_total_tokens=30),
            )
            process = runtime.process.get(spawned)

            assert process.resource_budget.max_tool_calls == 3
            assert process.resource_budget.max_llm_total_tokens == 30
            assert process.resource_usage.tool_calls == 0
            assert process.resource_usage.llm_total_tokens == 0

    def test_committed_jit_tool_remains_process_local(self) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit source')
            tool_source = 'export function run(args, libos) { return { value: args.value }; }'
            candidate_id = runtime.tools.propose(
                source,
                {
                    'name': 'committed_echo_value',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code=tool_source,
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            runtime.tools.register(source, candidate_id)
            checkpoint_id = runtime.checkpoint.create(source, 'jit ready', actor=source)
            runtime.image_registry.grant_register(source, 'jit-commit:v0', issued_by='test')
            runtime.image_registry.commit_from_checkpoint(
                actor=source,
                checkpoint_id=checkpoint_id,
                image_id='jit-commit:v0',
                name='jit-commit',
            )

            booted = runtime.process.spawn(image='jit-commit:v0', goal='boot jit')
            other = runtime.process.spawn(image='toolmaker-agent:v0', goal='unrelated')

            assert 'committed_echo_value' in runtime.process.get(booted).tool_table
            assert 'committed_echo_value' not in runtime.process.get(other).tool_table
            with pytest.raises(NotFound):
                runtime.tools.resolve('committed_echo_value')
            other_call = runtime.tools.call(other, 'committed_echo_value', {'value': 'x'})
            assert not other_call.ok
            assert 'not in process tool table' in (other_call.error or '')

    def test_failed_committed_jit_launch_removes_new_tool_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit source')
            candidate_id = runtime.tools.propose(
                source,
                {
                    'name': 'rollback_committed_echo',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code=(
                    'export function run(args, libos) { return { value: args.value }; }'
                ),
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            runtime.tools.register(source, candidate_id)
            checkpoint_id = runtime.checkpoint.create(source, 'jit ready', actor=source)
            runtime.image_registry.grant_register(
                source,
                'jit-launch-rollback:v0',
                issued_by='test',
            )
            runtime.image_registry.commit_from_checkpoint(
                actor=source,
                checkpoint_id=checkpoint_id,
                image_id='jit-launch-rollback:v0',
                name='jit-launch-rollback',
            )
            before_tool_ids = {row['tool_id'] for row in runtime.store.list_tools()}
            before_loaded_jit = runtime.tools.loaded_jit_tool_ids()
            original_advance = runtime.store.advance_runtime_publication

            def reject_launch_commit(publication_id: str, **kwargs: Any) -> bool:
                publication = runtime.store.get_runtime_publication(publication_id)
                if (
                    kwargs.get('state') == 'committed'
                    and publication is not None
                    and publication['kind'] == 'process_launch'
                    and publication['plan'].get('image_id') == 'jit-launch-rollback:v0'
                ):
                    return False
                return original_advance(publication_id, **kwargs)

            monkeypatch.setattr(
                runtime.store,
                'advance_runtime_publication',
                reject_launch_commit,
            )

            with pytest.raises(ProcessError, match='cannot commit process publication'):
                runtime.process.spawn(
                    image='jit-launch-rollback:v0',
                    goal='must be compensated',
                )

            failed_publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'process_launch'
                and item['plan'].get('image_id') == 'jit-launch-rollback:v0'
            ][-1]
            failed_pid = failed_publication['pid']
            assert runtime.store.get_process(failed_pid) is None
            assert runtime.store.select_table_rows(
                'tool_candidates',
                'pid = ?',
                [failed_pid],
            ) == []
            assert {row['tool_id'] for row in runtime.store.list_tools()} == before_tool_ids
            assert runtime.tools.loaded_jit_tool_ids() == before_loaded_jit
            assert failed_publication['state'] == 'rolled_back'

    def test_checkpoint_image_final_audit_failure_leaves_no_jit_orphan(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with _runtime() as runtime:
            source = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit source')
            candidate_id = runtime.tools.propose(
                source,
                {
                    'name': 'audit_failure_committed_echo',
                    'description': 'Echo a value.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args) { return { value: args.value }; }',
            )
            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            candidate.status = ToolCandidateStatus.VALIDATED
            candidate.validation = {'ok': True, 'language': 'typescript'}
            runtime.store.update_tool_candidate(candidate)
            runtime.tools.register(source, candidate_id)
            checkpoint_id = runtime.checkpoint.create(source, 'jit ready', actor=source)
            runtime.image_registry.grant_register(
                source,
                'jit-audit-rollback:v0',
                issued_by='test',
            )
            runtime.image_registry.commit_from_checkpoint(
                actor=source,
                checkpoint_id=checkpoint_id,
                image_id='jit-audit-rollback:v0',
                name='jit-audit-rollback',
            )
            before_tool_ids = {row['tool_id'] for row in runtime.store.list_tools()}
            before_loaded_jit = runtime.tools.loaded_jit_tool_ids()
            original_record = runtime.audit.record

            def fail_checkpoint_boot_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') == 'image.boot.checkpoint_commit':
                    raise RuntimeError('injected checkpoint image final audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_checkpoint_boot_audit)

            with pytest.raises(RuntimeError, match='final audit failure'):
                runtime.process.spawn(
                    image='jit-audit-rollback:v0',
                    goal='must be compensated before receipt publication',
                )

            failed_publication = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'process_launch'
                and item['plan'].get('image_id') == 'jit-audit-rollback:v0'
            ][-1]
            assert failed_publication['state'] == 'rolled_back'
            assert runtime.store.get_process(failed_publication['pid']) is None
            assert {row['tool_id'] for row in runtime.store.list_tools()} == before_tool_ids
            assert runtime.tools.loaded_jit_tool_ids() == before_loaded_jit

    @pytest.mark.real_deno
    def test_committed_jit_tool_source_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            db_path = Path(temp_dir) / 'runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                source = runtime.process.spawn(image='toolmaker-agent:v0', goal='jit source')
                tool_source = COUNT_CHARS_SOURCE
                candidate_id = runtime.tools.propose(
                    source,
                    {
                        'name': 'committed_persistent_count',
                        'description': 'Count text characters.',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                    },
                    source_code=tool_source,
                )
                candidate = runtime.store.get_tool_candidate(candidate_id)
                assert candidate is not None
                candidate.status = ToolCandidateStatus.VALIDATED
                candidate.validation = {'ok': True, 'language': 'typescript'}
                runtime.store.update_tool_candidate(candidate)
                runtime.tools.register(source, candidate_id)
                checkpoint_id = runtime.checkpoint.create(source, 'jit ready', actor=source)
                runtime.image_registry.grant_register(source, 'jit-commit-reopen:v0', issued_by='test')
                runtime.image_registry.commit_from_checkpoint(
                    actor=source,
                    checkpoint_id=checkpoint_id,
                    image_id='jit-commit-reopen:v0',
                    name='jit-commit-reopen',
                )
                booted = runtime.process.spawn(image='jit-commit-reopen:v0', goal='boot jit')
                tool_id = runtime.process.get(booted).tool_table['committed_persistent_count']
            finally:
                runtime.shutdown(actor='test', reason='test complete')

            reopened = Runtime.open(db_path)
            try:
                assert reopened.tools.resolve('committed_persistent_count', pid=booted).tool_id == tool_id
                result = reopened.tools.call(booted, 'committed_persistent_count', {'text': 'hello'})
                assert result.ok, result.error
                assert result.payload == {'count': 5}
            finally:
                reopened.shutdown(actor='test', reason='test complete')

    def test_committed_image_does_not_package_external_registry_or_skill_trust(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            with _runtime() as runtime:
                pid = runtime.process.spawn(image='base-agent:v0', goal='external registries')
                runtime.jsonrpc.register_endpoint(
                    {
                        'schema_version': 1,
                        'endpoint_id': 'demo-endpoint',
                        'url': 'https://example.com/rpc',
                        'headers': {},
                        'methods': [
                            {
                                'method_id': 'read_status',
                                'rpc_method': 'status.read',
                                'right': 'read',
                                'rollback_class': 'no_rollback_required',
                                'state_mutation': False,
                                'information_flow': True,
                            }
                        ],
                    },
                    actor='cli',
                    require_capability=False,
                )
                runtime.capability.grant(pid, 'jsonrpc_endpoint:demo-endpoint', [CapabilityRight.READ], issued_by='test')

                skill_dir = write_skill_package(Path(temp_dir), 'trusted-image-skill', allowed_tools=['human_output'])
                skill = runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.skills.trust_skill_source(
                    actor='cli',
                    source_type=skill['source_type'],
                    source=skill['source'],
                    package_sha256=skill['package_sha256'],
                    require_capability=False,
                )
                runtime.capability.grant(pid, 'skill:trusted-image-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'trusted-image-skill', actor=pid)

                checkpoint_id = runtime.checkpoint.create(pid, 'before image commit', actor=pid)
                runtime.image_registry.grant_register(pid, 'registry-free:v0', issued_by='test')
                result = runtime.image_registry.commit_from_checkpoint(
                    actor=pid,
                    checkpoint_id=checkpoint_id,
                    image_id='registry-free:v0',
                    name='registry-free',
                )

                found = runtime.store.get_image_artifact(result.image.boot['artifact_id'])
                assert found is not None
                artifact, _metadata = found
                assert 'jsonrpc_endpoints' not in artifact['rows']
                assert 'skill_trust' not in artifact['rows']

    def test_committed_image_boot_keeps_loaded_skill_snapshot_without_overwriting_global_registry(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'snapshot-skill',
                allowed_tools=['human_output'],
                body='# Snapshot Skill\n\nOriginal checkpoint instructions.\n',
            )
            with _runtime() as runtime:
                runtime.skills.register_skill_from_path(skill_dir, actor='test', require_capability=False)
                source = runtime.process.spawn(image='base-agent:v0', goal='capture skill snapshot')
                runtime.capability.grant(source, 'skill:snapshot-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(source, 'snapshot-skill', actor=source)
                checkpoint_id = runtime.checkpoint.create(source, 'skill snapshot ready', actor=source)
                runtime.image_registry.grant_register(source, 'snapshot-skill-image:v0', issued_by='test')
                runtime.image_registry.commit_from_checkpoint(
                    actor=source,
                    checkpoint_id=checkpoint_id,
                    image_id='snapshot-skill-image:v0',
                    name='snapshot-skill-image',
                )

                write_skill_package(
                    root,
                    'snapshot-skill',
                    allowed_tools=['human_output'],
                    body='# Snapshot Skill\n\nCurrent global instructions.\n',
                )
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='test',
                    replace=True,
                    require_capability=False,
                )
                assert 'Current global instructions.' in runtime.skills.inspect_skill(
                    'snapshot-skill',
                    require_capability=False,
                )['instructions']

                booted = runtime.process.spawn(image='snapshot-skill-image:v0', goal='boot captured skill')

                assert 'Current global instructions.' in runtime.skills.inspect_skill(
                    'snapshot-skill',
                    require_capability=False,
                )['instructions']
                prompt_skill = next(
                    item for item in runtime.skills.prompt_context(booted)
                    if item['skill_id'] == 'snapshot-skill'
                )
                assert 'Original checkpoint instructions.' in prompt_skill['instructions']

    def test_duplicate_commit_requires_replace(self) -> None:
        with _runtime() as runtime:
            pid = runtime.process.spawn(image='base-agent:v0', goal='source')
            checkpoint_id = runtime.checkpoint.create(pid, 'commit', actor=pid)
            runtime.image_registry.grant_register(pid, 'dupe:v0', issued_by='test')
            runtime.image_registry.commit_from_checkpoint(actor=pid, checkpoint_id=checkpoint_id, image_id='dupe:v0', name='dupe')
            with pytest.raises(ValidationError):
                runtime.image_registry.commit_from_checkpoint(actor=pid, checkpoint_id=checkpoint_id, image_id='dupe:v0', name='dupe')

    def test_cli_images_commit_list_and_inspect(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            db_path = str(Path(temp_dir) / 'runtime.sqlite')
            spawned = _run_cli_json(['--db', db_path, 'spawn', '--goal', 'source'])
            created = _run_cli_json(['--db', db_path, 'checkpoint', 'create', spawned['pid'], 'commit'])
            committed = _run_cli_json(['--db', db_path, 'images', 'commit', created['checkpoint_id'], 'cli-committed:v0', '--name', 'cli-committed'])
            listed = _run_cli_json(['--db', db_path, 'images', 'list'])
            inspected = _run_cli_json(['--db', db_path, 'images', 'inspect', 'cli-committed:v0'])
            assert committed['boot']['kind'] == 'checkpoint_commit'
            assert 'cli-committed:v0' in {item['image_id'] for item in listed}
            assert inspected['image']['boot']['kind'] == 'checkpoint_commit'

def _run_cli_json(args: list[str]) -> object:
    result = subprocess.run([sys.executable, '-m', 'agent_libos.api.cli', *args], cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return json.loads(result.stdout)

@contextmanager
def _runtime() -> Iterator[Runtime]:
    runtime = Runtime.open(':memory:')
    try:
        yield runtime
    finally:
        runtime.shutdown(actor='test', reason='test complete')
