from __future__ import annotations
import pytest
import json
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ObjectMetadata, ObjectType, ResourceBudget, ResourceUsage, ToolCandidateStatus
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from tests.support.deno import COUNT_CHARS_SOURCE
from tests.support.skills import write_skill_package

class TestImageCommit:

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
