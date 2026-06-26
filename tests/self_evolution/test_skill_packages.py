from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.config import AgentLibOSConfig, SkillDefaults
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import LocalResourceProviderSubstrate
from tests.support.skills import write_raw_skill, write_skill_package


class TestSkillPackageLoading:

    def test_standard_package_validation_and_global_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_dir = root / 'global-skills'
            skill_dir = write_skill_package(global_dir, 'global-skill', allowed_tools=['echo'])
            config = AgentLibOSConfig(skills=replace(SkillDefaults(), global_dirs=(str(global_dir),)))
            runtime = Runtime.open('local', config=config)
            try:
                with pytest.raises(CapabilityDenied):
                    runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                trust = runtime.skills.global_package_info(skill_dir)
                runtime.skills.trust_skill_source(actor='cli', source_type='global', source=trust['source'], package_sha256=trust['package_sha256'], require_capability=False)
                registered = runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                assert registered['skill_id'] == 'global-skill'
                assert registered['source_type'] == 'global'
                assert 'package_sha256' in registered
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad', 'name: bad\ndescription: Bad\nunknown: nope\n'))
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'BadName', 'name: BadName\ndescription: Bad\n'))
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(write_raw_skill(root, 'bad-metadata', 'name: bad-metadata\ndescription: Bad\nmetadata: {agent-libos.version: 1}\n'))
                old_yaml = root / 'legacy.yaml'
                old_yaml.write_text('schema_version: 1\nskill_id: legacy:v0\nname: Legacy\n', encoding='utf-8')
                with pytest.raises(ValidationError):
                    runtime.skills.validate_package_path(old_yaml)
                with pytest.raises(ValidationError):
                    runtime.skills.register_skill_package({'schema_version': 1, 'skill_id': 'legacy', 'name': 'legacy', 'description': 'Legacy shape.', 'tools': ['echo']}, actor='cli', require_capability=False)
                with pytest.raises(ValidationError):
                    write_skill_package(root, 'bad-jit', jit_tools=[{'name': 'bad', 'description': 'bad', 'source_path': '../escaped.ts'}])
                with pytest.raises(ValidationError):
                    write_skill_package(root, 'bad-right', required_capabilities=[{'resource': 'filesystem:workspace:*', 'rights': ['*']}])
            finally:
                runtime.close()

    def test_workspace_register_and_activate_reads_via_filesystem_and_uses_human_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_skill_package(Path(temp_dir), 'workspace-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'Workspace resource guide.'})
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load workspace skill')
                runtime.filesystem.grant_path(pid, 'workspace-skill/SKILL.md', [CapabilityRight.READ], issued_by='test')
                runtime.filesystem.grant_directory(pid, 'workspace-skill/references', [CapabilityRight.READ], issued_by='test')
                with pytest.raises(HumanApprovalRequired) as raised:
                    runtime.skills.activate_skill_from_workspace_path(pid, 'workspace-skill')
                runtime.human.approve(raised.value.request_id)
                loaded = runtime.skills.activate_skill_from_workspace_path(pid, 'workspace-skill')
                assert loaded['skill_id'] == 'workspace-skill'
                assert 'echo' in runtime.process.get(pid).tool_table
                assert not runtime.capability.check(pid, 'skill:workspace-skill', CapabilityRight.EXECUTE)
                resource = runtime.skills.read_skill_resource(pid, 'workspace-skill', 'references/guide.md')
                assert resource['content'] == 'Workspace resource guide.'
            finally:
                runtime.close()

    def test_host_skill_package_rejects_hardlinked_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside:
            root = Path(temp_dir)
            skill_dir = write_skill_package(root, 'hardlink-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'resource'})
            outside_file = Path(outside) / 'external-secret.txt'
            outside_file.write_text('external secret\n', encoding='utf-8')
            resource = skill_dir / 'references' / 'guide.md'
            resource.unlink()
            try:
                os.link(outside_file, resource)
            except OSError:
                pytest.skip('hardlink creation is not available in this environment')
            runtime = Runtime.open('local')
            try:
                with pytest.raises(ValidationError, match='hard links'):
                    runtime.skills.validate_package_path(skill_dir)
            finally:
                runtime.close()

    def test_skill_syscalls_use_primitive_capabilities_not_tool_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            write_skill_package(Path(temp_dir), 'syscall-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='syscall skill')
                process = runtime.process.get(pid)
                process.tool_table.pop('activate_skill', None)
                runtime.store.update_process(process)
                runtime.filesystem.grant_path(pid, 'syscall-skill/SKILL.md', [CapabilityRight.READ], issued_by='test')
                runtime.capability.grant(pid, 'skill:syscall-skill', [CapabilityRight.WRITE, CapabilityRight.EXECUTE], issued_by='test')
                registered = self._run(LibOSSyscallSession(runtime, pid).handle('skill.register_path', {'path': 'syscall-skill'}))
                loaded = self._run(LibOSSyscallSession(runtime, pid).handle('skill.activate', {'skill_id': 'syscall-skill'}))
                assert registered['skill_id'] == 'syscall-skill'
                assert loaded['skill_id'] == 'syscall-skill'
                assert 'echo' in runtime.process.get(pid).tool_table
                with pytest.raises(NotFound):
                    self._run(LibOSSyscallSession(runtime, pid).handle('skill.register', {'skill': {'schema_version': 1, 'skill_id': 'inline-skill', 'name': 'inline-skill', 'description': 'Inline package should not be syscall-visible.', 'instructions': 'inline'}}))
            finally:
                runtime.close()

    def test_loaded_existing_tool_visibility_does_not_grant_resource_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'read-skill', allowed_tools=['read_text_file'])
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load read tool')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:read-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'read-skill', actor=pid)
                result = runtime.tools.call(pid, 'read_text_file', {'path': 'secret.txt'})
                assert 'read_text_file' in runtime.process.get(pid).tool_table
                assert not runtime.capability.check(pid, 'filesystem:workspace:secret.txt', CapabilityRight.READ)
                assert not result.ok
                assert 'lacks read' in (result.error or '')
            finally:
                runtime.close()

    def test_read_skill_resource_requires_loaded_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'resource-skill', allowed_tools=['echo'], extra_resources={'references/guide.md': 'Remember resource-token.\n'})
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='read resource')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                with pytest.raises(CapabilityDenied):
                    runtime.skills.read_skill_resource(pid, 'resource-skill', 'references/guide.md')
                runtime.capability.grant(pid, 'skill:resource-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'resource-skill', actor=pid)
                resource = runtime.skills.read_skill_resource(pid, 'resource-skill', 'references/guide.md')
                assert 'resource-token' in resource['content']
            finally:
                runtime.close()

    def test_loaded_skill_uses_activation_snapshot_after_registry_replace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'snapshot-skill',
                allowed_tools=['echo'],
                extra_resources={'references/guide.md': 'original-resource-token\n'},
                body='# snapshot-skill\n\nUse original-instruction-token.\n',
            )
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='snapshot skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:snapshot-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'snapshot-skill', actor=pid)

                write_skill_package(
                    root,
                    'snapshot-skill',
                    allowed_tools=['human_output'],
                    extra_resources={'references/guide.md': 'replaced-resource-token\n'},
                    body='# snapshot-skill\n\nUse replaced-instruction-token.\n',
                )
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', replace=True, require_capability=False)

                context = runtime.skills.prompt_context(pid)[0]
                resource = runtime.skills.read_skill_resource(pid, 'snapshot-skill', 'references/guide.md')

                assert 'original-instruction-token' in context['instructions']
                assert 'replaced-instruction-token' not in context['instructions']
                assert context['allowed_tools'] == ['echo']
                assert resource['content'].replace('\r\n', '\n') == 'original-resource-token\n'
            finally:
                runtime.close()

    def test_checkpoint_restore_and_fork_do_not_resurrect_global_skill_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            global_dir = root / 'global'
            skill_dir = write_skill_package(global_dir, 'trust-checkpoint-skill', allowed_tools=['echo'])
            config = AgentLibOSConfig(skills=replace(SkillDefaults(), global_dirs=(str(global_dir),)))
            runtime = Runtime.open('local', config=config)
            try:
                trust = runtime.skills.global_package_info(skill_dir)
                runtime.skills.trust_skill_source(
                    actor='cli',
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                    require_capability=False,
                )
                runtime.skills.register_global_skill_from_path(skill_dir, actor='cli', require_capability=False)
                pid = runtime.process.spawn(image='base-agent:v0', goal='checkpoint trust')
                runtime.capability.grant(pid, 'skill:trust-checkpoint-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'trust-checkpoint-skill', actor=pid)
                checkpoint_id = runtime.checkpoint.create(pid, 'trusted skill loaded', actor=pid)

                runtime.skills.untrust_skill_source(
                    actor='cli',
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                    require_capability=False,
                )
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )

                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )
                runtime.checkpoint.fork_from_checkpoint('cli', checkpoint_id, require_capability=False)
                assert not runtime.store.is_skill_trusted(
                    source_type='global',
                    source=trust['source'],
                    package_sha256=trust['package_sha256'],
                )
            finally:
                runtime.close()

    def test_cross_process_skill_activate_requires_target_process_admin(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'cross-load-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                actor = runtime.process.spawn(image='base-agent:v0', goal='actor')
                target = runtime.process.spawn(image='base-agent:v0', goal='target')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(actor, 'skill:cross-load-skill', [CapabilityRight.EXECUTE], issued_by='test')
                with pytest.raises(CapabilityDenied):
                    runtime.skills.activate_skill(target, 'cross-load-skill', actor=actor)
                runtime.capability.grant(actor, f'process:{target}', [CapabilityRight.ADMIN], issued_by='test')
                loaded = runtime.skills.activate_skill(target, 'cross-load-skill', actor=actor)
                assert loaded['pid'] == target
                assert 'echo' in runtime.process.get(target).tool_table
            finally:
                runtime.close()

    def test_unload_skill_consumes_one_time_execute_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'unload-skill', allowed_tools=['echo'])
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='unload skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.activate_skill(pid, 'unload-skill')
                runtime.capability.grant_once(pid, 'skill:unload-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.unload_skill(pid, 'unload-skill', actor=pid)
                assert not runtime.capability.check(pid, 'skill:unload-skill', CapabilityRight.EXECUTE)
                assert 'echo' not in runtime.process.get(pid).tool_table
            finally:
                runtime.close()

    def _run(self, awaitable: Any) -> Any:
        return asyncio.run(awaitable)
