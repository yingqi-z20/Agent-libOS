from __future__ import annotations
import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import CommandResult, LocalFilesystemProvider, LocalResourceProviderSubstrate
from tests.support.public_errors import assert_public_error_message


class CountingFilesystemProvider(LocalFilesystemProvider):

    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.state_calls: list[str] = []

    def state(self, path):
        self.state_calls.append(path.relative)
        return super().state(path)

class RecordingShellProvider:

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], str | None]] = []

    def run(
        self,
        argv: list[str],
        *,
        timeout: float = 30.0,
        cwd: str | None = None,
        limits: object | None = None,
        stdout_limit_chars: int | None = None,
        stderr_limit_chars: int | None = None,
    ) -> CommandResult:
        self.calls.append((list(argv), cwd))
        return CommandResult(argv=list(argv), returncode=0, stdout='ok', stderr='')

    def classify_external_effect(self, operation: str, context: dict, result: object) -> ExternalEffectClassification:
        return ExternalEffectClassification(rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE, rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED, state_mutation=True, information_flow=True, metadata={'operation': operation})

class TestProcessWorkingDirectory:

    def test_working_directory_tools_reject_unknown_fields_before_state_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'known').mkdir()
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='reject unknown cwd fields')
                runtime.filesystem.grant_directory(
                    pid,
                    'known',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                read = runtime.tools.call(
                    pid,
                    'get_working_directory',
                    {'unexpected': True},
                )
                changed = runtime.tools.call(
                    pid,
                    'set_working_directory',
                    {'path': 'known', 'unexpected': True},
                )

                assert not read.ok
                assert read.error == 'Invalid arguments for tool `get_working_directory`.'
                assert not changed.ok
                assert changed.error == 'Invalid arguments for tool `set_working_directory`.'
                assert provider.state_calls == []
                assert runtime.process.working_directory(pid) == '.'
            finally:
                runtime.close()

    @pytest.mark.skipif(os.name != 'posix', reason='POSIX path identities required')
    @pytest.mark.parametrize(
        ('identity', 'legacy_retarget'),
        [
            (r'decoy\..\secret', 'secret'),
            (' directory with boundary spaces ', 'directory with boundary spaces'),
        ],
    )
    def test_set_working_directory_preserves_provider_canonical_identity(
        self,
        identity: str,
        legacy_retarget: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / identity).mkdir()
            (root / legacy_retarget).mkdir()
            shell = RecordingShellProvider()
            substrate = LocalResourceProviderSubstrate(root)
            substrate.shell = shell
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='preserve canonical cwd')
                expected_resource = runtime.filesystem.directory_resource_for_path(identity)
                capability = runtime.filesystem.grant_directory(
                    pid,
                    identity,
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                runtime.shell.grant_policy(pid, 'always_allow', issued_by='test')

                changed = runtime.tools.call(pid, 'set_working_directory', {'path': identity})
                shell_result = runtime.tools.call(
                    pid,
                    'run_shell_command',
                    {'argv': ['echo', 'cwd']},
                )

                assert changed.ok, changed.error
                assert changed.payload['working_directory'] == identity
                assert runtime.process.working_directory(pid) == identity
                assert capability.resource == expected_resource
                state_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=pid)
                    if effect.provider == 'filesystem' and effect.operation == 'state'
                ]
                assert len(state_effects) == 1
                assert state_effects[0].target == expected_resource
                assert shell_result.ok, shell_result.error
                assert shell.calls == [(['echo', 'cwd'], identity)]
            finally:
                runtime.close()

    @pytest.mark.skipif(os.name != 'posix', reason='POSIX path identities required')
    @pytest.mark.parametrize('tool_name', ['spawn_child_process', 'fork_child_process'])
    @pytest.mark.parametrize(
        ('identity', 'legacy_retarget'),
        [
            (r'decoy\..\secret', 'secret'),
            (' child cwd ', 'child cwd'),
        ],
    )
    def test_explicit_child_working_directory_preserves_provider_canonical_identity(
        self,
        tool_name: str,
        identity: str,
        legacy_retarget: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / identity).mkdir()
            (root / legacy_retarget).mkdir()
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                parent = runtime.process.spawn(image='review-agent:v0', goal='canonical child cwd')
                runtime.capability.grant(
                    parent,
                    'process:spawn',
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                expected_resource = runtime.filesystem.directory_resource_for_path(identity)
                capability = runtime.filesystem.grant_directory(
                    parent,
                    identity,
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                launched = runtime.tools.call(
                    parent,
                    tool_name,
                    {'goal': 'preserve cwd identity', 'working_directory': identity},
                )

                assert launched.ok, launched.error
                child = runtime.process.get(launched.payload['child_pid'])
                assert child.working_directory == identity
                assert capability.resource == expected_resource
                state_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=parent)
                    if effect.provider == 'filesystem' and effect.operation == 'state'
                ]
                assert len(state_effects) == 1
                assert state_effects[0].target == expected_resource
            finally:
                runtime.close()

    def test_set_working_directory_rejects_contained_absolute_path_before_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'known').mkdir()
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='reject absolute cwd')
                runtime.filesystem.grant_directory(
                    pid,
                    'known',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                changed = runtime.tools.call(
                    pid,
                    'set_working_directory',
                    {'path': str(root / 'known')},
                )

                assert not changed.ok
                assert (changed.error or '').startswith('validation_error: ValidationError')
                assert provider.state_calls == []
                assert runtime.store.list_external_effects(pid=pid) == []
                assert runtime.process.working_directory(pid) == '.'
            finally:
                runtime.close()

    @pytest.mark.parametrize('tool_name', ['spawn_child_process', 'fork_child_process'])
    def test_explicit_child_working_directory_rejects_contained_absolute_path(
        self,
        tool_name: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'child-cwd').mkdir()
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                parent = runtime.process.spawn(image='review-agent:v0', goal='reject absolute child cwd')
                runtime.capability.grant(
                    parent,
                    'process:spawn',
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                runtime.filesystem.grant_directory(
                    parent,
                    'child-cwd',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                launched = runtime.tools.call(
                    parent,
                    tool_name,
                    {
                        'goal': 'reject absolute child cwd',
                        'working_directory': str(root / 'child-cwd'),
                    },
                )

                assert not launched.ok
                assert (launched.error or '').startswith('validation_error: ValidationError')
                assert provider.state_calls == []
                assert runtime.process.list_children(parent) == []
            finally:
                runtime.close()

    def test_set_working_directory_requires_filesystem_read_before_state_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'known').mkdir()
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='probe cwd without authority')

                known = runtime.tools.call(pid, 'set_working_directory', {'path': 'known'})
                missing = runtime.tools.call(pid, 'set_working_directory', {'path': 'missing'})

                assert not known.ok
                assert_public_error_message(
                    known.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('lacks read', 'known'),
                )
                assert not missing.ok
                assert_public_error_message(
                    missing.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('lacks read', 'missing'),
                )
                assert provider.state_calls == []
                assert runtime.store.list_external_effects(pid=pid) == []
                assert runtime.process.working_directory(pid) == '.'
            finally:
                runtime.close()

    def test_working_directory_symlinks_do_not_leak_targets_before_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside_dir:
            root = Path(temp_dir)
            inside_target = root / 'private-target'
            inside_target.mkdir()
            outside_target = Path(outside_dir) / 'outside-target'
            outside_target.mkdir()
            try:
                os.symlink(inside_target, root / 'inside-link', target_is_directory=True)
                os.symlink(outside_target, root / 'outside-link', target_is_directory=True)
            except OSError:
                pytest.skip('symlink creation is not available in this environment')
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='probe cwd symlink')

                inside_denied = runtime.tools.call(pid, 'set_working_directory', {'path': 'inside-link'})
                outside_denied = runtime.tools.call(pid, 'set_working_directory', {'path': 'outside-link'})

                assert not inside_denied.ok
                assert_public_error_message(
                    inside_denied.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('inside-link', 'private-target'),
                )
                assert not outside_denied.ok
                assert_public_error_message(
                    outside_denied.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('outside-link', str(outside_target)),
                )
                assert provider.state_calls == []

                if os.name == 'nt':
                    for path, target_name in (
                        ('inside-link', 'private-target'),
                        ('outside-link', 'outside-target'),
                    ):
                        with pytest.raises(
                            CapabilityDenied,
                            match='reparse point',
                        ) as denied:
                            runtime.filesystem.grant_directory(
                                pid,
                                path,
                                [CapabilityRight.READ],
                                issued_by='test',
                            )
                        assert target_name not in str(denied.value)
                    assert provider.state_calls == []
                    assert runtime.store.list_external_effects(pid=pid) == []
                    assert runtime.process.working_directory(pid) == '.'
                    return

                runtime.filesystem.grant_directory(pid, 'inside-link', [CapabilityRight.READ], issued_by='test')
                runtime.filesystem.grant_directory(pid, 'outside-link', [CapabilityRight.READ], issued_by='test')
                inside_authorized = runtime.tools.call(pid, 'set_working_directory', {'path': 'inside-link'})
                outside_authorized = runtime.tools.call(pid, 'set_working_directory', {'path': 'outside-link'})

                assert not inside_authorized.ok
                assert_public_error_message(
                    inside_authorized.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('symlink', 'private-target'),
                )
                assert not outside_authorized.ok
                assert_public_error_message(
                    outside_authorized.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('escapes filesystem adapter root', str(outside_target)),
                )
                assert provider.state_calls == ['inside-link', 'outside-link']
                assert runtime.process.working_directory(pid) == '.'
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 2
                assert all(effect.operation == 'state' and effect.effect_state == 'finalized' for effect in effects)
            finally:
                runtime.close()

    @pytest.mark.parametrize('tool_name', ['spawn_child_process', 'fork_child_process'])
    def test_child_cwd_probe_occurs_only_after_spawn_and_image_authority(self, tool_name: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'child-cwd').mkdir()
            provider = CountingFilesystemProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                parent = runtime.process.spawn(image='review-agent:v0', goal='ordered child cwd validation')
                runtime.filesystem.grant_directory(
                    parent,
                    'child-cwd',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                denied_spawn = runtime.tools.call(
                    parent,
                    tool_name,
                    {'goal': 'denied child', 'image': 'missing-image:v0', 'working_directory': 'child-cwd'},
                )
                assert not denied_spawn.ok
                assert_public_error_message(
                    denied_spawn.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('process:spawn', 'not found'),
                )
                assert provider.state_calls == []

                runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                denied_image = runtime.tools.call(
                    parent,
                    tool_name,
                    {'goal': 'denied image', 'image': 'missing-image:v0', 'working_directory': 'child-cwd'},
                )
                assert not denied_image.ok
                assert_public_error_message(
                    denied_image.error,
                    code='permission_denied',
                    error_type='CapabilityDenied',
                    forbidden=('image:missing-image:v0', 'not found'),
                )
                assert provider.state_calls == []
            finally:
                runtime.close()

    def test_fork_syscall_validates_explicit_working_directory_through_filesystem_primitive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'syscall-cwd').mkdir()
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                parent = runtime.process.spawn(image='base-agent:v0', goal='fork with explicit cwd')
                runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                runtime.filesystem.grant_directory(
                    parent,
                    'syscall-cwd',
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                session = LibOSSyscallSession(runtime, parent)

                result = asyncio.run(
                    session.handle(
                        'process.fork',
                        {'goal': 'child', 'working_directory': 'syscall-cwd'},
                    )
                )

                child = runtime.process.get(result['child_pid'])
                assert child.working_directory == 'syscall-cwd'
                state_effects = [
                    effect
                    for effect in runtime.store.list_external_effects(pid=parent)
                    if effect.provider == 'filesystem' and effect.operation == 'state'
                ]
                assert len(state_effects) == 1
                assert state_effects[0].effect_state == 'finalized'
            finally:
                runtime.close()

    def test_filesystem_tools_resolve_paths_from_process_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg').mkdir()
            (root / 'pkg' / 'module.py').write_text("print('pkg')\n", encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='read from cwd')
                runtime.filesystem.grant_directory(pid, 'pkg', [CapabilityRight.READ, CapabilityRight.WRITE], issued_by='test')
                changed = runtime.tools.call(pid, 'set_working_directory', {'path': 'pkg'})
                read = runtime.tools.call(pid, 'read_text_file', {'path': 'module.py'})
                written = runtime.tools.call(pid, 'write_text_file', {'path': 'created.txt', 'content': 'ok'})
                assert changed.ok, changed.error
                assert changed.payload['working_directory'] == 'pkg'
                assert read.ok, read.error
                assert read.payload['path'] == 'pkg/module.py'
                assert written.ok, written.error
                assert (root / 'pkg' / 'created.txt').exists()
            finally:
                runtime.close()

    def test_filesystem_tools_preserve_cwd_relative_same_name_descendant_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg' / 'pkg').mkdir(parents=True)
            (root / 'pkg' / 'module.py').write_bytes(b"print('outer')\n")
            (root / 'pkg' / 'pkg' / 'module.py').write_bytes(b"print('nested')\n")
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='preserve cwd-relative path')
                runtime.filesystem.grant_directory(
                    pid,
                    'pkg',
                    [CapabilityRight.READ, CapabilityRight.WRITE],
                    issued_by='test',
                )
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'pkg'}).ok

                listed = runtime.tools.call(pid, 'read_directory', {'path': '.'})
                outer = runtime.tools.call(pid, 'read_text_file', {'path': 'module.py'})
                nested = runtime.tools.call(pid, 'read_text_file', {'path': 'pkg/module.py'})
                written = runtime.tools.call(
                    pid,
                    'write_text_file',
                    {'path': 'pkg/created.txt', 'content': 'ok'},
                )

                assert listed.ok, listed.error
                assert listed.payload['path'] == 'pkg'
                assert {entry['path'] for entry in listed.payload['entries']} == {
                    'pkg/module.py',
                    'pkg/pkg',
                }
                assert outer.ok, outer.error
                assert outer.payload['path'] == 'pkg/module.py'
                assert outer.payload['content'] == "print('outer')\n"
                assert nested.ok, nested.error
                assert nested.payload['path'] == 'pkg/pkg/module.py'
                assert nested.payload['content'] == "print('nested')\n"
                assert written.ok, written.error
                assert written.payload['path'] == 'pkg/pkg/created.txt'
                assert (root / 'pkg' / 'pkg' / 'created.txt').read_text(encoding='utf-8') == 'ok'
                assert not (root / 'pkg' / 'created.txt').exists()
            finally:
                runtime.close()

    def test_children_inherit_parent_working_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'child-cwd').mkdir()
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                parent = runtime.process.spawn(image='review-agent:v0', goal='spawn child')
                runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                runtime.filesystem.grant_directory(parent, 'child-cwd', [CapabilityRight.READ], issued_by='test')
                assert runtime.tools.call(parent, 'set_working_directory', {'path': 'child-cwd'}).ok
                spawned = runtime.tools.call(parent, 'spawn_child_process', {'goal': 'inherit cwd'})
                forked = runtime.tools.call(parent, 'fork_child_process', {'goal': 'inherit cwd'})
                assert spawned.ok, spawned.error
                assert forked.ok, forked.error
                assert runtime.process.get(spawned.payload['child_pid']).working_directory == 'child-cwd'
                assert runtime.process.get(forked.payload['child_pid']).working_directory == 'child-cwd'
            finally:
                runtime.close()

    def test_process_working_directory_persists_in_sqlite(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            tempfile.TemporaryDirectory() as store_dir,
        ):
            root = Path(temp_dir)
            (root / 'persisted').mkdir()
            db_path = Path(store_dir) / 'runtime.sqlite'
            runtime = Runtime.open(db_path, substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='persist cwd')
                runtime.filesystem.grant_directory(pid, 'persisted', [CapabilityRight.READ], issued_by='test')
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'persisted'}).ok
            finally:
                runtime.close()
            reopened = Runtime.open(db_path, substrate=LocalResourceProviderSubstrate(root))
            try:
                assert reopened.process.get(pid).working_directory == 'persisted'
            finally:
                reopened.close()

    def test_shell_tool_runs_from_process_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'commands').mkdir()
            shell = RecordingShellProvider()
            substrate = LocalResourceProviderSubstrate(root)
            substrate.shell = shell
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='run from cwd')
                runtime.shell.grant_policy(pid, 'always_allow', issued_by='test')
                runtime.filesystem.grant_directory(pid, 'commands', [CapabilityRight.READ], issued_by='test')
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'commands'}).ok
                result = runtime.tools.call(pid, 'run_shell_command', {'argv': ['echo', 'hello']})
                assert result.ok, result.error
                assert shell.calls == [(['echo', 'hello'], 'commands')]
            finally:
                runtime.close()
