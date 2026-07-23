from __future__ import annotations
import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import CommandResult, LocalFilesystemProvider, LocalResourceProviderSubstrate


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

                assert not known.ok and 'lacks read' in (known.error or '')
                assert not missing.ok and 'lacks read' in (missing.error or '')
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

                assert not inside_denied.ok and 'filesystem:workspace:inside-link/*' in (inside_denied.error or '')
                assert not outside_denied.ok and 'filesystem:workspace:outside-link/*' in (outside_denied.error or '')
                assert 'private-target' not in (inside_denied.error or '')
                assert str(outside_target) not in (outside_denied.error or '')
                assert provider.state_calls == []

                runtime.filesystem.grant_directory(pid, 'inside-link', [CapabilityRight.READ], issued_by='test')
                runtime.filesystem.grant_directory(pid, 'outside-link', [CapabilityRight.READ], issued_by='test')
                inside_authorized = runtime.tools.call(pid, 'set_working_directory', {'path': 'inside-link'})
                outside_authorized = runtime.tools.call(pid, 'set_working_directory', {'path': 'outside-link'})

                assert not inside_authorized.ok
                assert 'symlink' in (inside_authorized.error or '').lower()
                assert not outside_authorized.ok
                assert 'escapes filesystem adapter root' in (outside_authorized.error or '')
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
                assert 'process:spawn' in (denied_spawn.error or '')
                assert 'not found' not in (denied_spawn.error or '').lower()
                assert provider.state_calls == []

                runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                denied_image = runtime.tools.call(
                    parent,
                    tool_name,
                    {'goal': 'denied image', 'image': 'missing-image:v0', 'working_directory': 'child-cwd'},
                )
                assert not denied_image.ok
                assert 'image:missing-image:v0' in (denied_image.error or '')
                assert 'not found' not in (denied_image.error or '').lower()
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

    def test_filesystem_tools_accept_redundant_workspace_relative_cwd_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg').mkdir()
            (root / 'pkg' / 'module.py').write_text("print('pkg')\n", encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='accept redundant cwd prefix')
                runtime.filesystem.grant_directory(
                    pid,
                    'pkg',
                    [CapabilityRight.READ, CapabilityRight.WRITE],
                    issued_by='test',
                )
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'pkg'}).ok

                read = runtime.tools.call(pid, 'read_text_file', {'path': 'pkg/module.py'})
                written = runtime.tools.call(
                    pid,
                    'write_text_file',
                    {'path': 'pkg/created.txt', 'content': 'ok'},
                )

                assert read.ok, read.error
                assert read.payload['path'] == 'pkg/module.py'
                assert written.ok, written.error
                assert written.payload['path'] == 'pkg/created.txt'
                assert (root / 'pkg' / 'created.txt').read_text(encoding='utf-8') == 'ok'
                assert not (root / 'pkg' / 'pkg').exists()
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
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'persisted').mkdir()
            db_path = root / 'runtime.sqlite'
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
