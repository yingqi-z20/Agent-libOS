from __future__ import annotations
import pytest
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.substrate import CommandResult, LocalResourceProviderSubstrate

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

    def test_children_inherit_parent_working_directory_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'child-cwd').mkdir()
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                parent = runtime.process.spawn(image='review-agent:v0', goal='spawn child')
                runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
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
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'commands'}).ok
                result = runtime.tools.call(pid, 'run_shell_command', {'argv': ['echo', 'hello']})
                assert result.ok, result.error
                assert shell.calls == [(['echo', 'hello'], 'commands')]
            finally:
                runtime.close()
