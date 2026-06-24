from __future__ import annotations
import pytest
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.api.cli import main as cli_main
from agent_libos.models import CapabilityRight, ObjectMetadata, ObjectType, ProcessMessageKind, ProcessStatus
from agent_libos.substrate import LocalResourceProviderSubstrate

class TestCLIBuiltinCommand:

    def test_cli_cd_changes_process_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg').mkdir()
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'review-agent:v0', '--goal', 'set cwd'])
                result = _run_cli_json(['--db', str(db), 'cd', spawn['pid'], 'pkg'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                assert result['pid'] == spawn['pid']
                assert result['working_directory'] == 'pkg'
                assert runtime.process.get(spawn['pid']).working_directory == 'pkg'
            finally:
                runtime.close()

    def test_cli_spawn_and_exec_accept_llm_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json([
                    '--db',
                    str(db),
                    'spawn',
                    '--image',
                    'base-agent:v0',
                    '--goal',
                    'profiled',
                    '--llm-profile',
                    'cli-spawn',
                ])
                result = _run_cli_json([
                    '--db',
                    str(db),
                    'exec',
                    'base-agent:v0',
                    'profiled exec',
                    '--pid',
                    spawn['pid'],
                    '--llm-profile',
                    'cli-exec',
                    '--no-run',
                ])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert spawn['llm_profile_id'] == 'cli-spawn'
                assert result['exec']['old_llm_profile_id'] == 'cli-spawn'
                assert result['exec']['new_llm_profile_id'] == 'cli-exec'
                assert process.llm_profile_id == 'cli-exec'
            finally:
                runtime.close()

    def test_cli_exit_marks_process_exited_with_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'finish'])
                result = _run_cli_json(['--db', str(db), 'exit', spawn['pid'], '--payload', '{"done": true}'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert result['pid'] == spawn['pid']
                assert result['status'] == ProcessStatus.EXITED.value
                assert result['result_oid'] is not None
                assert process.status == ProcessStatus.EXITED
                assert (process.status_message or '').startswith('result_oid:')
            finally:
                runtime.close()

    def test_cli_exec_loads_image_package_from_first_arg_and_uses_second_arg_as_goal(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            package = root / 'cli-image'
            _write_cli_image_package(package)
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'old goal'])
                before = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
                try:
                    old_goal_oid = before.process.get(spawn['pid']).goal_oid
                finally:
                    before.close()
                result = _run_cli_json(['--db', str(db), 'exec', str(package), 'new goal from first arg', '--pid', spawn['pid'], '--no-run'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                process = runtime.process.get(spawn['pid'])
                assert result['goal'] == 'new goal from first arg'
                assert result['image_arg'] == str(package)
                assert result['loaded_image']['image_id'] == 'cli-package-agent:v0'
                assert result['process']['image'] == 'cli-package-agent:v0'
                assert not result['ran']
                assert process.image_id == 'cli-package-agent:v0'
                assert process.goal_oid != old_goal_oid
                assert 'human_output' in process.tool_table
            finally:
                runtime.close()

    def test_cli_message_and_interrupt_post_human_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                spawn = _run_cli_json(['--db', str(db), 'spawn', '--image', 'base-agent:v0', '--goal', 'listen'])
                normal = _run_cli_json(['--db', str(db), 'message', spawn['pid'], 'please inspect the latest result', '--subject', 'status'])
                interrupt = _run_cli_json(['--db', str(db), 'interrupt', spawn['pid'], 'stop and read this first'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                unread = runtime.messages.unread(spawn['pid'])
                assert normal['message']['kind'] == ProcessMessageKind.NORMAL.value
                assert interrupt['message']['kind'] == ProcessMessageKind.INTERRUPT.value
                assert [message.message_id for message in unread] == [normal['message']['message_id'], interrupt['message']['message_id']]
                assert unread[0].sender == 'human:owner'
                assert unread[0].subject == 'status'
                assert unread[1].subject == 'Human interrupt'
            finally:
                runtime.close()

    def test_cli_workflow_run_prints_result_and_persists_exited_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                result = _run_cli_json(['--db', str(db), 'workflow', 'run', 'get_working_directory'])
            runtime = Runtime.open(db, substrate=LocalResourceProviderSubstrate(root))
            try:
                assert result['ok'] is True
                assert result['tool'] == 'get_working_directory'
                assert result['status'] == ProcessStatus.EXITED.value
                assert result['result_oid'] is not None
                assert runtime.process.get(str(result['pid'])).status == ProcessStatus.EXITED
            finally:
                runtime.close()

    def test_cli_workflow_run_failure_exits_nonzero_after_printing_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            with _temporary_cwd(root):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
                    cli_main([
                        '--db',
                        str(db),
                        'workflow',
                        'run',
                        'parse_pytest_log',
                        '--args-json',
                        '{"log": "FAILED tests/x.py::test_y"}',
                    ])
            assert raised.value.code == 1
            result = json.loads(stdout.getvalue())
            assert result['ok'] is False
            assert result['status'] == ProcessStatus.FAILED.value
            assert 'not in process tool table' in result['error']

    def test_cli_object_task_start_outputs_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='object task cli')
        runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        result = _run_cli_json([
            'object-task',
            'start',
            '--pid',
            pid,
            '--owner-oid',
            owner.oid,
            '--watch-owner',
            '--watch-events',
            'updated',
            'get_working_directory',
            '--wait',
            '--timeout',
            '2',
        ])

        assert result['status'] == 'succeeded'
        assert result['owner_oid'] == owner.oid
        assert result['owner_watch']['enabled'] is True
        assert result['owner_watch']['events'] == ['updated']
        assert result['result_oid'] is not None

    def test_cli_object_task_start_requires_wait(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_main([
                'object-task',
                'start',
                '--pid',
                'pid-1',
                '--owner-oid',
                'oid-1',
                'get_working_directory',
            ])

        assert 'requires --wait' in str(raised.value)

    def test_cli_object_task_wait_rejects_non_finite_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), pytest.raises(SystemExit) as raised:
            cli_main(['object-task', 'wait', 'task-1', '--timeout', 'nan'])

        assert '--timeout must be a finite non-negative number' in str(raised.value)

    def test_cli_object_task_watch_owner_updates_existing_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        pid = runtime.process.spawn(image='base-agent:v0', goal='object task watch cli')
        runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        task = runtime.object_tasks.start(pid, owner, 'receive_process_messages', {'channel': 'owner-watch'})
        runtime.object_tasks.wait(task.task_id, actor_pid=pid, timeout=2)
        monkeypatch.setattr('agent_libos.api.cli.Runtime.open', lambda *args, **kwargs: runtime)

        result = _run_cli_json([
            'object-task',
            'watch-owner',
            task.task_id,
            '--pid',
            pid,
            '--watch-events',
            'updated',
            '--watch-channel',
            'owner-watch',
            '--watch-kind',
            'interrupt',
        ])

        assert result['task_id'] == task.task_id
        assert result['owner_watch']['enabled'] is True
        assert result['owner_watch']['events'] == ['updated']
        assert result['owner_watch']['channel'] == 'owner-watch'
        assert result['owner_watch']['kind'] == 'interrupt'

@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)

def _run_cli_json(argv: list[str]) -> dict[str, object]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        cli_main(argv)
    return json.loads(stdout.getvalue())


def _write_cli_image_package(root: Path) -> None:
    root.mkdir(parents=True)
    root.joinpath('IMAGE.yaml').write_text("""
image_id: cli-package-agent:v0
name: cli-package-agent
prompt: prompt.md
default_tools:
  - human_output
context_policy: evidence_first
""".lstrip(), encoding='utf-8')
    root.joinpath('prompt.md').write_text('CLI loaded image.\n', encoding='utf-8')
