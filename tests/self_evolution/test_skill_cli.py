from __future__ import annotations

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
from agent_libos.api.cli import main as cli_main
from agent_libos.models import CapabilityRight
from agent_libos.models.exceptions import CapabilityDenied
from agent_libos.skills.manager import SkillManager
from tests.support.skills import write_skill_package


class TestSkillCli:

    def test_skill_cli_outputs_stable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = str(root / 'runtime.sqlite')
            skill_dir = write_skill_package(root, 'cli-skill', allowed_tools=['echo'])
            validated = self._cli_json(['--db', db_path, 'skills', 'validate', str(skill_dir)])
            registered = self._cli_json(['--db', db_path, 'skills', 'register', str(skill_dir)])
            discovered = self._cli_json(['--db', db_path, 'skills', 'discover', '--text', 'cli'])
            spawned = self._cli_json(['--db', db_path, 'spawn', '--goal', 'cli skill'])
            loaded = self._cli_json(['--db', db_path, 'skills', 'activate', spawned['pid'], 'cli-skill'])
            assert validated['skill_id'] == 'cli-skill'
            assert registered['skill_id'] == 'cli-skill'
            assert any(item['skill_id'] == 'cli-skill' for item in discovered)
            assert loaded['skill_id'] == 'cli-skill'
            assert 'echo' in loaded['tool_names']

    def test_skill_cli_actor_pid_register_reads_workspace_package_through_primitive(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir).resolve()
            db_path = str(root / 'runtime.sqlite')
            skill_dir = write_skill_package(root, 'cli-actor-skill', allowed_tools=['echo'])
            relative_skill = skill_dir.relative_to(Path.cwd().resolve()).as_posix()
            skill_md = f'{relative_skill}/SKILL.md'
            runtime = Runtime.open(db_path)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='actor cli skill')
                runtime.capability.grant(pid, 'skill:cli-actor-skill', [CapabilityRight.WRITE], issued_by='test')
            finally:
                runtime.close()
            with pytest.raises(CapabilityDenied):
                self._cli_json(['--db', db_path, 'skills', '--actor-pid', pid, 'register', relative_skill])
            runtime = Runtime.open(db_path)
            try:
                runtime.filesystem.grant_path(pid, skill_md, [CapabilityRight.READ], issued_by='test')
            finally:
                runtime.close()
            registered = self._cli_json(['--db', db_path, 'skills', '--actor-pid', pid, 'register', relative_skill])
            assert registered['skill_id'] == 'cli-actor-skill'

    def test_skill_cli_actor_pid_rejects_host_filesystem_validation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db_path = str(root / 'runtime.sqlite')
            skill_dir = write_skill_package(root, 'cli-host-validation', allowed_tools=['echo'])
            runtime = Runtime.open(db_path)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='actor validation')
            finally:
                runtime.close()

            validate_calls = 0

            def fail_if_called(*_args: object, **_kwargs: object) -> object:
                nonlocal validate_calls
                validate_calls += 1
                raise AssertionError('actor rejection must precede Host-path validation')

            monkeypatch.setattr(SkillManager, 'validate_package_path', fail_if_called)

            with pytest.raises(SystemExit, match='Host filesystem operation'):
                self._cli_json(
                    [
                        '--db',
                        db_path,
                        'skills',
                        '--actor-pid',
                        pid,
                        'validate',
                        str(skill_dir),
                    ]
                )
            assert validate_calls == 0

    def _cli_json(self, argv: list[str]) -> Any:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            cli_main(argv)
        return json.loads(stdout.getvalue())
