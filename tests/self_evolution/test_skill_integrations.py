from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.models import CapabilityRight, ResourceBudget, ValidationResult
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.substrate import SubprocessLimits
from agent_libos.tools.sandbox import DenoTypescriptSandbox
from tests.support.deno import COUNT_CHARS_SOURCE
from tests.support.fakes import RecordingActionClient
from tests.support.skills import write_skill_package


class TestSkillIntegration:

    @pytest.mark.real_deno
    def test_jit_skill_tool_is_process_local_and_uses_deno_validation_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'jit-skill', jit_tools=[{'name': 'skill_count', 'description': 'Count text characters.', 'source_path': 'scripts/skill_count.ts', 'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}, 'tests': [{'args': {'text': 'abc'}, 'expected': {'count': 3}}]}], scripts={'scripts/skill_count.ts': COUNT_CHARS_SOURCE})
            runtime = Runtime.open('local')
            try:
                owner = runtime.process.spawn(image='base-agent:v0', goal='load jit skill')
                other = runtime.process.spawn(image='base-agent:v0', goal='other')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(owner, 'skill:jit-skill', [CapabilityRight.EXECUTE], issued_by='test')
                loaded = runtime.skills.activate_skill(owner, 'jit-skill', actor=owner)
                result = runtime.tools.call(owner, 'skill_count', {'text': 'hello'})
                assert 'skill_count' in loaded['jit_tool_ids']
                assert result.ok, result.error
                assert result.payload == {'count': 5}
                assert 'skill_count' in runtime.process.get(owner).tool_table
                assert 'skill_count' not in runtime.process.get(other).tool_table
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_jit_skill_validation_uses_broker_resource_limits_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'limited-jit-skill',
                jit_tools=[
                    {
                        'name': 'limited_count',
                        'description': 'Count text characters.',
                        'source_path': 'scripts/limited_count.ts',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                        'tests': [{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
                    }
                ],
                scripts={'scripts/limited_count.ts': COUNT_CHARS_SOURCE},
            )
            runtime = Runtime.open('local')
            sandbox = RecordingLimitDenoSandbox()
            runtime.tools.sandbox = sandbox
            try:
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='load limited jit skill',
                    resource_budget=ResourceBudget(
                        max_subprocess_wall_seconds=5.0,
                        max_subprocess_cpu_seconds=5.0,
                        max_subprocess_memory_bytes=512_000_000,
                    ),
                )
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:limited-jit-skill', [CapabilityRight.EXECUTE], issued_by='test')

                runtime.skills.activate_skill(pid, 'limited-jit-skill', actor=pid)

                assert sandbox.run_tests_calls == 1
                assert sandbox.last_limits is not None
                assert sandbox.last_return_metrics is True
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_skill_jit_activation_rolls_back_visible_jit_on_register_failure(self, monkeypatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'rollback-jit-skill',
                jit_tools=[
                    {
                        'name': 'first_count',
                        'description': 'First counter.',
                        'source_path': 'scripts/first_count.ts',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                        'tests': [{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
                    },
                    {
                        'name': 'second_count',
                        'description': 'Second counter.',
                        'source_path': 'scripts/second_count.ts',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                        'tests': [{'args': {'text': 'abcd'}, 'expected': {'count': 4}}],
                    },
                ],
                scripts={
                    'scripts/first_count.ts': COUNT_CHARS_SOURCE,
                    'scripts/second_count.ts': COUNT_CHARS_SOURCE,
                },
            )
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='load rollback skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:rollback-jit-skill', [CapabilityRight.EXECUTE], issued_by='test')
                real_register = runtime.tools.register
                calls: list[str] = []

                def fail_second_register(*args, **kwargs):
                    calls.append(str(args[1]))
                    if len(calls) == 2:
                        raise ValidationError('simulated second JIT registration failure')
                    return real_register(*args, **kwargs)

                monkeypatch.setattr(runtime.tools, 'register', fail_second_register)

                with pytest.raises(ValidationError, match='simulated second'):
                    runtime.skills.activate_skill(pid, 'rollback-jit-skill', actor=pid)

                process = runtime.process.get(pid)
                assert 'rollback-jit-skill' not in process.loaded_skills
                assert 'first_count' not in process.tool_table
                assert 'second_count' not in process.tool_table
            finally:
                runtime.close()

    def test_image_default_skills_spawn_fork_spawn_child_and_exec_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_skill = write_skill_package(Path(temp_dir), 'image-skill', allowed_tools=['echo'])
            extra_skill = write_skill_package(Path(temp_dir), 'parent-extra', allowed_tools=['read_text_file'])
            runtime = Runtime.open('local')
            try:
                runtime.skills.register_skill_from_path(image_skill, actor='cli', require_capability=False)
                runtime.skills.register_skill_from_path(extra_skill, actor='cli', require_capability=False)
                runtime.register_image(AgentImage(image_id='skill-image:v0', name='skill-image', default_tools=['human_output'], default_skills=['image-skill']), actor='cli')
                root = runtime.process.spawn(image='skill-image:v0', goal='root')
                runtime.capability.grant(root, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
                runtime.capability.grant(root, 'image:base-agent:v0', [CapabilityRight.READ], issued_by='test')
                runtime.capability.grant(root, 'skill:parent-extra', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(root, 'parent-extra', actor=root)
                forked = runtime.process.fork(root, 'forked')
                spawned = runtime.spawn_child_process(root, 'spawned', image='base-agent:v0')
                runtime.capability.grant(spawned, 'image:skill-image:v0', [CapabilityRight.READ], issued_by='test')
                runtime.exec_process(spawned, 'skill-image:v0', goal='exec')
                assert 'echo' in runtime.process.get(root).tool_table
                assert 'read_text_file' in runtime.process.get(forked).tool_table
                assert 'read_text_file' not in runtime.process.get(spawned).tool_table
                assert 'echo' in runtime.process.get(spawned).tool_table
            finally:
                runtime.close()

    def test_image_default_skill_spawn_fails_closed_when_skill_missing(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_image(
                AgentImage(
                    image_id='missing-skill-image:v0',
                    name='missing-skill-image',
                    default_skills=['missing-image-skill'],
                ),
                actor='cli',
            )

            with pytest.raises(NotFound):
                runtime.process.spawn(image='missing-skill-image:v0', goal='missing default skill')

            failed = [
                record for record in runtime.audit.trace()
                if record.action == 'image.boot.failed'
                and record.decision.get('phase') == 'image.default_skills'
            ]
            assert failed
        finally:
            runtime.close()

    def test_checkpoint_restore_preserves_loaded_skill_records_and_tool_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'checkpoint-skill', allowed_tools=['read_text_file'])
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='checkpoint skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:checkpoint-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'checkpoint-skill', actor=pid)
                checkpoint_id = runtime.checkpoint.create(pid, 'skill loaded', actor=pid)
                runtime.skills.unload_skill(pid, 'checkpoint-skill', actor=pid)
                runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert 'checkpoint-skill' in runtime.process.get(pid).loaded_skills
                assert 'read_text_file' in runtime.process.get(pid).tool_table
            finally:
                runtime.close()

    def test_loaded_skill_instructions_are_materialized_into_llm_prompt_and_persisted_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'prompt-skill', allowed_tools=['echo'], body='Always preserve the phrase skill-instruction-token in planning context.\n', actions=[{'name': 'prompt_action', 'use_cases': ['prompt testing']}])
            runtime = Runtime.open('local')
            try:
                runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'done': True}}])
                pid = runtime.process.spawn(image='base-agent:v0', goal='use skill prompt')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:prompt-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'prompt-skill', actor=pid)
                runtime.run_next_process_once()
                assert 'skill-instruction-token' in runtime.llm.client.user_prompts[0]
                persisted = runtime.store.list_llm_calls(pid)
                assert len(persisted) == 1
                assert persisted[0].messages['sha256']
                assert 'skill-instruction-token' not in str(persisted[0].messages)
            finally:
                runtime.close()


class RecordingLimitDenoSandbox(DenoTypescriptSandbox):
    def __init__(self) -> None:
        super().__init__()
        self.run_tests_calls = 0
        self.last_limits: SubprocessLimits | None = None
        self.last_return_metrics = False

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        self.run_tests_calls += 1
        self.last_limits = limits
        self.last_return_metrics = return_metrics
        return super().run_tests(source_code, tests, timeout, limits=limits, return_metrics=return_metrics)
