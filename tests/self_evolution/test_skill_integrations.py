from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.models import CapabilityRight, ResourceBudget, ValidationResult
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.substrate import SubprocessLimits
from agent_libos.tools.sandbox import DenoTypescriptSandbox, SandboxBackend
from tests.support.deno import COUNT_CHARS_SOURCE
from tests.support.fakes import RecordingActionClient
from tests.support.skills import write_skill_package


class TestSkillIntegration:

    def test_jit_skill_reactivation_replaces_alias_and_unload_removes_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'replace-jit-skill',
                jit_tools=[_jit_spec('replace_count', 'scripts/replace_count.ts')],
                scripts={'scripts/replace_count.ts': _jit_source('v1')},
            )
            runtime = Runtime.open('local')
            runtime.tools.sandbox = PassingSandbox()
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='replace jit skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:replace-jit-skill', [CapabilityRight.EXECUTE], issued_by='test')
                first = runtime.skills.activate_skill(pid, 'replace-jit-skill', actor=pid)
                old_tool_id = first['jit_tool_ids']['replace_count']

                write_skill_package(
                    root,
                    'replace-jit-skill',
                    jit_tools=[_jit_spec('replace_count', 'scripts/replace_count.ts')],
                    scripts={'scripts/replace_count.ts': _jit_source('v2')},
                )
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    replace=True,
                    require_capability=False,
                )
                second = runtime.skills.activate_skill(pid, 'replace-jit-skill', actor=pid)
                new_tool_id = second['jit_tool_ids']['replace_count']

                assert new_tool_id != old_tool_id
                assert runtime.process.get(pid).tool_table['replace_count'] == new_tool_id
                assert old_tool_id not in {str(row['tool_id']) for row in runtime.store.list_tools()}
                assert runtime.tools.loaded_tool_handle(old_tool_id) is None
                assert not runtime.tools.is_jit_tool_id(old_tool_id)
                assert not runtime.store.select_table_rows(
                    'tool_candidates',
                    'registered_tool_id = ?',
                    [old_tool_id],
                )

                runtime.skills.unload_skill(pid, 'replace-jit-skill', actor=pid)
                assert 'replace_count' not in runtime.process.get(pid).tool_table
                assert new_tool_id not in {str(row['tool_id']) for row in runtime.store.list_tools()}
                assert runtime.tools.loaded_tool_handle(new_tool_id) is None
                assert not runtime.tools.is_jit_tool_id(new_tool_id)
                assert not runtime.store.select_table_rows(
                    'tool_candidates',
                    'registered_tool_id = ?',
                    [new_tool_id],
                )
            finally:
                runtime.close()

    def test_jit_activation_failure_rolls_back_alias_rows_and_one_time_execute(self, monkeypatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'atomic-jit-skill',
                jit_tools=[_jit_spec('atomic_count', 'scripts/atomic_count.ts')],
                scripts={'scripts/atomic_count.ts': _jit_source('atomic')},
            )
            runtime = Runtime.open('local')
            runtime.tools.sandbox = PassingSandbox()
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='atomic jit skill')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                cap = runtime.capability.grant_once(
                    pid,
                    'skill:atomic-jit-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                )
                real_emit = runtime.events.emit

                def fail_loaded_event(event_type, *args, **kwargs):
                    if str(getattr(event_type, 'value', event_type)) == 'skill_loaded':
                        raise RuntimeError('skill loaded event failed')
                    return real_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', fail_loaded_event)
                with pytest.raises(RuntimeError, match='skill loaded event failed'):
                    runtime.skills.activate_skill(pid, 'atomic-jit-skill', actor=pid)

                process = runtime.process.get(pid)
                assert 'atomic-jit-skill' not in process.loaded_skills
                assert 'atomic_count' not in process.tool_table
                assert not [row for row in runtime.store.list_tools() if row['name'] == 'atomic_count']
                candidates = runtime.store.select_table_rows(
                    'tool_candidates',
                    'pid = ?',
                    [pid],
                )
                assert candidates == []
                assert not any(
                    isinstance(obj.payload, dict) and obj.payload.get('candidate_id')
                    for obj in runtime.store.list_objects()
                )
                assert not [
                    handle
                    for handle in runtime.tools.loaded_tool_handles()
                    if handle.name == 'atomic_count'
                ]
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            finally:
                runtime.close()

    def test_tool_registration_failure_does_not_leave_partial_jit_state(self, monkeypatch) -> None:
        runtime = Runtime.open('local')
        runtime.tools.sandbox = PassingSandbox()
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='partial tool registration')
            candidate_id = runtime.tools.propose(
                pid,
                {
                    'name': 'partial_tool',
                    'description': 'Partial tool.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code=_jit_source('partial'),
            )
            assert runtime.tools.validate(candidate_id, pid=pid).ok
            real_patch_process = runtime.tools.jit.processes.patch_process
            calls = 0

            def fail_once(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError('process update failed')
                return real_patch_process(*args, **kwargs)

            monkeypatch.setattr(
                runtime.tools.jit.processes,
                'patch_process',
                fail_once,
            )
            with pytest.raises(RuntimeError, match='process update failed'):
                runtime.tools.register(pid, candidate_id)
            assert calls == 1

            candidate = runtime.store.get_tool_candidate(candidate_id)
            assert candidate is not None
            assert candidate.status.value == 'validated'
            assert candidate.registered_tool_id is None
            assert 'partial_tool' not in runtime.process.get(pid).tool_table
            assert not [row for row in runtime.store.list_tools() if row['name'] == 'partial_tool']
            assert not [
                handle
                for handle in runtime.tools.loaded_tool_handles()
                if handle.name == 'partial_tool'
            ]
        finally:
            runtime.close()

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
    @pytest.mark.deno_resource_monitor
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

    @pytest.mark.parametrize('operation', ['restore', 'fork'])
    def test_checkpoint_operation_keeps_loaded_skill_snapshot_without_replacing_global_registry(
        self,
        operation: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            skill_dir = write_skill_package(
                root,
                'checkpoint-registry-skill',
                allowed_tools=['echo'],
                body='# Checkpoint Registry Skill\n\nOriginal checkpoint instructions.\n',
            )
            runtime = Runtime.open('local')
            try:
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                pid = runtime.process.spawn(image='base-agent:v0', goal=f'{operation} skill registry')
                runtime.capability.grant(
                    pid,
                    'skill:checkpoint-registry-skill',
                    [CapabilityRight.EXECUTE],
                    issued_by='test',
                )
                runtime.skills.activate_skill(pid, 'checkpoint-registry-skill', actor=pid)
                checkpoint_id = runtime.checkpoint.create(pid, 'skill registry snapshot', actor=pid)

                write_skill_package(
                    root,
                    'checkpoint-registry-skill',
                    allowed_tools=['echo'],
                    body='# Checkpoint Registry Skill\n\nCurrent global instructions.\n',
                )
                runtime.skills.register_skill_from_path(
                    skill_dir,
                    actor='cli',
                    replace=True,
                    require_capability=False,
                )

                if operation == 'restore':
                    runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                    target_pid = pid
                else:
                    runtime.capability.grant(
                        pid,
                        f'checkpoint:{checkpoint_id}',
                        [CapabilityRight.EXECUTE],
                        issued_by='test',
                    )
                    target_pid = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)['fork_root_pid']

                assert 'Current global instructions.' in runtime.skills.inspect_skill(
                    'checkpoint-registry-skill',
                    require_capability=False,
                )['instructions']
                prompt_skill = next(
                    item
                    for item in runtime.skills.prompt_context(target_pid)
                    if item['skill_id'] == 'checkpoint-registry-skill'
                )
                assert 'Original checkpoint instructions.' in prompt_skill['instructions']
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
                assert persisted[0].observability['messages']['sha256']
                assert 'skill-instruction-token' in str(persisted[0].messages)
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


class PassingSandbox(SandboxBackend):
    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {'marker': source_code, **args}

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        return ValidationResult(ok=True, metadata={})


def _jit_spec(name: str, source_path: str) -> dict[str, Any]:
    return {
        'name': name,
        'description': f'{name} test tool.',
        'source_path': source_path,
        'input_schema': {'type': 'object'},
        'output_schema': {'type': 'object'},
        'tests': [],
    }


def _jit_source(marker: str) -> str:
    return f'export async function run(args: unknown, libos: unknown) {{ return {{ marker: {marker!r} }}; }}\n'
