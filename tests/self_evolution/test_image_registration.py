from __future__ import annotations
import json
import os
import pytest
import tempfile
from pathlib import Path
from typing import Any
from agent_libos import AgentImage, Runtime
from agent_libos.models import (
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ProcessStatus,
    ResourceBudget,
    ValidationResult,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.substrate import LocalResourceProviderSubstrate, SubprocessLimits
from agent_libos.tools.sandbox import DenoTypescriptSandbox


class RejectingValidationSandbox(DenoTypescriptSandbox):
    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        return ValidationResult(ok=False, errors=['package validation failed'])


class TestImageRegistration:

    def test_register_image_primitive_validates_tools_and_emits_audit(self) -> None:
        runtime = Runtime.open('local')
        try:
            image = AgentImage(image_id='custom-review:v0', name='custom-review', system_prompt='Custom review image.', default_tools=['read_memory_object', 'human_output'], safety_profile='review')
            runtime.register_image(image, actor='cli')
            assert runtime.get_image('custom-review:v0') is image
            assert 'image.register' in [record.action for record in runtime.audit.trace()]
            assert EventType.IMAGE_REGISTERED in [event.type for event in runtime.events.list()]
        finally:
            runtime.close()

    def test_register_image_rejects_unknown_default_tool(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError):
                runtime.register_image({'image_id': 'bad-image:v0', 'name': 'bad-image', 'default_tools': ['not_a_real_tool']}, actor='cli')
        finally:
            runtime.close()

    def test_image_default_tools_are_not_implicitly_augmented(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_image(AgentImage(image_id='empty-tools:v0', name='empty-tools'), actor='cli')
            runtime.register_image(
                AgentImage(image_id='one-tool:v0', name='one-tool', default_tools=['human_output']),
                actor='cli',
            )

            empty = runtime.process.spawn(image='empty-tools:v0', goal='no implicit tools')
            one = runtime.process.spawn(image='one-tool:v0', goal='single explicit tool')

            assert runtime.process.get(empty).tool_table == {}
            assert set(runtime.process.get(one).tool_table) == {'human_output'}
            assert 'process_exit' not in runtime.process.get(one).tool_table
            assert 'create_memory_object' not in runtime.process.get(one).tool_table
        finally:
            runtime.close()

    def test_register_image_rejects_invalid_required_capability_right(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError):
                runtime.register_image({'image_id': 'bad-right-image:v0', 'name': 'bad-right-image', 'required_capabilities': [{'resource': 'filesystem:workspace:*', 'rights': ['*']}]}, actor='cli')
        finally:
            runtime.close()

    def test_register_image_rejects_unknown_jit_tool_exposure(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError, match='unknown jit_tool_exposure'):
                runtime.register_image(
                    {'image_id': 'bad-jit-exposure:v0', 'name': 'bad-jit-exposure', 'jit_tool_exposure': 'ambient'},
                    actor='cli',
                )
        finally:
            runtime.close()

    def test_multiplexed_image_rejects_reserved_default_tool(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError, match=JIT_MULTIPLEXER_TOOL_NAME):
                runtime.register_image(
                    AgentImage(
                        image_id='reserved-jit-protocol:v0',
                        name='reserved-jit-protocol',
                        jit_tool_exposure=JIT_TOOL_EXPOSURE_MULTIPLEXED,
                        default_tools=[JIT_MULTIPLEXER_TOOL_NAME],
                    ),
                    actor='cli',
                )
        finally:
            runtime.close()

    def test_register_image_rejects_oversized_manifest_fields(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError, match='system_prompt exceeds'):
                runtime.register_image(
                    AgentImage(
                        image_id='huge-prompt:v0',
                        name='huge-prompt',
                        system_prompt='x' * (runtime.config.image.prompt_max_chars + 1),
                    ),
                    actor='cli',
                )
            with pytest.raises(ValidationError, match='metadata exceeds'):
                runtime.register_image(
                    AgentImage(
                        image_id='huge-metadata:v0',
                        name='huge-metadata',
                        metadata={'blob': 'x' * runtime.config.image.structured_field_hard_limit_bytes},
                    ),
                    actor='cli',
                )
        finally:
            runtime.close()

    def test_spawn_rejects_unknown_image_instead_of_defaulting_tools(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(NotFound):
                runtime.process.spawn(image='missing-image:v0', goal='should fail')
        finally:
            runtime.close()

    def test_load_image_package_tool_reads_workspace_package_and_registers_image(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            package = _write_image_package(Path(temp_dir) / 'images' / 'package-agent')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='load image')
                runtime.filesystem.grant_directory(pid, 'images/package-agent', [CapabilityRight.READ], issued_by='test')
                runtime.image_registry.grant_register(pid, issued_by='test')
                result = runtime.tools.call(pid, 'load_image_package', {'path': 'images/package-agent'})
                assert result.ok, result.error
                assert result.payload['image_id'] == 'package-agent:v0'
                assert result.payload['boot_kind'] == 'image_package'
                assert result.payload['package_sha256']
                image = runtime.get_image('package-agent:v0')
                assert image.system_prompt.replace('\r\n', '\n') == 'Package registered image.\nKeep responses concise.\n'
                assert image.default_tools == ['human_output', 'read_memory_object']
                assert image.metadata['role'] == 'test'
                assert image.metadata['package_kind'] == 'image_package'
                assert package.exists()
            finally:
                runtime.close()

    def test_load_image_package_tool_requires_image_write_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='load image without authority')
                runtime.filesystem.grant_directory(pid, 'package-agent', [CapabilityRight.READ], issued_by='test')
                result = runtime.tools.call(pid, 'load_image_package', {'path': 'package-agent'})
                assert not result.ok
                assert 'lacks write on image:package-agent:v0' in (result.error or '')
                with pytest.raises(KeyError):
                    runtime.get_image('package-agent:v0')
            finally:
                runtime.close()

    def test_image_package_workspace_is_private_and_manifest_granted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                first = runtime.process.spawn(image='package-agent:v0', goal='first')
                second = runtime.process.spawn(image='package-agent:v0', goal='second')
                first_cwd = runtime.process.working_directory(first)
                second_cwd = runtime.process.working_directory(second)

                assert first_cwd != second_cwd
                assert runtime.filesystem.read_text(first, 'seed.txt', cwd=first_cwd).content.replace('\r\n', '\n') == 'seed\n'
                runtime.filesystem.write_text(first, 'seed.txt', 'changed\n', cwd=first_cwd)
                assert runtime.filesystem.read_text(first, 'seed.txt', cwd=first_cwd).content.replace('\r\n', '\n') == 'changed\n'
                assert runtime.filesystem.read_text(second, 'seed.txt', cwd=second_cwd).content.replace('\r\n', '\n') == 'seed\n'
            finally:
                runtime.close()

    def test_image_package_preserves_llm_profile_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', llm_profile='package-review')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='package-agent:v0', goal='profile default')

                assert result.image.llm_profile_id == 'package-review'
                assert runtime.get_image('package-agent:v0').llm_profile_id == 'package-review'
                assert runtime.process.get(pid).llm_profile_id == 'package-review'
            finally:
                runtime.close()

    def test_image_package_without_workspace_grants_cannot_read_materialized_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', workspace_grants=False)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='package-agent:v0', goal='no grant')
                cwd = runtime.process.working_directory(pid)
                resource = runtime.filesystem.resource_for_path('seed.txt', cwd=cwd)

                assert not runtime.capability.check(pid, resource, CapabilityRight.READ)
                with pytest.raises((CapabilityDenied, HumanApprovalRequired)):
                    runtime.filesystem.read_text(pid, 'seed.txt', cwd=cwd)
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_image_package_jit_tools_are_process_local_and_not_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                assert result.image.metadata['package_jit_tools'] == ['package_count']
                pid = runtime.process.spawn(image='package-agent:v0', goal='jit')
                visible = {row['name'] for row in runtime.tools.visible_tools(pid)}

                assert 'package_count' in visible
                assert 'package_count' in runtime.process.get(pid).tool_table
                assert not (Path(temp_dir) / runtime.process.working_directory(pid) / 'tools').exists()
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_image_package_jit_boot_validation_uses_broker_resource_limits_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            sandbox = RecordingLimitDenoSandbox()
            runtime.tools.sandbox = sandbox
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                runtime.process.spawn(
                    image='package-agent:v0',
                    goal='limited package jit',
                    resource_budget=ResourceBudget(
                        max_subprocess_wall_seconds=5.0,
                        max_subprocess_cpu_seconds=5.0,
                        max_subprocess_memory_bytes=512_000_000,
                    ),
                )

                assert sandbox.run_tests_calls == 1
                assert sandbox.last_limits is not None
                assert sandbox.last_return_metrics is True
            finally:
                runtime.close()

    def test_image_package_boot_failure_cleans_materialized_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = RejectingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                with pytest.raises(ValidationError, match='package validation failed'):
                    runtime.process.spawn(image='package-agent:v0', goal='failed boot')

                materialized = Path(temp_dir) / runtime.config.image.materialized_workspace_root
                seed_files = list(materialized.rglob('seed.txt')) if materialized.exists() else []
                assert seed_files == []
                assert not [
                    row for row in runtime.store.list_tools()
                    if row['name'] == 'package_count' and row['ephemeral']
                ]
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_image_package_multiplexed_jit_exposure_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                jit_tool_exposure=JIT_TOOL_EXPOSURE_MULTIPLEXED,
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(
                    Path(temp_dir) / 'package-agent',
                    actor='cli',
                )
                image = runtime.get_image('package-agent:v0')
                pid = runtime.process.spawn(image='package-agent:v0', goal='multiplexed package')
                schema_names = {schema['function']['name'] for schema in runtime.tools.openai_tool_schemas(pid)}

                assert result.image.jit_tool_exposure == JIT_TOOL_EXPOSURE_MULTIPLEXED
                assert image.jit_tool_exposure == JIT_TOOL_EXPOSURE_MULTIPLEXED
                assert runtime.image_registry.inspect('package-agent:v0')['image']['jit_tool_exposure'] == JIT_TOOL_EXPOSURE_MULTIPLEXED
                assert JIT_MULTIPLEXER_TOOL_NAME in schema_names
                assert 'package_count' not in schema_names
            finally:
                runtime.close()

    def test_multiplexed_image_package_rejects_jit_multiplexer_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                jit_name=JIT_MULTIPLEXER_TOOL_NAME,
                jit_tool_exposure=JIT_TOOL_EXPOSURE_MULTIPLEXED,
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match=JIT_MULTIPLEXER_TOOL_NAME):
                    runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
            finally:
                runtime.close()

    def test_image_package_rejects_provider_invalid_jit_name_and_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root, with_jit=True, jit_name='bad name')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='OpenAI tool name syntax'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
            finally:
                runtime.close()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root, with_jit=True)
            jit_path = root / 'tools' / 'jit-tools.json'
            jit_tools = json.loads(jit_path.read_text(encoding='utf-8'))
            jit_tools[0]['input_schema'] = {'type': 'definitely-not-a-json-schema-type'}
            jit_path.write_text(json.dumps(jit_tools), encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='valid JSON schema'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_image_package_jit_tool_name_does_not_become_global_default_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                owner = runtime.process.spawn(image='package-agent:v0', goal='owner')
                other = runtime.process.spawn(image='base-agent:v0', goal='other')

                assert 'package_count' in runtime.process.get(owner).tool_table
                with pytest.raises(ValidationError):
                    runtime.register_image(
                        AgentImage(
                            image_id='leak-image:v0',
                            name='leak-image',
                            default_tools=['package_count'],
                        ),
                        actor='cli',
                    )
                other_call = runtime.tools.call(other, 'package_count', {'text': 'abcd'})
                assert not other_call.ok
                assert 'not in process tool table' in (other_call.error or '')
            finally:
                runtime.close()

    def test_image_package_prompt_mode_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', prompt_mode='minimal_runtime')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                inspected = runtime.image_registry.inspect('package-agent:v0')

                assert result.image.prompt_mode == 'minimal_runtime'
                assert inspected['image']['prompt_mode'] == 'minimal_runtime'
                listed = {image['image_id']: image for image in runtime.image_registry.list_images()}
                assert listed['package-agent:v0']['prompt_mode'] == 'minimal_runtime'
            finally:
                runtime.close()

    def test_image_package_rejects_jit_name_shadowing_static_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True, jit_name='process_exit')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='conflicts with static tool'):
                    runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                with pytest.raises(KeyError):
                    runtime.get_image('package-agent:v0')
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_exec_process_instantiates_image_package_workspace_and_jit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='base-agent:v0', goal='before exec')
                runtime.capability.grant(pid, runtime.image_registry.resource_for('package-agent:v0'), [CapabilityRight.READ], issued_by='test')
                runtime.exec_process(pid, 'package-agent:v0', goal='after exec', preserve_capabilities=False)
                process = runtime.process.get(pid)

                assert process.status == ProcessStatus.RUNNABLE
                assert process.image_id == 'package-agent:v0'
                assert process.working_directory != '.'
                assert 'package_count' in process.tool_table
                assert runtime.filesystem.read_text(pid, 'seed.txt', cwd=process.working_directory).content.replace('\r\n', '\n') == 'seed\n'
            finally:
                runtime.close()

    def test_exec_process_image_package_failure_restores_state_and_cleans_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = RejectingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='base-agent:v0', goal='before exec')
                runtime.capability.grant(pid, runtime.image_registry.resource_for('package-agent:v0'), [CapabilityRight.READ], issued_by='test')
                before = runtime.process.get(pid)

                with pytest.raises(ValidationError, match='package validation failed'):
                    runtime.exec_process(pid, 'package-agent:v0', goal='after failed exec', preserve_capabilities=False)

                after = runtime.process.get(pid)
                materialized = Path(temp_dir) / runtime.config.image.materialized_workspace_root
                seed_files = list(materialized.rglob('seed.txt')) if materialized.exists() else []
                assert after.image_id == before.image_id
                assert after.working_directory == before.working_directory
                assert 'package_count' not in after.tool_table
                assert seed_files == []
            finally:
                runtime.close()

    def test_image_package_workspace_grants_are_relative_to_source_root_not_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root, workspace_grants=False)
            (root / 'workspace' / 'app').mkdir()
            (root / 'workspace' / 'data').mkdir()
            (root / 'workspace' / 'data' / 'x.txt').write_text('x\n', encoding='utf-8')
            root.joinpath('IMAGE.yaml').write_text("""
image_id: package-agent:v0
name: package-agent
prompt: prompt.md
workspace:
  source: workspace
  working_directory: app
  grants:
    - path: data
      rights: [read]
      recursive: true
""".lstrip(), encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                runtime.image_registry.register_from_package_path(root, actor='cli')
                pid = runtime.process.spawn(image='package-agent:v0', goal='cwd grant')
                cwd = runtime.process.working_directory(pid)

                assert cwd.endswith('/workspace/app')
                assert runtime.filesystem.read_text(pid, '../data/x.txt', cwd=cwd).content.replace('\r\n', '\n') == 'x\n'
                with pytest.raises((CapabilityDenied, HumanApprovalRequired, NotFound)):
                    runtime.filesystem.read_text(pid, 'data/x.txt', cwd=cwd)
            finally:
                runtime.close()

    def test_image_package_rejects_workspace_source_that_points_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root, workspace_grants=False)
            root.joinpath('IMAGE.yaml').write_text("""
image_id: package-agent:v0
name: package-agent
prompt: prompt.md
workspace:
  source: workspace/seed.txt
  grants:
    - path: .
      rights: [read]
      recursive: true
""".lstrip(), encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='workspace.source must point to a directory'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
                with pytest.raises(KeyError):
                    runtime.get_image('package-agent:v0')
            finally:
                runtime.close()

    def test_image_package_rejects_secret_or_cache_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root)
            root.joinpath('.env').write_text('TOKEN=secret\n', encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='secret material'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
            finally:
                runtime.close()

    def test_image_package_rejects_host_hardlinked_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as outside:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root)
            outside_file = Path(outside) / 'external-secret.txt'
            outside_file.write_text('external secret\n', encoding='utf-8')
            package_file = root / 'workspace' / 'seed.txt'
            package_file.unlink()
            try:
                os.link(outside_file, package_file)
            except OSError:
                pytest.skip('hardlink creation is not available in this environment')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='hard links'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
            finally:
                runtime.close()

    def test_image_package_rejects_undeclared_root_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'package-agent'
            _write_image_package(root)
            root.joinpath('notes.txt').write_text('not part of the image contract\n', encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                with pytest.raises(ValidationError, match='undeclared files'):
                    runtime.image_registry.register_from_package_path(root, actor='cli')
            finally:
                runtime.close()

    def test_image_package_rejects_windows_unsafe_paths_from_file_payloads(self) -> None:
        runtime = Runtime.open('local')
        try:
            files = {
                'IMAGE.yaml': """
image_id: unsafe-package:v0
name: unsafe-package
prompt: prompt.md
workspace:
  source: workspace
""".lstrip(),
                'prompt.md': 'Prompt\n',
                'workspace/a:stream.txt': 'unsafe\n',
            }
            with pytest.raises(ValidationError, match='Windows-unsafe'):
                runtime.image_registry.register_from_package_files(files, actor='cli')
        finally:
            runtime.close()


def _write_image_package(
    root: Path,
    *,
    workspace_grants: bool = True,
    with_jit: bool = False,
    jit_name: str = 'package_count',
    prompt_mode: str | None = None,
    jit_tool_exposure: str | None = None,
    llm_profile: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    grants = """
  grants:
    - path: .
      rights: [read, write]
      recursive: true
""".rstrip() if workspace_grants else "  grants: []"
    jit_line = "\njit_tools: tools/jit-tools.json" if with_jit else ""
    prompt_mode_line = f"prompt_mode: {prompt_mode}\n" if prompt_mode else ""
    jit_tool_exposure_line = f"jit_tool_exposure: {jit_tool_exposure}\n" if jit_tool_exposure else ""
    llm_profile_line = f"llm_profile: {llm_profile}\n" if llm_profile else ""
    root.joinpath('IMAGE.yaml').write_text(f"""
image_id: package-agent:v0
name: package-agent
version: v0
prompt: prompt.md
{prompt_mode_line}{jit_tool_exposure_line}{llm_profile_line}default_tools:
  - human_output
  - read_memory_object
context_policy: evidence_first
safety_profile: package-test
metadata:
  role: test{jit_line}
workspace:
  source: workspace
  working_directory: .
{grants}
""".lstrip(), encoding='utf-8')
    root.joinpath('prompt.md').write_text('Package registered image.\nKeep responses concise.\n', encoding='utf-8')
    workspace = root / 'workspace'
    workspace.mkdir()
    workspace.joinpath('seed.txt').write_text('seed\n', encoding='utf-8')
    if with_jit:
        scripts = root / 'tools' / 'scripts'
        scripts.mkdir(parents=True)
        root.joinpath('tools', 'jit-tools.json').write_text(
            f'[{{"name":"{jit_name}","description":"Count text characters.","source_path":"tools/scripts/package_count.ts","input_schema":{{"type":"object"}},"output_schema":{{"type":"object"}},"tests":[]}}]',
            encoding='utf-8',
        )
        scripts.joinpath('package_count.ts').write_text(
            'export function run(args, libos) { return { count: String(args.text || "").length }; }\n',
            encoding='utf-8',
        )
    return root


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
