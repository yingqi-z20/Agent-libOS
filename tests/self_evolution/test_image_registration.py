from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectType,
    PROMPT_MODE_LIBOS_DEFAULT,
    ProcessStatus,
    ResourceBudget,
    ToolCandidateStatus,
    ValidationResult,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ProcessError, RuntimeRecoveryRequired, ValidationError
from agent_libos.process_execution import bind_process_execution
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


def _abort_exec_before_compensation(
    *,
    error: BaseException,
    **_kwargs: object,
) -> None:
    """Model host process loss after durable exec effects are published."""

    raise error


def _group_contains(error: BaseException, kind: type[BaseException]) -> bool:
    if isinstance(error, BaseExceptionGroup):
        return any(_group_contains(item, kind) for item in error.exceptions)
    return isinstance(error, kind)


def _release_fenced_runtime_or_close(runtime: Runtime) -> None:
    reason = runtime.lifecycle.shutdown_reason
    if (
        runtime.lifecycle.state == "close_failed"
        and isinstance(reason, str)
        and reason.startswith("runtime.recovery_required:")
    ):
        result = runtime.release_recovery_diagnostics()
        assert result["ok"] is True, result
        assert result["recovery_diagnostics_released"] is True
        assert runtime.lifecycle.closed
        return
    runtime.close()


class TestImageRegistration:

    @pytest.mark.parametrize(
        'metadata, message',
        [
            ({'tool_projection': False}, "tool_projection must be 'skills'"),
            ({'tool_projection': 'groups'}, "tool_projection must be 'skills'"),
            ({'tool_projection': ['skills']}, "tool_projection must be 'skills'"),
        ],
    )
    def test_invalid_projection_metadata_is_rejected_before_registration(
        self,
        metadata: dict[str, object],
        message: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            image_id = 'invalid-projection:v0'
            with pytest.raises(ValidationError, match=message):
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name='invalid-projection',
                        default_tools=['process_exit'],
                        metadata=metadata,
                    ),
                    actor='test',
                )

            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None
        finally:
            runtime.close()

    @pytest.mark.parametrize("legacy_key", ["lazy_tool_groups", "initial_tool_groups"])
    def test_removed_tool_group_metadata_is_rejected_before_registration(
        self,
        legacy_key: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            image_id = f'removed-{legacy_key}:v0'
            with pytest.raises(
                ValidationError,
                match=rf'removed tool-group fields: {legacy_key}',
            ):
                runtime.register_image(
                    AgentImage(
                        image_id=image_id,
                        name=f'removed-{legacy_key}',
                        default_tools=['process_exit'],
                        metadata={
                            legacy_key: (
                                True
                                if legacy_key == 'lazy_tool_groups'
                                else ['filesystem']
                            ),
                        },
                    ),
                    actor='test',
                )

            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None
        finally:
            runtime.close()

    @pytest.mark.parametrize("legacy_key", ["lazy_tool_groups", "initial_tool_groups"])
    def test_removed_tool_group_metadata_is_rejected_when_persisted_image_is_loaded(
        self,
        legacy_key: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            image_id = f'persisted-{legacy_key}:v0'
            runtime.register_image(
                AgentImage(image_id=image_id, name=f'persisted-{legacy_key}'),
                actor='test',
            )
            runtime.store.conn.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (
                    json.dumps(
                        {
                            'image_id': image_id,
                            'name': f'persisted-{legacy_key}',
                            'metadata': {
                                legacy_key: (
                                    True
                                    if legacy_key == 'lazy_tool_groups'
                                    else ['filesystem']
                                ),
                            },
                        },
                    ),
                    image_id,
                ),
            )
            runtime.store.conn.commit()

            with pytest.raises(
                ValidationError,
                match=rf'removed tool-group fields: {legacy_key}',
            ):
                runtime.image_registry.load_persisted_images()
        finally:
            runtime.close()

    def test_concurrent_register_without_replace_has_single_winner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            barrier = threading.Barrier(2)
            original_validate = runtime.image_registry._validate_image

            def synchronized_validate(image: AgentImage) -> None:
                original_validate(image)
                if image.image_id == 'concurrent-image:v0':
                    barrier.wait(timeout=5)

            monkeypatch.setattr(runtime.image_registry, '_validate_image', synchronized_validate)
            outcomes: list[object] = []

            def register(version: str) -> None:
                try:
                    outcomes.append(
                        runtime.image_registry.register(
                            AgentImage(
                                image_id='concurrent-image:v0',
                                name='concurrent-image',
                                version=version,
                            ),
                            actor=f'thread-{version}',
                            replace=False,
                        )
                    )
                except Exception as exc:
                    outcomes.append(exc)

            threads = [threading.Thread(target=register, args=(version,)) for version in ('v1', 'v2')]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            assert all(not thread.is_alive() for thread in threads)
            assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
            assert sum(isinstance(outcome, ValidationError) for outcome in outcomes) == 1
            assert len(
                [record for record in runtime.audit.trace() if record.target == 'image:concurrent-image:v0']
            ) == 1
            persisted = runtime.store.get_image('concurrent-image:v0')
            assert persisted is not None
            assert runtime.get_image('concurrent-image:v0') == persisted[0]
        finally:
            runtime.close()

    def test_register_image_primitive_validates_tools_and_emits_audit(self) -> None:
        runtime = Runtime.open('local')
        try:
            image = AgentImage(image_id='custom-review:v0', name='custom-review', system_prompt='Custom review image.', default_tools=['read_memory_object', 'human_output'], safety_profile='review')
            runtime.register_image(image, actor='cli')
            assert runtime.get_image('custom-review:v0') == image
            assert runtime.get_image('custom-review:v0') is not image
            assert 'image.register' in [record.action for record in runtime.audit.trace()]
            assert EventType.IMAGE_REGISTERED in [event.type for event in runtime.events.list()]
        finally:
            runtime.close()

    def test_registered_image_isolated_from_caller_and_getter_mutation(self) -> None:
        runtime = Runtime.open('local')
        try:
            source = AgentImage(
                image_id='immutable-manifest:v0',
                name='immutable-manifest',
                default_tools=['human_output'],
                metadata={'nested': {'version': 1}},
            )
            registration = runtime.image_registry.register(source, actor='test')

            source.default_tools.append('read_memory_object')
            source.metadata['nested']['version'] = 2
            registration.image.default_tools.append('write_memory_object')
            fetched = runtime.get_image('immutable-manifest:v0')
            fetched.default_tools.append('read_text_file')
            fetched.metadata['nested']['version'] = 3

            canonical = runtime.get_image('immutable-manifest:v0')
            persisted = runtime.store.get_image('immutable-manifest:v0')
            pid = runtime.process.spawn(image='immutable-manifest:v0', goal='use canonical manifest')

            assert canonical.default_tools == ['human_output']
            assert canonical.metadata == {'nested': {'version': 1}}
            assert persisted is not None
            assert persisted[0].default_tools == ['human_output']
            assert runtime.process.get(pid).tool_table.keys() == {'human_output'}
        finally:
            runtime.close()

    def test_module_image_rehydrate_is_internal_exact_and_provenance_bound(self) -> None:
        runtime = Runtime.open('local')
        image = AgentImage(image_id='module-replay:v0', name='module-replay')
        owner = 'module:owner:v0'
        try:
            created = runtime.image_registry.register_module_image(image, actor=owner)
            assert created.disposition == 'created'

            with pytest.raises(ValidationError, match='already exists'):
                runtime.image_registry.register(image, actor=owner)
            with pytest.raises(ValidationError, match='already exists'):
                runtime.image_registry.register_module_image(image, actor=owner)

            runtime.images.pop(image.image_id)
            before = runtime.store.get_image(image.image_id)
            with pytest.raises(ValidationError, match='already exists'):
                runtime.image_registry.register_module_image(
                    image,
                    actor='module:other:v0',
                )
            assert image.image_id not in runtime.images
            assert runtime.store.get_image(image.image_id) == before

            rehydrated = runtime.image_registry.register_module_image(image, actor=owner)
            assert rehydrated.disposition == 'rehydrated'
            assert runtime.store.get_image(image.image_id) == before
            assert runtime.get_image(image.image_id) == image
        finally:
            runtime.close()

    def test_image_replace_failure_restores_cache_store_event_and_audit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            original = AgentImage(image_id='atomic-image:v0', name='atomic-image', version='v1')
            runtime.image_registry.register(original, actor='test')
            before_events = list(runtime.events.list())
            before_audit = list(runtime.audit.trace())
            real_record = runtime.audit.record

            def fail_replace_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') == 'image.replace':
                    raise RuntimeError('image replace audit failed')
                return real_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_replace_audit)
            with pytest.raises(RuntimeError, match='image replace audit failed'):
                runtime.image_registry.register(
                    AgentImage(image_id='atomic-image:v0', name='atomic-image', version='v2'),
                    actor='test',
                    replace=True,
                )

            persisted = runtime.store.get_image('atomic-image:v0')
            assert runtime.get_image('atomic-image:v0') == original
            assert persisted is not None and persisted[0].version == 'v1'
            assert runtime.events.list() == before_events
            assert runtime.audit.trace() == before_audit
        finally:
            runtime.close()

    def test_image_registry_mutations_require_admin_for_replace_and_settle_one_shot_authority(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        image_id = 'one-shot-image-registry:v0'
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='one-shot image registry')
            original_record = runtime.audit.record
            failing_actions = {'image.register'}

            def fail_selected_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') in failing_actions:
                    raise RuntimeError(f"injected {kwargs['action']} audit failure")
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_selected_audit)
            write_cap = runtime.capability.grant_once(
                actor,
                runtime.image_registry.resource_for(image_id),
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            original = AgentImage(image_id=image_id, name='one-shot-image-registry', version='v1')

            with pytest.raises(RuntimeError, match='image.register'):
                runtime.image_registry.register(original, actor=actor, require_capability=True)

            assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 1
            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None

            failing_actions.clear()
            runtime.image_registry.register(original, actor=actor, require_capability=True)
            assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 0

            runtime.capability.grant(
                actor,
                runtime.image_registry.resource_for(image_id),
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            with pytest.raises(CapabilityDenied):
                runtime.image_registry.register(
                    AgentImage(image_id=image_id, name='one-shot-image-registry', version='v2'),
                    actor=actor,
                    replace=True,
                    require_capability=True,
                )

            admin_cap = runtime.capability.grant_once(
                actor,
                runtime.image_registry.resource_for(image_id),
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            failing_actions.add('image.replace')
            replacement = AgentImage(image_id=image_id, name='one-shot-image-registry', version='v2')
            with pytest.raises(RuntimeError, match='image.replace'):
                runtime.image_registry.register(
                    replacement,
                    actor=actor,
                    replace=True,
                    require_capability=True,
                )

            assert runtime.store.get_capability(admin_cap.cap_id).uses_remaining == 1
            assert runtime.get_image(image_id).version == 'v1'

            failing_actions.clear()
            runtime.image_registry.register(
                replacement,
                actor=actor,
                replace=True,
                require_capability=True,
            )
            assert runtime.store.get_capability(admin_cap.cap_id).uses_remaining == 0
            assert runtime.get_image(image_id).version == 'v2'
        finally:
            runtime.close()

    def test_image_package_registration_failure_removes_new_artifact_and_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        package = _write_image_package(tmp_path / 'package-agent')
        runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(tmp_path))
        try:
            before_artifacts = runtime.store.list_image_artifacts()
            real_record = runtime.audit.record

            def fail_package_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') == 'image.package.register':
                    raise RuntimeError('package registration audit failed')
                return real_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_package_audit)
            with pytest.raises(RuntimeError, match='package registration audit failed'):
                runtime.image_registry.register_from_package_path(package, actor='test')

            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.get_image('package-agent:v0') is None
            assert runtime.store.list_image_artifacts() == before_artifacts
            assert not [event for event in runtime.events.list() if event.target == 'image:package-agent:v0']
            assert not [record for record in runtime.audit.trace() if record.target == 'image:package-agent:v0']
        finally:
            runtime.close()

    def test_checkpoint_image_commit_failure_removes_new_artifact_and_manifest(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='atomic image commit')
            checkpoint_id = runtime.checkpoint.create(pid, 'atomic image commit', actor=pid)
            before_artifacts = runtime.store.list_image_artifacts()
            real_emit = runtime.events.emit

            def fail_commit_event(event_type: EventType | str, *args: Any, **kwargs: Any) -> Any:
                if EventType(event_type) == EventType.IMAGE_COMMITTED:
                    raise RuntimeError('image commit event failed')
                return real_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_commit_event)
            with pytest.raises(RuntimeError, match='image commit event failed'):
                runtime.image_registry.commit_from_checkpoint(
                    actor='test',
                    checkpoint_id=checkpoint_id,
                    image_id='atomic-commit:v0',
                    name='atomic-commit',
                    require_capability=False,
                )

            assert 'atomic-commit:v0' not in runtime.images
            assert runtime.store.get_image('atomic-commit:v0') is None
            assert runtime.store.list_image_artifacts() == before_artifacts
            assert not [event for event in runtime.events.list() if event.target == 'image:atomic-commit:v0']
            assert not [record for record in runtime.audit.trace() if record.target == 'image:atomic-commit:v0']
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

    def test_image_context_management_is_strict_without_restricting_other_planner_keys(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_image(
                AgentImage(
                    image_id='context-policy:v0',
                    name='context-policy',
                    prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
                    planner={
                        'custom_planner_extension': {'enabled': True},
                        'context_management': {
                            'mode': 'prompt',
                            'threshold_ratio': 0.75,
                            'prompt': 'Reduce context.',
                        },
                    },
                ),
                actor='cli',
            )
            with pytest.raises(ValidationError, match='unknown planner.context_management'):
                runtime.register_image(
                    AgentImage(
                        image_id='bad-context-policy:v0',
                        name='bad-context-policy',
                        planner={'context_management': {'ambient_tool_grant': True}},
                    ),
                    actor='cli',
                )
        finally:
            runtime.close()

    def test_register_image_rejects_invalid_required_capability_right(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError):
                runtime.register_image({'image_id': 'bad-right-image:v0', 'name': 'bad-right-image', 'required_capabilities': [{'resource': 'filesystem:workspace:*', 'rights': ['*']}]}, actor='cli')
        finally:
            runtime.close()

    def test_register_image_rejects_invalid_required_module_spec(self) -> None:
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError, match='source_sha256'):
                runtime.register_image(
                    {
                        'image_id': 'bad-module-image:v0',
                        'name': 'bad-module-image',
                        'required_modules': [{'module_id': 'module:v0', 'source_sha256': 'not-a-sha'}],
                    },
                    actor='cli',
                )
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

    def test_image_package_required_modules_round_trips_and_boot_requires_loaded_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            required_module = {'module_id': 'missing-module:v0', 'source_sha256': '0' * 64}
            _write_image_package(Path(temp_dir) / 'package-agent', required_modules=[required_module])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                inspected = runtime.image_registry.inspect('package-agent:v0')

                assert result.image.required_modules == [required_module]
                assert inspected['image']['required_modules'] == [required_module]
                assert inspected['artifact']['required_modules'] == [required_module]
                with pytest.raises(ValidationError, match='image requires startup modules'):
                    runtime.process.spawn(image='package-agent:v0', goal='missing module')
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
                assert (result.error or '').startswith(
                    'permission_denied: CapabilityDenied'
                )
                assert 'image:package-agent:v0' not in (result.error or '')
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

    def test_checkpoint_snapshot_captures_image_package_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='package-agent:v0', goal='checkpoint package artifact')
                checkpoint_id = runtime.checkpoint.create(pid, 'package artifact', actor=pid)

                snapshot = runtime.store.get_checkpoint_snapshot(checkpoint_id)[1]

                artifact_id = result.image.boot['artifact_id']
                assert snapshot['images']['package-agent:v0']['boot']['artifact_id'] == artifact_id
                assert snapshot['image_artifacts'][artifact_id]['kind'] == 'image_package'
                assert snapshot['image_artifacts'][artifact_id]['artifact']['manifest_path'] == 'IMAGE.yaml'
                assert snapshot['image_artifacts'][artifact_id]['artifact']['workspace']['source'] == 'workspace'
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

    @pytest.mark.parametrize(
        ('field', 'yaml_value'),
        [
            ('recursive', '"false"'),
            ('recursive', '1'),
            ('recursive', 'null'),
            ('delegable', '"false"'),
            ('delegable', '1'),
            ('delegable', 'null'),
        ],
        ids=[
            'recursive-string',
            'recursive-int',
            'recursive-null',
            'delegable-string',
            'delegable-int',
            'delegable-null',
        ],
    )
    def test_image_package_rejects_non_boolean_workspace_grant_flags_before_effects(
        self,
        tmp_path: Path,
        field: str,
        yaml_value: str,
    ) -> None:
        root = _write_image_package(
            tmp_path / 'package-agent',
            workspace_grants=False,
        )
        root.joinpath('IMAGE.yaml').write_text(
            f"""
image_id: package-agent:v0
name: package-agent
prompt: prompt.md
workspace:
  source: workspace
  grants:
    - path: .
      rights: [read]
      {field}: {yaml_value}
""".lstrip(),
            encoding='utf-8',
        )
        runtime = Runtime.open(
            'local',
            substrate=LocalResourceProviderSubstrate(tmp_path),
        )
        try:
            before_artifacts = runtime.store.list_image_artifacts()
            before_capabilities = runtime.store.list_capabilities()
            before_audit_ids = [record.record_id for record in runtime.audit.trace()]
            before_event_ids = [event.event_id for event in runtime.events.list()]
            before_publications = runtime.store.list_runtime_publications()
            materialized = (
                tmp_path / runtime.config.image.materialized_workspace_root
            )

            with pytest.raises(
                ValidationError,
                match=rf'workspace\.grants\[\]\.{field} must be a boolean',
            ):
                runtime.image_registry.register_from_package_path(
                    root,
                    actor='cli',
                )

            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.get_image('package-agent:v0') is None
            assert runtime.store.list_image_artifacts() == before_artifacts
            assert runtime.store.list_capabilities() == before_capabilities
            assert [
                record.record_id for record in runtime.audit.trace()
            ] == before_audit_ids
            assert [event.event_id for event in runtime.events.list()] == before_event_ids
            assert runtime.store.list_runtime_publications() == before_publications
            assert not materialized.exists()
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('grant_flags', 'expected_recursive', 'expected_delegable'),
        [
            ('', False, False),
            ('      recursive: false\n      delegable: false\n', False, False),
            ('      recursive: true\n      delegable: true\n', True, True),
        ],
        ids=['defaults', 'explicit-false', 'explicit-true'],
    )
    def test_image_package_workspace_grant_boolean_defaults_and_execution(
        self,
        tmp_path: Path,
        grant_flags: str,
        expected_recursive: bool,
        expected_delegable: bool,
    ) -> None:
        root = _write_image_package(
            tmp_path / 'package-agent',
            workspace_grants=False,
        )
        root.joinpath('IMAGE.yaml').write_text(
            f"""
image_id: package-agent:v0
name: package-agent
prompt: prompt.md
workspace:
  source: workspace
  grants:
    - path: .
      rights: [read]
{grant_flags}""".lstrip(),
            encoding='utf-8',
        )
        runtime = Runtime.open(
            'local',
            substrate=LocalResourceProviderSubstrate(tmp_path),
        )
        try:
            result = runtime.image_registry.register_from_package_path(
                root,
                actor='cli',
            )
            persisted = runtime.store.get_image_artifact(
                str(result.image.boot['artifact_id'])
            )
            assert persisted is not None
            artifact, _metadata = persisted
            [grant] = artifact['workspace']['grants']
            assert grant['recursive'] is expected_recursive
            assert grant['delegable'] is expected_delegable

            pid = runtime.process.spawn(
                image='package-agent:v0',
                goal='validated workspace grant booleans',
            )
            workspace_root = runtime.process.working_directory(pid)
            package_caps = [
                capability
                for capability in runtime.store.list_capabilities(subject=pid)
                if capability.issued_by == 'image.package:package-agent:v0'
            ]
            assert len(package_caps) == 1
            [capability] = package_caps
            expected_resource = (
                runtime.filesystem.directory_resource_for_path(workspace_root)
                if expected_recursive
                else runtime.filesystem.resource_for_path(workspace_root)
            )
            assert capability.resource == expected_resource
            assert capability.delegable is expected_delegable
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
                assert runtime.store.select_table_rows('tool_candidates') == []
                assert not [obj for obj in runtime.store.list_objects() if obj.type == ObjectType.TOOL_CANDIDATE]
            finally:
                runtime.close()

    def test_image_package_default_skill_failure_cleans_materialized_workspace_and_jit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(Path(temp_dir) / 'package-agent', with_jit=True, default_skills=['missing-package-skill'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                with pytest.raises(Exception, match='missing-package-skill'):
                    runtime.process.spawn(image='package-agent:v0', goal='failed default skill')

                materialized = Path(temp_dir) / runtime.config.image.materialized_workspace_root
                seed_files = list(materialized.rglob('seed.txt')) if materialized.exists() else []
                assert seed_files == []
                assert not [
                    row for row in runtime.store.list_tools()
                    if row['name'] == 'package_count' and row['ephemeral']
                ]
                assert runtime.store.select_table_rows('tool_candidates') == []
                assert not [obj for obj in runtime.store.list_objects() if obj.type == ObjectType.TOOL_CANDIDATE]
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

    def test_image_package_jit_manifest_rejects_pathological_json_parser_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_roots: list[Path] = []
            for index, payload in enumerate(_pathological_json_payloads()):
                package_root = root / f'package-agent-{index}'
                _write_image_package(package_root, with_jit=True)
                package_root.joinpath('tools', 'jit-tools.json').write_text(
                    payload,
                    encoding='utf-8',
                )
                package_roots.append(package_root)

            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                for package_root in package_roots:
                    with pytest.raises(ValidationError, match='invalid image package jit_tools JSON'):
                        runtime.image_registry.register_from_package_path(package_root, actor='cli')
                    with pytest.raises(KeyError):
                        runtime.get_image('package-agent:v0')
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

    def test_image_package_context_management_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = _write_image_package(
                Path(temp_dir) / 'package-agent',
                prompt_mode=PROMPT_MODE_LIBOS_DEFAULT,
            )
            manifest = root / 'IMAGE.yaml'
            manifest.write_text(
                manifest.read_text(encoding='utf-8').replace(
                    'metadata:\n',
                    'planner:\n'
                    '  context_management:\n'
                    '    mode: prompt\n'
                    '    threshold_ratio: 0.7\n'
                    '    prompt: Preserve package state.\n'
                    'metadata:\n',
                ),
                encoding='utf-8',
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                result = runtime.image_registry.register_from_package_path(root, actor='cli')
                inspected = runtime.image_registry.inspect('package-agent:v0')

                expected = {
                    'mode': 'prompt',
                    'threshold_ratio': 0.7,
                    'prompt': 'Preserve package state.',
                }
                assert result.image.planner['context_management'] == expected
                assert inspected['image']['planner']['context_management'] == expected
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

    def test_failed_package_launch_compensates_exact_publication_artifacts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_image_package(root / 'package-agent')
            runtime = Runtime.open(
                'local',
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                runtime.image_registry.register_from_package_path(
                    root / 'package-agent',
                    actor='cli',
                )
                original_advance = runtime.store.advance_runtime_publication

                def fail_launch_commit(publication_id: str, **kwargs: Any) -> bool:
                    publication = runtime.store.get_runtime_publication(publication_id)
                    if (
                        kwargs.get('state') == 'committed'
                        and publication is not None
                        and publication['kind'] == 'process_launch'
                        and publication['plan'].get('image_id') == 'package-agent:v0'
                    ):
                        return False
                    return original_advance(publication_id, **kwargs)

                monkeypatch.setattr(
                    runtime.store,
                    'advance_runtime_publication',
                    fail_launch_commit,
                )
                with pytest.raises(ProcessError, match='cannot commit process publication'):
                    runtime.process.spawn(
                        image='package-agent:v0',
                        goal='publication artifact rollback',
                    )

                publication = [
                    item
                    for item in runtime.store.list_runtime_publications()
                    if item['kind'] == 'process_launch'
                    and item['plan'].get('image_id') == 'package-agent:v0'
                ][-1]
                artifacts = publication['receipt']['artifacts']
                kinds = {item['kind'] for item in artifacts}
                workspace_root = publication['plan']['materialized_workspace_root']

                assert publication['state'] == 'rolled_back'
                assert {'workspace', 'capability'} <= kinds
                assert not (root / workspace_root).exists()
                assert runtime.store.get_process(publication['pid']) is None
                for artifact in artifacts:
                    if artifact['kind'] == 'capability':
                        assert runtime.store.get_capability(artifact['capability_id']) is None
            finally:
                runtime.close()

    def test_reopen_retries_failed_package_launch_compensation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / 'runtime.sqlite'
            _write_image_package(root / 'package-agent')
            runtime = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                runtime.image_registry.register_from_package_path(
                    root / 'package-agent',
                    actor='cli',
                )
                original_advance = runtime.store.advance_runtime_publication

                def fail_launch_commit(publication_id: str, **kwargs: Any) -> bool:
                    publication = runtime.store.get_runtime_publication(publication_id)
                    if (
                        kwargs.get('state') == 'committed'
                        and publication is not None
                        and publication['kind'] == 'process_launch'
                        and publication['plan'].get('image_id') == 'package-agent:v0'
                    ):
                        return False
                    return original_advance(publication_id, **kwargs)

                def fail_first_compensation(_publication: dict[str, Any]) -> None:
                    raise RuntimeError('injected first compensation failure')

                monkeypatch.setattr(
                    runtime.store,
                    'advance_runtime_publication',
                    fail_launch_commit,
                )
                monkeypatch.setattr(
                    runtime.image_boot,
                    'cleanup_failed_launch_artifacts',
                    fail_first_compensation,
                )

                with pytest.raises(ExceptionGroup, match='launch and compensation failed'):
                    runtime.process.spawn(
                        image='package-agent:v0',
                        goal='retry failed publication compensation after reopen',
                    )

                publication = [
                    item
                    for item in runtime.store.list_runtime_publications()
                    if item['kind'] == 'process_launch'
                    and item['plan'].get('image_id') == 'package-agent:v0'
                ][-1]
                publication_id = publication['publication_id']
                pid = publication['pid']
                workspace_root = publication['plan']['materialized_workspace_root']
                assert publication['state'] == 'failed'
                assert runtime.store.get_process(pid) is not None
                assert (root / workspace_root / 'seed.txt').exists()
            finally:
                _release_fenced_runtime_or_close(runtime)

            reopened = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'rolled_back'
                assert publication_id in reopened.recovered_runtime_publications
                assert reopened.store.get_process(pid) is None
                assert not (root / workspace_root).exists()
                for artifact in publication['receipt']['artifacts']:
                    if artifact['kind'] == 'capability':
                        assert reopened.store.get_capability(artifact['capability_id']) is None
            finally:
                reopened.close()

    def test_reopen_cleans_image_package_workspace_after_interrupted_exec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class SimulatedCrash(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / 'runtime.sqlite'
            _write_image_package(root / 'package-agent')
            runtime = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                runtime.image_registry.register_from_package_path(
                    root / 'package-agent',
                    actor='cli',
                )
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='before interrupted exec',
                )
                pre_exec_capability = runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                original_instantiate = runtime.image_boot._instantiate_boot

                def crash_after_image_boot(*args: Any, **kwargs: Any) -> None:
                    original_instantiate(*args, **kwargs)
                    raise SimulatedCrash('interrupted after image workspace materialization')

                monkeypatch.setattr(
                    runtime.image_boot,
                    '_instantiate_boot',
                    crash_after_image_boot,
                )
                monkeypatch.setattr(
                    runtime.image_boot,
                    '_rollback_failed_exec',
                    _abort_exec_before_compensation,
                )
                with pytest.raises(BaseExceptionGroup) as caught:
                    runtime.exec_process(
                        pid,
                        'package-agent:v0',
                        goal='interrupted exec',
                    )
                assert _group_contains(caught.value, SimulatedCrash)

                materialized = root / runtime.config.image.materialized_workspace_root
                assert list(materialized.rglob('seed.txt'))
                publications = [
                    item
                    for item in runtime.store.list_runtime_publications(pid=pid)
                    if item['kind'] == 'process_exec'
                ]
                assert len(publications) == 1
                publication_id = publications[0]['publication_id']
                capability_artifact_ids = {
                    str(artifact['capability_id'])
                    for artifact in publications[0]['receipt']['artifacts']
                    if artifact['kind'] == 'capability'
                }
                assert capability_artifact_ids
                assert pre_exec_capability.cap_id not in capability_artifact_ids
            finally:
                _release_fenced_runtime_or_close(runtime)

            reopened = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                assert reopened.process.get(pid).image_id == 'base-agent:v0'
                assert list(materialized.rglob('seed.txt')) == []
                publication = reopened.store.get_runtime_publication(publication_id)
                assert publication is not None
                assert publication['state'] == 'rolled_back'
                restored_pre_exec_capability = reopened.store.get_capability(
                    pre_exec_capability.cap_id
                )
                assert restored_pre_exec_capability is not None
                assert restored_pre_exec_capability.active
                for capability_id in capability_artifact_ids:
                    assert reopened.store.get_capability(capability_id) is None
            finally:
                reopened.close()

    def test_reopen_restores_pre_exec_jit_handle_after_interrupted_exec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class SimulatedCrash(BaseException):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / 'runtime.sqlite'
            runtime = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                pid = runtime.process.spawn(
                    image='toolmaker-agent:v0',
                    goal='pre-exec process-local JIT',
                )
                candidate_id = runtime.tools.propose(
                    pid,
                    {
                        'name': 'pre_exec_jit_echo',
                        'description': 'Echo before an interrupted exec.',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                    },
                    'export function run(args) { return args; }',
                )
                candidate = runtime.store.get_tool_candidate(candidate_id)
                assert candidate is not None
                candidate.status = ToolCandidateStatus.VALIDATED
                candidate.validation = {'ok': True, 'language': 'typescript'}
                runtime.store.update_tool_candidate(candidate)
                handle = runtime.tools.register(pid, candidate_id)
                runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('review-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                original_configure = runtime.image_boot._configure_tools

                def crash_after_exec_tool_replacement(*args: Any, **kwargs: Any) -> Any:
                    original_configure(*args, **kwargs)
                    raise SimulatedCrash('interrupted after exec tool replacement')

                monkeypatch.setattr(
                    runtime.image_boot,
                    '_configure_tools',
                    crash_after_exec_tool_replacement,
                )
                monkeypatch.setattr(
                    runtime.image_boot,
                    '_rollback_failed_exec',
                    _abort_exec_before_compensation,
                )
                with pytest.raises(BaseExceptionGroup) as caught:
                    runtime.exec_process(pid, 'review-agent:v0', goal='interrupted exec')
                assert _group_contains(caught.value, SimulatedCrash)
            finally:
                _release_fenced_runtime_or_close(runtime)

            reopened = Runtime.open(
                target,
                substrate=LocalResourceProviderSubstrate(root),
            )
            try:
                restored = reopened.process.get(pid)
                assert restored.image_id == 'toolmaker-agent:v0'
                assert restored.tool_table['pre_exec_jit_echo'] == handle.tool_id
                restored_handle = reopened.tools.loaded_tool_handle(handle.tool_id)
                assert restored_handle is not None
                assert restored_handle.name == 'pre_exec_jit_echo'
                publication = [
                    item
                    for item in reopened.store.list_runtime_publications(pid=pid)
                    if item['kind'] == 'process_exec'
                ][-1]
                assert publication['state'] == 'rolled_back'
            finally:
                reopened.close()

    def test_exec_publication_commit_failure_compensates_before_return(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_image(
                AgentImage(
                    image_id='exec-commit-failure:v0',
                    name='exec-commit-failure',
                    default_tools=['human_output'],
                ),
                actor='test',
            )
            pid = runtime.process.spawn(goal='before failed publication commit')
            runtime.capability.grant(
                pid,
                runtime.image_registry.resource_for('exec-commit-failure:v0'),
                [CapabilityRight.READ],
                issued_by='test',
            )
            before = runtime.process.get(pid)
            original_advance = runtime.store.advance_runtime_publication

            def fail_process_exec_commit(publication_id: str, **kwargs: Any) -> bool:
                if kwargs.get('state') == 'committed':
                    publication = runtime.store.get_runtime_publication(publication_id)
                    if publication is not None and publication['kind'] == 'process_exec':
                        return False
                return original_advance(publication_id, **kwargs)

            monkeypatch.setattr(
                runtime.store,
                'advance_runtime_publication',
                fail_process_exec_commit,
            )

            with pytest.raises(ValidationError, match='cannot commit process exec publication'):
                runtime.exec_process(pid, 'exec-commit-failure:v0', goal='must be compensated')

            after = runtime.process.get(pid)
            assert after.image_id == before.image_id
            assert after.goal_oid == before.goal_oid
            assert after.tool_table == before.tool_table
            publications = [
                item
                for item in runtime.store.list_runtime_publications()
                if item['kind'] == 'process_exec' and item['pid'] == pid
            ]
            assert publications[-1]['state'] == 'rolled_back'
            exec_operations = [
                operation
                for operation in runtime.store.list_operations(pid=pid)
                if operation.name == 'process.exec'
            ]
            assert len(exec_operations) == 1
            assert exec_operations[0].outcome.value == 'failed'
            assert publications[-1]['plan']['operation_id'] == exec_operations[0].operation_id
            assert not [
                record
                for record in runtime.audit.trace(actor=pid, target=f'process:{pid}')
                if record.action == 'process.exec'
            ]
            assert not [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.PROCESS_EXEC
            ]
        finally:
            runtime.close()

    def test_failed_exec_inside_claim_fences_execution_token_and_returns_to_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                default_skills=['missing-package-skill'],
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(
                    Path(temp_dir) / 'package-agent',
                    actor='cli',
                )
                pid = runtime.process.spawn(goal='exec rollback fences scheduler token')
                runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                token = runtime.store.claim_execution(pid, owner_id='test.runtime')
                assert token is not None

                with bind_process_execution(token):
                    with pytest.raises(Exception, match='missing-package-skill'):
                        runtime.exec_process(
                            pid,
                            'package-agent:v0',
                            goal='fail in claimed quantum',
                        )

                restored = runtime.process.get(pid)
                assert restored.status == ProcessStatus.RUNNABLE
                assert restored.execution_generation > token.generation
                assert restored.execution_owner_id is None
                assert restored.execution_lease_id is None
                assert (
                    runtime.store.complete_execution(
                        token,
                        status=ProcessStatus.RUNNABLE,
                    )
                    is False
                )
            finally:
                runtime.close()

    def test_exec_process_late_package_failure_cleans_registered_jit_candidate_and_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                default_skills=['missing-package-skill'],
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(image='base-agent:v0', goal='before late failed exec')
                pre_exec_capability = runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                before = runtime.process.get(pid)

                with pytest.raises(Exception, match='missing-package-skill'):
                    runtime.exec_process(pid, 'package-agent:v0', goal='after late failed exec')

                after = runtime.process.get(pid)
                materialized = Path(temp_dir) / runtime.config.image.materialized_workspace_root
                seed_files = list(materialized.rglob('seed.txt')) if materialized.exists() else []
                assert after.image_id == before.image_id
                assert after.working_directory == before.working_directory
                assert 'package_count' not in after.tool_table
                assert seed_files == []
                assert runtime.store.select_table_rows('tool_candidates', 'pid = ?', [pid]) == []
                assert not [
                    obj
                    for obj in runtime.store.list_objects_owned_by('process', pid)
                    if obj.type == ObjectType.TOOL_CANDIDATE
                ]
                assert not [row for row in runtime.store.list_tools() if row['name'] == 'package_count']
                publication = [
                    item
                    for item in runtime.store.list_runtime_publications(pid=pid)
                    if item['kind'] == 'process_exec'
                ][-1]
                capability_artifact_ids = {
                    str(artifact['capability_id'])
                    for artifact in publication['receipt']['artifacts']
                    if artifact['kind'] == 'capability'
                }
                assert publication['state'] == 'rolled_back'
                assert capability_artifact_ids
                assert pre_exec_capability.cap_id not in capability_artifact_ids
                restored_pre_exec_capability = runtime.store.get_capability(
                    pre_exec_capability.cap_id
                )
                assert restored_pre_exec_capability is not None
                assert restored_pre_exec_capability.active
                for capability_id in capability_artifact_ids:
                    assert runtime.store.get_capability(capability_id) is None
            finally:
                runtime.close()

    def test_failed_exec_preserves_external_borrower_capability(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                default_skills=['missing-package-skill'],
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                target = runtime.process.spawn(goal='exec target with borrowed object')
                borrower = runtime.process.spawn(goal='external borrower')
                shared = runtime.memory.create_object(target, ObjectType.ARTIFACT, {'shared': True})
                borrower_cap = runtime.capability.issue_trusted(
                    borrower,
                    f'object:{shared.oid}',
                    [CapabilityRight.READ],
                    issued_by='external.borrower',
                )
                runtime.capability.grant(
                    target,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                with pytest.raises(Exception, match='missing-package-skill'):
                    runtime.exec_process(target, 'package-agent:v0', goal='fail after package install')

                persisted = runtime.store.get_capability(borrower_cap.cap_id)
                assert persisted is not None and persisted.active
                assert runtime.store.get_object(shared.oid) is not None
            finally:
                runtime.close()

    @pytest.mark.parametrize('revoke_mode', ['capability', 'resource'])
    def test_failed_exec_does_not_resurrect_concurrent_revoke(
        self,
        monkeypatch: pytest.MonkeyPatch,
        revoke_mode: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                default_skills=['missing-package-skill'],
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(goal='exec revoke winner')
                cap = runtime.capability.grant(
                    pid,
                    'custom:concurrent-revoke',
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                original_activate = runtime.skills.activate_skill

                def revoke_then_fail(*args: Any, **kwargs: Any) -> Any:
                    if revoke_mode == 'capability':
                        runtime.capability.revoke(
                            cap.cap_id,
                            revoked_by='external.concurrent',
                            reason='concurrent revoke wins',
                            require_authority=False,
                        )
                    else:
                        runtime.capability.revoke_resource_trusted(
                            cap.resource,
                            revoked_by='external.concurrent',
                            reason='concurrent resource revoke wins',
                        )
                    return original_activate(*args, **kwargs)

                monkeypatch.setattr(runtime.skills, 'activate_skill', revoke_then_fail)
                with pytest.raises(Exception, match='missing-package-skill'):
                    runtime.exec_process(
                        pid,
                        'package-agent:v0',
                        goal='fail after concurrent revoke',
                        preserve_capabilities=False,
                    )

                persisted = runtime.store.get_capability(cap.cap_id)
                assert persisted is not None and not persisted.active
                assert '_agent_libos_exec_rollback_token' not in persisted.metadata
            finally:
                runtime.close()

    def test_failed_exec_preserves_external_terminal_process_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _write_image_package(
                Path(temp_dir) / 'package-agent',
                with_jit=True,
                default_skills=['missing-package-skill'],
            )
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            runtime.tools.sandbox = AcceptingValidationSandbox()
            try:
                runtime.image_registry.register_from_package_path(Path(temp_dir) / 'package-agent', actor='cli')
                pid = runtime.process.spawn(goal='exec terminal winner')
                runtime.capability.grant(
                    pid,
                    runtime.image_registry.resource_for('package-agent:v0'),
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                original_activate = runtime.skills.activate_skill

                def cancel_then_fail(*args: Any, **kwargs: Any) -> Any:
                    runtime.process.cancel(pid, 'external terminal mutation')
                    return original_activate(*args, **kwargs)

                monkeypatch.setattr(runtime.skills, 'activate_skill', cancel_then_fail)
                with pytest.raises(RuntimeRecoveryRequired) as caught:
                    runtime.exec_process(pid, 'package-agent:v0', goal='fail after cancellation')

                assert runtime.process.get(pid).status == ProcessStatus.KILLED
                publication = runtime.store.get_runtime_publication(
                    caught.value.publication_id,
                )
                assert publication is not None
                assert publication['state'] == 'failed'
                assert publication['phase'] == 'compensation_failed'
                assert not any(
                    phase.get('phase') == 'compensation_applied'
                    for phase in publication['receipt']['phases']
                )
                assert runtime.lifecycle.state == 'close_failed'
            finally:
                _release_fenced_runtime_or_close(runtime)

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

    def test_host_image_package_growth_is_bounded_after_descriptor_open(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        root = _write_image_package(tmp_path / 'package-agent')
        target = root / 'workspace' / 'seed.txt'
        file_limit = 512
        config = replace(
            DEFAULT_CONFIG,
            image=replace(
                DEFAULT_CONFIG.image,
                package_file_max_bytes=file_limit,
            ),
        )
        runtime = Runtime.open('local', config=config)
        real_fstat = os.fstat
        real_fdopen = os.fdopen
        target_identity = (target.stat().st_dev, target.stat().st_ino)
        grew = False
        read_sizes: list[int] = []

        class RecordingHandle:
            def __init__(self, handle: Any):
                self.handle = handle

            def __enter__(self) -> 'RecordingHandle':
                self.handle.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self.handle.__exit__(*args)

            def fileno(self) -> int:
                return self.handle.fileno()

            def read(self, size: int = -1) -> bytes:
                read_sizes.append(size)
                return self.handle.read(size)

        def grow_after_open(fd: int) -> os.stat_result:
            nonlocal grew
            result = real_fstat(fd)
            if not grew and (result.st_dev, result.st_ino) == target_identity:
                grew = True
                with target.open('ab') as output:
                    output.write(b'x' * file_limit)
            return result

        def record_reads(fd: int, *args: object, **kwargs: object) -> Any:
            handle = real_fdopen(fd, *args, **kwargs)
            opened = real_fstat(fd)
            if (opened.st_dev, opened.st_ino) == target_identity:
                return RecordingHandle(handle)
            return handle

        monkeypatch.setattr(os, 'fstat', grow_after_open)
        monkeypatch.setattr(os, 'fdopen', record_reads)
        try:
            with pytest.raises(
                ValidationError,
                match=r'package_file_max_bytes=512',
            ):
                runtime.image_registry.validate_package_path(root)

            assert grew
            assert read_sizes
            assert all(0 < size <= file_limit + 1 for size in read_sizes)
            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.list_image_artifacts() == []
        finally:
            runtime.close()

    def test_host_image_package_swap_after_open_fails_before_authority_or_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        root = _write_image_package(tmp_path / 'package-agent')
        prompt = root / 'prompt.md'
        original_prompt = tmp_path / 'original-prompt.md'
        replacement = tmp_path / 'replacement-prompt.md'
        replacement.write_text('Replaced after descriptor open.\n', encoding='utf-8')
        prompt_identity = (prompt.stat().st_dev, prompt.stat().st_ino)
        real_fstat = os.fstat
        swapped = False
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(goal='host package swap must fail closed')
            capability = runtime.capability.grant_once(
                actor,
                runtime.image_registry.resource_for('package-agent:v0'),
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            before_artifacts = runtime.store.list_image_artifacts()
            before_audit = list(runtime.audit.trace())
            before_events = list(runtime.events.list())

            def swap_after_open(fd: int) -> os.stat_result:
                nonlocal swapped
                result = real_fstat(fd)
                if not swapped and (result.st_dev, result.st_ino) == prompt_identity:
                    swapped = True
                    os.replace(prompt, original_prompt)
                    os.replace(replacement, prompt)
                return result

            monkeypatch.setattr(os, 'fstat', swap_after_open)

            with pytest.raises(ValidationError, match='changed during read'):
                runtime.image_registry.register_from_package_path(
                    root,
                    actor=actor,
                    require_capability=True,
                )

            assert swapped
            assert runtime.store.get_capability(capability.cap_id).uses_remaining == 1
            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.get_image('package-agent:v0') is None
            assert runtime.store.list_image_artifacts() == before_artifacts
            assert runtime.audit.trace() == before_audit
            assert runtime.events.list() == before_events
        finally:
            runtime.close()

    def test_host_image_package_secure_open_failure_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        root = _write_image_package(tmp_path / 'package-agent')
        runtime = Runtime.open('local')
        try:
            def unavailable(*_args: object, **_kwargs: object) -> object:
                raise OSError('secure Host image package file open failed')

            monkeypatch.setattr(
                'agent_libos.runtime.image_registry.open_secure_directory',
                unavailable,
            )
            with pytest.raises(ValidationError, match='securely open image package'):
                runtime.image_registry.validate_package_path(root)
            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.list_image_artifacts() == []
        finally:
            runtime.close()

    @pytest.mark.skipif(os.name == 'nt', reason='requires POSIX symlink semantics')
    def test_host_image_package_rejects_symlinked_intermediate_ancestor(
        self,
        tmp_path: Path,
    ) -> None:
        real_parent = tmp_path / 'real-parent'
        root = _write_image_package(real_parent / 'package-agent')
        alias_parent = tmp_path / 'alias-parent'
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        runtime = Runtime.open('local')
        try:
            with pytest.raises(
                ValidationError,
                match='securely open image package',
            ):
                runtime.image_registry.validate_package_path(
                    alias_parent / root.name
                )
            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.list_image_artifacts() == []
        finally:
            runtime.close()

    def test_host_image_package_directory_replacement_during_enumeration_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        if os.name == 'nt':
            pytest.skip('POSIX descriptor replacement fixture')
        root = _write_image_package(tmp_path / 'package-agent')
        parked = tmp_path / 'package-agent-original'
        root_identity = (root.stat().st_dev, root.stat().st_ino)
        real_fstat = os.fstat
        swapped = False

        def replace_after_directory_open(fd: int) -> os.stat_result:
            nonlocal swapped
            result = real_fstat(fd)
            if not swapped and (result.st_dev, result.st_ino) == root_identity:
                swapped = True
                os.replace(root, parked)
                shutil.copytree(parked, root)
            return result

        monkeypatch.setattr(os, 'fstat', replace_after_directory_open)
        runtime = Runtime.open('local')
        try:
            with pytest.raises(ValidationError, match='changed during enumeration'):
                runtime.image_registry.validate_package_path(root)
            assert swapped
            assert 'package-agent:v0' not in runtime.images
            assert runtime.store.list_image_artifacts() == []
        finally:
            runtime.close()

    @pytest.mark.skipif(
        sys.platform != 'win32',
        reason='requires real Win32 package handles',
    )
    def test_windows_image_package_reads_through_secure_handle_chain(
        self,
        tmp_path: Path,
    ) -> None:
        root = _write_image_package(tmp_path / 'package-agent')
        runtime = Runtime.open('local')
        try:
            summary = runtime.image_registry.validate_package_path(root)

            assert summary['image_id'] == 'package-agent:v0'
            assert summary['counts']['files'] > 0
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
    required_modules: list[dict[str, str]] | None = None,
    default_skills: list[str] | None = None,
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
    required_modules_block = ""
    if required_modules:
        lines = ["required_modules:"]
        for module in required_modules:
            lines.append(f"  - module_id: {module['module_id']}")
            lines.append(f"    source_sha256: \"{module['source_sha256']}\"")
        required_modules_block = "\n".join(lines) + "\n"
    default_skills_block = ""
    if default_skills:
        lines = ["default_skills:"]
        for skill_id in default_skills:
            lines.append(f"  - {skill_id}")
        default_skills_block = "\n".join(lines) + "\n"
    root.joinpath('IMAGE.yaml').write_text(f"""
image_id: package-agent:v0
name: package-agent
version: v0
prompt: prompt.md
{prompt_mode_line}{jit_tool_exposure_line}{llm_profile_line}{required_modules_block}{default_skills_block}default_tools:
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


def _pathological_json_payloads() -> tuple[str, str]:
    oversized_integer = '9' * 5_000
    excessively_nested = ('[' * 2_000) + '0' + (']' * 2_000)
    return oversized_integer, excessively_nested


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


class AcceptingValidationSandbox(DenoTypescriptSandbox):
    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        return ValidationResult(ok=True, metadata={'language': 'typescript'})
