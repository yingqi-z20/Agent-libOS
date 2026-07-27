from __future__ import annotations
import pytest
import json
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    ProcessStatus,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.observability import json_size_bytes
from tests.support.public_errors import assert_public_error_message

class TestObjectFileTool:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_copy_file_via_named_object_without_materializing_content_to_process(self) -> None:
        sentinel = f'OBJECT_COPY_SENTINEL_{uuid4().hex}'
        source = f'agent_outputs/object_copy_source_{uuid4().hex}.txt'
        target = f'agent_outputs/object_copy_target_{uuid4().hex}.txt'
        object_name = f'copy.object.{uuid4().hex}'
        source_path = self.runtime.workspace_root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f'alpha\n{sentinel}\nomega\n', encoding='utf-8')
        client = GuardedActionClient(actions=[{'action': 'create_object_from_file', 'name': object_name, 'path': source}, {'action': 'write_object_to_file', 'name': object_name, 'path': target}, {'action': 'process_exit', 'payload': {'copied': True, 'object_name': object_name}}], forbidden_text=sentinel)
        self.runtime.llm.client = client
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='copy a file through Object Memory')
        self.runtime.skills.activate_skill(pid, 'agent-libos-object-file-transfer', actor=pid)
        self.runtime.filesystem.grant_path(pid, source, [CapabilityRight.READ], issued_by='test')
        self.runtime.filesystem.grant_path(pid, target, [CapabilityRight.WRITE], issued_by='test')
        results = []
        for _ in range(5):
            result = self.runtime.run_next_process_once()
            if result is None:
                break
            results.append(result)
            if self.runtime.process.get(pid).status == ProcessStatus.EXITED:
                break
        action_names = [result['action']['action'] for result in results if 'action' in result]
        create_result, write_result = (results[0]['result'], results[1]['result'])
        assert action_names == ['create_object_from_file', 'write_object_to_file', 'process_exit']
        assert create_result['ok']
        assert write_result['ok']
        assert sentinel not in json.dumps(create_result, ensure_ascii=False)
        assert sentinel not in json.dumps(write_result, ensure_ascii=False)
        assert (self.runtime.workspace_root / target).read_text(encoding='utf-8') == source_path.read_text(encoding='utf-8')
        assert client.calls == 3

    def test_object_file_tools_enforce_filesystem_and_object_capabilities(self) -> None:
        source = f'agent_outputs/object_tool_source_{uuid4().hex}.txt'
        target = f'agent_outputs/object_tool_target_{uuid4().hex}.txt'
        object_name = f'secure.object.{uuid4().hex}'
        source_path = self.runtime.workspace_root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text('capability checked', encoding='utf-8')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='use object file tools')
        denied_read = self.runtime.tools.call(pid, 'create_object_from_file', {'name': object_name, 'path': source})
        assert not denied_read.ok
        assert_public_error_message(
            denied_read.error,
            code='permission_denied',
            error_type='CapabilityDenied',
            forbidden=('lacks read', source),
        )
        self.runtime.filesystem.grant_path(pid, source, [CapabilityRight.READ], issued_by='test')
        created = self.runtime.tools.call(pid, 'create_object_from_file', {'name': object_name, 'path': source})
        assert created.ok
        assert 'capability checked' not in json.dumps(created.payload, ensure_ascii=False)
        denied_write = self.runtime.tools.call(pid, 'write_object_to_file', {'name': object_name, 'path': target})
        assert not denied_write.ok
        assert_public_error_message(
            denied_write.error,
            code='permission_denied',
            error_type='CapabilityDenied',
            forbidden=('lacks write', target),
        )
        self.runtime.filesystem.grant_path(pid, target, [CapabilityRight.WRITE], issued_by='test')
        written = self.runtime.tools.call(pid, 'write_object_to_file', {'name': object_name, 'path': target})
        assert written.ok
        assert (self.runtime.workspace_root / target).read_text(encoding='utf-8') == 'capability checked'

    def test_write_object_to_file_reuses_exact_one_time_read_snapshot(self) -> None:
        pid = self.runtime.process.spawn(
            image='review-agent:v0',
            goal='export one-time Object snapshot',
        )
        name = f'one-time.export.{uuid4().hex}'
        target = f'agent_outputs/one_time_export_{uuid4().hex}.txt'
        original = self.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'content': 'read exactly once'},
            name=name,
        )
        self.runtime.capability.revoke(
            original.capability_id,
            revoked_by='test.host',
            reason='replace with one-time Object read',
            require_authority=False,
        )
        one_time_read = self.runtime.capability.issue_trusted(
            pid,
            f'object:{original.oid}',
            [ObjectRight.READ],
            issued_by='test.host',
            uses_remaining=1,
        )
        self.runtime.filesystem.grant_path(
            pid,
            target,
            [CapabilityRight.WRITE],
            issued_by='test.host',
        )

        result = self.runtime.tools.call(
            pid,
            'write_object_to_file',
            {'name': name, 'path': target},
        )

        assert result.ok, result.error
        assert (self.runtime.workspace_root / target).read_text(
            encoding='utf-8'
        ) == 'read exactly once'
        persisted = self.runtime.store.get_capability(one_time_read.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0

    def test_write_object_to_file_rejects_changed_source_snapshot_before_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pid = self.runtime.process.spawn(
            image='review-agent:v0',
            goal='reject stale Object export',
        )
        name = f'stale.export.{uuid4().hex}'
        target = f'agent_outputs/stale_export_{uuid4().hex}.txt'
        handle = self.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'content': 'old snapshot bytes'},
            name=name,
            immutable=False,
        )
        self.runtime.filesystem.grant_path(
            pid,
            target,
            [CapabilityRight.WRITE],
            issued_by='test.host',
        )
        original_write = self.runtime.filesystem.write_text
        update_count = 0

        def update_before_write(*args, **kwargs):
            nonlocal update_count
            update_count += 1
            self.runtime.memory.update_object(
                pid,
                handle,
                ObjectPatch(payload={'content': 'new snapshot bytes'}),
            )
            return original_write(*args, **kwargs)

        monkeypatch.setattr(self.runtime.filesystem, 'write_text', update_before_write)
        effects_before = tuple(self.runtime.store.list_external_effects(pid=pid))

        result = self.runtime.tools.call(
            pid,
            'write_object_to_file',
            {'name': name, 'path': target},
        )

        assert not result.ok
        assert update_count == 1
        assert not (self.runtime.workspace_root / target).exists()
        assert tuple(self.runtime.store.list_external_effects(pid=pid)) == effects_before
        assert self.runtime.store.get_file_label_binding(target) is None

    def test_create_object_from_file_rejects_limit_above_filesystem_read_boundary(self) -> None:
        pid = self.runtime.process.spawn(
            image='review-agent:v0',
            goal='reject an impossible Object-file read ceiling',
        )

        rejected = self.runtime.tools.call(
            pid,
            'create_object_from_file',
            {
                'name': 'impossible.limit',
                'path': 'does-not-need-to-exist.txt',
                'max_bytes': (
                    self.runtime.config.tools.filesystem_read_hard_limit_bytes + 1
                ),
            },
        )

        assert not rejected.ok
        assert_public_error_message(
            rejected.error,
            code='validation_error',
            error_type='ToolExecutionError',
            forbidden=('effective Object-file read limit',),
        )
        assert not self.runtime.capability.check(
            pid,
            'filesystem:workspace:does-not-need-to-exist.txt',
            CapabilityRight.READ,
        )

    def test_create_object_from_file_accepts_runtime_limits_above_builtin_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        self.runtime.close()
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(
                DEFAULT_CONFIG.tools,
                object_file_max_bytes=15_000_000,
                object_file_hard_limit_bytes=20_000_000,
                filesystem_read_hard_limit_bytes=20_000_000,
            ),
        )
        self.runtime = Runtime.open(
            'local',
            config=config,
            substrate=LocalResourceProviderSubstrate(tmp_path),
        )
        source = 'runtime-configured-object-file.txt'
        (tmp_path / source).write_text('bounded payload', encoding='utf-8')
        pid = self.runtime.process.spawn(
            image='review-agent:v0',
            goal='use the active runtime Object-file limits',
        )
        self.runtime.filesystem.grant_path(
            pid,
            source,
            [CapabilityRight.READ],
            issued_by='test',
        )

        defaulted = self.runtime.tools.call(
            pid,
            'create_object_from_file',
            {'name': 'runtime.default.limit', 'path': source},
        )
        explicit = self.runtime.tools.call(
            pid,
            'create_object_from_file',
            {
                'name': 'runtime.explicit.limit',
                'path': source,
                'max_bytes': 16_000_000,
            },
        )

        assert defaulted.ok, defaulted.error
        assert explicit.ok, explicit.error

    def test_file_object_token_estimate_limits_prompt_materialization(self) -> None:
        sentinel = f'FILE_OBJECT_BUDGET_SENTINEL_{uuid4().hex}'
        source = f'agent_outputs/object_budget_source_{uuid4().hex}.txt'
        object_name = f'budget.object.{uuid4().hex}'
        source_path = self.runtime.workspace_root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text((sentinel + ' ') * 200, encoding='utf-8')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='import file with real token estimate')
        self.runtime.filesystem.grant_path(pid, source, [CapabilityRight.READ], issued_by='test')

        created = self.runtime.tools.call(pid, 'create_object_from_file', {'name': object_name, 'path': source})

        assert created.ok, created.error
        oid = created.payload['oid']
        obj = self.runtime.store.get_object(oid)
        assert obj is not None
        assert obj.metadata.token_estimate is not None
        assert obj.metadata.token_estimate > 1
        handle = self.runtime.memory.handle_for_name(pid, object_name, rights=['read', 'materialize'])
        view = self.runtime.memory.create_view(pid, [handle])
        context = self.runtime.memory.materialize_context(pid, view, budget_tokens=1)
        assert oid in context.omitted_objects
        assert sentinel not in context.text

    def test_create_object_from_file_enforces_memory_payload_limit_before_create_and_can_truncate(self) -> None:
        self.runtime.close()
        limit = 1_200
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(
                DEFAULT_CONFIG.tools,
                memory_payload_hard_limit_bytes=limit,
                object_file_max_bytes=2_000,
                object_file_hard_limit_bytes=2_000,
            ),
            llm_context=replace(
                DEFAULT_CONFIG.llm_context,
                storage_compaction_threshold_bytes=limit - 1,
            ),
        )
        self.runtime = Runtime.open('local', config=config)
        source = f'agent_outputs/object_large_source_{uuid4().hex}.txt'
        source_path = self.runtime.workspace_root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text('x' * 1_500, encoding='utf-8')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='import bounded file')
        self.runtime.filesystem.grant_path(pid, source, [CapabilityRight.READ], issued_by='test')

        failed = self.runtime.tools.call(
            pid,
            'create_object_from_file',
            {'name': 'large.no.truncate', 'path': source, 'max_bytes': 1_500},
        )
        truncated = self.runtime.tools.call(
            pid,
            'create_object_from_file',
            {'name': 'large.truncated', 'path': source, 'max_bytes': 1_500, 'allow_truncated': True},
        )

        assert not failed.ok
        assert_public_error_message(
            failed.error,
            code='execution_error',
            error_type='ToolExecutionError',
            forbidden=('payload limit', source),
        )
        assert truncated.ok, truncated.error
        assert truncated.payload['truncated'] is True
        obj = self.runtime.store.get_object(truncated.payload['oid'])
        assert obj.payload['truncated'] is True
        assert len(obj.payload['content']) < 1_500
        assert json_size_bytes(obj.payload) <= limit

class GuardedActionClient:

    def __init__(self, actions: list[dict[str, object]], forbidden_text: str):
        self.actions = list(actions)
        self.forbidden_text = forbidden_text
        self.calls = 0

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        serialized_messages = json.dumps(messages, ensure_ascii=False)
        if self.forbidden_text and self.forbidden_text in serialized_messages:
            raise AssertionError('file content was materialized into the process prompt')
        action = self.actions.pop(0)
        name = str(action['action'])
        args = {key: value for key, value in action.items() if key != 'action'}
        return LLMCompletion(content='', tool_calls=[{'id': f'fake_{self.calls}', 'name': name, 'arguments': json.dumps(args)}])
