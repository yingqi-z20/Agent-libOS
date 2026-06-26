from __future__ import annotations
import pytest
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.llm.context_memory import context_object_name
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    PROMPT_MODE_LIBOS_DEFAULT,
    ProcessStatus,
    ResourceBudget,
    ViewMode,
)
from tests.support.deno import COUNT_CHARS_SOURCE
from tests.support.fakes import RecordingActionClient
from tests.support.skills import write_skill_package


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')


def _grant_context_compressor_authority(runtime: Runtime, pid: str) -> None:
    _grant_process_spawn(runtime, pid)
    runtime.capability.grant(pid, 'image:context-compressor:v0', [CapabilityRight.READ], issued_by='test')


class TestLLMContextMemory:

    def test_llm_context_is_process_readable_writable_memory_object(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([{'action': 'create_memory_object', 'type': 'observation', 'payload': {'seen': 1}}])
            pid = runtime.process.spawn(image='base-agent:v0', goal='create context')
            runtime.run_next_process_once()
            name = context_object_name(pid)
            obj = runtime.store.get_object_by_name(name, namespace=runtime.memory.resolve_namespace(pid))
            assert obj is not None
            assert obj is not None
            assert not obj.immutable
            assert obj.payload['kind'] == 'llm_context'
            assert runtime.capability.check(pid, f'object:{obj.oid}', ObjectRight.READ)
            assert runtime.capability.check(pid, f'object:{obj.oid}', ObjectRight.WRITE)
            process = runtime.process.get(pid)
            assert obj.oid in [handle.oid for handle in process.memory_view.roots]
            read = runtime.tools.call(pid, 'read_memory_object', {'name': name})
            appended = runtime.tools.call(pid, 'append_memory_object', {'name': name, 'entry': {'kind': 'agent_note', 'text': 'keep this in context'}})
            updated = runtime.store.get_object_by_name(name, namespace=runtime.memory.resolve_namespace(pid))
            assert read.ok, read.error
            assert appended.ok, appended.error
            assert updated.payload['entries'][-1]['kind'] == 'agent_note'
        finally:
            runtime.close()

    def test_llm_context_prompt_grows_by_appending_to_preserve_cache_prefix(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([{'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}}, {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 2}}])
            pid = runtime.process.spawn(image='base-agent:v0', goal='append context')
            runtime.run_next_process_once()
            runtime.run_next_process_once()
            first, second = runtime.llm.client.user_prompts
            assert 'Cache strategy: append_only_stable_prefix' in first
            assert 'LLM context object' in first
            assert second.startswith(first)
            context = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            kinds = [entry['kind'] for entry in context.payload['entries']]
            assert 'memory_delta' in kinds
            assert len(second) > len(first)
        finally:
            runtime.close()

    def test_llm_context_rendered_prompt_is_charged_to_materialization_budget_before_model_call(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'should_not_run': True}}])
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='budget context before model call',
                resource_budget=ResourceBudget(
                    max_context_materialization_tokens=100_000,
                    max_context_materialization_total_tokens=1,
                ),
            )

            result = runtime.run_next_process_once()

            assert result['resource_limit_exceeded']
            assert runtime.llm.client.tool_batches == []
            assert runtime.process.get(pid).status == ProcessStatus.KILLED
        finally:
            runtime.close()

    def test_llm_prompt_lists_only_process_visible_tools(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'done': True}}])
            pid = runtime.process.spawn(image='base-agent:v0', goal='exit')
            runtime.run_next_process_once()
            tool_names = {tool['function']['name'] for tool in runtime.llm.client.tool_batches[0]}
            assert 'process_exit' in tool_names
            assert 'read_text_file' not in tool_names
            assert 'read_text_file' not in runtime.llm.client.user_prompts[0]
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_multiplexed_jit_tool_call_dispatches_real_jit_tool(self) -> None:
        runtime = Runtime.open('local')
        try:
            _register_multiplexed_image(runtime)
            pid = runtime.process.spawn(image='multiplexed-jit:v0', goal='count with a JIT tool')
            _register_count_tool(runtime, pid, 'count_chars')
            runtime.llm.client = RecordingActionClient([
                {
                    'action': JIT_MULTIPLEXER_TOOL_NAME,
                    'tool_name': 'count_chars',
                    'arguments': {'text': 'hello'},
                }
            ])

            result = runtime.run_next_process_once()
            tool_names = {tool['function']['name'] for tool in runtime.llm.client.tool_batches[0]}

            assert result['ok'], result
            assert result['action']['action'] == 'count_chars'
            assert result['result']['payload'] == {'count': 5}
            assert JIT_MULTIPLEXER_TOOL_NAME in tool_names
            assert 'count_chars' not in tool_names
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_multiplexed_jit_direct_name_and_bad_args_are_repairable(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, action_repair_attempts=3))
        runtime = Runtime.open('local', config=config)
        try:
            _register_multiplexed_image(runtime)
            pid = runtime.process.spawn(image='multiplexed-jit:v0', goal='repair bad JIT action')
            _register_count_tool(
                runtime,
                pid,
                'strict_count',
                input_schema={
                    'type': 'object',
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                    'additionalProperties': False,
                },
            )
            runtime.llm.client = RecordingActionClient([
                {'action': 'strict_count', 'text': 'hello'},
                {
                    'action': JIT_MULTIPLEXER_TOOL_NAME,
                    'tool_name': 'strict_count',
                    'arguments': {'extra': 'rejected'},
                },
                {'action': 'process_exit', 'payload': {'done': True}},
            ])

            result = runtime.run_next_process_once()

            assert result['ok'], result
            assert result['action']['action'] == 'process_exit'
            repairs = [record for record in runtime.audit.trace() if record.action == 'llm.action_repair_requested']
            assert any('strict_count' in str(record.decision) for record in repairs)
            assert any('Additional properties' in str(record.decision) for record in repairs)
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'strict_count'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_multiplexed_prompt_context_hides_jit_catalog(self) -> None:
        runtime = Runtime.open('local')
        try:
            _register_multiplexed_image(runtime, prompt_mode=PROMPT_MODE_LIBOS_DEFAULT)
            pid = runtime.process.spawn(image='multiplexed-jit:v0', goal='hide catalog')
            _register_count_tool(runtime, pid, 'secret_count')
            runtime.events.emit(
                EventType.TOOL_COMPLETED,
                source='tool:secret_count',
                target=pid,
                payload={'secret_count': {'action': 'secret_count'}},
            )
            runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'done': True}}])

            runtime.run_next_process_once()

            prompt = runtime.llm.client.user_prompts[0]
            tool_names = {tool['function']['name'] for tool in runtime.llm.client.tool_batches[0]}
            assert JIT_MULTIPLEXER_TOOL_NAME in prompt
            assert 'secret_count' not in prompt
            assert JIT_MULTIPLEXER_TOOL_NAME in tool_names
            assert 'secret_count' not in tool_names
            assert JIT_MULTIPLEXER_TOOL_NAME in runtime.tools.model_tool_names(pid)
            assert 'secret_count' not in runtime.tools.model_tool_names(pid)
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_multiplexed_prompt_context_hides_skill_jit_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(
                Path(temp_dir),
                'secret-jit-skill',
                jit_tools=[
                    {
                        'name': 'skill_secret_count',
                        'description': 'Count text characters.',
                        'source_path': 'scripts/skill_secret_count.ts',
                        'input_schema': {'type': 'object'},
                        'output_schema': {'type': 'object'},
                        'tests': [{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
                    }
                ],
                scripts={
                    'scripts/skill_secret_count.ts': COUNT_CHARS_SOURCE
                },
                body='Use this skill without relying on an automatic JIT catalog.\n',
            )
            runtime = Runtime.open('local')
            try:
                _register_multiplexed_image(runtime, prompt_mode=PROMPT_MODE_LIBOS_DEFAULT)
                pid = runtime.process.spawn(image='multiplexed-jit:v0', goal='hide skill catalog')
                runtime.skills.register_skill_from_path(skill_dir, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'skill:secret-jit-skill', [CapabilityRight.EXECUTE], issued_by='test')
                runtime.skills.activate_skill(pid, 'secret-jit-skill', actor=pid)
                runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'done': True}}])

                runtime.run_next_process_once()

                prompt = runtime.llm.client.user_prompts[0]
                assert JIT_MULTIPLEXER_TOOL_NAME in prompt
                assert 'skill_secret_count' not in prompt
                assert 'tool_secret' not in prompt
                assert 'secret-jit-skill' in prompt
            finally:
                runtime.close()

    def test_llm_context_appends_updated_object_version(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                {'action': 'process_exit', 'payload': {'done': True}},
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='track object updates')
            handle = runtime.memory.create_object(
                pid=pid,
                object_type=ObjectType.OBSERVATION,
                payload={'value': 'old-object-token'},
                immutable=False,
                name='changing-observation',
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(pid, [handle], mode=ViewMode.READ_ONLY)
            runtime.store.update_process(process)

            runtime.run_next_process_once()
            runtime.memory.update_object(pid, handle, ObjectPatch(payload={'value': 'new-object-token'}))
            runtime.run_next_process_once()

            first, second = runtime.llm.client.user_prompts
            assert 'old-object-token' in first
            assert 'new-object-token' in second
            assert second.startswith(first)
        finally:
            runtime.close()

    def test_llm_executor_fails_closed_when_process_image_is_missing(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = ExplodingClient()
            pid = runtime.process.spawn(image='base-agent:v0', goal='missing image')
            process = runtime.process.get(pid)
            process.image_id = 'missing-image:v0'
            runtime.store.update_process(process)

            result = runtime.run_process_once(pid)

            assert not result['ok']
            assert 'agent image not found' in result['error']
            assert runtime.process.get(pid).status == ProcessStatus.FAILED
            assert 'llm.image_missing' in [record.action for record in runtime.audit.trace()]
        finally:
            runtime.close()

    def test_llm_retries_malformed_empty_tool_name_once(self) -> None:
        runtime = Runtime.open('local')
        try:
            secret = 'SECRET_REPAIR_ARGUMENT_SHOULD_NOT_APPEAR'
            runtime.llm.client = RecordingActionClient([{'action': '', 'path': '.', 'token': secret}, {'action': 'process_exit', 'payload': {'done': True}}])
            pid = runtime.process.spawn(image='base-agent:v0', goal='recover malformed action')
            result = runtime.run_next_process_once()
            assert result['ok']
            assert result['action']['action'] == 'process_exit'
            assert len(runtime.llm.client.user_prompts) == 2
            assert 'could not be dispatched' in runtime.llm.client.user_prompts[1]
            repairs = [record for record in runtime.audit.trace() if record.action == 'llm.action_repair_requested']
            assert len(repairs) == 1
            assert repairs[0].decision is not None
            preview = repairs[0].decision['tool_calls_preview'][0]
            assert preview['name'] == ''
            assert '"path"' in preview['arguments_preview']
            assert secret not in preview['arguments_preview']
            assert preview['arguments_redacted']
            assert preview['arguments_sha256']
            assert preview['arguments_bytes'] > 0
        finally:
            runtime.close()

    def test_llm_call_records_persist_sanitized_prompt_output_usage_and_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = MetadataActionClient()
                pid = runtime.process.spawn(image='base-agent:v0', goal='persist llm calls')
                runtime.run_next_process_once()
                calls = runtime.store.list_llm_calls(pid)
                assert len(calls) == 1
                call = calls[0]
                assert call.pid == pid
                assert call.purpose == 'action_selection'
                assert call.status == 'ok'
                assert call.api == 'chat'
                assert call.model == 'test-model'
                assert call.request_id == 'req_123'
                assert call.response_id == 'resp_123'
                assert call.response_content == 'visible assistant text'
                assert call.usage['total_tokens'] == 17
                assert call.messages['sha256']
                assert call.tools['sha256']
                assert call.tool_calls['sha256']
                assert call.raw_response['sha256']
                assert call.reasoning['sha256']
                assert call.observability['response_content']['sha256']
                serialized = json.dumps(call.__dict__, sort_keys=True)
                assert 'persist llm calls' not in serialized
                assert '"payload": {"done": true}' not in serialized
            finally:
                runtime.close()
            reopened = Runtime.open(db)
            try:
                persisted = reopened.store.list_llm_calls()
                assert len(persisted) == 1
                assert persisted[0].usage['prompt_tokens'] == 13
                assert persisted[0].observability['messages']['bytes'] > 0
            finally:
                reopened.close()

    def test_llm_call_records_can_persist_full_io_when_configured(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=True))
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = MetadataActionClient()
                pid = runtime.process.spawn(image='base-agent:v0', goal='persist full llm calls')

                runtime.run_next_process_once()
                call = runtime.store.list_llm_calls(pid)[0]

                assert call.messages[1]['content']
                assert 'persist full llm calls' in call.messages[1]['content']
                assert any((tool['function']['name'] == 'process_exit' for tool in call.tools))
                assert call.response_content == 'visible assistant text'
                assert call.tool_calls[0]['name'] == 'process_exit'
                assert call.tool_calls[0]['arguments'] == json.dumps({'payload': {'done': True}})
                assert call.reasoning == {'summary': 'selected process_exit'}
                assert call.raw_response['id'] == 'raw_resp'
                assert call.observability['messages']['sha256']
            finally:
                runtime.close()

            reopened = Runtime.open(db, config=config)
            try:
                persisted = reopened.store.list_llm_calls(pid)[0]
                assert 'persist full llm calls' in persisted.messages[1]['content']
                assert persisted.raw_response['provider'] == 'fake'
            finally:
                reopened.close()

    def test_pending_human_llm_action_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            path = 'agent_outputs/pending_llm_action_reopen.txt'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = RecordingActionClient([
                    {'action': 'write_text_file', 'path': path, 'content': 'persisted approval action'},
                ])
                pid = runtime.process.spawn(image='review-agent:v0', goal='write after approval')
                runtime.capability.set_permission_policy(
                    subject=pid,
                    resource=runtime.filesystem.resource_for(path),
                    rights=[CapabilityRight.WRITE],
                    policy='ask_each_time',
                    issued_by='test',
                )
                waiting = runtime.run_next_process_once()
                assert waiting['waiting_human']
                assert runtime.store.get_llm_pending_action(pid)['wait_type'] == 'human'
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                reopened.llm.client = ExplodingClient()
                reopened.human.drain_terminal_queue(auto_approve=True)
                resumed = reopened.run_next_process_once()

                assert resumed['resumed_after_human']
                assert resumed['action']['action'] == 'write_text_file'
                assert resumed['result']['ok']
                assert (reopened.workspace_root / path).read_text(encoding='utf-8') == 'persisted approval action'
                assert reopened.store.get_llm_pending_action(pid)['status'] == 'completed'
            finally:
                reopened.close()

    def test_compact_process_context_waits_for_compressor_child_and_replaces_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'force': True,
                    'target_tokens': 512,
                    'max_chunks': 1,
                    'preserve_recent_entries': 1,
                },
                {'action': 'process_exit', 'payload': _compact_summary('compressed state')},
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='compact current context')
            _grant_context_compressor_authority(runtime, pid)

            results = runtime.run_until_idle(max_quanta=3)

            completed = _last_action_result(results, 'compact_process_context')
            assert completed['result']['ok']
            output = completed['result']['payload']
            assert output['compacted'] is True
            assert len(output['compressor_pids']) == 1
            child = runtime.process.get(output['compressor_pids'][0])
            assert child.image_id == 'context-compressor:v0'
            assert set(child.tool_table) == {'process_exit'}

            context = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            assert context is not None
            assert context.version == output['new_version']
            compacted_entry = context.payload['entries'][0]
            assert compacted_entry['kind'] == 'context_compacted'
            assert compacted_entry['summary']['goal'] == 'compressed state'
            assert compacted_entry['compaction_method'] == 'agent_image_child'
            assert compacted_entry['compaction_metadata']['compressor_image_id'] == 'context-compressor:v0'
            assert compacted_entry['compaction_metadata']['tool_name'] == 'compact_process_context'
            assert 'compacted' in context.metadata.tags
            assert 'compaction_method:agent_image_child' in context.metadata.tags
            assert any(isinstance(result, dict) and result.get('waiting_event') for result in results)
            child_tool_names = {tool['function']['name'] for tool in runtime.llm.client.tool_batches[1]}
            assert child_tool_names == {'process_exit'}
        finally:
            runtime.close()

    def test_compact_process_context_skips_small_context_without_force(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'target_tokens': 64_000,
                    'force': False,
                }
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='small context')

            result = runtime.run_next_process_once()

            assert result['action']['action'] == 'compact_process_context'
            assert result['result']['ok']
            output = result['result']['payload']
            assert output['compacted'] is False
            assert output['reason'] == 'context_under_target'
            assert runtime.process.list_children(pid) == []
            context = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            assert context is not None
            assert not any(entry.get('kind') == 'context_compacted' for entry in context.payload['entries'])
        finally:
            runtime.close()

    def test_compact_process_context_invalid_child_output_does_not_replace_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'force': True,
                    'target_tokens': 512,
                    'max_chunks': 1,
                },
                {'action': 'process_exit', 'payload': {'goal': 'missing required fields'}},
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='invalid compressor output')
            _grant_context_compressor_authority(runtime, pid)

            waiting = runtime.run_next_process_once()
            assert waiting['waiting_event']
            before = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            assert before is not None
            before_version = before.version
            before_payload = json.loads(json.dumps(before.payload))

            results = runtime.run_until_idle(max_quanta=2)

            completed = _last_action_result(results, 'compact_process_context')
            assert completed['result']['ok'] is False
            assert 'missing required fields' in (completed['result']['error'] or '')
            after = runtime.store.get_object(before.oid)
            assert after is not None
            assert after.version == before_version
            assert after.payload == before_payload
        finally:
            runtime.close()

    def test_compact_process_context_rejects_forged_resume_job(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='reject forged resume')
            context = _seed_context_entries(runtime, pid, count=1)
            forged_job = {
                'kind': 'context_compaction_job',
                'schema_version': 1,
                'status': 'active',
                'caller_pid': pid,
                'context_oid': context.oid,
                'source_version': context.version,
                'source_payload': json.loads(json.dumps(context.payload)),
                'source_tokens': 1,
                'target_tokens': 512,
                'preserve_recent_entries': 0,
                'max_chunks': 1,
                'stage_index': 1,
                'current_child_pid': 'pid_forged_child',
                'compressor_pids': [],
                'summaries': [_compact_summary('forged state')],
            }

            result = runtime.tools.call(pid, 'compact_process_context', {'_resume_job': forged_job})

            assert result.ok is False
            assert 'not callable directly' in (result.error or '')
            after = runtime.store.get_object(context.oid)
            assert after is not None
            assert not any(entry.get('kind') == 'context_compacted' for entry in after.payload['entries'])
            assert runtime.process.list_children(pid) == []
        finally:
            runtime.close()

    def test_compact_process_context_spawn_failure_marks_job_failed(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'force': True,
                    'target_tokens': 512,
                    'max_chunks': 64,
                    'preserve_recent_entries': 0,
                },
                *[
                    {'action': 'process_exit', 'payload': _compact_summary(f'stage {index}')}
                    for index in range(runtime.config.process.max_child_processes)
                ],
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='fail exhausted compaction')
            _grant_context_compressor_authority(runtime, pid)
            _seed_context_entries(runtime, pid, count=40)

            failed = None
            for _ in range(80):
                result = runtime.run_next_process_once()
                if (
                    isinstance(result, dict)
                    and result.get('action', {}).get('action') == 'compact_process_context'
                    and result.get('result', {}).get('ok') is False
                ):
                    failed = result
                    break

            assert failed is not None
            assert 'exhausted child process budget' in (failed['result']['error'] or '')
            job = runtime.store.get_object_by_name(
                f'context_compaction_job:{pid}',
                namespace=runtime.memory.resolve_namespace(pid),
            )
            assert job is not None
            assert job.payload['status'] == 'failed'
            assert 'exhausted child process budget' in job.payload['error']
            assert len(runtime.process.list_children(pid)) == runtime.config.process.max_child_processes
        finally:
            runtime.close()

    def test_compact_process_context_version_race_does_not_overwrite_new_context(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'force': True,
                    'target_tokens': 512,
                    'max_chunks': 1,
                },
                {'action': 'process_exit', 'payload': _compact_summary('stale summary')},
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='race context')
            _grant_context_compressor_authority(runtime, pid)

            waiting = runtime.run_next_process_once()
            assert waiting['waiting_event']
            context = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            assert context is not None
            handle = runtime.memory.handle_for_name(
                pid,
                context_object_name(pid),
                rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
            )
            changed_payload = json.loads(json.dumps(context.payload))
            changed_payload['entries'].append({'kind': 'external_update', 'value': 'must survive'})
            runtime.memory.update_object(pid, handle, ObjectPatch(payload=changed_payload))

            results = runtime.run_until_idle(max_quanta=2)

            completed = _last_action_result(results, 'compact_process_context')
            assert completed['result']['ok'] is False
            assert 'changed during compaction' in (completed['result']['error'] or '')
            after = runtime.store.get_object(context.oid)
            assert after is not None
            assert after.payload['entries'][-1] == {'kind': 'external_update', 'value': 'must survive'}
            assert not any(entry.get('kind') == 'context_compacted' for entry in after.payload['entries'])
        finally:
            runtime.close()

    def test_compact_process_context_uses_multiple_chunks(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {
                    'action': 'compact_process_context',
                    'force': True,
                    'target_tokens': 512,
                    'max_chunks': 2,
                    'preserve_recent_entries': 0,
                },
                {'action': 'process_exit', 'payload': _compact_summary('stage one')},
                {'action': 'process_exit', 'payload': _compact_summary('stage two')},
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='multi chunk context')
            _grant_context_compressor_authority(runtime, pid)

            results = runtime.run_until_idle(max_quanta=5)

            completed = _last_action_result(results, 'compact_process_context')
            assert completed['result']['ok']
            output = completed['result']['payload']
            assert output['compacted'] is True
            assert len(output['compressor_pids']) == 2
            context = runtime.store.get_object_by_name(context_object_name(pid), namespace=runtime.memory.resolve_namespace(pid))
            assert context is not None
            summary_goal = context.payload['entries'][0]['summary']['goal']
            assert summary_goal == ['stage one', 'stage two']
            assert context.payload['entries'][0]['compaction_metadata']['stage_count'] == 2
        finally:
            runtime.close()

    def test_pending_context_compaction_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = RecordingActionClient([
                    {
                        'action': 'compact_process_context',
                        'force': True,
                        'target_tokens': 512,
                        'max_chunks': 1,
                    }
                ])
                pid = runtime.process.spawn(image='base-agent:v0', goal='reopen compaction')
                _grant_context_compressor_authority(runtime, pid)
                waiting = runtime.run_next_process_once()
                assert waiting['waiting_event']
                assert runtime.store.get_llm_pending_action(pid)['wait_type'] == 'child'
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                reopened.llm.client = RecordingActionClient([
                    {'action': 'process_exit', 'payload': _compact_summary('reopened state')},
                ])
                results = reopened.run_until_idle(max_quanta=2)
                completed = _last_action_result(results, 'compact_process_context')
                assert completed['result']['ok']
                context = reopened.store.get_object_by_name(context_object_name(pid), namespace=reopened.memory.resolve_namespace(pid))
                assert context is not None
                assert context.payload['entries'][0]['summary']['goal'] == 'reopened state'
                assert reopened.store.get_llm_pending_action(pid)['status'] == 'completed'
            finally:
                reopened.close()

    def test_reopen_after_compressor_exit_reruns_missing_result_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = RecordingActionClient([
                    {
                        'action': 'compact_process_context',
                        'force': True,
                        'target_tokens': 512,
                        'max_chunks': 1,
                    },
                    {'action': 'process_exit', 'payload': _compact_summary('lost result')},
                ])
                pid = runtime.process.spawn(image='base-agent:v0', goal='rerun missing child result')
                _grant_context_compressor_authority(runtime, pid)
                waiting = runtime.run_next_process_once()
                child_pid = waiting['child_pid']
                child_exit = runtime.run_process_once(child_pid)
                assert child_exit['result']['ok']
                assert runtime.store.get_llm_pending_action(pid)['status'] == 'pending'
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                reopened.llm.client = RecordingActionClient([
                    {'action': 'process_exit', 'payload': _compact_summary('rerun result')},
                ])
                results = reopened.run_until_idle(max_quanta=3)
                completed = _last_action_result(results, 'compact_process_context')
                assert completed['result']['ok']
                output = completed['result']['payload']
                assert len(output['compressor_pids']) == 2
                context = reopened.store.get_object_by_name(context_object_name(pid), namespace=reopened.memory.resolve_namespace(pid))
                assert context is not None
                assert context.payload['entries'][0]['summary']['goal'] == 'rerun result'
                assert context.payload['entries'][0]['compaction_metadata']['discarded_compressor_pids']
            finally:
                reopened.close()

def _register_multiplexed_image(
    runtime: Runtime,
    *,
    prompt_mode: str | None = None,
) -> None:
    runtime.register_image(
        AgentImage(
            image_id='multiplexed-jit:v0',
            name='multiplexed-jit',
            system_prompt='Use run_jit_tool for JIT tools.',
            prompt_mode=prompt_mode or 'image_only',
            default_tools=['process_exit'],
            jit_tool_exposure=JIT_TOOL_EXPOSURE_MULTIPLEXED,
        ),
        actor='test',
    )


def _register_count_tool(
    runtime: Runtime,
    pid: str,
    name: str,
    *,
    input_schema: dict[str, Any] | None = None,
) -> None:
    candidate = runtime.tools.propose(
        pid,
        {
            'name': name,
            'description': 'Count characters in text.',
            'input_schema': input_schema
            or {'type': 'object', 'properties': {'text': {'type': 'string'}}},
            'output_schema': {'type': 'object'},
        },
        source_code=COUNT_CHARS_SOURCE,
        tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
    )
    assert runtime.tools.validate(candidate).ok
    runtime.tools.register(pid, candidate)


class MetadataActionClient:

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        return LLMCompletion(content='visible assistant text', tool_calls=[{'id': 'tool_123', 'name': 'process_exit', 'arguments': json.dumps({'payload': {'done': True}})}], raw=SimpleNamespace(id='raw_resp', provider='fake'), api='chat', response_id='resp_123', request_id='req_123', model='test-model', usage={'prompt_tokens': 13, 'completion_tokens': 4, 'total_tokens': 17}, reasoning={'summary': 'selected process_exit'})


class ExplodingClient:

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        raise AssertionError('LLM client should not be called when the process image is missing')


def _compact_summary(goal: str) -> dict[str, Any]:
    return {
        'goal': goal,
        'constraints': ['preserve exact ids'],
        'user_preferences': [],
        'completed': [],
        'pending': ['continue from compacted state'],
        'key_references': {},
        'recent_decisions': [],
        'risks': [],
        'uncertainties': [],
        'next_steps': ['resume caller process'],
    }


def _last_action_result(results: list[Any], action: str) -> dict[str, Any]:
    for result in reversed(results):
        if isinstance(result, dict) and result.get('action', {}).get('action') == action:
            return result
    raise AssertionError(f'action result not found: {action}')


def _seed_context_entries(runtime: Runtime, pid: str, *, count: int) -> Any:
    process = runtime.process.get(pid)
    handle = runtime.llm.context_memory.ensure(
        pid,
        runtime.images[process.image_id],
        process,
        runtime.tools.visible_tools(pid),
    )
    context = runtime.memory.get_object(pid, handle)
    payload = json.loads(json.dumps(context.payload))
    payload['entries'].extend({'kind': 'seed_entry', 'index': index} for index in range(count))
    write_handle = runtime.memory.handle_for_name(
        pid,
        context_object_name(pid),
        rights={ObjectRight.READ.value, ObjectRight.WRITE.value},
    )
    updated = runtime.memory.update_object(pid, write_handle, ObjectPatch(payload=payload))
    return runtime.memory.get_object(pid, updated)
