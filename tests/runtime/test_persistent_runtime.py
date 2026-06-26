from __future__ import annotations
import pytest
import tempfile
from pathlib import Path
from agent_libos import Runtime
from agent_libos.models import ToolHandle, ToolSpec
from tests.support.deno import COUNT_CHARS_SOURCE

class TestPersistentRuntime:

    def test_static_tool_ids_survive_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                pid = runtime.process.spawn(image='review-agent:v0', goal='persistent tool table')
                tool_id = runtime.process.get(pid).tool_table['get_current_time']
                assert runtime.tools.call(pid, 'get_current_time', {}).ok
            finally:
                runtime.close()
            reopened = Runtime.open(db_path)
            try:
                resolved = reopened.tools.resolve('get_current_time', pid=pid)
                result = reopened.tools.call(pid, 'get_current_time', {})
                assert resolved.tool_id == tool_id
                assert result.ok, result.error
                assert len([row for row in reopened.tools.list() if row['name'] == 'get_current_time']) == 1
            finally:
                reopened.close()

    @pytest.mark.real_deno
    def test_registered_jit_tool_source_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='persistent jit tool')
                candidate_id = runtime.tools.propose(
                    pid,
                    {
                        'name': 'persistent_count_chars',
                        'description': 'Count text characters.',
                        'input_schema': {
                            'type': 'object',
                            'properties': {'text': {'type': 'string'}},
                        },
                        'output_schema': {'type': 'object'},
                    },
                    source_code=COUNT_CHARS_SOURCE,
                    tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
                )
                assert runtime.tools.validate(candidate_id, pid=pid).ok
                handle = runtime.tools.register(pid, candidate_id)
                candidate = runtime.store.get_tool_candidate(candidate_id)
                assert candidate is not None
                assert candidate.registered_tool_id == handle.tool_id
                assert 'persistent_count_chars' in _schema_names(runtime, pid)
            finally:
                runtime.close()

            reopened = Runtime.open(db_path)
            try:
                assert reopened.tools.resolve('persistent_count_chars', pid=pid).tool_id == handle.tool_id
                assert 'persistent_count_chars' in _schema_names(reopened, pid)
                result = reopened.tools.call(pid, 'persistent_count_chars', {'text': 'hello'})
                assert result.ok, result.error
                assert result.payload == {'count': 5}
                records = [record for record in reopened.audit.trace() if record.action == 'runtime.jit.rehydrate']
                assert records
                assert records[-1].decision['restored'][0]['name'] == 'persistent_count_chars'
            finally:
                reopened.close()

    def test_stale_ephemeral_jit_reference_is_pruned_on_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / 'runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='stale jit')
                handle = ToolHandle(
                    tool_id='tool_stale_jit',
                    name='stale_jit',
                    capability_id=None,
                    scope='ephemeral_process',
                )
                runtime.store.insert_tool(
                    handle,
                    ToolSpec(
                        name='stale_jit',
                        description='Missing JIT implementation.',
                        input_schema={'type': 'object'},
                        output_schema={'type': 'object'},
                    ),
                    registered_by='test',
                    created_at='2026-01-01T00:00:00Z',
                    ephemeral=True,
                )
                process = runtime.process.get(pid)
                process.tool_table['stale_jit'] = handle.tool_id
                runtime.store.update_process(process)
            finally:
                runtime.close()

            reopened = Runtime.open(db_path)
            try:
                process = reopened.process.get(pid)
                assert 'stale_jit' not in process.tool_table
                assert 'stale_jit' not in {row['name'] for row in reopened.tools.visible_tools(pid)}
                records = [record for record in reopened.audit.trace() if record.action == 'runtime.jit.rehydrate']
                assert records
                assert records[-1].decision['pruned_stale'] == [
                    {'pid': pid, 'tool_id': 'tool_stale_jit', 'name': 'stale_jit'}
                ]
            finally:
                reopened.close()


def _schema_names(runtime: Runtime, pid: str) -> set[str]:
    return {schema['function']['name'] for schema in runtime.tools.openai_tool_schemas(pid)}
