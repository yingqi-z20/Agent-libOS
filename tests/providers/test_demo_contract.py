from __future__ import annotations
import pytest
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.api.cli import DEMO_PATCH_PREVIEW_CONTENT, DEMO_PATCH_PREVIEW_PATH, run_demo
from tests.support.public_errors import assert_public_error_message

class TestDemoContract:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_run_demo_returns_auditable_contract(self) -> None:
        result = run_demo(self.runtime)
        assert result['root'].startswith('pid_')
        assert result['worker'].startswith('pid_')
        assert result['checkpoint'].startswith('ckpt_')
        assert result['final_report_oid'].startswith('obj_')
        assert result['approval_request'] is not None
        assert result['audit_records'] > 0
        assert not result['filesystem_write_denial']['ok']
        assert_public_error_message(
            result['filesystem_write_denial']['error'],
            code='permission_denied',
            error_type='CapabilityDenied',
            forbidden=('lacks write', DEMO_PATCH_PREVIEW_PATH),
        )
        assert result['write_result']['ok']
        assert result['write_result']['payload']['path'] == DEMO_PATCH_PREVIEW_PATH
        assert result['target_file_exists']
        assert result['target_file_content_matches']
        target = self.runtime.workspace_root / DEMO_PATCH_PREVIEW_PATH
        assert target.read_text(encoding='utf-8') == DEMO_PATCH_PREVIEW_CONTENT
        tool_names = [entry['tool'] for entry in result['tool_sequence']]
        assert 'parse_pytest_log' in tool_names
        if result['jit_validation_ok']:
            assert 'extract_failed_tests' in tool_names
        else:
            assert result['jit_validation_errors']
        assert tool_names.count('write_text_file') >= 2
        report = self.runtime.store.get_object(result['final_report_oid'])
        assert report is not None
        assert report is not None
        payload = report.payload
        assert payload['problem']['failed_test'] == 'tests/test_math.py::test_add'
        assert payload['authorization']['filesystem_write_approval_request'] == result['approval_request']
        assert not payload['authorization']['filesystem_write_denied_before_grant']['ok']
        assert payload['external_side_effects'][0]['path'] == DEMO_PATCH_PREVIEW_PATH
        assert payload['target_file']['content_matches']
        assert 'not a production automatic repair system' in payload['limits']
        audit_actions = [record.action for record in self.runtime.audit.trace()]
        for action in ['checkpoint.create', 'human.query', 'human.response', 'primitive.filesystem.write_text', 'tool.call', 'process.exit']:
            assert action in audit_actions
        event_types = [event.type.value for event in self.runtime.events.list()]
        assert 'external_write' in event_types
        assert 'human_query' in event_types
        assert 'human_response' in event_types

    def test_tool_outside_process_tool_table_is_denied_without_human_approval(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='write a demo file')
        path = f'agent_outputs/demo_missing_tool_{uuid4().hex}.txt'
        result = self.runtime.tools.call(pid, 'write_text_file', {'path': path, 'content': 'denied'})
        assert not result.ok
        assert 'not in process tool table' in (result.error or '')
        assert not (self.runtime.workspace_root / path).exists()
        assert 'human.query' not in [record.action for record in self.runtime.audit.trace()]
