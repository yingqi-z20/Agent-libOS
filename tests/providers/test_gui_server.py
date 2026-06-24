from __future__ import annotations
import pytest
import http.client
import json
import tempfile
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any
from agent_libos.api.gui.server import create_gui_http_server
from agent_libos.config import AgentLibOSConfig, DEFAULT_CONFIG, GuiDefaults, RuntimeDefaults
from agent_libos.models import CapabilityRight, EventType, ObjectMetadata, ObjectType
from agent_libos.runtime.runtime import Runtime
from tests.support.skills import write_skill_package

class TestGuiServer:

    def setup_method(self) -> None:
        self.server = create_gui_http_server(db='local', port=0, token='test-token', auto_run=False)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def teardown_method(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.service.shutdown()
        self.server.server_close()

    def request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        *,
        token: str = 'test-token',
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        headers = {'Authorization': f'Bearer {token}'}
        headers.update(extra_headers or {})
        payload = None
        if body is not None:
            payload = json.dumps(body).encode('utf-8')
            headers['Content-Type'] = 'application/json'
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded = json.loads(data.decode('utf-8')) if data else None
        return (response.status, decoded)

    def request_raw(
        self,
        method: str,
        path: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        headers = {'Authorization': 'Bearer test-token'}
        headers.update(extra_headers or {})
        conn.request(method, path, headers=headers)
        response = conn.getresponse()
        data = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
        conn.close()
        return response.status, response_headers, data

    def test_auth_health_snapshot_and_process_flow(self) -> None:
        status, _body = self.request('GET', '/api/health', token='wrong')
        assert status == 401
        status, health = self.request('GET', '/api/health')
        assert status == 200
        assert health['ok']
        assert not health['scheduler']['auto_run']
        assert health['scheduler']['default_max_quanta'] is None
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'inspect README', 'auto_run': False, 'llm_profile': 'gui-spawn'},
        )
        assert status == 200
        pid = spawned['pid']
        assert spawned['process']['llm_profile_id'] == 'gui-spawn'
        status, message = self.request('POST', f'/api/processes/{pid}/message', {'body': 'hello', 'auto_run': False})
        assert status == 200
        assert message['message']['body'] == 'hello'
        status, interrupt = self.request('POST', f'/api/processes/{pid}/interrupt', {'body': 'stop', 'auto_run': False})
        assert status == 200
        assert interrupt['message']['kind'] == 'interrupt'
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        assert len(snapshot['processes']) == 1
        assert snapshot['processes'][0]['llm_profile_id'] == 'gui-spawn'
        assert snapshot['processes'][0]['unread_message_count'] >= 2
        assert 'tools' in snapshot
        assert 'images' in snapshot
        assert any((image['image_id'] == 'base-agent:v0' for image in snapshot['images']))

    def test_encoded_route_segments_are_decoded(self) -> None:
        status, inspected = self.request('GET', '/api/images/base-agent%3Av0')

        assert status == 200
        assert inspected['image']['image_id'] == 'base-agent:v0'

    def test_cors_is_limited_to_local_gui_origins(self) -> None:
        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'http://127.0.0.1:5173'},
        )
        assert status == 204
        assert headers['access-control-allow-origin'] == 'http://127.0.0.1:5173'

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'https://example.test'},
        )
        assert status == 204
        assert 'access-control-allow-origin' not in headers

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'null'},
        )
        assert status == 204
        assert 'access-control-allow-origin' not in headers

    def test_sse_replays_snapshot_event(self) -> None:
        request = urllib.request.Request(f'http://{self.host}:{self.port}/api/events/stream?cursor=0', headers={'Authorization': 'Bearer test-token'})
        with urllib.request.urlopen(request, timeout=10) as response:
            assert response.status == 200
            frame_lines: list[str] = []
            while len(frame_lines) < 3:
                line = response.readline().decode('utf-8').strip()
                if line:
                    frame_lines.append(line)
            assert frame_lines[0].startswith('id: ')
            assert frame_lines[1] == 'event: snapshot'
            assert frame_lines[2].startswith('data: ')

    def test_snapshot_audit_window_contains_latest_records(self) -> None:
        for index in range(205):
            self.server.service.runtime.audit.record(
                actor='test',
                action=f'audit.window.{index}',
                target='process:audit-window',
            )

        status, snapshot = self.request('GET', '/api/snapshot')
        actions = [record['action'] for record in snapshot['audit']]

        assert status == 200
        assert 'audit.window.204' in actions
        assert 'audit.window.0' not in actions

    def test_snapshot_truncates_model_amplified_event_payloads(self) -> None:
        huge = 'x' * (self.server.service.runtime.config.gui.snapshot_string_max_chars + 100)
        self.server.service.runtime.events.emit(
            EventType.EXTERNAL_WRITE,
            source='gui-test',
            target='gui-test',
            payload={'blob': huge},
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        serialized = json.dumps(snapshot)
        assert huge not in serialized
        event = snapshot['events'][-1]
        assert event['payload']['blob']['truncated'] is True
        assert len(event['payload']['blob']['preview']) == self.server.service.runtime.config.gui.snapshot_string_max_chars

    def test_process_audit_filters_before_limit(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'audit target', 'auto_run': False})
        pid = spawned['pid']
        self.server.service.runtime.audit.record(
            actor=pid,
            action='process.audit.target',
            target=f'process:{pid}',
        )
        for index in range(205):
            self.server.service.runtime.audit.record(
                actor='noise',
                action=f'process.audit.noise.{index}',
                target='process:noise',
            )

        status, records = self.request('GET', f'/api/processes/{pid}/audit?limit=1')

        assert status == 200
        assert [record['action'] for record in records] == ['process.audit.target']

    def test_high_risk_exec_requires_confirmation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'goal', 'auto_run': False})
        pid = spawned['pid']
        status, denied = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'image': 'base-agent:v0', 'goal': 'new', 'llm_profile': 'gui-exec'},
        )
        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['preview']['llm_profile'] == 'gui-exec'
        status, string_confirmed = self.request('POST', f'/api/processes/{pid}/exec', {'image': 'base-agent:v0', 'goal': 'new', 'confirmed': 'true'})
        assert status == 409
        assert string_confirmed['error']['confirmation_required']
        status, allowed = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {
                'image': 'base-agent:v0',
                'goal': 'new',
                'confirmed': True,
                'auto_run': False,
                'llm_profile': 'gui-exec',
            },
        )
        assert status == 200
        assert allowed['process']['image_id'] == 'base-agent:v0'
        assert allowed['process']['llm_profile_id'] == 'gui-exec'

    def test_high_risk_image_commit_requires_confirmation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'commit source', 'auto_run': False})
        pid = spawned['pid']
        status, created = self.request('POST', '/api/checkpoints/create', {'pid': pid, 'reason': 'commit'})
        assert status == 200
        status, denied = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed'})
        assert status == 409
        assert denied['error']['confirmation_required']
        status, forbidden = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed', 'actor': pid, 'confirmed': True})
        assert status == 403
        assert 'lacks write' in forbidden['error']['message']
        status, committed = self.request('POST', '/api/images/commit', {'checkpoint_id': created['checkpoint_id'], 'image_id': 'gui-committed:v0', 'name': 'gui-committed', 'confirmed': True})
        assert status == 200
        assert committed['image_id'] == 'gui-committed:v0'
        status, inspected = self.request('GET', '/api/images/gui-committed:v0')
        assert status == 200
        assert inspected['image']['boot']['kind'] == 'checkpoint_commit'

    def test_checkpoint_actor_mode_enforces_restore_capability(self) -> None:
        _status, owner = self.request('POST', '/api/processes', {'goal': 'checkpoint owner', 'auto_run': False})
        _status, other = self.request('POST', '/api/processes', {'goal': 'unprivileged actor', 'auto_run': False})
        status, created = self.request('POST', '/api/checkpoints/create', {'pid': owner['pid'], 'reason': 'admin checkpoint'})
        assert status == 200
        status, body = self.request(
            'POST',
            f"/api/checkpoints/{created['checkpoint_id']}/restore",
            {'actor': other['pid'], 'confirmed': True},
        )
        assert status == 403
        assert 'checkpoint' in body['error']['message']

    def test_capability_actor_mode_enforces_process_authority(self) -> None:
        runtime = self.server.service.runtime
        _status, actor = self.request('POST', '/api/processes', {'goal': 'capability actor', 'auto_run': False})
        _status, subject = self.request('POST', '/api/processes', {'goal': 'capability subject', 'auto_run': False})

        status, denied = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-actor-grant',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'lacks grant/admin authority' in denied['error']['message']

        status, admin_granted = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-admin-grant',
                'rights': ['read'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert admin_granted['subject'] == subject['pid']

        runtime.capability.grant(actor['pid'], 'object:gui-actor-grant', [CapabilityRight.READ], issued_by='test')
        runtime.capability.grant(actor['pid'], 'object:gui-actor-grant', [CapabilityRight.GRANT], issued_by='test')
        status, granted = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-actor-grant',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert granted['subject'] == subject['pid']
        assert granted['parent_cap_id']

        runtime.capability.grant(actor['pid'], 'object:gui-delegate', [CapabilityRight.READ], issued_by='test', delegable=True)
        status, mismatched_parent = self.request(
            'POST',
            '/api/capabilities/delegate',
            {
                'parent': subject['pid'],
                'child': actor['pid'],
                'resource': 'object:gui-delegate',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'actor-mode delegation' in mismatched_parent['error']['message']

        status, delegated = self.request(
            'POST',
            '/api/capabilities/delegate',
            {
                'parent': actor['pid'],
                'child': subject['pid'],
                'resource': 'object:gui-delegate',
                'rights': ['read'],
                'actor': actor['pid'],
                'confirmed': True,
            },
        )
        assert status == 200
        assert delegated['subject'] == subject['pid']

        cap = runtime.capability.grant(subject['pid'], 'object:gui-revoke', [CapabilityRight.READ], issued_by='test')
        status, revoke_denied = self.request(
            'POST',
            f"/api/capabilities/{cap.cap_id}/revoke",
            {'actor': actor['pid'], 'confirmed': True},
        )
        assert status == 403
        assert 'lacks revoke/admin authority' in revoke_denied['error']['message']

        runtime.capability.grant(actor['pid'], 'object:gui-revoke', [CapabilityRight.REVOKE], issued_by='test')
        status, revoked = self.request(
            'POST',
            f"/api/capabilities/{cap.cap_id}/revoke",
            {'actor': actor['pid'], 'confirmed': True},
        )
        assert status == 200
        assert revoked['status'] == 'revoked'

    def test_image_register_accepts_package_files_and_rejects_host_file_path(self) -> None:
        files = _gui_image_package_files()
        status, denied = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent'})
        assert status == 409
        assert denied['error']['confirmation_required']
        status, string_confirmed = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent', 'confirmed': 'true'})
        assert status == 409
        assert string_confirmed['error']['confirmation_required']
        status, path_rejected = self.request('POST', '/api/images/register', {'path': 'image-package', 'confirmed': True})
        assert status == 400
        assert 'package files' in path_rejected['error']['message']
        status, registered = self.request('POST', '/api/images/register', {'files': files, 'source': 'gui-package-agent', 'confirmed': True})
        assert status == 200
        assert registered['image_id'] == 'gui-package-agent:v0'
        assert registered['boot']['kind'] == 'image_package'
        status, listed = self.request('GET', '/api/images')
        assert status == 200
        assert 'gui-package-agent:v0' in {item['image_id'] for item in listed}

    def test_scheduler_requests_are_serialized(self) -> None:
        first_status, first = self.request('POST', '/api/processes', {'goal': 'goal', 'auto_run': False})
        assert first_status == 200
        pid = first['pid']
        self.server.service.scheduler.running = True
        status, duplicate = self.request('POST', f'/api/processes/{pid}/run', {'max_quanta': 1})
        assert status == 200
        assert duplicate['running']
        self.server.service.scheduler.running = False

    def test_process_run_targets_selected_process(self) -> None:
        _first_status, first = self.request('POST', '/api/processes', {'goal': 'first', 'auto_run': False})
        _second_status, second = self.request('POST', '/api/processes', {'goal': 'second', 'auto_run': False})
        seen: list[str] = []
        seen_event = threading.Event()

        async def fake_quantum(pid: str) -> dict[str, str]:
            seen.append(pid)
            self.server.service.runtime.process.pause(pid, 'fake quantum completed')
            seen_event.set()
            return {'pid': pid}
        self.server.service.runtime.arun_process_once = fake_quantum
        status, body = self.request('POST', f"/api/processes/{second['pid']}/run", {'max_quanta': 1})
        assert seen_event.wait(timeout=2.0)
        assert status == 200
        assert body['reason'] == f"run:{second['pid']}"
        assert seen == [second['pid']]
        records = self.server.service.runtime.audit.trace()
        assert not any((record.target == f"process:{first['pid']}" and record.action == 'scheduler.run_quantum' for record in records))
        assert any((record.target == f"process:{second['pid']}" and record.action == 'scheduler.run_quantum' for record in records))

    def test_workflow_run_endpoint_returns_result_and_snapshot_process(self) -> None:
        status, result = self.request('POST', '/api/workflows/run', {'tool': 'get_working_directory', 'args': {}})

        assert status == 200
        assert result['ok'] is True
        assert result['tool'] == 'get_working_directory'
        assert result['status'] == 'exited'
        assert result['result_oid'] is not None
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        processes = {process['pid']: process for process in snapshot['processes']}
        assert processes[result['pid']]['status'] == 'exited'

    def test_object_task_endpoint_runs_task_and_exposes_snapshot(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'object task', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        self.server.service.runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )

        status, started = self.request(
            'POST',
            '/api/object-tasks/start',
            {
                'pid': pid,
                'owner_oid': owner.oid,
                'tool': 'get_working_directory',
                'args': {},
                'owner_watch': True,
                'watch_events': ['updated'],
                'watch_channel': 'owner-watch',
            },
        )
        assert status == 200
        assert started['owner_watch']['enabled'] is True
        assert started['owner_watch']['events'] == ['updated']
        assert started['owner_watch']['channel'] == 'owner-watch'
        status, waited = self.request('POST', f"/api/object-tasks/{started['task_id']}/wait", {'pid': pid, 'timeout_s': 2})
        assert status == 200
        assert waited['status'] == 'succeeded'
        assert waited['result_oid'] is not None
        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        assert any(
            task['task_id'] == started['task_id']
            and task['status'] == 'succeeded'
            and task['owner_watch']['enabled'] is True
            for task in snapshot['object_tasks']
        )

    def test_object_task_watch_owner_endpoint_updates_existing_task(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'object task watch', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        self.server.service.runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )
        status, started = self.request(
            'POST',
            '/api/object-tasks/start',
            {'pid': pid, 'owner_oid': owner.oid, 'tool': 'receive_process_messages', 'args': {'channel': 'owner-watch'}},
        )
        assert status == 200
        status, waited = self.request('POST', f"/api/object-tasks/{started['task_id']}/wait", {'pid': pid, 'timeout_s': 2})
        assert status == 200
        assert waited['status'] == 'waiting_message'

        status, watched = self.request(
            'POST',
            f"/api/object-tasks/{started['task_id']}/watch-owner",
            {
                'pid': pid,
                'enabled': True,
                'watch_events': ['updated'],
                'watch_channel': 'owner-watch',
                'watch_kind': 'interrupt',
            },
        )

        assert status == 200
        assert watched['owner_watch']['enabled'] is True
        assert watched['owner_watch']['events'] == ['updated']
        assert watched['owner_watch']['channel'] == 'owner-watch'
        assert watched['owner_watch']['kind'] == 'interrupt'

    def test_object_task_start_rejects_invalid_watch_kind_as_bad_request(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'bad watch kind', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']
        owner = self.server.service.runtime.memory.create_object(
            pid,
            ObjectType.ARTIFACT,
            {'name': 'owner'},
            metadata=ObjectMetadata(title='owner'),
            immutable=False,
        )

        status, body = self.request(
            'POST',
            '/api/object-tasks/start',
            {
                'pid': pid,
                'owner_oid': owner.oid,
                'tool': 'get_working_directory',
                'args': {},
                'owner_watch': True,
                'watch_kind': 'bad-kind',
            },
        )

        assert status == 400
        assert 'owner watch kind' in body['error']['message']

    def test_object_task_wait_uses_bounded_timeout(self) -> None:
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        self.server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]

        status, body = self.request('POST', '/api/object-tasks/task-1/wait', {'pid': 'pid-1'})
        assert status == 200
        assert body['timeout'] == DEFAULT_CONFIG.gui.object_task_wait_default_timeout_s
        assert seen == [DEFAULT_CONFIG.gui.object_task_wait_default_timeout_s]

        status, body = self.request('POST', '/api/object-tasks/task-1/wait', {'timeout_s': 'nan'})
        assert status == 400
        assert 'finite' in body['error']['message']

        status, body = self.request(
            'POST',
            '/api/object-tasks/task-1/wait',
            {'timeout_s': DEFAULT_CONFIG.gui.object_task_wait_max_timeout_s + 1},
        )
        assert status == 400
        assert 'at most' in body['error']['message']

    def test_injected_runtime_config_controls_spawn_and_wait_defaults(self) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(default_image_id='gui-base:v0', coding_image_id='gui-coding:v0'),
            gui=replace(DEFAULT_CONFIG.gui, object_task_wait_default_timeout_s=0.25, object_task_wait_max_timeout_s=0.5),
        )
        runtime = Runtime.open(config=config)
        server = create_gui_http_server(runtime=runtime, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]
        thread.start()
        try:
            status, spawned = _request_to_server(server, 'POST', '/api/processes', {'goal': 'custom', 'auto_run': False}, token='custom-token')
            assert status == 200
            assert spawned['process']['image_id'] == 'gui-base:v0'

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'pid': spawned['pid']}, token='custom-token')
            assert status == 200
            assert body['timeout'] == 0.25
            assert seen == [0.25]

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'timeout_s': 0.75}, token='custom-token')
            assert status == 400
            assert '0.5 seconds' in body['error']['message']
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_injected_runtime_config_controls_request_body_limit(self) -> None:
        runtime = Runtime.open(config=AgentLibOSConfig(gui=GuiDefaults(request_body_max_bytes=8)))
        server = create_gui_http_server(runtime=runtime, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, body = _request_to_server(server, 'POST', '/api/scheduler/auto', {'enabled': True}, token='custom-token')
            assert status == 413
            assert '8 bytes' in body['error']['message']
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_jsonrpc_register_rejects_host_file_path(self) -> None:
        status, body = self.request('POST', '/api/jsonrpc/register', {'path': 'secrets.yaml', 'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_jsonrpc_register_requires_manifest_text(self) -> None:
        status, body = self.request('POST', '/api/jsonrpc/register', {'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_jsonrpc_register_actor_mode_requires_endpoint_write_capability(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'jsonrpc actor', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_jsonrpc_manifest('gui-actor-jsonrpc')

        status, denied = self.request(
            'POST',
            '/api/jsonrpc/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 403
        assert 'jsonrpc_endpoint:gui-actor-jsonrpc' in denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'jsonrpc_endpoint:gui-actor-jsonrpc',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        status, registered = self.request(
            'POST',
            '/api/jsonrpc/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 200
        assert registered['endpoint_id'] == 'gui-actor-jsonrpc'

    def test_skill_register_actor_mode_requires_skill_write_capability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'gui-actor-skill', allowed_tools=['echo'])
            _status, spawned = self.request('POST', '/api/processes', {'goal': 'skill actor', 'auto_run': False})
            pid = spawned['pid']

            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': str(skill_dir), 'actor': pid, 'confirmed': True},
            )

            assert status == 409
            assert denied['error']['type'] == 'HumanApprovalRequired'
            assert denied['error']['request_id']

            self.server.service.runtime.capability.grant(
                pid,
                'skill:gui-actor-skill',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            status, registered = self.request(
                'POST',
                '/api/skills/register',
                {'path': str(skill_dir), 'actor': pid, 'confirmed': True},
            )

            assert status == 200
            assert registered['skill_id'] == 'gui-actor-skill'

    def test_human_request_respond_rejects_non_pending_request(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='gui human conflict')
        request_id = runtime.human.ask(pid, 'Approve once?', blocking=True)
        status, approved = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'yes', 'auto_run': False},
        )
        status_again, conflict = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'again', 'auto_run': False},
        )

        assert status == 200
        assert approved['request']['status'] == 'approved'
        assert status_again == 409
        assert 'not pending' in conflict['error']['message']

    def test_invalid_max_quanta_is_rejected(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'goal', 'max_quanta': 0})
        assert status == 400
        assert 'max_quanta' in body['error']['message']

    def test_request_body_size_is_bounded(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'x' * 1100000})
        assert status == 413
        assert 'exceeds' in body['error']['message']

    def test_shutdown_endpoint_stops_http_server(self) -> None:
        try:
            status, body = self.request('POST', '/api/shutdown', {})
            assert status == 200
            assert body['status'] == 'shutting_down'
        except ConnectionResetError:
            pass
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        self.server.service.shutdown()


def _request_to_server(
    server: Any,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    token: str,
) -> tuple[int, Any]:
    host, port = server.server_address
    conn = http.client.HTTPConnection(host, port, timeout=10)
    headers = {'Authorization': f'Bearer {token}'}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    conn.request(method, path, body=payload, headers=headers)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    decoded = json.loads(data.decode('utf-8')) if data else None
    return response.status, decoded


def _gui_image_package_files() -> dict[str, str]:
    return {
        "IMAGE.yaml": """
image_id: gui-package-agent:v0
name: gui-package-agent
version: v0
prompt: prompt.md
default_tools:
  - human_output
""".lstrip(),
        "prompt.md": "Registered from GUI package files.\n",
    }


def _gui_jsonrpc_manifest(endpoint_id: str) -> str:
    return f"""
schema_version: 1
endpoint_id: {endpoint_id}
url: https://api.example.test/jsonrpc
methods:
  - method_id: echo
    rpc_method: echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
""".lstrip()
