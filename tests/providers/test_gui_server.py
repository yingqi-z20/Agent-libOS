from __future__ import annotations
import asyncio
import pytest
import http.client
import json
import tempfile
import threading
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any
from jsonschema import Draft202012Validator
from agent_libos.api.gui.server import (
    GuiEventBroadcaster,
    GuiRuntimeService,
    GuiServerError,
    _BoundedSeenKeys,
    _shutdown_gui_service_before_exit,
    _sse_payload_data,
    create_gui_http_server,
)
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import (
    AgentLibOSConfig,
    DEFAULT_CONFIG,
    GuiDefaults,
    LLMProfile,
    RuntimeDefaults,
)
from agent_libos.models import (
    AuditRecord,
    CapabilityRight,
    Event,
    EventPriority,
    EventType,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequest,
    HumanRequestStatus,
    ObjectMetadata,
    ObjectPatch,
    ObjectType,
    McpProviderTool,
    McpToolListResult,
    ProcessSignal,
    ProcessStatus,
    SinkTrustLevel,
    SinkTrustRule,
    process_outcome_to_mapping,
    process_wait_state_to_mapping,
)
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, HumanResponseRequired, ProcessWaitRequired, ValidationError
from agent_libos.utils.ids import utc_now
from agent_libos.utils.serde import to_jsonable
from agent_libos.runtime.runtime import Runtime
from agent_libos.runtime.syscalls import LibOSSyscallSession
from tests.support.checkpoints import checkpoint_cli_json
from tests.support.skills import write_skill_package


def test_gui_validates_user_profiles_before_opening_owned_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = tmp_path / "invalid-profiles.json"
    profiles.write_text("not-json", encoding="utf-8")
    opened = False
    original_open = Runtime.open

    def record_open(*args: object, **kwargs: object) -> Runtime:
        nonlocal opened
        opened = True
        return original_open("local")

    monkeypatch.setattr("agent_libos.api.gui.server.Runtime.open", record_open)

    with pytest.raises(ValidationError, match="invalid LLM profiles JSON"):
        GuiRuntimeService(db="local", auto_run=False, llm_profiles_file=profiles)

    assert opened is False


def test_gui_closes_owned_runtime_when_profile_registration_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[Runtime] = []
    original_open = Runtime.open

    def record_open(*args: object, **kwargs: object) -> Runtime:
        runtime = original_open("local")
        opened.append(runtime)
        return runtime

    def fail_registration(*args: object, **kwargs: object) -> dict[str, Any]:
        raise RuntimeError("injected profile registration failure")

    monkeypatch.setattr("agent_libos.api.gui.server.Runtime.open", record_open)
    monkeypatch.setattr(GuiRuntimeService, "_register_user_llm_profiles", fail_registration)

    with pytest.raises(RuntimeError, match="injected profile registration failure"):
        GuiRuntimeService(
            db="local",
            auto_run=False,
            llm_profiles_file=tmp_path / "missing-profiles.json",
        )

    assert len(opened) == 1
    assert opened[0].lifecycle.closed is True


def test_gui_rejects_mismatched_config_for_borrowed_runtime(
    tmp_path: Path,
) -> None:
    runtime_config = replace(
        DEFAULT_CONFIG,
        llm=replace(
            DEFAULT_CONFIG.llm,
            profiles={
                **DEFAULT_CONFIG.llm.profiles,
                "runtime-owned": LLMProfile(model="runtime-model"),
            },
        ),
    )
    runtime = Runtime.open("local", config=runtime_config)
    profiles = tmp_path / "llm-profiles.json"
    profiles.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profiles": {
                    "runtime-owned": {
                        "model": "shadow-model",
                        "api_key_env": "SHADOW_API_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        with pytest.raises(ValidationError, match="must match the supplied Runtime config"):
            GuiRuntimeService(
                runtime=runtime,
                config=DEFAULT_CONFIG,
                auto_run=False,
                llm_profiles_file=profiles,
            )
        assert runtime.llms.profile("runtime-owned").model == "runtime-model"
    finally:
        runtime.shutdown(actor="test", reason="test.cleanup")

class TestGuiServer:

    def setup_method(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.llm_profiles_file = Path(self.temp_dir.name) / 'llm-profiles.json'
        self.server = create_gui_http_server(
            db='local',
            port=0,
            token='test-token',
            auto_run=False,
            llm_profiles_file=self.llm_profiles_file,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address

    def teardown_method(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.service.shutdown()
        self.server.server_close()
        self.temp_dir.cleanup()

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

    def request_json_text(self, method: str, path: str, raw: str) -> tuple[int, Any]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
        conn.request(
            method,
            path,
            body=raw.encode('utf-8'),
            headers={'Authorization': 'Bearer test-token', 'Content-Type': 'application/json'},
        )
        response = conn.getresponse()
        data = response.read()
        conn.close()
        decoded = json.loads(data.decode('utf-8')) if data else None
        return response.status, decoded

    def test_auth_health_snapshot_and_process_flow(self) -> None:
        status, _body = self.request('GET', '/api/health', token='wrong')
        assert status == 401
        status, health = self.request('GET', '/api/health')
        assert status == 200
        assert health['ok']
        assert not health['scheduler']['auto_run']
        assert health['scheduler']['default_max_quanta'] is None
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'gui-spawn', 'model': 'gui-spawn-model', 'api_key_env': 'GUI_SPAWN_API_KEY'},
        )
        assert status == 200
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
        schema = json.loads((Path(__file__).resolve().parents[2] / 'docs' / 'gui_api_schema.json').read_text(encoding='utf-8'))
        Draft202012Validator({**schema, '$ref': '#/$defs/snapshotResponse'}).validate(snapshot)
        assert len(snapshot['processes']) == 1
        assert snapshot['processes'][0]['llm_profile_id'] == 'gui-spawn'
        assert snapshot['processes'][0]['unread_message_count'] >= 2
        assert 'tools' in snapshot
        assert 'images' in snapshot
        assert any((profile['profile_id'] == 'gui-spawn' for profile in snapshot['llm_profiles']))
        assert any((image['image_id'] == 'base-agent:v0' for image in snapshot['images']))

    def test_process_handlers_preserve_typed_wait_and_outcome_discriminators(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        status, spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "typed GUI state", "auto_run": False},
        )
        assert status == 200
        waiting_pid = spawned["pid"]
        runtime.process.pause(waiting_pid, "review")

        status, listed = self.request("GET", "/api/processes")
        assert status == 200
        waiting = next(process for process in listed if process["pid"] == waiting_pid)
        assert waiting["wait_state"]["schema_version"] == 1
        assert waiting["wait_state"]["kind"] == "paused"
        assert waiting["outcome"] is None

        monkeypatch.setattr(
            runtime,
            "set_process_working_directory",
            lambda *args, **kwargs: runtime.process.get(waiting_pid),
        )
        status, changed_directory = self.request(
            "POST",
            f"/api/processes/{waiting_pid}/cd",
            {"path": "."},
        )
        assert status == 200
        assert changed_directory["wait_state"]["schema_version"] == 1
        assert changed_directory["wait_state"]["kind"] == "paused"

        monkeypatch.setattr(
            runtime,
            "exec_process",
            lambda *args, **kwargs: runtime.process.get(waiting_pid),
        )
        status, executed = self.request(
            "POST",
            f"/api/processes/{waiting_pid}/exec",
            {
                "image": "base-agent:v0",
                "goal": "typed exec response",
                "confirmed": True,
                "auto_run": False,
            },
        )
        assert status == 200
        assert executed["process"]["wait_state"]["schema_version"] == 1
        assert executed["process"]["wait_state"]["kind"] == "paused"

        status, terminal_spawned = self.request(
            "POST",
            "/api/processes",
            {"goal": "typed GUI outcome", "auto_run": False},
        )
        assert status == 200
        terminal_pid = terminal_spawned["pid"]
        runtime.process.exit(terminal_pid, message="done")
        status, terminal = self.request("GET", f"/api/processes/{terminal_pid}")
        assert status == 200
        assert terminal["wait_state"] is None
        assert terminal["outcome"] == {
            "schema_version": 1,
            "kind": "exited",
            "result_oid": terminal["outcome"]["result_oid"],
        }
        assert terminal["outcome"]["result_oid"]

    def test_operation_list_detail_and_evidence_resolution_endpoints(self) -> None:
        status, created = self.request(
            'POST',
            '/api/processes',
            {'goal': 'explain endpoint', 'image': 'base-agent:v0', 'auto_run': False},
        )
        assert status == 200
        pid = created['pid']

        status, listed = self.request('GET', f'/api/operations?pid={pid}&limit=100')
        assert status == 200
        operation = next(item for item in listed['operations'] if item['name'] == 'process.spawn')
        status, explained = self.request('GET', f"/api/operations/{operation['operation_id']}")
        assert status == 200
        assert explained['root']['operation_id'] == operation['operation_id']
        assert explained['evidence_complete'] is True
        audit = next(item for item in explained['evidence'] if item['evidence_type'] == 'audit')

        status, resolved = self.request(
            'GET',
            f"/api/operations/resolve?kind=audit&id={audit['evidence_id']}",
        )
        assert status == 200
        assert resolved['root']['operation_id'] == operation['operation_id']
        status, missing = self.request('GET', '/api/operations/op_missing')
        assert status == 404
        assert missing['error']['type'] == 'NotFound'

        runtime = self.server.service.runtime
        first = runtime.operations.start(kind='runtime', name='first', actor=pid, pid=pid)
        second = runtime.operations.start(kind='runtime', name='second', actor=pid, pid=pid)
        runtime.operations.link_evidence('audit', 'shared-http-audit', 'audit', operation_id=first.operation_id)
        runtime.operations.link_evidence('audit', 'shared-http-audit', 'audit', operation_id=second.operation_id)
        runtime.operations.finish('succeeded', operation_id=first.operation_id)
        runtime.operations.finish('succeeded', operation_id=second.operation_id)
        status, ambiguous = self.request(
            'GET',
            '/api/operations/resolve?kind=audit&id=shared-http-audit',
        )
        assert status == 409
        assert set(ambiguous['error']['candidates']) == {first.operation_id, second.operation_id}

    def test_snapshot_keeps_new_pending_human_request_ahead_of_bounded_history(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='pending must stay visible')
        now = utc_now()
        for index in range(runtime.config.gui.snapshot_collection_max_items + 1):
            runtime.store.insert_human_request(
                HumanRequest(
                    request_id=f'hreq_history_{index:04d}',
                    pid=pid,
                    human='owner',
                    payload={'type': 'question', 'question': f'history {index}'},
                    status=HumanRequestStatus.REJECTED,
                    decision={'approved': False},
                    blocking=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        pending_id = 'hreq_pending_latest'
        runtime.store.insert_human_request(
            HumanRequest(
                request_id=pending_id,
                pid=pid,
                human='owner',
                payload={'type': 'question', 'question': 'must remain visible'},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at=now,
                updated_at=now,
            )
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert pending_id in {request['request_id'] for request in snapshot['human_requests']}

    def test_repeated_snapshot_reuses_unchanged_human_presentation_evidence(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='reuse an unchanged GUI presentation',
        )
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {'type': 'question', 'question': 'UNCHANGED_GUI_PRESENTATION'},
            blocking=False,
        )

        first = self.server.service.snapshot()
        effects_after_first = runtime.store.list_external_effects(pid=pid)
        events_after_first = len(runtime.events.list())
        audit_after_first = len(runtime.audit.trace())
        decisions_after_first = len(runtime.store.list_data_flow_decisions(pid=pid))

        second = self.server.service.snapshot()

        assert next(
            item for item in first['human_requests'] if item['request_id'] == request_id
        ) == next(
            item for item in second['human_requests'] if item['request_id'] == request_id
        )
        presentation_effects = [
            effect
            for effect in effects_after_first
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]
        assert len(presentation_effects) == 1
        assert runtime.store.list_external_effects(pid=pid) == effects_after_first
        assert len(runtime.events.list()) == events_after_first
        assert len(runtime.audit.trace()) == audit_after_first
        assert len(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_after_first

    def test_delivered_output_snapshot_remains_visible_after_source_mutation_and_exit(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='present a frozen output after the process advances',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.PROCESS_STATE,
            {'step': 1},
            metadata=ObjectMetadata(sensitivity='normal'),
            immutable=False,
        )
        delivered: list[str] = []
        runtime.human.provider.output_sink = delivered.append
        sentinel = 'GUI_DELIVERED_OUTPUT_SNAPSHOT_SENTINEL'

        output = runtime.human.output(
            pid,
            sentinel,
            source_oids=[source.oid],
        )
        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={'step': 2}),
            expected_version=1,
        )
        runtime.process.exit(pid, message='advance after Human output')

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert delivered == [sentinel]
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['message'] == sentinel
        assert '_agent_libos_output_snapshot_sha256' not in presented['payload']
        assert runtime.human.is_request_withheld_for_presentation(
            output['request_id'],
            presentation='gui',
        ) is False

    def test_delivered_output_snapshot_digest_mismatch_fails_closed(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='reject a mutated delivered output snapshot',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        runtime.human.provider.output_sink = lambda _message: None
        output = runtime.human.output(pid, 'ORIGINAL_OUTPUT_SNAPSHOT')
        request = runtime.human.get(output['request_id'])
        request.payload = dict(request.payload)
        request.payload['message'] = 'MUTATED_OUTPUT_SNAPSHOT_SENTINEL'
        runtime.human.requests.update(request)

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        encoded = json.dumps(snapshot, sort_keys=True)
        assert 'MUTATED_OUTPUT_SNAPSHOT_SENTINEL' not in encoded
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['release_required'] is True
        assert runtime.human.is_request_withheld_for_presentation(
            output['request_id'],
            presentation='gui',
        ) is True

    def test_delivered_output_snapshot_still_obeys_current_sink_clearance(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='retain labels on a frozen output snapshot',
        )
        runtime.capability.grant(
            pid,
            runtime.config.runtime.default_human_resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        pattern = f'human:{human}:{channel}'
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=pattern,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='secret',
            ),
            actor='test.host',
            replace=True,
            require_capability=False,
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_FROZEN_SECRET_OUTPUT_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.human.provider.output_sink = lambda _message: None
        output = runtime.human.output(
            pid,
            'GUI_FROZEN_SECRET_OUTPUT_SENTINEL',
            source_oids=[source.oid],
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=pattern,
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='normal',
            ),
            actor='test.host',
            replace=True,
            require_capability=False,
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        encoded = json.dumps(snapshot, sort_keys=True)
        assert 'GUI_FROZEN_SECRET_OUTPUT_SENTINEL' not in encoded
        presented = next(
            item
            for item in snapshot['human_requests']
            if item['request_id'] == output['request_id']
        )
        assert presented['payload']['release_required'] is True

    def test_new_gui_presentation_session_does_not_reuse_an_old_receipt(self) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='bind GUI presentation receipts to one service session',
        )
        request_id = runtime.human.query(
            pid,
            runtime.config.runtime.default_human,
            {'type': 'question', 'question': 'SESSION_BOUND_GUI_PRESENTATION'},
            blocking=False,
        )

        service.snapshot()
        first_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]
        old_provider = service._human_presentation_provider
        new_provider = type(old_provider)()
        old_key = ('gui', request_id, id(old_provider))
        receipts = runtime.human.presentation._receipts
        old_receipt = receipts[old_key]
        # Deterministically simulate a recycled object-id key.  Receipt
        # identity, not just the integer key, must bind the old session.
        receipts[
            ('gui', request_id, id(new_provider))
        ] = old_receipt
        service._human_presentation_provider = new_provider

        service.snapshot()
        second_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose')
            == 'gui_presentation'
        ]

        assert len(first_effects) == 1
        assert len(second_effects) == 2

    def test_cached_gui_projection_linearizes_with_sink_policy_changes(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='linearize cached GUI presentation policy',
        )
        sentinel = 'GUI_CACHED_POLICY_LINEARIZATION_SENTINEL'
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': sentinel},
            metadata=ObjectMetadata(sensitivity='normal'),
        )
        human = runtime.config.runtime.default_human
        pattern = f'human:{human}:{runtime.config.runtime.terminal_channel}'

        def register(max_sensitivity: str) -> None:
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity=max_sensitivity,
                ),
                actor='test.host',
                replace=True,
                require_capability=False,
            )

        register('normal')
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': sentinel},
            source_oids=[source.oid],
            blocking=False,
        )
        assert sentinel in json.dumps(service.snapshot(), sort_keys=True)

        original_receipt_check = runtime.human._presentation_was_delivered
        receipt_checked = threading.Event()
        allow_cached_return = threading.Event()
        blocked_once = False

        def checked_receipt(*args: object, **kwargs: object) -> bool:
            nonlocal blocked_once
            delivered = original_receipt_check(*args, **kwargs)
            selected = args[0] if args else None
            if (
                isinstance(selected, HumanRequest)
                and selected.request_id == request_id
                and not blocked_once
            ):
                blocked_once = True
                receipt_checked.set()
                assert allow_cached_return.wait(timeout=5)
            return delivered

        monkeypatch.setattr(
            runtime.human,
            '_presentation_was_delivered',
            checked_receipt,
        )
        snapshot_box: list[dict[str, Any]] = []
        snapshot_thread = threading.Thread(
            target=lambda: snapshot_box.append(service.snapshot()),
            daemon=True,
        )
        snapshot_thread.start()
        assert receipt_checked.wait(timeout=5)

        mutation_started = threading.Event()
        mutation_done = threading.Event()

        def downgrade_sink() -> None:
            mutation_started.set()
            register('public')
            mutation_done.set()

        mutation_thread = threading.Thread(target=downgrade_sink, daemon=True)
        mutation_thread.start()
        assert mutation_started.wait(timeout=5)
        assert mutation_done.wait(timeout=0.1) is False

        allow_cached_return.set()
        snapshot_thread.join(timeout=5)
        mutation_thread.join(timeout=5)

        assert snapshot_thread.is_alive() is False
        assert mutation_thread.is_alive() is False
        assert sentinel in json.dumps(snapshot_box[0], sort_keys=True)
        assert sentinel not in json.dumps(service.snapshot(), sort_keys=True)

    def test_human_presentation_evidence_does_not_starve_snapshot_causal_windows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='preserve causal snapshot markers',
        )
        marker_event = runtime.events.emit(
            EventType.PROCESS_CREATED,
            source=pid,
            target=pid,
            payload={'marker': 'before-gui-presentation-burst'},
        )
        marker_audit = runtime.audit.record(
            actor=pid,
            action='test.gui.causal_marker',
            target=f'process:{pid}',
            decision={'marker': 'before-gui-presentation-burst'},
        )
        old_event_window = (
            runtime.config.gui.snapshot_event_limit
            + runtime.config.gui.snapshot_collection_max_items * 8
        )
        old_audit_window = (
            runtime.config.gui.snapshot_audit_limit
            + runtime.config.gui.snapshot_collection_max_items * 8
        )
        burst_size = max(old_event_window, old_audit_window) * 2 + 1
        burst_timestamp = '9999-12-31T23:59:59+00:00'
        for index in range(burst_size):
            runtime.store.insert_event(
                Event(
                    event_id=f'evt_gui_presentation_burst_{index:05d}',
                    type=EventType.HUMAN_OUTPUT,
                    source='human:owner',
                    target=pid,
                    payload={'purpose': 'gui_presentation'},
                    priority=EventPriority.NORMAL,
                    created_at=burst_timestamp,
                )
            )
            runtime.store.insert_audit(
                AuditRecord(
                    record_id=f'audit_gui_presentation_burst_{index:05d}',
                    timestamp=burst_timestamp,
                    actor='human:owner',
                    action='human.output',
                    target=f'human:{runtime.config.runtime.default_human}:terminal',
                    input_refs=[],
                    output_refs=[],
                    capability_refs=[],
                    decision={'purpose': 'gui_presentation'},
                    correlation_id=None,
                )
            )

        event_queries: list[dict[str, Any]] = []
        audit_queries: list[dict[str, Any]] = []
        original_list_events = runtime.store.list_events
        original_list_audit = runtime.store.list_audit

        def tracked_events(*args: Any, **kwargs: Any) -> list[Event]:
            event_queries.append(dict(kwargs))
            return original_list_events(*args, **kwargs)

        def tracked_audit(*args: Any, **kwargs: Any) -> list[AuditRecord]:
            audit_queries.append(dict(kwargs))
            return original_list_audit(*args, **kwargs)

        monkeypatch.setattr(runtime.store, 'list_events', tracked_events)
        monkeypatch.setattr(runtime.store, 'list_audit', tracked_audit)
        snapshot = self.server.service.snapshot()

        assert marker_event.event_id in {item['event_id'] for item in snapshot['events']}
        assert marker_audit.record_id in {item['record_id'] for item in snapshot['audit']}
        assert event_queries == [
            {
                'target': None,
                'limit': runtime.config.gui.snapshot_event_limit,
                'before_event_id': None,
                'after_event_id': None,
                'include_gui_presentation': False,
            }
        ]
        assert audit_queries == [
            {
                'limit': runtime.config.gui.snapshot_audit_limit,
                'actor': None,
                'target': None,
                'match_any': False,
                'include_gui_presentation': False,
            }
        ]

    def test_snapshot_completed_release_history_cannot_crowd_out_withheld_parent(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='pending Human release pair must stay visible',
        )
        human = runtime.config.runtime.default_human
        now = utc_now()
        limit = runtime.config.gui.snapshot_collection_max_items
        for index in range(limit + 2):
            runtime.store.insert_human_request(
                HumanRequest(
                    request_id=f'hreq_completed_release_{index:04d}',
                    pid=pid,
                    human=human,
                    payload={
                        'type': 'data_release_approval',
                        'question': f'completed release {index}',
                    },
                    status=HumanRequestStatus.APPROVED,
                    decision={'approved': True},
                    blocking=False,
                    created_at=now,
                    updated_at=now,
                )
            )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_PENDING_PAIR_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        parent_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_PENDING_PAIR_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        requests = snapshot['human_requests']
        assert len(requests) == limit
        assert 'GUI_PENDING_PAIR_SECRET_SENTINEL' not in json.dumps(
            requests,
            sort_keys=True,
        )
        parent = next(item for item in requests if item['request_id'] == parent_id)
        release = next(
            item
            for item in requests
            if item.get('release_for_request_id') == parent_id
            and item['status'] == HumanRequestStatus.PENDING.value
        )
        assert parent['payload']['release_required'] is True
        assert requests.index(release) < requests.index(parent)

    def test_presentation_window_does_not_release_a_cropped_approved_parent(self) -> None:
        service = self.server.service
        runtime = service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='cropped GUI parent must remain withheld',
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        first_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_FIRST_WINDOW_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        cropped_source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_CROPPED_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_FIRST_WINDOW_SECRET_SENTINEL'},
            source_oids=[first_source.oid],
        )
        cropped_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_CROPPED_SECRET_SENTINEL'},
            source_oids=[cropped_source.oid],
        )
        cropped_view = service.human_request_view(runtime.human.get(cropped_id))
        assert cropped_view['payload']['release_required'] is True
        cropped_release_id = cropped_view['release_request_id']
        cropped_release = runtime.human.approve(
            cropped_release_id,
            {'approved': True, 'source': 'test.gui'},
        )
        release_resource = cropped_release.payload['requested_once_capability']['resource']
        release_capability = next(
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == release_resource
        )
        assert release_capability.uses_remaining == 1
        cropped_effects_before = [
            effect.effect_id
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider_metadata.get('context', {}).get('request_id') == cropped_id
        ]

        views, has_more = runtime.human.list_for_presentation_window(
            presentation='gui',
            provider=service._human_presentation_provider,
            limit=2,
        )

        assert has_more is True
        assert cropped_id not in {
            request['request_id'] for request in views
        }
        assert len(views) == 2
        assert views[0].get('release_for_request_id') == views[1]['request_id']
        assert views[1]['payload']['release_required'] is True
        release_capability_after = next(
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.cap_id == release_capability.cap_id
        )
        assert release_capability_after.uses_remaining == 1
        assert [
            effect.effect_id
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider_metadata.get('context', {}).get('request_id') == cropped_id
        ] == cropped_effects_before
        assert runtime.human.is_request_withheld_for_presentation(
            cropped_id,
            presentation='gui',
        ) is True

        parent_before_denial = to_jsonable(runtime.human.get(cropped_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        denied_status, denied = self.request(
            'POST',
            f'/api/human-requests/{cropped_id}/respond',
            {'approved': True, 'answer': 'not presented', 'auto_run': False},
        )
        assert denied_status == 409
        assert 'not been released' in denied['error']['message']
        assert to_jsonable(runtime.human.get(cropped_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial

    def test_presentation_window_reports_pair_expansion_without_raw_lookahead(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='logical Human presentation expansion',
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_EXPANSION_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        first_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_EXPANSION_SECRET_SENTINEL'},
            source_oids=[source.oid],
        )
        second_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'ordinary second parent'},
        )

        views, has_more = runtime.human.list_for_presentation_window(
            presentation='gui',
            provider=self.server.service._human_presentation_provider,
            limit=2,
        )

        assert has_more is True
        assert [view.get('release_for_request_id') for view in views] == [first_id, None]
        assert views[1]['request_id'] == first_id
        assert second_id not in {view['request_id'] for view in views}

    def test_human_request_views_redact_conditional_payload_before_exact_release(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='redact conditional Human request')
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI DATA_FLOW_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI DATA_FLOW_SECRET_SENTINEL',
                'note': 'GUI_NOTE_SECRET_SENTINEL',
                'custom': {
                    'arbitrary_leaf': 'GUI_NESTED_SECRET_SENTINEL',
                },
            },
            source_oids=[source.oid],
        )

        status, snapshot = self.request('GET', '/api/snapshot')
        list_status, listed = self.request('GET', '/api/human-requests')
        process_status, process_list = self.request(
            'GET',
            f'/api/processes/{pid}/human-requests',
        )

        assert status == list_status == process_status == 200
        snapshot_encoded = json.dumps(snapshot, sort_keys=True)
        assert 'GUI DATA_FLOW_SECRET_SENTINEL' not in snapshot_encoded
        assert 'GUI_NOTE_SECRET_SENTINEL' not in snapshot_encoded
        assert 'GUI_NESTED_SECRET_SENTINEL' not in snapshot_encoded
        for projection in (snapshot['human_requests'], listed, process_list):
            encoded = json.dumps(projection, sort_keys=True)
            assert 'GUI DATA_FLOW_SECRET_SENTINEL' not in encoded
            assert 'GUI_NOTE_SECRET_SENTINEL' not in encoded
            assert 'GUI_NESTED_SECRET_SENTINEL' not in encoded
            parent = next(item for item in projection if item['request_id'] == request_id)
            assert parent['payload']['release_required'] is True
            assert parent['payload']['payload_observation']['redacted'] is True
            assert parent['payload']['payload_observation']['metadata_only'] is True
            release = next(
                item
                for item in projection
                if item['payload'].get('type') == 'data_release_approval'
            )
            assert release['payload']['context']['payload_sha256']
            assert release['release_for_request_id'] == request_id

        releases = [
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]
        assert len(releases) == 1
        release = releases[0]
        assert release.payload['context']['sink'] == f'human:{human}:gui'
        assert release.payload['context']['operation'] == 'human.gui.present'

        parent_before_denial = to_jsonable(runtime.human.get(request_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        requests_before_denial = to_jsonable(runtime.human.list(pid=pid))
        capabilities_before_denial = to_jsonable(
            runtime.store.list_capabilities(subject=pid)
        )
        decisions_before_denial = to_jsonable(
            runtime.store.list_data_flow_decisions(pid=pid)
        )
        withheld_status, withheld = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'too early', 'auto_run': False},
        )
        assert withheld_status == 409
        assert 'not been released' in withheld['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial
        assert to_jsonable(runtime.human.list(pid=pid)) == requests_before_denial
        assert to_jsonable(runtime.store.list_capabilities(subject=pid)) == capabilities_before_denial
        assert to_jsonable(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

        response_status, _response = self.request(
            'POST',
            f'/api/human-requests/{release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert response_status == 200

        released_status, released_snapshot = self.request('GET', '/api/snapshot')
        repeated_status, repeated_snapshot = self.request('GET', '/api/snapshot')
        assert released_status == repeated_status == 200
        for projection in (released_snapshot['human_requests'], repeated_snapshot['human_requests']):
            parent = next(item for item in projection if item['request_id'] == request_id)
            assert parent['payload']['question'] == 'GUI DATA_FLOW_SECRET_SENTINEL'
            assert parent['payload']['note'] == 'GUI_NOTE_SECRET_SENTINEL'
            assert parent['payload']['custom']['arbitrary_leaf'] == 'GUI_NESTED_SECRET_SENTINEL'
        assert len([
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]) == 1
        release_caps = [
            capability
            for capability in runtime.store.list_capabilities(subject=pid)
            if capability.resource == release.payload['requested_once_capability']['resource']
        ]
        assert len(release_caps) == 1
        assert release_caps[0].uses_remaining == 0
        presentation_effects = [
            effect
            for effect in runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human'
            and effect.provider_metadata.get('context', {}).get('purpose') == 'gui_presentation'
        ]
        assert len(presentation_effects) == 1
        presentation_flow = presentation_effects[0].provider_metadata['data_flow']
        assert presentation_flow['sink'] == f'human:{human}:gui'
        assert presentation_flow['release_capability_id'] == release_caps[0].cap_id

        answered_status, answered = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'released answer', 'auto_run': False},
        )
        assert answered_status == 200
        assert answered['request']['status'] == 'approved'
        assert 'answer' not in answered['request']['decision']
        assert answered['request']['payload']['release_required'] is True
        decision_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != release.request_id
        )
        assert decision_release.status == HumanRequestStatus.PENDING
        runtime.human.approve(
            decision_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        decision_status, decision_snapshot = self.request('GET', '/api/snapshot')
        assert decision_status == 200
        decision_parent = next(
            item
            for item in decision_snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert decision_parent['decision']['answer'] == 'released answer'
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_gui_presentation_release_does_not_suspend_runnable_process(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='keep runnable during GUI presentation release',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_PRESENTATION_RELEASE_SECRET'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        channel = runtime.config.runtime.terminal_channel
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_PRESENTATION_RELEASE_SECRET',
            },
            blocking=False,
            source_oids=[source.oid],
        )
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        parent = next(
            item for item in snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert parent['payload']['release_required'] is True
        release = next(
            item for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
        )
        assert release.payload['_agent_libos_data_release_presentation'] == 'gui'
        assert release.blocking is False
        assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_gui_conditional_release_survives_reopen_without_duplicate(self) -> None:
        db_path = Path(self.temp_dir.name) / 'gui-human-release.sqlite'
        runtime = Runtime.open(str(db_path))
        service = GuiRuntimeService(
            runtime=runtime,
            token='reopen-one',
            auto_run=False,
        )
        pid = runtime.process.spawn(image='base-agent:v0', goal='reopen GUI release')
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_REOPEN_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{runtime.config.runtime.terminal_channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_REOPEN_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )
        first = service.snapshot()
        first_encoded = json.dumps(first['human_requests'], sort_keys=True)
        assert 'GUI_REOPEN_SECRET_SENTINEL' not in first_encoded
        release_ids = [
            item.request_id
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        ]
        assert len(release_ids) == 1
        release_id = release_ids[0]
        service.close()
        runtime.close()

        reopened = Runtime.open(str(db_path))
        reopened_server = create_gui_http_server(
            runtime=reopened,
            port=0,
            token='reopen-two',
            auto_run=False,
        )
        reopened_thread = threading.Thread(
            target=reopened_server.serve_forever,
            daemon=True,
        )
        reopened_thread.start()
        reopened_host, reopened_port = reopened_server.server_address

        def reopened_request(
            method: str,
            path: str,
            body: dict[str, Any] | None = None,
        ) -> tuple[int, Any]:
            conn = http.client.HTTPConnection(reopened_host, reopened_port, timeout=10)
            headers = {'Authorization': 'Bearer reopen-two'}
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

        try:
            reopened_status, reopened_snapshot = reopened_request('GET', '/api/snapshot')
            assert reopened_status == 200
            assert 'GUI_REOPEN_SECRET_SENTINEL' not in json.dumps(
                reopened_snapshot['human_requests'],
                sort_keys=True,
            )
            reopened_release_ids = [
                item.request_id
                for item in reopened.human.list()
                if item.payload.get('type') == 'data_release_approval'
            ]
            assert reopened_release_ids == [release_id]

            parent_before_denial = to_jsonable(reopened.human.get(request_id))
            process_before_denial = to_jsonable(reopened.process.get(pid))
            requests_before_denial = to_jsonable(reopened.human.list(pid=pid))
            capabilities_before_denial = to_jsonable(
                reopened.store.list_capabilities(subject=pid)
            )
            decisions_before_denial = to_jsonable(
                reopened.store.list_data_flow_decisions(pid=pid)
            )
            withheld_status, withheld = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'too early', 'auto_run': False},
            )
            assert withheld_status == 409
            assert 'not been released' in withheld['error']['message']
            assert to_jsonable(reopened.human.get(request_id)) == parent_before_denial
            assert to_jsonable(reopened.process.get(pid)) == process_before_denial
            assert to_jsonable(reopened.human.list(pid=pid)) == requests_before_denial
            assert to_jsonable(reopened.store.list_capabilities(subject=pid)) == capabilities_before_denial
            assert to_jsonable(reopened.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

            reopened.human.approve(
                release_id,
                {'approved': True, 'source': 'test.gui'},
            )

            parent_before_consumption = to_jsonable(reopened.human.get(request_id))
            process_before_consumption = to_jsonable(reopened.process.get(pid))
            requests_before_consumption = to_jsonable(reopened.human.list(pid=pid))
            capabilities_before_consumption = to_jsonable(
                reopened.store.list_capabilities(subject=pid)
            )
            decisions_before_consumption = to_jsonable(
                reopened.store.list_data_flow_decisions(pid=pid)
            )
            unconsumed_status, unconsumed = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'still too early', 'auto_run': False},
            )
            assert unconsumed_status == 409
            assert 'not been released' in unconsumed['error']['message']
            assert to_jsonable(reopened.human.get(request_id)) == parent_before_consumption
            assert to_jsonable(reopened.process.get(pid)) == process_before_consumption
            assert to_jsonable(reopened.human.list(pid=pid)) == requests_before_consumption
            assert to_jsonable(reopened.store.list_capabilities(subject=pid)) == capabilities_before_consumption
            assert to_jsonable(reopened.store.list_data_flow_decisions(pid=pid)) == decisions_before_consumption

            released_status, released = reopened_request('GET', '/api/snapshot')
            assert released_status == 200
            parent = next(
                item
                for item in released['human_requests']
                if item['request_id'] == request_id
            )
            assert parent['payload']['question'] == 'GUI_REOPEN_SECRET_SENTINEL'
            assert [
                item.request_id
                for item in reopened.human.list()
                if item.payload.get('type') == 'data_release_approval'
            ] == [release_id]

            answered_status, answered = reopened_request(
                'POST',
                f'/api/human-requests/{request_id}/respond',
                {'approved': True, 'answer': 'after reopen', 'auto_run': False},
            )
            assert answered_status == 200
            assert answered['request']['status'] == 'approved'
            assert 'answer' not in answered['request']['decision']
            assert answered['request']['payload']['release_required'] is True
            decision_release = next(
                item
                for item in reopened.human.list(pid=pid)
                if item.payload.get('type') == 'data_release_approval'
                and item.request_id != release_id
            )
            assert decision_release.status == HumanRequestStatus.PENDING
            reopened.human.approve(
                decision_release.request_id,
                {'approved': True, 'source': 'test.gui'},
            )
            decision_status, decision_snapshot = reopened_request(
                'GET',
                '/api/snapshot',
            )
            assert decision_status == 200
            decision_parent = next(
                item
                for item in decision_snapshot['human_requests']
                if item['request_id'] == request_id
            )
            assert decision_parent['decision']['answer'] == 'after reopen'
            assert reopened.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            reopened_server.shutdown()
            reopened_thread.join(timeout=5)
            reopened_server.service.shutdown()
            reopened_server.server_close()
            reopened.close()

    def test_gui_visible_release_is_invalidated_by_sink_registry_generation(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='invalidate stale GUI release',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_STALE_RELEASE_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
            immutable=False,
        )
        human = runtime.config.runtime.default_human
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=f'human:{human}:{runtime.config.runtime.terminal_channel}',
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {
                'type': 'question',
                'question': 'GUI_STALE_RELEASE_SECRET_SENTINEL',
            },
            source_oids=[source.oid],
        )

        initial_status, initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(initial, sort_keys=True)
        first_release = next(
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
        )
        approval_status, _ = self.request(
            'POST',
            f'/api/human-requests/{first_release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert approval_status == 200
        visible_status, visible = self.request('GET', '/api/snapshot')
        assert visible_status == 200
        visible_parent = next(
            item for item in visible['human_requests'] if item['request_id'] == request_id
        )
        assert visible_parent['payload']['question'] == 'GUI_STALE_RELEASE_SECRET_SENTINEL'

        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern='human:gui-release-generation-bump:terminal',
                trust_level=SinkTrustLevel.TRUSTED,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        parent_before_denial = to_jsonable(runtime.human.get(request_id))
        process_before_denial = to_jsonable(runtime.process.get(pid))
        requests_before_denial = to_jsonable(runtime.human.list(pid=pid))
        capabilities_before_denial = to_jsonable(
            runtime.store.list_capabilities(subject=pid)
        )
        decisions_before_denial = to_jsonable(
            runtime.store.list_data_flow_decisions(pid=pid)
        )
        stale_status, stale = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'stale release', 'auto_run': False},
        )
        assert stale_status == 409
        assert 'not been released' in stale['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_denial
        assert to_jsonable(runtime.human.list(pid=pid)) == requests_before_denial
        assert to_jsonable(runtime.store.list_capabilities(subject=pid)) == capabilities_before_denial
        assert to_jsonable(runtime.store.list_data_flow_decisions(pid=pid)) == decisions_before_denial

        redacted_status, redacted = self.request('GET', '/api/snapshot')
        assert redacted_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(redacted, sort_keys=True)
        redacted_parent = next(
            item for item in redacted['human_requests'] if item['request_id'] == request_id
        )
        assert redacted_parent['payload']['release_required'] is True
        second_release = next(
            item
            for item in runtime.human.list()
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != first_release.request_id
        )
        assert second_release.status == HumanRequestStatus.PENDING

        renewed_status, renewed = self.request(
            'POST',
            f'/api/human-requests/{second_release.request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        assert renewed_status == 200, renewed
        renewed_snapshot_status, renewed_snapshot = self.request('GET', '/api/snapshot')
        assert renewed_snapshot_status == 200
        renewed_parent = next(
            item
            for item in renewed_snapshot['human_requests']
            if item['request_id'] == request_id
        )
        assert renewed_parent['payload']['question'] == 'GUI_STALE_RELEASE_SECRET_SENTINEL'

        runtime.memory.update_object(
            pid,
            source,
            ObjectPatch(payload={'value': 'source changed after GUI release'}),
        )
        parent_before_source_denial = to_jsonable(runtime.human.get(request_id))
        process_before_source_denial = to_jsonable(runtime.process.get(pid))
        source_stale_status, source_stale = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'source is stale', 'auto_run': False},
        )
        assert source_stale_status == 409
        assert 'not been released' in source_stale['error']['message']
        assert to_jsonable(runtime.human.get(request_id)) == parent_before_source_denial
        assert to_jsonable(runtime.process.get(pid)) == process_before_source_denial

        source_redacted_status, source_redacted = self.request('GET', '/api/snapshot')
        assert source_redacted_status == 200
        assert 'GUI_STALE_RELEASE_SECRET_SENTINEL' not in json.dumps(
            source_redacted,
            sort_keys=True,
        )
        source_redacted_parent = next(
            item
            for item in source_redacted['human_requests']
            if item['request_id'] == request_id
        )
        assert source_redacted_parent['payload']['release_required'] is True

    @pytest.mark.parametrize(
        ('sensitivity', 'initial_max', 'downgraded_max'),
        [
            ('secret', 'secret', 'normal'),
            ('normal', 'normal', 'public'),
        ],
    )
    def test_gui_projection_revalidates_trusted_sink_clearance(
        self,
        sensitivity: str,
        initial_max: str,
        downgraded_max: str,
    ) -> None:
        runtime = self.server.service.runtime
        sentinel = f'GUI_TRUSTED_DOWNGRADE_{sensitivity.upper()}_SENTINEL'
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='revalidate trusted GUI projection clearance',
        )
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': sentinel},
            metadata=ObjectMetadata(sensitivity=sensitivity),
        )
        human = runtime.config.runtime.default_human
        pattern = f'human:{human}:{runtime.config.runtime.terminal_channel}'

        def register(max_sensitivity: str) -> None:
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern=pattern,
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity=max_sensitivity,
                ),
                actor='test.host',
                replace=True,
                require_capability=False,
            )

        register(initial_max)
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': sentinel},
            source_oids=[source.oid],
        )

        initial_status, initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        initial_parent = next(
            item for item in initial['human_requests'] if item['request_id'] == request_id
        )
        assert initial_parent['payload']['question'] == sentinel

        register(downgraded_max)
        decisions_before = {
            item.decision_id for item in runtime.store.list_data_flow_decisions(pid=pid)
        }
        downgraded_status, downgraded = self.request('GET', '/api/snapshot')

        assert downgraded_status == 200
        assert sentinel not in json.dumps(downgraded, sort_keys=True)
        downgraded_parent = next(
            item
            for item in downgraded['human_requests']
            if item['request_id'] == request_id
        )
        assert downgraded_parent['payload']['release_required'] is True
        denials = [
            item
            for item in runtime.store.list_data_flow_decisions(pid=pid, outcome='deny')
            if item.decision_id not in decisions_before
            and item.sink == f'human:{human}:gui'
        ]
        assert len(denials) == 1
        denial = denials[0]
        assert denial.labels.sensitivity.value == sensitivity
        assert f'exceeds Sink maximum {downgraded_max}' in denial.reason
        assert any(
            record.action == 'data_flow.egress'
            and record.target == f'human:{human}:gui'
            and record.decision.get('decision_id') == denial.decision_id
            and record.decision.get('outcome') == 'deny'
            for record in runtime.audit.trace()
        )
        assert any(
            event.type == EventType.DATA_FLOW_DECISION
            and event.payload.get('decision_id') == denial.decision_id
            and event.payload.get('outcome') == 'deny'
            for event in runtime.events.list(target=f'data_flow_sink:human:{human}:gui')
        )

        register(initial_max)
        restored_status, restored = self.request('GET', '/api/snapshot')
        assert restored_status == 200, restored
        restored_parent = next(
            item for item in restored['human_requests'] if item['request_id'] == request_id
        )
        assert restored_parent['payload']['question'] == sentinel
        repeated_status, repeated = self.request('GET', '/api/snapshot')
        assert repeated_status == 200, repeated
        repeated_parent = next(
            item for item in repeated['human_requests'] if item['request_id'] == request_id
        )
        assert repeated_parent['payload']['question'] == sentinel

    def test_gui_response_guard_and_decision_share_binding_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='serialize GUI presentation guard with Human decision',
        )
        human = runtime.config.runtime.default_human
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'},
            source_oids=[source.oid],
        )
        withheld = self.server.service.human_request_view(runtime.human.get(request_id))
        release_id = withheld['release_request_id']
        runtime.human.approve(release_id, {'approved': True, 'source': 'test.gui'})
        visible = self.server.service.human_request_view(runtime.human.get(request_id))
        assert visible['payload']['question'] == 'GUI_ATOMIC_GUARD_SECRET_SENTINEL'

        original_guard = runtime.human.is_request_withheld_for_presentation
        guard_checked = threading.Event()
        allow_decision = threading.Event()
        blocked_once = False

        def guarded(request: HumanRequest | str, *, presentation: str) -> bool:
            nonlocal blocked_once
            result = original_guard(request, presentation=presentation)
            selected_id = request.request_id if isinstance(request, HumanRequest) else request
            if selected_id == request_id and not blocked_once:
                blocked_once = True
                guard_checked.set()
                assert allow_decision.wait(timeout=5)
            return result

        monkeypatch.setattr(runtime.human, 'is_request_withheld_for_presentation', guarded)
        response_box: list[tuple[int, Any]] = []
        response_thread = threading.Thread(
            target=lambda: response_box.append(
                self.request(
                    'POST',
                    f'/api/human-requests/{request_id}/respond',
                    {'approved': True, 'answer': 'atomic answer', 'auto_run': False},
                )
            ),
            daemon=True,
        )
        response_thread.start()
        assert guard_checked.wait(timeout=5)

        mutation_started = threading.Event()
        mutation_done = threading.Event()
        status_seen_after_mutation: list[HumanRequestStatus] = []

        def mutate_registry() -> None:
            mutation_started.set()
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='human:gui-atomic-generation-bump:terminal',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                ),
                actor='test.host',
                require_capability=False,
            )
            status_seen_after_mutation.append(runtime.human.get(request_id).status)
            mutation_done.set()

        mutation_thread = threading.Thread(target=mutate_registry, daemon=True)
        mutation_thread.start()
        assert mutation_started.wait(timeout=5)
        assert mutation_done.wait(timeout=0.1) is False

        allow_decision.set()
        response_thread.join(timeout=5)
        mutation_thread.join(timeout=5)

        assert response_thread.is_alive() is False
        assert mutation_thread.is_alive() is False
        assert response_box[0][0] == 200
        assert status_seen_after_mutation == [HumanRequestStatus.APPROVED]
        assert runtime.human.get(request_id).decision['answer'] == 'atomic answer'

    def test_gui_release_binds_the_returned_decision_view(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='bind GUI release to the complete returned view',
        )
        human = runtime.config.runtime.default_human
        source = runtime.memory.create_object(
            pid,
            ObjectType.EVIDENCE,
            {'value': 'GUI_COMPLETE_VIEW_SECRET'},
            metadata=ObjectMetadata(sensitivity='secret'),
        )
        runtime.data_flow.register_sink_trust(
            SinkTrustRule(
                pattern=(
                    f'human:{human}:'
                    f'{runtime.config.runtime.terminal_channel}'
                ),
                trust_level=SinkTrustLevel.CONDITIONAL,
                max_sensitivity='secret',
            ),
            actor='test.host',
            require_capability=False,
        )
        request_id = runtime.human.query(
            pid,
            human,
            {'type': 'question', 'question': 'GUI_COMPLETE_VIEW_SECRET'},
            source_oids=[source.oid],
        )

        initial_status, _initial = self.request('GET', '/api/snapshot')
        assert initial_status == 200
        first_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
        )
        runtime.human.approve(
            first_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        visible_status, visible = self.request('GET', '/api/snapshot')
        assert visible_status == 200
        visible_parent = next(
            item for item in visible['human_requests'] if item['request_id'] == request_id
        )
        assert visible_parent['payload']['question'] == 'GUI_COMPLETE_VIEW_SECRET'
        decisions_before_response = len(
            runtime.store.list_data_flow_decisions(pid=pid)
        )

        answered_status, answered = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'approved': True,
                'answer': 'GUI_DECISION_SECRET_SENTINEL',
                'auto_run': False,
            },
        )

        assert answered_status == 200
        assert 'GUI_DECISION_SECRET_SENTINEL' not in json.dumps(
            answered['request'],
            sort_keys=True,
        )
        assert answered['request']['payload']['release_required'] is True
        assert len(runtime.store.list_data_flow_decisions(pid=pid)) > decisions_before_response
        second_release = next(
            item
            for item in runtime.human.list(pid=pid)
            if item.payload.get('type') == 'data_release_approval'
            and item.request_id != first_release.request_id
        )
        assert second_release.status == HumanRequestStatus.PENDING

        runtime.human.approve(
            second_release.request_id,
            {'approved': True, 'source': 'test.gui'},
        )
        released_status, released = self.request('GET', '/api/snapshot')
        assert released_status == 200
        released_parent = next(
            item for item in released['human_requests'] if item['request_id'] == request_id
        )
        assert released_parent['decision']['answer'] == 'GUI_DECISION_SECRET_SENTINEL'

    def test_snapshot_source_bounds_process_and_registry_reads(self, monkeypatch: pytest.MonkeyPatch) -> None:
        service = self.server.service
        runtime = service.runtime
        for index in range(17):
            runtime.process.spawn(image='base-agent:v0', goal=f'source-bound-{index}')
        runtime.config = replace(
            runtime.config,
            gui=replace(runtime.config.gui, snapshot_collection_max_items=16),
        )
        service._static_snapshot_dirty = True
        seen: dict[str, list[int | None]] = {}

        def spy_limit(owner: object, attribute: str, label: str) -> None:
            original = getattr(owner, attribute)

            def wrapped(*args: object, **kwargs: object) -> object:
                seen.setdefault(label, []).append(kwargs.get('limit'))
                return original(*args, **kwargs)

            monkeypatch.setattr(owner, attribute, wrapped)

        spy_limit(runtime.process, 'list', 'processes')
        spy_limit(runtime.human, 'list', 'human_requests')
        spy_limit(runtime.tools, 'list', 'tools')
        spy_limit(runtime.image_registry, 'list_images', 'images')
        spy_limit(runtime.skills, 'discover_skills_window', 'skills')
        spy_limit(runtime.jsonrpc, 'list_endpoints_window', 'jsonrpc_endpoints')
        spy_limit(runtime.mcp, 'list_servers_window', 'mcp_servers')
        spy_limit(runtime.modules, 'loaded_module_summaries', 'modules')
        spy_limit(service, '_llm_profile_summaries', 'llm_profiles')

        snapshot = service.snapshot()

        assert len(snapshot['processes']) == 16
        assert len(snapshot['tools']) == 16
        assert seen == {
            'processes': [17],
            'human_requests': [17],
            'tools': [17],
            'images': [17],
            'skills': [17],
            'jsonrpc_endpoints': [17],
            'mcp_servers': [17],
            'modules': [17],
            'llm_profiles': [17],
        }
        assert snapshot['_truncated']['processes']['source_limited'] is True
        assert snapshot['_truncated']['processes']['omitted_is_lower_bound'] is True
        assert snapshot['_truncated']['tools']['source_limited'] is True

    def test_snapshot_reports_truncation_at_stricter_jsonrpc_source_limit(self) -> None:
        service = self.server.service
        runtime = service.runtime
        runtime.config = replace(
            runtime.config,
            jsonrpc=replace(runtime.config.jsonrpc, list_limit=2),
        )
        runtime.jsonrpc.config = runtime.config
        for index in range(3):
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _gui_jsonrpc_manifest(f'gui-source-limit-{index}'),
                actor='test',
                require_capability=False,
            )
        service._static_snapshot_dirty = True

        snapshot = service.snapshot()

        assert len(snapshot['jsonrpc_endpoints']) == 2
        assert snapshot['_truncated']['jsonrpc_endpoints'] == {
            'kind': 'array',
            'returned': 2,
            'omitted': 1,
            'omitted_is_lower_bound': True,
            'source_limited': True,
        }

    def test_snapshot_batches_process_activity_rating_and_resource_queries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = self.server.service
        runtime = service.runtime
        pids = [runtime.process.spawn(image='base-agent:v0', goal=f'batch-{index}') for index in range(3)]
        for index, pid in enumerate(pids):
            runtime.messages.post(sender='gui-test', recipient_pid=pid, body=f'message-{index}')
        runtime.ratings.upsert(pids[0], score=5, comment='batched')

        calls = {'activity': 0, 'remaining': 0, 'ratings': 0}
        original_activity = runtime.store.get_process_activity_summaries
        original_remaining = runtime.resources.remaining_budgets
        original_ratings = runtime.ratings.get_many
        original_list_llm_calls = runtime.store.list_llm_calls

        def activity(*args: object, **kwargs: object) -> object:
            calls['activity'] += 1
            return original_activity(*args, **kwargs)

        def remaining(*args: object, **kwargs: object) -> object:
            calls['remaining'] += 1
            return original_remaining(*args, **kwargs)

        def ratings(*args: object, **kwargs: object) -> object:
            calls['ratings'] += 1
            return original_ratings(*args, **kwargs)

        def list_llm_calls(pid: str | None = None, limit: int | None = None) -> object:
            assert pid is None, 'snapshot must not load LLM call rows once per process'
            return original_list_llm_calls(pid=pid, limit=limit)

        def unexpected_single_process_query(*_args: object, **_kwargs: object) -> object:
            raise AssertionError('snapshot used a per-process manager query')

        monkeypatch.setattr(runtime.store, 'get_process_activity_summaries', activity)
        monkeypatch.setattr(runtime.resources, 'remaining_budgets', remaining)
        monkeypatch.setattr(runtime.ratings, 'get_many', ratings)
        monkeypatch.setattr(runtime.store, 'list_llm_calls', list_llm_calls)
        monkeypatch.setattr(runtime.messages, 'list', unexpected_single_process_query)
        monkeypatch.setattr(runtime.resources, 'remaining_budget', unexpected_single_process_query)
        monkeypatch.setattr(runtime.ratings, 'get', unexpected_single_process_query)

        snapshot = service.snapshot()

        assert calls == {'activity': 1, 'remaining': 1, 'ratings': 1}
        by_pid = {process['pid']: process for process in snapshot['processes']}
        assert set(by_pid) == set(pids)
        assert all(process['unread_message_count'] == 1 for process in by_pid.values())
        assert by_pid[pids[0]]['rating']['score'] == 5

    def test_snapshot_selects_recent_process_messages_at_the_source(self) -> None:
        service = self.server.service
        runtime = service.runtime
        runtime.config = replace(
            runtime.config,
            gui=replace(runtime.config.gui, snapshot_process_message_limit=2),
        )
        pid = runtime.process.spawn(image='base-agent:v0', goal='recent message window')
        for index in range(5):
            runtime.messages.post(sender='gui-test', recipient_pid=pid, body=f'message-{index}')

        snapshot = service.snapshot()
        process = next(item for item in snapshot['processes'] if item['pid'] == pid)

        assert process['unread_message_count'] == 5
        assert [message['body'] for message in process['messages']] == ['message-3', 'message-4']

    def test_llm_profile_endpoints_persist_user_profiles_and_reject_secrets(self, monkeypatch) -> None:
        monkeypatch.setenv('KIMI_API_KEY', 'secret')

        status, profiles = self.request('GET', '/api/llm-profiles')
        assert status == 200
        assert any(profile['profile_id'] == 'default' and profile['source'] == 'config' for profile in profiles)

        status, created = self.request(
            'POST',
            '/api/llm-profiles',
            {
                'profile_id': 'kimi-k2.7-code',
                'model': 'kimi-k2.7-code',
                'base_url': 'https://kimi.example/v1',
                'api_key_env': 'KIMI_API_KEY',
                'api_mode': 'chat',
                'timeout_s': 17.5,
                'max_retries': 4,
                'store': False,
                'temperature': 0.1,
                'reasoning_effort': 'high',
                'verbosity': 'low',
                'safety_identifier_env': 'OPENAI_SAFETY_IDENTIFIER',
                'prompt_cache_retention': '24h',
                'responses_previous_response_id': True,
                'parallel_tool_calls': False,
                'auto_wait_on_empty_tool_calls': True,
                'max_tokens': 2048,
                'context_window_tokens': 100000,
                'allow_custom_base_url': False,
            },
        )
        assert status == 200
        assert created['profile_id'] == 'kimi-k2.7-code'
        assert created['source'] == 'user'
        assert created['editable'] is True
        assert created['api_key_env_present'] is True
        assert created['timeout_s'] == 17.5
        assert created['max_retries'] == 4
        assert created['store'] is False
        assert created['reasoning_effort'] == 'high'
        assert created['verbosity'] == 'low'
        assert created['safety_identifier_env'] == 'OPENAI_SAFETY_IDENTIFIER'
        assert created['prompt_cache_retention'] == '24h'
        assert created['responses_previous_response_id'] is True
        assert created['parallel_tool_calls'] is False
        assert created['auto_wait_on_empty_tool_calls'] is True
        assert created['allow_custom_base_url'] is False

        omitted_fields = {
            'kind': 'openai_compatible',
            'model': 'kimi-k2.7-code',
            'base_url': 'https://kimi.example/v1',
            'api_key_env': 'KIMI_API_KEY',
            'api_mode': 'chat',
            'timeout_s': 17.5,
            'max_retries': 4,
            'store': False,
            'reasoning_effort': 'high',
            'verbosity': 'low',
            'safety_identifier_env': 'OPENAI_SAFETY_IDENTIFIER',
            'prompt_cache_retention': '24h',
            'responses_previous_response_id': True,
            'parallel_tool_calls': False,
            'auto_wait_on_empty_tool_calls': True,
            'temperature': 0.1,
        }
        updates = {
            'max_tokens': 4096,
            'context_window_tokens': 200000,
            'allow_custom_base_url': True,
        }
        before_update = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        assert all(getattr(before_update, field) != value for field, value in updates.items())
        status, updated = self.request(
            'PUT',
            '/api/llm-profiles/kimi-k2.7-code',
            updates,
        )
        assert status == 200
        assert updated['max_tokens'] == 4096
        assert updated['context_window_tokens'] == 200000
        assert updated['allow_custom_base_url'] is True
        assert updated['reasoning_effort'] == 'high'
        assert updated['verbosity'] == 'low'
        assert updated['safety_identifier_env'] == 'OPENAI_SAFETY_IDENTIFIER'
        assert updated['prompt_cache_retention'] == '24h'
        assert updated['responses_previous_response_id'] is True
        updated_profile = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        for field, expected in {**omitted_fields, **updates}.items():
            assert getattr(updated_profile, field) == expected, field
        persisted_profile = json.loads(self.llm_profiles_file.read_text(encoding='utf-8'))['profiles']['kimi-k2.7-code']
        for field, expected in {**omitted_fields, **updates}.items():
            if field != 'kind':
                assert persisted_profile[field] == expected, field
        assert 'secret' not in self.llm_profiles_file.read_text(encoding='utf-8')

        status, rejected = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'bad-secret', 'model': 'bad', 'api_key_env': 'BAD_API_KEY', 'api_key': 'secret'},
        )
        assert status == 400
        assert 'API keys are not accepted' in rejected['error']['message']

        self.server.service.shutdown()
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

        self.server = create_gui_http_server(
            db='local',
            port=0,
            token='test-token',
            auto_run=False,
            llm_profiles_file=self.llm_profiles_file,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        status, profiles = self.request('GET', '/api/llm-profiles')
        assert status == 200
        assert any(
            profile['profile_id'] == 'kimi-k2.7-code'
            and profile['max_tokens'] == 4096
            and profile['context_window_tokens'] == 200000
            and profile['timeout_s'] == 17.5
            and profile['max_retries'] == 4
            and profile['store'] is False
            and profile['reasoning_effort'] == 'high'
            and profile['verbosity'] == 'low'
            and profile['parallel_tool_calls'] is False
            and profile['auto_wait_on_empty_tool_calls'] is True
            for profile in profiles
        )
        reopened_profile = self.server.service.runtime.llms.profile('kimi-k2.7-code')
        for field, expected in {**omitted_fields, **updates}.items():
            assert getattr(reopened_profile, field) == expected, field
        reopened_persisted = json.loads(self.llm_profiles_file.read_text(encoding='utf-8'))['profiles']['kimi-k2.7-code']
        for field, expected in {**omitted_fields, **updates}.items():
            if field != 'kind':
                assert reopened_persisted[field] == expected, field

    def test_llm_profile_spawn_exec_validation_and_delete_in_use(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'bad profile', 'auto_run': False, 'llm_profile': 'missing'})
        assert status == 400
        assert 'unknown LLM profile' in body['error']['message']

        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'glm-5.2', 'model': 'glm-5.2', 'api_key_env': 'GLM_API_KEY'},
        )
        assert status == 200
        status, spawned = self.request('POST', '/api/processes', {'goal': 'profile', 'auto_run': False, 'llm_profile': 'glm-5.2'})
        assert status == 200
        pid = spawned['pid']
        status, body = self.request('DELETE', '/api/llm-profiles/glm-5.2')
        assert status == 409
        assert pid in body['error']['pids']

        status, bad_exec = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'image': 'base-agent:v0', 'goal': 'new', 'llm_profile': 'missing', 'confirmed': True},
        )
        assert status == 400
        assert 'unknown LLM profile' in bad_exec['error']['message']

        self.server.service.runtime.process.exit(pid, message='done')
        status, deleted = self.request('DELETE', '/api/llm-profiles/glm-5.2')
        assert status == 409
        assert deleted['error']['profile_id'] == 'glm-5.2'

    def test_process_spawn_authority_manifest_keeps_workspace_access_request_only(self) -> None:
        runtime = self.server.service.runtime
        subtree = 'agent_outputs/gui_authority'
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {
                'image': 'coding-agent:v0',
                'goal': 'edit one scoped result',
                'working_directory': subtree,
                'auto_run': False,
                'authority_manifest': {
                    'authorized_capabilities': [
                        {
                            'resource': 'human:owner',
                            'rights': ['write'],
                            'delegable': False,
                        }
                    ],
                    'approval_policy': {
                        'requestable_capabilities': [
                            {
                                'resource': f'filesystem:workspace:{subtree}/*',
                                'rights': ['read', 'write'],
                                'delegable': False,
                            }
                        ]
                    },
                },
            },
        )

        assert status == 200
        pid = spawned['pid']
        inside = f'filesystem:workspace:{subtree}/result.txt'
        outside = 'filesystem:workspace:agent_outputs/outside.txt'
        assert runtime.capability.check(pid, 'human:owner', CapabilityRight.WRITE)
        assert not runtime.capability.check(pid, inside, CapabilityRight.READ)
        assert not runtime.capability.check(pid, inside, CapabilityRight.WRITE)

        with pytest.raises(CapabilityDenied):
            runtime.shell.run(pid, ['python', '-m', 'pytest', '-q'])
        assert runtime.human.pending() == []

        denied = runtime.tools.call(
            pid,
            'request_permission',
            {'resource': outside, 'rights': ['write'], 'reason': 'outside launch scope'},
        )
        assert not denied.ok
        assert runtime.human.pending() == []

        with pytest.raises(HumanResponseRequired):
            runtime.tools.call(
                pid,
                'request_permission',
                {'resource': inside, 'rights': ['write'], 'reason': 'create scoped result'},
            )
        assert [request.pid for request in runtime.human.pending()] == [pid]
        assert not runtime.capability.check(pid, inside, CapabilityRight.WRITE)

    def test_process_spawn_reviewed_shell_policy_requires_exact_command_approval(self) -> None:
        runtime = self.server.service.runtime
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {
                'image': 'coding-agent:v0',
                'goal': 'run reviewed tests',
                'auto_run': False,
                'authority_manifest': {
                    'authorized_capabilities': [
                        {
                            'resource': 'human:owner',
                            'rights': ['write'],
                            'delegable': False,
                        },
                        {
                            'resource': 'shell:*',
                            'rights': ['execute'],
                            'delegable': False,
                            'constraints': {
                                'shell_policy_level': 'allowlist_auto_else_ask',
                            },
                        },
                    ],
                    'approval_policy': {'requestable_capabilities': []},
                },
            },
        )

        assert status == 200
        pid = spawned['pid']
        policy = next(
            capability
            for capability in runtime.capability.capabilities_for(pid)
            if capability.resource == 'shell:*'
        )
        assert policy.constraints == {
            'shell_policy_level': 'allowlist_auto_else_ask'
        }

        argv = ['python', '-m', 'pytest', '-q']
        with pytest.raises(HumanApprovalRequired):
            runtime.shell.run(pid, argv)

        pending = runtime.human.pending()
        assert len(pending) == 1
        assert pending[0].pid == pid
        assert pending[0].payload['context']['argv'] == argv
        assert any(
            record.action == 'human.query' and record.actor == pid
            for record in runtime.audit.trace(actor=pid)
        )
        assert any(
            event.type == EventType.HUMAN_QUERY and event.source == pid
            for event in runtime.events.list()
        )

    def test_process_rating_endpoint_updates_snapshot_and_audit(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'rate agent', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']

        status, empty = self.request('GET', f'/api/processes/{pid}/rating')
        assert status == 200
        assert empty is None

        status, rating = self.request('POST', f'/api/processes/{pid}/rating', {'score': 5, 'comment': 'strong result'})
        assert status == 200
        assert rating['pid'] == pid
        assert rating['score'] == 5
        assert rating['comment'] == 'strong result'
        assert rating['rater'] == DEFAULT_CONFIG.runtime.default_human
        assert rating['source'] == 'gui'

        status, updated = self.request('POST', f'/api/processes/{pid}/rating', {'score': 3, 'comment': 'missed detail'})
        assert status == 200
        assert updated['rating_id'] == rating['rating_id']
        assert updated['score'] == 3

        status, snapshot = self.request('GET', '/api/snapshot')
        assert status == 200
        process = next(item for item in snapshot['processes'] if item['pid'] == pid)
        assert process['rating']['score'] == 3
        assert process['rating']['comment'] == 'missed detail'
        assert any(
            record.action == 'agent.rating.upsert'
            and record.target == f'process:{pid}'
            for record in self.server.service.runtime.audit.trace()
        )

    def test_process_rating_endpoint_rejects_invalid_requests(self) -> None:
        status, spawned = self.request('POST', '/api/processes', {'goal': 'bad rating', 'auto_run': False})
        assert status == 200
        pid = spawned['pid']

        status, body = self.request('POST', f'/api/processes/{pid}/rating', {'score': 0})
        assert status == 400
        assert 'between 1 and 5' in body['error']['message']

        status, body = self.request('GET', '/api/processes/missing/rating')
        assert status == 404
        assert 'process not found' in body['error']['message']

    def test_encoded_route_segments_are_decoded(self) -> None:
        status, inspected = self.request('GET', '/api/images/base-agent%3Av0')

        assert status == 200
        assert inspected['image']['image_id'] == 'base-agent:v0'

    def test_process_spawn_accepts_initial_working_directory(self) -> None:
        status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'cwd target', 'working_directory': 'src\\app', 'auto_run': False},
        )

        assert status == 200
        assert spawned['process']['working_directory'] == 'src/app'

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
            extra_headers={'Origin': 'agent-libos://app'},
        )
        assert status == 204
        assert headers['access-control-allow-origin'] == 'agent-libos://app'

        status, headers, _body = self.request_raw(
            'OPTIONS',
            '/api/health',
            extra_headers={'Origin': 'agent-libos://untrusted'},
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

    def test_sse_broadcaster_invalidates_evicted_and_restarted_cursors(self) -> None:
        broadcaster = GuiEventBroadcaster(max_events=2)
        broadcaster.publish('snapshot', {'version': 1})
        broadcaster.publish('snapshot', {'version': 2})
        broadcaster.publish('snapshot', {'version': 3})

        evicted = broadcaster.replay_after(0)

        assert [event.event for event in evicted] == ['event.invalidated', 'snapshot', 'snapshot']
        assert evicted[0].seq == 1
        assert evicted[0].data == {
            'invalidated': True,
            'reason': 'sse_cursor_not_replayable',
            'requested_cursor': 0,
            'reset_cursor': 1,
            'oldest_available': 2,
            'latest_available': 3,
        }
        assert [event.seq for event in evicted[1:]] == [2, 3]

        restarted = broadcaster.replay_after(99)

        assert [event.event for event in restarted] == ['event.invalidated', 'snapshot', 'snapshot']
        assert restarted[0].seq == 0
        assert restarted[0].data['requested_cursor'] == 99
        assert restarted[0].data['reset_cursor'] == 0

    def test_gui_delta_deduplication_is_bounded(self) -> None:
        seen = _BoundedSeenKeys(2)

        assert seen.add_if_new('first') is True
        assert seen.add_if_new('second') is True
        assert seen.add_if_new('second') is False
        assert seen.add_if_new('third') is True
        assert len(seen) == 2
        assert seen.add_if_new('first') is True
        assert len(seen) == 2

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
        assert isinstance(event['payload']['blob'], str)
        assert len(event['payload']['blob']) == self.server.service.runtime.config.gui.snapshot_string_max_chars
        truncation = {
            path: meta
            for path, meta in snapshot['_truncated'].items()
            if path.endswith('.payload.blob')
        }
        assert list(truncation.values())[0]['kind'] == 'string'
        assert list(truncation.values())[0]['chars'] == len(huge)

    def test_snapshot_array_truncation_uses_metadata_not_sentinel_items(self) -> None:
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(
                self.server.service.runtime.config.gui,
                snapshot_collection_max_items=20,
                snapshot_event_limit=25,
            ),
        )
        for index in range(25):
            self.server.service.runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source='gui-test',
                target='gui-test',
                payload={'index': index},
            )

        status, snapshot = self.request('GET', '/api/snapshot')

        assert status == 200
        assert len(snapshot['events']) == 20
        assert all('event_id' in event for event in snapshot['events'])
        assert not any(event.get('truncated') is True for event in snapshot['events'])
        assert snapshot['_truncated']['events']['kind'] == 'array'
        assert snapshot['_truncated']['events']['omitted'] == 5

    def test_process_events_endpoint_is_store_bounded(self) -> None:
        pid = self.server.service.runtime.process.spawn(goal='bounded event api')
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(self.server.service.runtime.config.gui, snapshot_event_limit=3),
        )
        emitted = [
            self.server.service.runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source='gui-test',
                target=pid,
                payload={'index': index},
            )
            for index in range(5)
        ]

        status, events = self.request('GET', f'/api/processes/{pid}/events?limit=2')
        previous_status, previous = self.request(
            'GET',
            f'/api/processes/{pid}/events?limit=2&before={events[0]["event_id"]}',
        )
        invalid_status, invalid = self.request('GET', f'/api/processes/{pid}/events?limit=4')

        assert status == 200
        assert [event['event_id'] for event in events] == [emitted[-2].event_id, emitted[-1].event_id]
        assert previous_status == 200
        assert [event['event_id'] for event in previous] == [emitted[-4].event_id, emitted[-3].event_id]
        assert invalid_status == 400
        assert 'at most 3' in invalid['error']['message']

    def test_oversized_snapshot_sse_payload_uses_explicit_truncated_event(self) -> None:
        event_name, payload = _sse_payload_data(
            'snapshot',
            {'snapshot': {'events': [{'payload': 'x' * 100}]}},
            max_bytes=50,
            string_limit=200,
            collection_limit=200,
        )

        assert event_name == 'snapshot_truncated'
        assert payload['invalidated'] is True
        assert payload['event'] == 'snapshot'

    def test_strict_json_bool_rejects_string_false(self) -> None:
        status, body = self.request('POST', '/api/processes', {'goal': 'strict bool', 'auto_run': 'false'})

        assert status == 400
        assert 'auto_run must be a JSON boolean' in body['error']['message']

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
        status, _profile = self.request(
            'POST',
            '/api/llm-profiles',
            {'profile_id': 'gui-exec', 'model': 'gui-exec-model', 'api_key_env': 'GUI_EXEC_API_KEY'},
        )
        assert status == 200
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
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']
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

    def test_destructive_process_signal_requires_confirmation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'signal target', 'auto_run': False})
        pid = spawned['pid']

        status, paused = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.PAUSE.value},
        )
        assert status == 200
        assert paused['status'] == ProcessStatus.PAUSED.value
        assert paused['wait_state']['kind'] == 'paused'
        assert paused['outcome'] is None
        paused_generation = paused['state_generation']

        status, denied = self.request('POST', f'/api/processes/{pid}/signal', {'signal': ProcessSignal.TERMINATE.value})
        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['preview']['signal'] == ProcessSignal.TERMINATE.value

        status, string_confirmed = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.TERMINATE.value, 'confirmed': 'true'},
        )
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']

        status, allowed = self.request(
            'POST',
            f'/api/processes/{pid}/signal',
            {'signal': ProcessSignal.TERMINATE.value, 'confirmed': True},
        )
        assert status == 200
        assert allowed['status'] == ProcessStatus.KILLED.value
        assert allowed['wait_state'] is None
        assert allowed['outcome']['schema_version'] == 1
        assert allowed['outcome']['kind'] == 'killed'
        assert allowed['state_generation'] > paused_generation

    def test_checkpoint_inspect_typed_state_survives_sqlite_restart_across_boundaries(
        self,
        tmp_path: Path,
    ) -> None:
        db_path = str(tmp_path / 'typed-checkpoint.sqlite')
        runtime = Runtime.open(db_path)
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='checkpoint parent')
            runtime.capability.grant(
                parent,
                'process:spawn',
                [CapabilityRight.WRITE],
                issued_by='test',
            )
            child = runtime.spawn_child_process(parent, 'checkpoint child')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            waiting = runtime.process.get(parent)
            expected_wait = process_wait_state_to_mapping(waiting.wait_state)
            waiting_generation = waiting.state_generation
            waiting_checkpoint = runtime.checkpoint.create(
                parent,
                'typed waiting snapshot',
                actor=parent,
            )

            runtime.process.signal_child(
                parent,
                child,
                ProcessSignal.TERMINATE,
                reason='typed terminal snapshot',
            )
            terminal = runtime.process.get(child)
            expected_outcome = process_outcome_to_mapping(terminal.outcome)
            terminal_generation = terminal.state_generation
            terminal_checkpoint = runtime.checkpoint.create(
                parent,
                'typed terminal snapshot',
                actor=parent,
            )
        finally:
            runtime.close()

        reopened = Runtime.open(db_path)
        try:
            manager_waiting = reopened.checkpoint.inspect(waiting_checkpoint, actor=parent)
            waiting_row = next(
                row for row in manager_waiting['processes'] if row['pid'] == parent
            )
            assert waiting_row['wait_state'] == expected_wait
            assert waiting_row['outcome'] is None
            assert waiting_row['state_generation'] == waiting_generation

            manager_terminal = reopened.checkpoint.inspect(terminal_checkpoint, actor=parent)
            terminal_row = next(
                row for row in manager_terminal['processes'] if row['pid'] == child
            )
            assert terminal_row['wait_state'] is None
            assert terminal_row['outcome'] == expected_outcome
            assert terminal_row['state_generation'] == terminal_generation

            tool_waiting = reopened.tools.call(
                parent,
                'inspect_checkpoint',
                {'checkpoint_id': waiting_checkpoint},
            )
            assert tool_waiting.ok, tool_waiting.error
            tool_waiting_row = next(
                row for row in tool_waiting.payload['processes'] if row['pid'] == parent
            )
            assert tool_waiting_row['wait_state'] == expected_wait
            assert tool_waiting_row['outcome'] is None
            assert tool_waiting_row['state_generation'] == waiting_generation

            tool_terminal = reopened.tools.call(
                parent,
                'inspect_checkpoint',
                {'checkpoint_id': terminal_checkpoint},
            )
            assert tool_terminal.ok, tool_terminal.error
            tool_terminal_row = next(
                row for row in tool_terminal.payload['processes'] if row['pid'] == child
            )
            assert tool_terminal_row['wait_state'] is None
            assert tool_terminal_row['outcome'] == expected_outcome
            assert tool_terminal_row['state_generation'] == terminal_generation

            syscall_terminal = asyncio.run(
                LibOSSyscallSession(reopened, parent).handle(
                    'checkpoint.inspect',
                    {'checkpoint_id': terminal_checkpoint},
                )
            )
            syscall_terminal_row = next(
                row for row in syscall_terminal['processes'] if row['pid'] == child
            )
            assert syscall_terminal_row['outcome'] == expected_outcome
            assert syscall_terminal_row['state_generation'] == terminal_generation
        finally:
            reopened.close()

        cli_waiting = checkpoint_cli_json(
            ['--db', db_path, 'checkpoint', 'inspect', waiting_checkpoint]
        )
        cli_waiting_row = next(
            row for row in cli_waiting['processes'] if row['pid'] == parent
        )
        assert cli_waiting_row['wait_state'] == expected_wait
        assert cli_waiting_row['state_generation'] == waiting_generation
        cli_terminal = checkpoint_cli_json(
            ['--db', db_path, 'checkpoint', 'inspect', terminal_checkpoint]
        )
        cli_terminal_row = next(
            row for row in cli_terminal['processes'] if row['pid'] == child
        )
        assert cli_terminal_row['outcome'] == expected_outcome
        assert cli_terminal_row['state_generation'] == terminal_generation

        typed_server = create_gui_http_server(
            db=db_path,
            port=0,
            token='typed-checkpoint-token',
            auto_run=False,
            llm_profiles_file=tmp_path / 'typed-llm-profiles.json',
        )
        typed_thread = threading.Thread(target=typed_server.serve_forever, daemon=True)
        typed_thread.start()
        try:
            host, port = typed_server.server_address
            def gui_inspect(checkpoint_id: str) -> dict[str, Any]:
                conn = http.client.HTTPConnection(host, port, timeout=10)
                conn.request(
                    'GET',
                    f'/api/checkpoints/{checkpoint_id}',
                    headers={'Authorization': 'Bearer typed-checkpoint-token'},
                )
                response = conn.getresponse()
                payload = json.loads(response.read().decode('utf-8'))
                conn.close()
                assert response.status == 200
                return payload

            gui_waiting = gui_inspect(waiting_checkpoint)
            gui_waiting_row = next(
                row for row in gui_waiting['processes'] if row['pid'] == parent
            )
            assert gui_waiting_row['wait_state'] == expected_wait
            assert gui_waiting_row['state_generation'] == waiting_generation

            gui_terminal = gui_inspect(terminal_checkpoint)
            gui_terminal_row = next(
                row for row in gui_terminal['processes'] if row['pid'] == child
            )
            assert gui_terminal_row['wait_state'] is None
            assert gui_terminal_row['outcome'] == expected_outcome
            assert gui_terminal_row['state_generation'] == terminal_generation
        finally:
            typed_server.shutdown()
            typed_thread.join(timeout=5)
            typed_server.service.shutdown()
            typed_server.server_close()

    def test_invalid_process_signal_is_a_bad_request_without_mutation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'invalid signal target', 'auto_run': False})
        pid = spawned['pid']

        status, body = self.request('POST', f'/api/processes/{pid}/signal', {'signal': 'not-a-signal'})

        assert status == 400
        assert 'unknown process signal' in body['error']['message']
        assert self.server.service.runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    def test_missing_required_mutation_fields_are_bad_requests(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'required fields', 'auto_run': False})
        pid = spawned['pid']

        exec_status, exec_body = self.request(
            'POST',
            f'/api/processes/{pid}/exec',
            {'goal': 'missing image', 'confirmed': True, 'auto_run': False},
        )
        cd_status, cd_body = self.request('POST', f'/api/processes/{pid}/cd', {})
        checkpoint_status, checkpoint_body = self.request('POST', '/api/checkpoints/create', {})

        assert exec_status == 400
        assert 'image must be a non-empty JSON string' in exec_body['error']['message']
        assert cd_status == 400
        assert 'path must be a non-empty JSON string' in cd_body['error']['message']
        assert checkpoint_status == 400
        assert 'pid must be a non-empty JSON string' in checkpoint_body['error']['message']
        assert self.server.service.runtime.process.get(pid).image_id == 'base-agent:v0'

    def test_actor_is_rejected_on_routes_that_do_not_apply_actor_authority(self) -> None:
        service = self.server.service
        runtime = service.runtime
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'actor contract target', 'auto_run': False},
        )
        pid = spawned['pid']
        request_id = 'hreq_actor_contract'
        now = utc_now()
        runtime.store.insert_human_request(
            HumanRequest(
                request_id=request_id,
                pid=pid,
                human=runtime.config.runtime.default_human,
                payload={'type': 'question', 'question': 'must remain pending'},
                status=HumanRequestStatus.PENDING,
                decision=None,
                blocking=True,
                created_at=now,
                updated_at=now,
            )
        )
        service.save_user_llm_profile(
            'actor-contract-profile',
            {'model': 'actor-contract-model', 'api_key_env': 'ACTOR_CONTRACT_API_KEY'},
        )

        def observable_state() -> dict[str, Any]:
            return {
                'processes': to_jsonable(runtime.process.list()),
                'messages': to_jsonable(runtime.messages.list(pid, include_acked=True)),
                'rating': to_jsonable(runtime.ratings.get(pid)),
                'human_requests': to_jsonable(runtime.human.list()),
                'object_tasks': to_jsonable(runtime.object_tasks.list()),
                'capabilities': to_jsonable(runtime.store.list_capabilities()),
                'external_effects': to_jsonable(runtime.store.list_external_effects()),
                'jsonrpc_endpoints': runtime.jsonrpc.list_endpoints(require_capability=False),
                'mcp_servers': runtime.mcp.list_servers(require_capability=False),
                'llm_profiles': service._llm_profile_summaries(),
                'llm_profiles_file': self.llm_profiles_file.read_text(encoding='utf-8'),
                'audit': to_jsonable(runtime.audit.trace()),
                'events': to_jsonable(runtime.events.list()),
                'gui_events': to_jsonable(service.broadcaster.replay_after(0)),
                'scheduler': service.scheduler.status(),
                'service_closing': service._closing,
                'service_closed': service.closed,
            }

        before = observable_state()
        unsupported_actor_routes = [
            ('process.spawn', 'POST', '/api/processes', {'goal': 'must not spawn', 'actor': pid}),
            ('workflow.run', 'POST', '/api/workflows/run', {'tool': 'get_working_directory', 'args': {}, 'actor': pid}),
            ('object_task.start', 'POST', '/api/object-tasks/start', {'pid': pid, 'tool': 'get_working_directory', 'args': {}, 'actor': pid}),
            ('object_task.cancel', 'POST', '/api/object-tasks/missing/cancel', {'pid': pid, 'actor': pid}),
            ('object_task.wait', 'POST', '/api/object-tasks/missing/wait', {'pid': pid, 'actor': pid}),
            ('object_task.watch_owner', 'POST', '/api/object-tasks/missing/watch-owner', {'pid': pid, 'enabled': True, 'actor': pid}),
            ('scheduler.auto', 'POST', '/api/scheduler/auto', {'enabled': True, 'actor': pid}),
            ('scheduler.pause', 'POST', '/api/scheduler/pause', {'actor': pid}),
            ('process.rating', 'POST', f'/api/processes/{pid}/rating', {'score': 5, 'actor': pid}),
            ('process.run', 'POST', f'/api/processes/{pid}/run', {'max_quanta': 1, 'actor': pid}),
            ('process.step', 'POST', f'/api/processes/{pid}/step', {'actor': pid}),
            ('process.pause', 'POST', f'/api/processes/{pid}/pause', {'reason': 'must not pause', 'actor': pid}),
            ('process.resume', 'POST', f'/api/processes/{pid}/resume', {'actor': pid}),
            ('process.signal', 'POST', f'/api/processes/{pid}/signal', {'signal': ProcessSignal.PAUSE.value, 'actor': pid}),
            ('process.message', 'POST', f'/api/processes/{pid}/message', {'body': 'must not deliver', 'actor': pid}),
            ('process.interrupt', 'POST', f'/api/processes/{pid}/interrupt', {'body': 'must not interrupt', 'actor': pid}),
            ('process.cd', 'POST', f'/api/processes/{pid}/cd', {'path': '.', 'actor': pid}),
            ('process.exec', 'POST', f'/api/processes/{pid}/exec', {'image': 'coding-agent:v0', 'confirmed': True, 'actor': pid}),
            ('process.exit', 'POST', f'/api/processes/{pid}/exit', {'confirmed': True, 'actor': pid}),
            ('human.respond', 'POST', f'/api/human-requests/{request_id}/respond', {'approved': True, 'answer': 'must not answer', 'actor': pid}),
            ('capability.explain', 'POST', '/api/capabilities/explain', {'subject': pid, 'resource': 'object:actor-contract', 'right': 'read', 'actor': pid}),
            ('llm_profile.create', 'POST', '/api/llm-profiles', {'profile_id': 'must-not-create', 'model': 'unused', 'api_key_env': 'UNUSED_API_KEY', 'actor': pid}),
            ('llm_profile.update', 'PUT', '/api/llm-profiles/actor-contract-profile', {'model': 'must-not-update', 'actor': pid}),
            ('llm_profile.delete', 'DELETE', '/api/llm-profiles/actor-contract-profile', {'actor': pid}),
            ('jsonrpc.call', 'POST', '/api/jsonrpc/missing/call', {'pid': pid, 'method_id': 'read', 'confirmed': True, 'actor': pid}),
            ('mcp.call', 'POST', '/api/mcp/missing/call', {'pid': pid, 'tool_id': 'read', 'confirmed': True, 'actor': pid}),
            ('shutdown', 'POST', '/api/shutdown', {'actor': pid}),
        ]
        for route_name, method, path, payload in unsupported_actor_routes:
            status, body = self.request(method, path, payload)
            assert status == 400, route_name
            assert 'does not accept actor' in body['error']['message'], route_name
            assert observable_state() == before, route_name

        status, invalid_actor = self.request(
            'POST',
            '/api/checkpoints/create',
            {'pid': pid, 'actor': None},
        )
        assert status == 400
        assert 'actor must be a non-empty JSON string' in invalid_actor['error']['message']
        process = runtime.process.get(pid)
        assert process.status == ProcessStatus.RUNNABLE
        assert process.image_id == 'base-agent:v0'
        assert runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert observable_state() == before

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

        status, spoofed = self.request(
            'POST',
            '/api/capabilities/grant',
            {
                'subject': subject['pid'],
                'resource': 'object:gui-spoofed-human-grant',
                'rights': ['read'],
                'actor': DEFAULT_CONFIG.runtime.default_human_actor,
                'confirmed': True,
            },
        )
        assert status == 403
        assert 'lacks grant/admin authority' in spoofed['error']['message']

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
        assert status == 400
        assert 'confirmed must be a JSON boolean' in string_confirmed['error']['message']
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

    def test_scheduler_background_releases_runtime_lock_between_quanta(self) -> None:
        calls: list[tuple[int | None, bool]] = []

        def fake_run_until_idle(
            *,
            max_quanta: int | None = None,
            process_human_queue: bool = True,
            cancel_inflight_on_budget_exhaustion: bool = True,
        ) -> list[dict[str, int]]:
            assert cancel_inflight_on_budget_exhaustion is False
            calls.append((max_quanta, process_human_queue))
            return [{'call': len(calls)}] if len(calls) == 1 else []

        self.server.service.runtime.run_until_idle = fake_run_until_idle

        status = self.server.service.scheduler.start(max_quanta=3, reason='test-batch')
        assert status['running']
        thread = self.server.service.scheduler._thread
        assert thread is not None
        thread.join(timeout=2)

        assert calls == [(1, False), (1, False)]
        assert self.server.service.scheduler.status()['last_result'] == [{'call': 1}]

    def test_scheduler_background_completes_slow_inflight_quantum_at_batch_boundary(self) -> None:
        runtime = self.server.service.runtime
        runtime.scheduler.poll_interval_s = 0.001
        runtime.scheduler.drain_window_s = 0.001
        pid = runtime.process.spawn(image='base-agent:v0', goal='slow GUI quantum')
        completed = threading.Event()

        async def slow_quantum(selected_pid: str) -> dict[str, str]:
            assert selected_pid == pid
            await asyncio.sleep(0.03)
            runtime.process.pause(selected_pid, 'slow quantum completed')
            completed.set()
            return {'pid': selected_pid, 'status': 'completed'}

        runtime.arun_process_once = slow_quantum

        status = self.server.service.scheduler.start(max_quanta=1, reason='slow-batch')
        assert status['running']
        thread = self.server.service.scheduler._thread
        assert thread is not None
        thread.join(timeout=2)

        assert completed.is_set()
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED
        cancellations = [
            record
            for record in runtime.audit.trace()
            if record.action == 'scheduler.process_task_cancelled'
            and record.target == f'process:{pid}'
        ]
        assert cancellations == []
        assert self.server.service.scheduler.status()['last_result'] == [
            {'pid': pid, 'status': 'completed'}
        ]

    def test_health_uses_fast_path_when_runtime_lock_is_busy(self) -> None:
        self.server.service.runtime_lock.acquire()
        try:
            status, health = self.request('GET', '/api/health')
        finally:
            self.server.service.runtime_lock.release()

        assert status == 200
        assert health['runtime_busy'] is True
        assert health['process_count'] is None

    def test_gui_shutdown_waits_for_runtime_users_and_can_retry_after_timeout(self) -> None:
        runtime = Runtime.open('local')
        service = GuiRuntimeService(runtime=runtime, auto_run=False, token='lifecycle-test')
        entered = threading.Event()
        release = threading.Event()
        worker_done = threading.Event()

        def runtime_user() -> None:
            with service.runtime_user():
                entered.set()
                release.wait(timeout=2.0)
            worker_done.set()

        worker = threading.Thread(target=runtime_user)
        worker.start()
        assert entered.wait(timeout=2.0)
        try:
            assert service.shutdown(timeout_s=0.01) is False
            assert not service._closed
            release.set()
            assert worker_done.wait(timeout=2.0)
            assert service.shutdown(timeout_s=1.0) is True
            assert service._closed
            assert runtime.process.list() == []
        finally:
            release.set()
            worker.join(timeout=2.0)
            service.shutdown(timeout_s=1.0)
            runtime.close()

    def test_owned_runtime_partial_shutdown_never_reopens_api(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = GuiRuntimeService(db='local', auto_run=False, token='partial-shutdown')
        original_shutdown = service.runtime.shutdown
        calls = 0

        def fail_once(*, actor: str, reason: str) -> dict[str, object]:
            nonlocal calls
            calls += 1
            if calls == 1:
                return {'ok': False, 'object_tasks_stopped': False}
            return original_shutdown(actor=actor, reason=reason)

        monkeypatch.setattr(service.runtime, 'shutdown', fail_once)
        try:
            assert service.shutdown(timeout_s=1.0) is False
            assert service._closing is True
            assert service._closed is False
            with pytest.raises(GuiServerError, match='shutting down'):
                with service.runtime_user():
                    pass

            assert service.shutdown(timeout_s=1.0) is True
            assert service._closed is True
            assert calls == 2
        finally:
            service.shutdown(timeout_s=1.0)

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

    def test_process_step_returns_and_publishes_final_scheduler_status(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'step once', 'auto_run': False})
        pid = spawned['pid']

        async def fake_quantum(selected_pid: str) -> dict[str, str]:
            assert selected_pid == pid
            return {'pid': selected_pid, 'status': 'completed'}

        self.server.service.runtime.arun_process_once = fake_quantum
        before = self.server.service.broadcaster.replay_after(0)[-1].seq

        status, body = self.request('POST', f'/api/processes/{pid}/step', {})

        assert status == 200
        assert body['started'] is True
        assert body['scheduler']['running'] is False
        snapshots = [
            event.data['snapshot']
            for event in self.server.service.broadcaster.replay_after(before)
            if event.event == 'snapshot'
        ]
        assert snapshots
        assert snapshots[-1]['scheduler']['running'] is False

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

    def test_side_effect_workflow_requires_confirmation(self) -> None:
        status, denied = self.request('POST', '/api/workflows/run', {'tool': 'ask_human', 'args': {'question': 'Continue?'}})

        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['action'] == 'workflow.run'
        assert denied['error']['preview']['tool'] == 'ask_human'

    def test_unknown_workflow_tool_requires_confirmation_fail_closed(self) -> None:
        status, denied = self.request('POST', '/api/workflows/run', {'tool': 'missing_workflow_tool', 'args': {}})

        assert status == 409
        assert denied['error']['confirmation_required']
        assert denied['error']['action'] == 'workflow.run'

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

    def test_config_argument_controls_gui_runtime_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        target = str(tmp_path / 'gui-memory.sqlite')
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                local_store_target=target,
                default_image_id='configured-gui-base:v0',
                coding_image_id='configured-gui-coding:v0',
            ),
            gui=replace(DEFAULT_CONFIG.gui, object_task_wait_default_timeout_s=0.2, object_task_wait_max_timeout_s=0.4),
        )
        server = create_gui_http_server(config=config, port=0, token='custom-token', auto_run=False)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        seen: list[float | None] = []

        def fake_wait(task_id: str, *, actor_pid: str | None = None, timeout: float | None = None) -> dict[str, object]:
            seen.append(timeout)
            return {'task_id': task_id, 'actor_pid': actor_pid, 'timeout': timeout, 'status': 'running'}

        server.service.runtime.object_tasks.wait = fake_wait  # type: ignore[method-assign]
        thread.start()
        try:
            assert server.service.db == target
            assert server.service.runtime.store.path == target

            status, spawned = _request_to_server(server, 'POST', '/api/processes', {'goal': 'custom', 'auto_run': False}, token='custom-token')
            assert status == 200
            assert spawned['process']['image_id'] == 'configured-gui-base:v0'

            status, body = _request_to_server(server, 'POST', '/api/object-tasks/task-1/wait', {'pid': spawned['pid']}, token='custom-token')
            assert status == 200
            assert body['timeout'] == 0.2
            assert seen == [0.2]
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.service.shutdown()
            server.server_close()

    def test_gui_runtime_service_redacts_postgres_dsn_in_status_payloads(self) -> None:
        dsn = 'postgresql://agent:secret@localhost:5432/agent_libos'
        runtime = Runtime.open('local')
        server = create_gui_http_server(runtime=runtime, db=dsn, port=0, token='custom-token', auto_run=False)
        try:
            redacted = 'postgresql://agent:***@localhost:5432/agent_libos'
            assert server.service.db == redacted
            assert server.service.health()['db'] == redacted
            assert server.service.snapshot()['db'] == redacted
        finally:
            server.service.shutdown()
            server.server_close()
            runtime.close()

    def test_gui_runtime_service_uses_configured_postgres_dsn_when_db_is_omitted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(
                store_backend='postgres',
                store_dsn='postgresql://agent:secret@localhost:5432/agent_libos',
            )
        )
        calls: dict[str, object] = {}
        original_open = Runtime.open

        def fake_open(target: object = None, **kwargs: object) -> Runtime:
            calls['target'] = target
            calls['config'] = kwargs.get('config')
            return original_open('local')

        monkeypatch.setattr(Runtime, 'open', staticmethod(fake_open))
        server = create_gui_http_server(config=config, port=0, token='custom-token', auto_run=False)
        try:
            redacted = 'postgresql://agent:***@localhost:5432/agent_libos'

            assert calls['target'] is None
            assert server.service.db == redacted
            assert server.service.health()['db'] == redacted
            assert server.service.snapshot()['db'] == redacted
        finally:
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

    def test_mcp_register_rejects_host_file_path(self) -> None:
        status, body = self.request('POST', '/api/mcp/register', {'path': 'secrets.yaml', 'confirmed': True})
        assert status == 400
        assert 'manifest_text' in body['error']['message']

    def test_mcp_register_actor_mode_requires_server_write_capability(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'mcp actor', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_mcp_manifest('gui-actor-mcp')

        status, denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert status == 403
        assert 'mcp_server:gui-actor-mcp' in denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'mcp_server:gui-actor-mcp',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        spawn_status, spawn_denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert spawn_status == 403
        assert 'process:spawn' in spawn_denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            'process:spawn',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        stdio_status, stdio_denied = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )

        assert stdio_status == 403
        assert 'mcp_stdio' in stdio_denied['error']['message']

        self.server.service.runtime.capability.grant(
            pid,
            self.server.service.runtime.mcp.stdio_resource_for_argv('python3', ['-m', 'demo_mcp']),
            [CapabilityRight.EXECUTE],
            issued_by='test',
        )
        register_status, registered = self.request(
            'POST',
            '/api/mcp/register',
            {'manifest_text': manifest, 'actor': pid, 'confirmed': True},
        )
        tools_status, tools = self.request('GET', '/api/mcp/gui-actor-mcp/tools')

        assert register_status == 200
        assert tools_status == 200
        assert registered['server_id'] == 'gui-actor-mcp'
        assert tools['tools'][0]['tool_id'] == 'echo'
        assert tools['tools'][0]['resource'] == 'mcp:gui-actor-mcp:echo'

    def test_mcp_call_preserves_invalid_arguments_for_primitive_validation(self) -> None:
        _status, spawned = self.request('POST', '/api/processes', {'goal': 'mcp invalid args', 'auto_run': False})
        pid = spawned['pid']
        manifest = _gui_mcp_manifest('gui-invalid-args-mcp')
        self.server.service.runtime.mcp.register_server_from_yaml_text(
            manifest,
            actor='test',
            require_capability=False,
        )
        self.server.service.runtime.capability.grant(
            pid,
            'mcp:gui-invalid-args-mcp:echo',
            [CapabilityRight.READ],
            issued_by='test',
        )

        status, body = self.request(
            'POST',
            '/api/mcp/gui-invalid-args-mcp/call',
            {'pid': pid, 'tool_id': 'echo', 'arguments': [], 'confirmed': True},
        )

        assert status == 400
        assert 'arguments must be a JSON object' in body['error']['message']

    def test_mcp_provider_exception_secret_is_absent_from_gui_response(self) -> None:
        secret = "GUI_MCP_HOST_EXCEPTION_SECRET_SENTINEL"

        class FailingProvider:
            def list_tools(self, server: Any, **_kwargs: Any) -> McpToolListResult:
                return McpToolListResult(
                    server_id=server.server_id,
                    tools=[
                        McpProviderTool(
                            name="demo.echo",
                            description="Echo",
                            input_schema={},
                        )
                    ],
                    response_bytes=32,
                    duration_s=0.01,
                )

            def call_tool(
                self,
                _server: Any,
                _tool: Any,
                _arguments: dict[str, Any],
                **_kwargs: Any,
            ) -> Any:
                raise RuntimeError(secret)

            def classify_external_effect(
                self,
                _operation: str,
                _context: dict[str, Any],
                _result: Any,
            ) -> ExternalEffectClassification:
                return ExternalEffectClassification(
                    rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
                    rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
                    state_mutation=False,
                    information_flow=True,
                )

        runtime = self.server.service.runtime
        runtime.mcp.provider = FailingProvider()
        _status, spawned = self.request(
            'POST',
            '/api/processes',
            {'goal': 'MCP GUI provider failure', 'auto_run': False},
        )
        pid = spawned['pid']
        runtime.mcp.register_server_from_yaml_text(
            _gui_mcp_manifest('gui-provider-failure'),
            actor='test',
            require_capability=False,
        )
        runtime.capability.grant(
            pid,
            'mcp:gui-provider-failure:echo',
            [CapabilityRight.READ],
            issued_by='test',
        )
        runtime.capability.grant(
            pid,
            'process:spawn',
            [CapabilityRight.WRITE],
            issued_by='test',
        )
        runtime.capability.grant(
            pid,
            runtime.mcp.stdio_resource_for_argv('python3', ['-m', 'demo_mcp']),
            [CapabilityRight.EXECUTE],
            issued_by='test',
        )

        status, body = self.request(
            'POST',
            '/api/mcp/gui-provider-failure/call',
            {
                'pid': pid,
                'tool_id': 'echo',
                'arguments': {'text': 'hello'},
                'confirmed': True,
            },
        )

        assert status == 200
        assert body['ok'] is False
        assert set(body['error']) == {'code', 'error_type', 'correlation_id'}
        assert secret not in json.dumps(body, sort_keys=True)

    def test_skill_register_without_actor_rejects_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            skill_dir = write_skill_package(Path(temp_dir), 'gui-host-path-skill', allowed_tools=['echo'])

            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': str(skill_dir), 'confirmed': True},
            )

            assert status == 400
            assert 'requires an actor' in denied['error']['message']

    def test_skill_register_actor_mode_requires_skill_write_capability(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir).resolve()
            skill_dir = write_skill_package(root, 'gui-actor-skill', allowed_tools=['echo'])
            relative_skill = skill_dir.relative_to(Path.cwd().resolve()).as_posix()
            skill_md = f'{relative_skill}/SKILL.md'
            _status, spawned = self.request('POST', '/api/processes', {'goal': 'skill actor', 'auto_run': False})
            pid = spawned['pid']

            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
            )

            assert status == 403
            assert 'filesystem:workspace' in denied['error']['message']

            self.server.service.runtime.filesystem.grant_path(
                pid,
                skill_md,
                [CapabilityRight.READ],
                issued_by='test',
            )
            status, denied = self.request(
                'POST',
                '/api/skills/register',
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
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
                {'path': relative_skill, 'actor': pid, 'confirmed': True},
            )

            assert status == 200
            assert registered['skill_id'] == 'gui-actor-skill'

    def test_human_request_respond_rejects_non_pending_request(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='gui human conflict',
            authority_manifest={
                'authorized_capabilities': [
                    {'resource': DEFAULT_CONFIG.runtime.default_human_resource, 'rights': ['write']},
                ],
            },
        )
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

    def test_permission_response_requires_explicit_valid_policy(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='typed gui permission')
        resource = runtime.filesystem.resource_for('agent_outputs/typed-gui.txt')
        request_id = runtime.human.query(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={
                'type': 'permission_request',
                'question': 'Allow write?',
                'requested_permission': {
                    'subject': pid,
                    'resource': resource,
                    'rights': ['write'],
                    'constraints': {},
                },
            },
            blocking=True,
        )

        missing_status, missing = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        invalid_status, invalid = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'decision': {'policy': 'sometimes'}, 'auto_run': False},
        )

        assert missing_status == 400
        assert 'policy' in missing['error']['message']
        assert invalid_status == 400
        assert 'policy' in invalid['error']['message']
        assert runtime.human.get(request_id).status.value == 'pending'

        approved_status, approved = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'approved': True,
                'decision': {'policy': CapabilityManager.ASK_EACH_TIME},
                'auto_run': False,
            },
        )
        assert approved_status == 200
        assert approved['request']['decision']['policy'] == CapabilityManager.ASK_EACH_TIME
        assert runtime.capability.permission_policy(pid, resource, CapabilityRight.WRITE) == CapabilityManager.ASK_EACH_TIME

    def test_question_response_requires_string_answer_before_commit(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='typed gui question')
        request_id = runtime.human.query(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={'type': 'question', 'question': 'Which region?'},
            blocking=True,
        )

        missing_status, missing = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'auto_run': False},
        )
        wrong_status, wrong = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 42, 'auto_run': False},
        )
        empty_status, empty = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': '   ', 'auto_run': False},
        )

        assert missing_status == 400
        assert 'answer' in missing['error']['message']
        assert wrong_status == 400
        assert 'answer' in wrong['error']['message']
        assert empty_status == 400
        assert 'answer' in empty['error']['message']
        assert runtime.human.get(request_id).status.value == 'pending'

        accepted_status, accepted = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {'approved': True, 'answer': 'eu-west', 'auto_run': False},
        )
        assert accepted_status == 200
        assert accepted['request']['decision']['answer'] == 'eu-west'

    def test_human_request_delta_is_emitted_for_each_changed_version(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(
            image='base-agent:v0',
            goal='gui human delta',
            authority_manifest={
                'authorized_capabilities': [
                    {'resource': DEFAULT_CONFIG.runtime.default_human_resource, 'rights': ['write']},
                ],
            },
        )
        cursor = self.server.service.broadcaster.replay_after(0)[-1].seq
        request_id = runtime.human.ask(pid, 'Emit both versions?', blocking=True)

        self.server.service.publish_runtime_changes('human.pending')
        pending_events = self.server.service.broadcaster.replay_after(cursor)
        pending_updates = [
            event
            for event in pending_events
            if event.event == 'human_request.updated' and event.data['request_id'] == request_id
        ]
        assert len(pending_updates) == 1
        assert pending_updates[0].data['status'] == 'pending'
        cursor = pending_events[-1].seq

        runtime.human.approve(request_id, {'approved': True, 'answer': 'yes', 'source': 'test'})
        self.server.service.publish_runtime_changes('human.approved')
        approved_events = self.server.service.broadcaster.replay_after(cursor)
        approved_updates = [
            event
            for event in approved_events
            if event.event == 'human_request.updated' and event.data['request_id'] == request_id
        ]
        assert len(approved_updates) == 1
        assert approved_updates[0].data['status'] == 'approved'
        cursor = approved_events[-1].seq

        self.server.service.publish_runtime_changes('human.unchanged')
        unchanged = self.server.service.broadcaster.replay_after(cursor)
        assert not any(
            event.event == 'human_request.updated' and event.data['request_id'] == request_id
            for event in unchanged
        )

    def test_permission_response_without_approved_uses_explicit_deny_policy(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='gui human default reject')
        request_id = runtime.human.query(
            pid=pid,
            human=DEFAULT_CONFIG.runtime.default_human,
            request={
                'type': 'permission_request',
                'question': 'Allow object read?',
                'requested_permission': {
                    'subject': pid,
                    'resource': 'object:gui-default-reject',
                    'rights': ['read'],
                },
            },
            blocking=True,
        )

        status, rejected = self.request(
            'POST',
            f'/api/human-requests/{request_id}/respond',
            {
                'decision': {'policy': CapabilityManager.ALWAYS_DENY},
                'auto_run': False,
            },
        )

        assert status == 200
        assert rejected['request']['status'] == 'rejected'
        assert (
            runtime.capability.permission_policy(pid, 'object:gui-default-reject', CapabilityRight.READ)
            == 'always_deny'
        )

    def test_invalid_max_quanta_is_rejected(self) -> None:
        before_count = len(self.server.service.runtime.process.list())
        status, body = self.request('POST', '/api/processes', {'goal': 'goal', 'max_quanta': 1.5})
        assert status == 400
        assert 'max_quanta' in body['error']['message']
        assert len(self.server.service.runtime.process.list()) == before_count

        status, body = self.request('POST', '/api/processes', {'goal': 'goal', 'max_quanta': 0})
        assert status == 400
        assert 'max_quanta' in body['error']['message']
        assert len(self.server.service.runtime.process.list()) == before_count

    def test_process_resume_validates_body_before_mutating_process(self) -> None:
        runtime = self.server.service.runtime
        pid = runtime.process.spawn(image='base-agent:v0', goal='resume validation')
        runtime.process.pause(pid, 'hold for invalid resume body')
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED

        status, body = self.request_json_text('POST', f'/api/processes/{pid}/resume', '[]')

        assert status == 400
        assert 'JSON object' in body['error']['message']
        assert runtime.process.get(pid).status == ProcessStatus.PAUSED

    def test_request_body_size_is_bounded(self) -> None:
        self.server.service.runtime.config = replace(
            self.server.service.runtime.config,
            gui=replace(self.server.service.runtime.config.gui, request_body_max_bytes=1024),
        )
        status, body = self.request('POST', '/api/processes', {'goal': 'x' * 1100})
        assert status == 413
        assert 'exceeds' in body['error']['message']

    def test_shutdown_endpoint_stops_http_server(self) -> None:
        try:
            status, body = self.request('POST', '/api/shutdown', {})
            assert status == 200
            assert body['status'] == 'stopped'
        except ConnectionResetError:
            pass
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        self.server.service.shutdown()

    def test_shutdown_endpoint_reports_incomplete_teardown_and_remains_retryable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_shutdown = self.server.service.shutdown
        monkeypatch.setattr(self.server.service, 'shutdown', lambda timeout_s=None: False)

        status, body = self.request('POST', '/api/shutdown', {})

        assert status == 503
        assert body['ok'] is False
        assert body['error']['retryable'] is True
        assert self.thread.is_alive()

        monkeypatch.setattr(self.server.service, 'shutdown', original_shutdown)
        status, body = self.request('POST', '/api/shutdown', {})
        assert status == 200
        assert body == {'ok': True, 'status': 'stopped'}
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()

    def test_serve_teardown_retries_and_fails_visibly_if_runtime_never_closes(self) -> None:
        class FakeService:
            def __init__(self, results: list[bool | Exception]):
                self.results = iter(results)
                self.calls = 0

            def shutdown(self) -> bool:
                self.calls += 1
                result = next(self.results)
                if isinstance(result, Exception):
                    raise result
                return result

        retrying = FakeService([False, True])
        _shutdown_gui_service_before_exit(retrying)
        assert retrying.calls == 2

        exception_retry = FakeService([RuntimeError('first teardown attempt failed'), True])
        _shutdown_gui_service_before_exit(exception_retry)
        assert exception_retry.calls == 2

        incomplete = FakeService([False, False])
        with pytest.raises(RuntimeError, match='teardown remained incomplete'):
            _shutdown_gui_service_before_exit(incomplete)
        assert incomplete.calls == 2


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


def _gui_mcp_manifest(server_id: str) -> str:
    return f"""
schema_version: 1
server_id: {server_id}
transport: stdio
stdio:
  command: python3
  args: ["-m", "demo_mcp"]
tools:
  - tool_id: echo
    mcp_name: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".lstrip()
