from __future__ import annotations
import pytest
import contextlib
import hashlib
import io
import json
import os
import socket
import tempfile
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from pytest import MonkeyPatch
from agent_libos import Runtime
from agent_libos.api.cli import main as cli_main
from agent_libos.capability.rules import AUTHORITY_RULES_KEY
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import (
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcTransportResult,
    JsonRpcEndpointSpec,
    JsonRpcMethodSpec,
    ObjectMetadata,
    ObjectType,
    SinkTrustLevel,
    SinkTrustRule,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    HumanApprovalRequired,
    NotFound,
    ProviderHostError,
    ValidationError,
)
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import HttpJsonRpcProvider, LocalResourceProviderSubstrate, ProviderEffectNotStarted
from agent_libos.utils.serde import dumps

class TestJsonRpcPrimitive:

    def test_async_call_returns_external_untrusted_ingress_context(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='JSON-RPC ingress')
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest('return-ingress', server.url, with_header=False),
                    actor='cli',
                    require_capability=False,
                )
                runtime.capability.grant(
                    pid,
                    'jsonrpc:return-ingress:echo',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                with runtime.data_flow.activate(
                    DataFlowContext(labels=DataLabels(origin=None))
                ):
                    async def call_and_capture_context() -> tuple[Any, DataFlowContext]:
                        result = await runtime.jsonrpc.acall(
                            pid,
                            'return-ingress',
                            'echo',
                            {'value': 'remote'},
                        )
                        return result, runtime.data_flow.current_context()

                    result, returned = self._run(call_and_capture_context())

                    assert result.ok
                    assert returned.labels.sensitivity.value == 'normal'
                    assert returned.labels.trust_level.value == 'untrusted'
                    assert returned.labels.integrity.value == 'untrusted'
                    assert returned.labels.origin == 'external:jsonrpc'
            finally:
                runtime.close()

    def test_labeled_params_require_matching_trusted_endpoint_identity(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='labeled JSON-RPC egress')
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    'labeled-endpoint',
                    'https://api.example.test/jsonrpc',
                    with_header=False,
                ),
                actor='cli',
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                'jsonrpc:labeled-endpoint:echo',
                [CapabilityRight.READ],
                issued_by='test',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'jsonrpc-data-flow-sentinel'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )

            with pytest.raises(CapabilityDenied, match='data-flow denied egress'):
                runtime.jsonrpc.call(
                    pid,
                    'labeled-endpoint',
                    'echo',
                    {'value': 'jsonrpc-data-flow-sentinel'},
                    source_oids=[source.oid],
                )
            assert provider.calls == []

            spec, _metadata = runtime.jsonrpc._load_endpoint('labeled-endpoint')
            method = spec.method_by_id('echo')
            assert method is not None
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='jsonrpc:labeled-endpoint:echo',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=runtime.jsonrpc._endpoint_identity_sha256(
                        spec,
                        method,
                    ),
                ),
                actor='test.host',
                require_capability=False,
            )
            monkeypatch.setattr(
                runtime.jsonrpc,
                '_validate_runtime_resolution',
                lambda _spec: ('93.184.216.34',),
            )

            result = runtime.jsonrpc.call(
                pid,
                'labeled-endpoint',
                'echo',
                {'value': 'jsonrpc-data-flow-sentinel'},
                source_oids=[source.oid],
            )

            assert result.ok
            assert len(provider.calls) == 1
        finally:
            runtime.close()

    @pytest.mark.parametrize('operation', ['inspect', 'unregister', 'register', 'replace'])
    @pytest.mark.parametrize('endpoint_id', ['secret-existing', 'secret-missing'])
    def test_registry_item_authority_precedes_endpoint_metadata_load(
        self,
        monkeypatch: MonkeyPatch,
        operation: str,
        endpoint_id: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc registry oracle')

            def fail_if_loaded(_endpoint_id: str) -> Any:
                raise AssertionError('endpoint metadata must not load before authority')

            monkeypatch.setattr(runtime.store, 'get_jsonrpc_endpoint', fail_if_loaded)
            with pytest.raises(CapabilityDenied):
                if operation == 'inspect':
                    runtime.jsonrpc.inspect_endpoint(endpoint_id, actor=pid)
                elif operation == 'unregister':
                    runtime.jsonrpc.unregister_endpoint(endpoint_id, actor=pid)
                else:
                    runtime.jsonrpc.register_endpoint_from_yaml_text(
                        _manifest(endpoint_id, 'https://safe.example.test/jsonrpc', with_header=False),
                        actor=pid,
                        replace=operation == 'replace',
                    )
        finally:
            runtime.close()

    def test_registry_register_audit_failure_rolls_back_endpoint(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        original_record = runtime.audit.record

        def fail_register_audit(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get('action') == 'jsonrpc.endpoint.register':
                raise RuntimeError('injected jsonrpc register audit failure')
            return original_record(*args, **kwargs)

        monkeypatch.setattr(runtime.audit, 'record', fail_register_audit)
        try:
            with pytest.raises(RuntimeError, match='register audit failure'):
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest('register-rollback', 'https://safe.example.test/jsonrpc', with_header=False),
                    actor='cli',
                    require_capability=False,
                )
            assert runtime.store.get_jsonrpc_endpoint('register-rollback') is None
        finally:
            runtime.close()

    def test_registry_unregister_audit_failure_rolls_back_endpoint_and_method_caps(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest('unregister-rollback', 'https://safe.example.test/jsonrpc', with_header=False),
                actor='cli',
                require_capability=False,
            )
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc unregister rollback')
            cap = runtime.capability.grant(
                pid,
                'jsonrpc:unregister-rollback:echo',
                [CapabilityRight.READ],
                issued_by='test',
            )
            original_record = runtime.audit.record

            def fail_unregister_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') == 'jsonrpc.endpoint.unregister':
                    raise RuntimeError('injected jsonrpc unregister audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_unregister_audit)
            with pytest.raises(RuntimeError, match='unregister audit failure'):
                runtime.jsonrpc.unregister_endpoint(
                    'unregister-rollback',
                    actor='cli',
                    require_capability=False,
                )

            assert runtime.store.get_jsonrpc_endpoint('unregister-rollback') is not None
            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.active and not persisted.revoked
        finally:
            runtime.close()

    def test_registry_mutations_settle_one_shot_authority_with_their_transaction(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='one-shot JSON-RPC registry')
            endpoint_id = 'one-shot-registry'
            original_record = runtime.audit.record
            failing_actions = {'jsonrpc.endpoint.register'}

            def fail_selected_audit(*args: Any, **kwargs: Any) -> Any:
                if kwargs.get('action') in failing_actions:
                    raise RuntimeError(f"injected {kwargs['action']} audit failure")
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_selected_audit)
            register_cap = runtime.capability.grant_once(
                actor,
                runtime.jsonrpc.endpoint_resource(endpoint_id),
                [CapabilityRight.WRITE],
                issued_by='test',
            )

            with pytest.raises(RuntimeError, match='endpoint.register'):
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest(endpoint_id, 'https://safe.example.test/jsonrpc', with_header=False),
                    actor=actor,
                )

            assert runtime.store.get_capability(register_cap.cap_id).uses_remaining == 1
            assert runtime.store.get_jsonrpc_endpoint(endpoint_id) is None

            failing_actions.clear()
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(endpoint_id, 'https://safe.example.test/jsonrpc', with_header=False),
                actor=actor,
            )
            assert runtime.store.get_capability(register_cap.cap_id).uses_remaining == 0

            unregister_cap = runtime.capability.grant_once(
                actor,
                runtime.jsonrpc.endpoint_resource(endpoint_id),
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            failing_actions.add('jsonrpc.endpoint.unregister')
            with pytest.raises(RuntimeError, match='endpoint.unregister'):
                runtime.jsonrpc.unregister_endpoint(endpoint_id, actor=actor)

            assert runtime.store.get_capability(unregister_cap.cap_id).uses_remaining == 1
            assert runtime.store.get_jsonrpc_endpoint(endpoint_id) is not None

            failing_actions.clear()
            assert runtime.jsonrpc.unregister_endpoint(endpoint_id, actor=actor)['deleted'] is True
            assert runtime.store.get_capability(unregister_cap.cap_id).uses_remaining == 0
            assert runtime.store.get_jsonrpc_endpoint(endpoint_id) is None
        finally:
            runtime.close()

    @pytest.mark.parametrize('operation', ['register', 'unregister'])
    def test_registry_reauthorizes_unlimited_authority_before_mutation(
        self,
        monkeypatch: MonkeyPatch,
        operation: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(
                image='base-agent:v0',
                goal=f'JSON-RPC registry authority race {operation}',
            )
            endpoint_id = f'reauthorization-{operation}'
            if operation == 'unregister':
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest(
                        endpoint_id,
                        'https://safe.example.test/jsonrpc',
                        with_header=False,
                    ),
                    actor='test.host',
                    require_capability=False,
                )
            authority = runtime.capability.grant(
                actor,
                runtime.jsonrpc.endpoint_resource(endpoint_id),
                [
                    CapabilityRight.WRITE
                    if operation == 'register'
                    else CapabilityRight.ADMIN
                ],
                issued_by='test.host',
            )
            original_require = runtime.capability.require

            def revoke_after_outer_authorization(*args: Any, **kwargs: Any):
                decision = original_require(*args, **kwargs)
                runtime.capability.revoke(
                    authority.cap_id,
                    revoked_by='test.host',
                    reason='JSON-RPC registry revocation race regression',
                    require_authority=False,
                )
                return decision

            monkeypatch.setattr(runtime.capability, 'require', revoke_after_outer_authorization)

            with pytest.raises(CapabilityDenied, match='authority changed'):
                if operation == 'register':
                    runtime.jsonrpc.register_endpoint_from_yaml_text(
                        _manifest(
                            endpoint_id,
                            'https://safe.example.test/jsonrpc',
                            with_header=False,
                        ),
                        actor=actor,
                    )
                else:
                    runtime.jsonrpc.unregister_endpoint(endpoint_id, actor=actor)

            persisted = runtime.store.get_jsonrpc_endpoint(endpoint_id)
            assert (persisted is None) is (operation == 'register')
        finally:
            runtime.close()

    def test_manifest_validation_rejects_unsafe_endpoint_shapes(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('valid', 'https://api.example.test/jsonrpc'), actor='cli', require_capability=False)
            invalid_cases = [_manifest('bad-http', 'http://api.example.test/jsonrpc'), _manifest('bad-userinfo', 'https://user:pass@example.test/jsonrpc'), _manifest('bad-fragment', 'https://api.example.test/jsonrpc#secret'), _manifest('bad:colon', 'https://api.example.test/jsonrpc'), _manifest('bad-private-ip', 'https://10.0.0.10/jsonrpc'), _manifest('bad-metadata-ip', 'https://169.254.169.254/jsonrpc'), _manifest('bad-metadata-host', 'https://metadata.google.internal/jsonrpc'), _manifest('missing-class', 'https://api.example.test/jsonrpc', rollback_class=None), _manifest('empty-status', 'https://api.example.test/jsonrpc', rollback_status='""'), _manifest('literal-header', 'https://api.example.test/jsonrpc', literal_header=True), _manifest('bad-prefix', 'https://api.example.test/jsonrpc', header_prefix='Bearer literal-secret '), _manifest('bad-suffix', 'https://api.example.test/jsonrpc', header_suffix=' literal-secret'), _manifest('bad-bool', 'https://api.example.test/jsonrpc', state_mutation='"false"'), _manifest('dup-method', 'https://api.example.test/jsonrpc', duplicate_method=True), _manifest('bad-port', 'https://api.example.test:99999/jsonrpc'), _manifest('bad-timeout', 'https://api.example.test/jsonrpc', timeout_s='.nan'), _manifest('bad-bool-bytes', 'https://api.example.test/jsonrpc', max_request_bytes='true')]
            for text in invalid_cases:
                with pytest.raises(ValidationError):
                    runtime.jsonrpc.register_endpoint_from_yaml_text(text, actor='cli', require_capability=False)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("scope", "field", "value"),
        [
            pytest.param("endpoint", "headers", [], id="headers-list"),
            pytest.param("endpoint", "metadata", [], id="endpoint-metadata-list"),
            pytest.param("method", "params_schema", [], id="params-schema-list"),
            pytest.param("method", "params_schema", False, id="params-schema-bool"),
            pytest.param("method", "metadata", [], id="method-metadata-list"),
        ],
    )
    def test_manifest_validation_rejects_explicit_wrong_container_types(
        self,
        scope: str,
        field: str,
        value: Any,
    ) -> None:
        manifest = _manifest_mapping("strict-containers")
        target = manifest if scope == "endpoint" else manifest["methods"][0]
        target[field] = value
        runtime = Runtime.open("local")
        try:
            with pytest.raises(ValidationError, match="must be a mapping"):
                runtime.jsonrpc.register_endpoint(
                    manifest,
                    actor="cli",
                    require_capability=False,
                )
        finally:
            runtime.close()

    def test_call_params_require_jsonrpc_structured_value_or_null(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="strict JSON-RPC params",
            )
            for invalid in (True, 7, "scalar"):
                with pytest.raises(
                    ValidationError,
                    match="object, array, or null",
                ):
                    runtime.jsonrpc.call(pid, "hidden", "method", invalid)

            for valid in (None, {}, []):
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.call(pid, "hidden", "method", valid)
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param("remote failure", id="scalar"),
            pytest.param({}, id="missing-members"),
            pytest.param({"code": True, "message": "bad"}, id="boolean-code"),
            pytest.param({"code": "-32000", "message": "bad"}, id="string-code"),
            pytest.param({"code": -32000}, id="missing-message"),
            pytest.param({"code": -32000, "message": 7}, id="numeric-message"),
        ],
    )
    def test_invalid_jsonrpc_error_object_is_invalid_response(
        self,
        monkeypatch: MonkeyPatch,
        error: Any,
    ) -> None:
        runtime = Runtime.open("local")
        provider = _EnvelopeJsonRpcProvider(error=error)
        runtime.jsonrpc.provider = provider
        monkeypatch.setattr(
            "agent_libos.primitives.jsonrpc.socket.getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ],
        )
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="strict JSON-RPC error object",
            )
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    "strict-error",
                    "https://safe.example.test/jsonrpc",
                    with_header=False,
                ),
                actor="cli",
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                "jsonrpc:strict-error:echo",
                [CapabilityRight.READ],
                issued_by="test",
            )

            result = runtime.jsonrpc.call(pid, "strict-error", "echo", {})

            assert result.status.value == "invalid_response"
            assert not result.ok
            assert result.error == {"message": "JSON-RPC error object is invalid"}
        finally:
            runtime.close()

    def test_valid_jsonrpc_error_object_preserves_remote_error(self, monkeypatch: MonkeyPatch) -> None:
        remote_error = {
            "code": -32000,
            "message": "remote failure",
            "data": {"retryable": False},
        }
        runtime = Runtime.open("local")
        runtime.jsonrpc.provider = _EnvelopeJsonRpcProvider(error=remote_error)
        monkeypatch.setattr(
            "agent_libos.primitives.jsonrpc.socket.getaddrinfo",
            lambda *_args, **_kwargs: [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
            ],
        )
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="valid JSON-RPC error object",
            )
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    "valid-error",
                    "https://safe.example.test/jsonrpc",
                    with_header=False,
                ),
                actor="cli",
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                "jsonrpc:valid-error:echo",
                [CapabilityRight.READ],
                issued_by="test",
            )

            result = runtime.jsonrpc.call(pid, "valid-error", "echo", {})

            assert result.status.value == "jsonrpc_error"
            assert result.error == remote_error
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('rollback_class', 'rollback_status', 'expected_status'),
        [
            pytest.param('irreversible', None, 'not_supported', id='irreversible-default'),
            pytest.param('rollbackable', None, 'not_applied', id='rollbackable-default'),
            pytest.param('no_rollback_required', None, 'not_required', id='not-required-default'),
            pytest.param('unknown', None, 'unknown', id='unknown-default'),
            pytest.param('rollbackable', 'unknown', 'unknown', id='explicit-override'),
        ],
    )
    def test_manifest_rollback_status_defaults_and_explicit_override(
        self,
        rollback_class: str,
        rollback_status: str | None,
        expected_status: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            endpoint_id = f'rollback-status-{rollback_class}-{rollback_status or "default"}'
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    endpoint_id,
                    'https://api.example.test/jsonrpc',
                    rollback_class=rollback_class,
                    rollback_status=rollback_status,
                ),
                actor='cli',
                require_capability=False,
            )

            endpoint = runtime.jsonrpc.inspect_endpoint(
                endpoint_id,
                require_capability=False,
            )

            assert endpoint['methods'][0]['rollback_class'] == rollback_class
            assert endpoint['methods'][0]['rollback_status'] == expected_status
        finally:
            runtime.close()

    def test_list_endpoints_rejects_unbounded_primitive_limits(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('valid', 'https://api.example.test/jsonrpc'), actor='cli', require_capability=False)
            with pytest.raises(ValidationError, match='limit'):
                runtime.jsonrpc.list_endpoints(require_capability=False, limit=0)
            with pytest.raises(ValidationError, match='limit'):
                runtime.jsonrpc.list_endpoints(require_capability=False, limit=runtime.config.jsonrpc.list_limit + 1)
        finally:
            runtime.close()

    def test_list_endpoints_window_reports_rows_beyond_requested_limit(self) -> None:
        runtime = Runtime.open('local')
        try:
            for index in range(3):
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest(f'window-{index}', 'https://api.example.test/jsonrpc'),
                    actor='cli',
                    require_capability=False,
                )

            bounded, has_more = runtime.jsonrpc.list_endpoints_window(require_capability=False, limit=2)
            complete, complete_has_more = runtime.jsonrpc.list_endpoints_window(require_capability=False, limit=3)

            assert len(bounded) == 2
            assert has_more is True
            assert len(complete) == 3
            assert complete_has_more is False
        finally:
            runtime.close()

    def test_call_requires_method_capability_and_records_http_effect(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            monkeypatch.setenv('AGENT_LIBOS_JSONRPC_TEST_TOKEN', 'secret-token')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='remote call')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('demo', server.url), actor='cli', require_capability=False)
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.call(pid, 'demo', 'echo', {'city': 'Beijing'})
                runtime.capability.grant(pid, 'jsonrpc:demo:echo', [CapabilityRight.READ], issued_by='test')
                result = runtime.jsonrpc.call(pid, 'demo', 'echo', {'city': 'Beijing'})
                assert result.ok
                assert result.result['method'] == 'demo.echo'
                assert server.requests[0]['authorization'] == 'Bearer secret-token'
                assert server.requests[0]['body']['jsonrpc'] == '2.0'
                assert server.requests[0]['body']['method'] == 'demo.echo'
                assert 'url' not in (runtime.audit.trace()[-1].decision or {})
                assert 'secret-token' not in json.dumps([record.__dict__ for record in runtime.audit.trace()])
                effect = [item for item in runtime.store.list_external_effects() if item.provider == 'jsonrpc'][0]
                assert effect.rollback_class.value == 'no_rollback_required'
                assert not effect.state_mutation
                assert effect.information_flow
            finally:
                runtime.close()

    def test_header_environment_is_snapshotted_before_provider_dispatch(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            monkeypatch.setenv('AGENT_LIBOS_JSONRPC_TEST_TOKEN', 'approved-token')
            provider = _EnvironmentMutatingJsonRpcProvider(
                'AGENT_LIBOS_JSONRPC_TEST_TOKEN',
            )
            runtime.jsonrpc.provider = provider
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='credential snapshot')
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest('credential-snapshot', server.url),
                    actor='cli',
                    require_capability=False,
                )
                runtime.capability.grant(
                    pid,
                    'jsonrpc:credential-snapshot:echo',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                result = runtime.jsonrpc.call(
                    pid,
                    'credential-snapshot',
                    'echo',
                    {'city': 'Beijing'},
                )

                assert result.ok
                assert provider.resolved_headers == {
                    'Authorization': 'Bearer approved-token',
                }
                assert provider.snapshot_was_immutable
                assert server.requests[0]['authorization'] == 'Bearer approved-token'
            finally:
                runtime.close()

    def test_default_provider_ignores_ambient_http_proxy(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        with _jsonrpc_server() as server:
            provider = HttpJsonRpcProvider()
            endpoint = JsonRpcEndpointSpec(
                schema_version=1,
                endpoint_id='no-ambient-proxy',
                url=server.url,
                headers={},
                methods=[
                    JsonRpcMethodSpec(
                        method_id='echo',
                        rpc_method='demo.echo',
                        right='read',
                        rollback_class='no_rollback_required',
                        state_mutation=False,
                        information_flow=True,
                    )
                ],
                timeout_s=1,
                max_request_bytes=1024,
                max_response_bytes=1024,
            )
            monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
            monkeypatch.setenv('no_proxy', '')

            def fail_pinned(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError('unpinned provider call must use the urllib opener')

            monkeypatch.setattr(provider, '_call_pinned', fail_pinned)
            request_body = json.dumps(
                {
                    'jsonrpc': '2.0',
                    'id': 'direct-no-proxy',
                    'method': 'demo.echo',
                    'params': {'ok': True},
                }
            ).encode('utf-8')

            result = provider.call(
                endpoint,
                endpoint.methods[0],
                request_body,
                timeout_s=endpoint.timeout_s,
                max_response_bytes=endpoint.max_response_bytes,
                resolved_addresses=None,
                resolved_headers={},
            )

            assert result.status_code == 200, result.error
            assert len(server.requests) == 1

    @pytest.mark.parametrize('sink', ['event', 'audit'])
    def test_call_post_provider_sink_failure_leaves_pending_effect_intent(
        self,
        monkeypatch: MonkeyPatch,
        sink: str,
    ) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        resource = 'jsonrpc:pending-sink:echo'
        monkeypatch.setattr(
            'agent_libos.primitives.jsonrpc.socket.getaddrinfo',
            lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))],
        )
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'jsonrpc {sink} sink failure')
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest('pending-sink', 'https://safe.example.test/jsonrpc', with_header=False),
                actor='cli',
                require_capability=False,
            )
            runtime.capability.grant(pid, resource, [CapabilityRight.READ], issued_by='test')
            if sink == 'event':
                original_emit = runtime.events.emit

                def fail_result_event(event_type: Any, *args: Any, **kwargs: Any) -> Any:
                    if kwargs.get('target') == resource:
                        raise RuntimeError('injected jsonrpc event failure')
                    return original_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', fail_result_event)
            else:
                original_record = runtime.audit.record

                def fail_result_audit(*args: Any, **kwargs: Any) -> Any:
                    if kwargs.get('action') == 'primitive.jsonrpc.call':
                        raise RuntimeError('injected jsonrpc audit failure')
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', fail_result_audit)

            with pytest.raises(RuntimeError, match=f'injected jsonrpc {sink} failure'):
                runtime.jsonrpc.call(pid, 'pending-sink', 'echo', {'x': 1})

            assert len(provider.calls) == 1
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].provider == 'jsonrpc'
            assert effects[0].operation == 'call'
            assert effects[0].effect_state == 'pending'
            assert effects[0].provider_metadata['effect_state'] == 'pending'
        finally:
            runtime.close()

    def test_call_provider_not_started_after_dns_finalizes_information_flow(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        provider = _NotStartedJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        monkeypatch.setattr(
            'agent_libos.primitives.jsonrpc.socket.getaddrinfo',
            lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))],
        )
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc provider not started')
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest('not-started', 'https://safe.example.test/jsonrpc', with_header=False),
                actor='cli',
                require_capability=False,
            )
            cap = runtime.capability.grant_once(
                pid,
                'jsonrpc:not-started:echo',
                [CapabilityRight.READ],
                issued_by='test',
            )

            with pytest.raises(ProviderHostError) as caught:
                runtime.jsonrpc.call(pid, 'not-started', 'echo', {'x': 1})
            assert 'jsonrpc failed before transport' not in str(caught.value)
            assert caught.value.code == 'jsonrpc_provider_not_started'
            assert caught.value.error_type == 'ProviderEffectNotStarted'
            assert caught.value.correlation_id

            assert provider.calls == 1
            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.uses_remaining == 0
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].effect_state == 'finalized'
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].information_flow
            assert effects[0].provider_metadata['phase'] == 'transport_not_started_after_dns'
        finally:
            runtime.close()

    def test_malformed_transport_result_is_sanitized_before_field_access(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        secret = "JSONRPC_MALFORMED_RESULT_SECRET_SENTINEL"
        runtime = Runtime.open("local")
        runtime.jsonrpc.provider = _MalformedJsonRpcProvider(secret)
        monkeypatch.setattr(
            "agent_libos.primitives.jsonrpc.socket.getaddrinfo",
            lambda *_args, **_kwargs: [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    6,
                    "",
                    ("93.184.216.34", 443),
                )
            ],
        )
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="sanitize malformed JSON-RPC result",
            )
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    "malformed-result",
                    "https://safe.example.test/jsonrpc",
                    with_header=False,
                ),
                actor="cli",
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                "jsonrpc:malformed-result:echo",
                [CapabilityRight.READ],
                issued_by="test",
            )

            with pytest.raises(ProviderHostError) as caught:
                runtime.jsonrpc.call(
                    pid,
                    "malformed-result",
                    "echo",
                    {"x": 1},
                )

            observed = dumps(
                {
                    "exception": caught.value.to_dict(),
                    "exception_text": str(caught.value),
                    "audit": runtime.audit.trace(),
                    "events": runtime.events.list(),
                    "effects": runtime.store.list_external_effects(pid=pid),
                }
            )
            assert caught.value.code == "jsonrpc_provider_error"
            assert caught.value.error_type == "RuntimeError"
            assert caught.value.correlation_id
            assert secret not in observed
            process = runtime.process.get(pid)
            assert (
                process.resource_usage.jsonrpc_response_bytes
                == runtime.config.jsonrpc.max_response_bytes
            )
            effect = runtime.store.list_external_effects(pid=pid)[0]
            assert effect.transaction_state == "unknown"
            assert (
                effect.provider_metadata["phase"]
                == "transport_not_started_after_dns"
            )
        finally:
            runtime.close()

    def test_resolution_certified_not_started_restores_one_shot_and_abandons_intent(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc resolution not started')
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest('resolution-not-started', 'https://safe.example.test/jsonrpc', with_header=False),
                actor='cli',
                require_capability=False,
            )
            cap = runtime.capability.grant_once(
                pid,
                'jsonrpc:resolution-not-started:echo',
                [CapabilityRight.READ],
                issued_by='test',
            )
            monkeypatch.setattr(
                runtime.jsonrpc,
                '_validate_runtime_resolution',
                lambda _spec: (_ for _ in ()).throw(ProviderEffectNotStarted('resolution did not start')),
            )

            with pytest.raises(ProviderHostError) as caught:
                runtime.jsonrpc.call(pid, 'resolution-not-started', 'echo', {'x': 1})
            assert 'resolution did not start' not in str(caught.value)
            assert caught.value.code == 'jsonrpc_provider_not_started'

            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.uses_remaining == 1
            assert provider.calls == []
            assert runtime.store.list_external_effects(pid=pid) == []
        finally:
            runtime.close()

    def test_call_denies_before_loading_endpoint_metadata_without_visibility(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc hidden manifest')

            def fail_if_manifest_loaded(_endpoint_id: str) -> Any:
                raise AssertionError('endpoint manifest should stay hidden before capability gate')

            monkeypatch.setattr(runtime.store, 'get_jsonrpc_endpoint', fail_if_manifest_loaded)
            monkeypatch.setattr(
                runtime.store,
                'get_jsonrpc_registry_binding',
                lambda _endpoint_id: (_ for _ in ()).throw(
                    AssertionError('registry binding should stay hidden without matching authority')
                ),
            )

            with pytest.raises(CapabilityDenied, match='JSON-RPC call authority'):
                runtime.jsonrpc.call(pid, 'secret-endpoint', 'hidden-method', {'city': 'Beijing'})
        finally:
            runtime.close()

    def test_call_ask_visibility_prompts_before_loading_endpoint_metadata(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc ask hidden manifest')
            runtime.capability.set_permission_policy(
                pid,
                'jsonrpc:secret-endpoint:hidden-method',
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by='test',
            )

            def fail_if_manifest_loaded(_endpoint_id: str) -> Any:
                raise AssertionError('endpoint manifest should stay hidden before human approval')

            monkeypatch.setattr(runtime.store, 'get_jsonrpc_endpoint', fail_if_manifest_loaded)

            with pytest.raises(HumanApprovalRequired):
                runtime.jsonrpc.call(pid, 'secret-endpoint', 'hidden-method', {'city': 'Beijing'})
        finally:
            runtime.close()

    def test_call_visibility_honors_params_scoped_authority_rule(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc scoped visibility')
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest('scoped-visibility', server.url, with_header=False),
                    actor='cli',
                    require_capability=False,
                )
                params = {'x': 1}
                params_sha = hashlib.sha256(dumps(params).encode('utf-8')).hexdigest()
                runtime.capability.grant(
                    pid,
                    'jsonrpc:scoped-visibility:echo',
                    [CapabilityRight.READ],
                    issued_by='test',
                    constraints={
                        AUTHORITY_RULES_KEY: [
                            {
                                'rule_id': 'jsonrpc.scoped.visibility',
                                'operation': 'jsonrpc.call',
                                'effect': 'allow',
                                'risk': 'low',
                                'conditions': {
                                    'endpoint_id': 'scoped-visibility',
                                    'method_id': 'echo',
                                    'params_sha256': params_sha,
                                },
                            }
                        ]
                    },
                )

                result = runtime.jsonrpc.call(pid, 'scoped-visibility', 'echo', params)

                assert result.ok
                assert server.requests[0]['body']['params'] == params
            finally:
                runtime.close()

    def test_call_rejects_dns_resolution_to_private_address_before_provider_attempt(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        monkeypatch.setattr(
            'agent_libos.primitives.jsonrpc.socket.getaddrinfo',
            lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('169.254.169.254', 443))],
        )
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='dns pinning')
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('dns-safe-name', 'https://safe.example.test/jsonrpc', with_header=False), actor='cli', require_capability=False)
            cap = runtime.capability.grant_once(pid, 'jsonrpc:dns-safe-name:echo', [CapabilityRight.READ], issued_by='test')
            with pytest.raises(ValidationError, match='non-public|loopback|not allowed'):
                runtime.jsonrpc.call(pid, 'dns-safe-name', 'echo', {'x': 1})
            assert provider.calls == []
            persisted = runtime.store.get_capability(cap.cap_id)
            assert persisted is not None and persisted.uses_remaining == 0
            effects = runtime.store.list_external_effects(pid=pid)
            assert len(effects) == 1
            assert effects[0].effect_state == 'finalized'
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].information_flow
            assert effects[0].provider_metadata['phase'] == 'dns_resolution'
            flow = runtime.data_flow.current_context()
            assert flow.labels.trust_level.value == 'untrusted'
            assert flow.labels.integrity.value == 'untrusted'
        finally:
            runtime.close()

    def test_call_passes_validated_resolved_addresses_to_provider(self, monkeypatch: MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        monkeypatch.setattr(
            'agent_libos.primitives.jsonrpc.socket.getaddrinfo',
            lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', ('93.184.216.34', 443))],
        )
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='dns pinned provider')
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('pinned', 'https://safe.example.test/jsonrpc', with_header=False), actor='cli', require_capability=False)
            runtime.capability.grant(pid, 'jsonrpc:pinned:echo', [CapabilityRight.READ], issued_by='test')

            result = runtime.jsonrpc.call(pid, 'pinned', 'echo', {'x': 1})

            assert result.ok
            assert provider.kwargs[0]['resolved_addresses'] == ('93.184.216.34',)
        finally:
            runtime.close()

    def test_http_jsonrpc_provider_connects_to_pinned_address(self, monkeypatch: MonkeyPatch) -> None:
        provider = HttpJsonRpcProvider()
        endpoint = JsonRpcEndpointSpec(
            schema_version=1,
            endpoint_id='direct-pin',
            url='https://api.example.test/rpc',
            headers={},
            methods=[
                JsonRpcMethodSpec(
                    method_id='echo',
                    rpc_method='demo.echo',
                    right='read',
                    rollback_class='no_rollback_required',
                    state_mutation=False,
                    information_flow=True,
                )
            ],
            timeout_s=1,
            max_request_bytes=1024,
            max_response_bytes=1024,
        )
        called: list[tuple[str, int]] = []

        def fail_connect(address: tuple[str, int], *_args: Any, **_kwargs: Any) -> Any:
            called.append(address)
            raise OSError('stop before network')

        monkeypatch.setattr('agent_libos.substrate.local.socket.create_connection', fail_connect)

        result = provider.call(
            endpoint,
            endpoint.methods[0],
            b'{}',
            timeout_s=1,
            max_response_bytes=1024,
            resolved_addresses=('93.184.216.34',),
        )

        assert not result.status_code
        assert called == [('93.184.216.34', 443)]

    def test_jsonrpc_params_are_sanitized_in_audit_and_external_effects(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='remote secret params')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('secret-params', server.url, with_header=False), actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'jsonrpc:secret-params:echo', [CapabilityRight.READ], issued_by='test')
                result = runtime.jsonrpc.call(pid, 'secret-params', 'echo', {'token': 'jsonrpc-secret-token', 'city': 'Beijing'})
                assert result.ok
                assert server.requests[0]['body']['params']['token'] == 'jsonrpc-secret-token'
                persisted = json.dumps(
                    {
                        'audit': [record.__dict__ for record in runtime.audit.trace()],
                        'effects': [effect.__dict__ for effect in runtime.store.list_external_effects()],
                    },
                    sort_keys=True,
                    default=str,
                )
                assert 'jsonrpc-secret-token' not in persisted
                assert 'params_observation' in persisted
                assert 'redacted' in persisted
            finally:
                runtime.close()

    def test_params_schema_is_enforced_before_provider_call_and_one_shot_consumption(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='schema params')
                manifest = _manifest(
                    'schema-demo',
                    server.url,
                    with_header=False,
                    params_schema='\n      type: object\n      required: [city]\n      properties:\n        city:\n          type: string\n      additionalProperties: false\n',
                )
                runtime.jsonrpc.register_endpoint_from_yaml_text(manifest, actor='cli', require_capability=False)
                cap = runtime.capability.grant_once(pid, 'jsonrpc:schema-demo:echo', [CapabilityRight.READ], issued_by='test')

                with pytest.raises(ValidationError, match='params_schema'):
                    runtime.jsonrpc.call(pid, 'schema-demo', 'echo', {'city': 3})

                assert server.requests == []
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
                assert runtime.jsonrpc.call(pid, 'schema-demo', 'echo', {'city': 'Beijing'}).ok
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
            finally:
                runtime.close()

    def test_invalid_params_schema_is_rejected_at_registration(self) -> None:
        runtime = Runtime.open('local')
        try:
            manifest = _manifest(
                'bad-schema',
                'https://api.example.test/jsonrpc',
                with_header=False,
                params_schema='\n      type: definitely-not-a-json-schema-type\n',
            )
            with pytest.raises(ValidationError, match='params_schema'):
                runtime.jsonrpc.register_endpoint_from_yaml_text(manifest, actor='cli', require_capability=False)
        finally:
            runtime.close()

    def test_jsonrpc_errors_and_http_failures_are_structured_results(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='remote errors')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('errors', server.url, with_header=False), actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'jsonrpc:errors:echo', [CapabilityRight.READ], issued_by='test')
                server.mode = 'jsonrpc_error'
                rpc_error = runtime.jsonrpc.call(pid, 'errors', 'echo', {})
                assert not rpc_error.ok
                assert rpc_error.status.value == 'jsonrpc_error'
                assert rpc_error.error['code'] == -32000
                server.mode = 'http_500'
                http_error = runtime.jsonrpc.call(pid, 'errors', 'echo', {})
                assert not http_error.ok
                assert http_error.status.value == 'http_error'
                assert http_error.http_status == 500
            finally:
                runtime.close()

    def test_missing_header_env_fails_before_http_attempt_and_one_shot_consumption(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            monkeypatch.delenv('AGENT_LIBOS_JSONRPC_TEST_TOKEN', raising=False)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='missing env')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('needs-token', server.url), actor='cli', require_capability=False)
                cap = runtime.capability.grant_once(pid, 'jsonrpc:needs-token:echo', [CapabilityRight.READ], issued_by='test')
                with pytest.raises(ValidationError):
                    runtime.jsonrpc.call(pid, 'needs-token', 'echo', {})
                assert server.requests == []
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            finally:
                runtime.close()

    def test_invalid_header_env_value_fails_before_http_attempt_and_one_shot_consumption(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            monkeypatch.setenv('AGENT_LIBOS_JSONRPC_TEST_TOKEN', 'token\r\nX-Injected: yes')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='bad header env')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('bad-header-env', server.url), actor='cli', require_capability=False)
                cap = runtime.capability.grant_once(pid, 'jsonrpc:bad-header-env:echo', [CapabilityRight.READ], issued_by='test')
                with pytest.raises(ValidationError, match='header environment'):
                    runtime.jsonrpc.call(pid, 'bad-header-env', 'echo', {})
                assert server.requests == []
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            finally:
                runtime.close()

    def test_provider_secret_header_env_is_rejected_at_registration(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            monkeypatch.setenv('OPENAI_API_KEY', 'provider-secret')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='secret env registration')
                cap = runtime.capability.grant_once(pid, 'jsonrpc:secret-header:echo', [CapabilityRight.READ], issued_by='test')
                manifest = _manifest('secret-header', server.url).replace(
                    'AGENT_LIBOS_JSONRPC_TEST_TOKEN',
                    'OPENAI_API_KEY',
                )

                with pytest.raises(ValidationError, match='header env is not allowed'):
                    runtime.jsonrpc.register_endpoint_from_yaml_text(manifest, actor='cli', require_capability=False)

                assert server.requests == []
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 1
            finally:
                runtime.close()

    def test_configured_header_env_allowlist_permits_test_secret(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            config = replace(
                DEFAULT_CONFIG,
                jsonrpc=replace(DEFAULT_CONFIG.jsonrpc, header_env_allowlist=('CUSTOM_JSONRPC_TOKEN',)),
            )
            runtime = Runtime.open('local', config=config)
            monkeypatch.setenv('CUSTOM_JSONRPC_TOKEN', 'custom-token')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='custom env registration')
                manifest = _manifest('custom-header', server.url).replace(
                    'AGENT_LIBOS_JSONRPC_TEST_TOKEN',
                    'CUSTOM_JSONRPC_TOKEN',
                )
                runtime.jsonrpc.register_endpoint_from_yaml_text(manifest, actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'jsonrpc:custom-header:echo', [CapabilityRight.READ], issued_by='test')

                result = runtime.jsonrpc.call(pid, 'custom-header', 'echo', {'ok': True})

                assert result.ok
                assert server.requests[0]['authorization'] == 'Bearer custom-token'
            finally:
                runtime.close()

    def test_human_jsonrpc_approval_is_bound_to_params_hash(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='jsonrpc approval')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('approval', server.url, with_header=False), actor='cli', require_capability=False)
                runtime.capability.set_permission_policy(
                    pid,
                    'jsonrpc:approval:echo',
                    [CapabilityRight.READ],
                    runtime.capability.ASK_EACH_TIME,
                    issued_by='test',
                )
                with pytest.raises(HumanApprovalRequired):
                    runtime.jsonrpc.call(pid, 'approval', 'echo', {'x': 1})
                runtime.human.drain_terminal_queue(auto_approve=True)
                with pytest.raises(HumanApprovalRequired):
                    runtime.jsonrpc.call(pid, 'approval', 'echo', {'x': 2})
                assert server.requests == []
                result = runtime.jsonrpc.call(pid, 'approval', 'echo', {'x': 1})
                assert result.ok
                assert len(server.requests) == 1
            finally:
                runtime.close()

    def test_human_jsonrpc_approval_cannot_authorize_a_later_first_registration(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        provider = _RecordingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='future jsonrpc approval')
            resource = 'jsonrpc:future-approval:echo'
            runtime.capability.set_permission_policy(
                pid,
                resource,
                [CapabilityRight.READ],
                runtime.capability.ASK_EACH_TIME,
                issued_by='test',
            )

            with pytest.raises(HumanApprovalRequired):
                runtime.jsonrpc.call(pid, 'future-approval', 'echo', {'x': 1})
            first = runtime.human.pending()[0]
            first_context = first.payload['context']
            first_conditions = first.payload['requested_once_capability']['constraints'][
                AUTHORITY_RULES_KEY
            ][0]['conditions']
            assert len(first_context['registry_spec_sha256']) == 64
            assert first_conditions['registry_spec_sha256'] == first_context['registry_spec_sha256']
            assert first_conditions['registry_generation'] == first_context['registry_generation']
            runtime.human.drain_terminal_queue(auto_approve=True)

            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    'future-approval',
                    'https://safe.example.test/jsonrpc',
                    with_header=False,
                    rpc_method='dangerous.mutate',
                ),
                actor='cli',
                require_capability=False,
            )

            with pytest.raises(HumanApprovalRequired):
                runtime.jsonrpc.call(pid, 'future-approval', 'echo', {'x': 1})
            second = runtime.human.pending()[0]
            second_context = second.payload['context']
            assert second_context['registry_generation'] > first_context['registry_generation']
            assert second_context['registry_spec_sha256'] != first_context['registry_spec_sha256']
            assert provider.calls == []

            runtime.human.drain_terminal_queue(auto_approve=True)
            monkeypatch.setattr(
                runtime.jsonrpc,
                '_validate_runtime_resolution',
                lambda _spec: ('93.184.216.34',),
            )
            result = runtime.jsonrpc.call(pid, 'future-approval', 'echo', {'x': 1})
            assert result.ok
            assert len(provider.calls) == 1
        finally:
            runtime.close()

    def test_effectful_provider_without_classifier_fails_closed(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='missing classifier')
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('no-classifier', 'https://api.example.test/jsonrpc', with_header=False), actor='cli', require_capability=False)
            runtime.capability.grant(pid, 'jsonrpc:no-classifier:echo', [CapabilityRight.READ], issued_by='test')
            runtime.jsonrpc.provider = _ProviderWithoutClassifier()
            with pytest.raises(ValidationError):
                runtime.jsonrpc.call(pid, 'no-classifier', 'echo', {})
        finally:
            runtime.close()

    def test_syscall_bypasses_tool_table_but_not_capabilities(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='syscall remote')
                process = runtime.process.get(pid)
                process.tool_table.pop('call_jsonrpc_method', None)
                runtime.store.update_process(process)
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('sys', server.url, with_header=False), actor='cli', require_capability=False)
                session = LibOSSyscallSession(runtime, pid)
                with pytest.raises(CapabilityDenied):
                    self._run(session.handle('jsonrpc.call', {'endpoint_id': 'sys', 'method_id': 'echo', 'params': {}}))
                runtime.capability.grant(pid, 'jsonrpc:sys:echo', [CapabilityRight.READ], issued_by='test')
                result = self._run(session.handle('jsonrpc.call', {'endpoint_id': 'sys', 'method_id': 'echo', 'params': {}}))
                assert result['ok']
                assert 'call_jsonrpc_method' not in runtime.process.get(pid).tool_table
            finally:
                runtime.close()

    def test_checkpoint_reports_remote_effect_but_does_not_restore_endpoint_registry(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='checkpoint remote')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('ckpt', server.url, with_header=False), actor='cli', require_capability=False)
                runtime.capability.grant(pid, 'jsonrpc:ckpt:echo', [CapabilityRight.READ], issued_by='test')
                checkpoint_id = runtime.checkpoint.create(pid, 'before remote', actor=pid)
                runtime.jsonrpc.call(pid, 'ckpt', 'echo', {'x': 1})
                runtime.jsonrpc.unregister_endpoint('ckpt', actor='cli', require_capability=False)
                with pytest.raises(NotFound):
                    runtime.jsonrpc.inspect_endpoint('ckpt', require_capability=False)
                restored = runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
                assert restored['external_effect_summary']['by_provider_operation']['jsonrpc.call'] == 1
                with pytest.raises(NotFound):
                    runtime.jsonrpc.inspect_endpoint('ckpt', require_capability=False)
            finally:
                runtime.close()

    def test_endpoint_replace_requires_admin_and_disables_existing_method_grants(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                actor = runtime.process.spawn(image='base-agent:v0', goal='endpoint registrar')
                pid = runtime.process.spawn(image='base-agent:v0', goal='endpoint caller')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('replace-demo', server.url, with_header=False), actor='cli', require_capability=False)
                runtime.capability.grant(actor, 'jsonrpc_endpoint:replace-demo', [CapabilityRight.WRITE], issued_by='test')
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('replace-demo', server.url, with_header=False, rpc_method='demo.replaced'), actor=actor, replace=True)
                cap = runtime.capability.grant(pid, 'jsonrpc:replace-demo:echo', [CapabilityRight.READ], issued_by='test')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('replace-demo', server.url, with_header=False, rpc_method='demo.replaced'), actor='cli', replace=True, require_capability=False)
                assert not runtime.store.get_capability(cap.cap_id).active
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.call(pid, 'replace-demo', 'echo', {})
            finally:
                runtime.close()

    def test_endpoint_replace_disables_endpoint_wildcard_but_preserves_namespace_wildcard(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='endpoint wildcard caller')
                global_pid = runtime.process.spawn(image='base-agent:v0', goal='namespace wildcard caller')
                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest('replace-wildcard', server.url, with_header=False),
                    actor='cli',
                    require_capability=False,
                )
                cap = runtime.capability.grant(
                    pid,
                    'jsonrpc:replace-wildcard:*',
                    [CapabilityRight.READ],
                    issued_by='test',
                )
                global_cap = runtime.capability.grant(
                    global_pid,
                    'jsonrpc:*',
                    [CapabilityRight.READ],
                    issued_by='test',
                )

                runtime.jsonrpc.register_endpoint_from_yaml_text(
                    _manifest(
                        'replace-wildcard',
                        server.url,
                        with_header=False,
                        rpc_method='demo.replaced',
                    ),
                    actor='cli',
                    replace=True,
                    require_capability=False,
                )

                assert not runtime.store.get_capability(cap.cap_id).active
                assert runtime.store.get_capability(global_cap.cap_id).active
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.call(pid, 'replace-wildcard', 'echo', {})
                assert runtime.jsonrpc.call(
                    global_pid,
                    'replace-wildcard',
                    'echo',
                    {},
                ).ok
            finally:
                runtime.close()

    def test_endpoint_replace_rolls_back_spec_when_stale_grant_disable_fails(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='endpoint caller')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('replace-rollback', server.url, with_header=False), actor='cli', require_capability=False)
                cap = runtime.capability.grant(pid, 'jsonrpc:replace-rollback:echo', [CapabilityRight.READ], issued_by='test')

                def fail_disable(endpoint_id: str, *, actor: str) -> None:
                    raise RuntimeError(f'cannot disable grants for {endpoint_id}')

                monkeypatch.setattr(runtime.jsonrpc, '_disable_replaced_endpoint_method_capabilities', fail_disable)
                with pytest.raises(RuntimeError, match='cannot disable grants'):
                    runtime.jsonrpc.register_endpoint_from_yaml_text(
                        _manifest('replace-rollback', server.url, with_header=False, rpc_method='demo.replaced'),
                        actor='cli',
                        replace=True,
                        require_capability=False,
                    )

                spec, _metadata = runtime.store.get_jsonrpc_endpoint('replace-rollback')
                assert spec.methods[0].rpc_method == 'demo.echo'
                assert runtime.store.get_capability(cap.cap_id).active
            finally:
                runtime.close()

    def test_endpoint_unregister_disables_existing_method_grants_before_reuse(self) -> None:
        with _jsonrpc_server() as server:
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='endpoint reuse')
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('reuse-demo', server.url, with_header=False), actor='cli', require_capability=False)
                cap = runtime.capability.grant(pid, 'jsonrpc:reuse-demo:echo', [CapabilityRight.READ], issued_by='test')
                runtime.jsonrpc.unregister_endpoint('reuse-demo', actor='cli', require_capability=False)
                runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('reuse-demo', server.url, with_header=False), actor='cli', require_capability=False)

                assert not runtime.store.get_capability(cap.cap_id).active
                with pytest.raises(CapabilityDenied):
                    runtime.jsonrpc.call(pid, 'reuse-demo', 'echo', {})
            finally:
                runtime.close()

    def test_post_call_classifier_failure_records_conservative_external_effect(self) -> None:
        runtime = Runtime.open('local')
        provider = _PostCallClassifierFailsProvider()
        runtime.jsonrpc.provider = provider
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='classifier failure')
            runtime.jsonrpc.register_endpoint_from_yaml_text(_manifest('classify-after', 'http://127.0.0.1:9/rpc', with_header=False), actor='cli', require_capability=False)
            runtime.capability.grant(pid, 'jsonrpc:classify-after:echo', [CapabilityRight.READ], issued_by='test')
            result = runtime.jsonrpc.call(pid, 'classify-after', 'echo', {'x': 1})
            assert result.ok
            assert provider.calls == 1
            effect = runtime.store.list_external_effects()[0]
            assert effect.rollback_class == ExternalEffectRollbackClass.IRREVERSIBLE
            assert effect.rollback_status == ExternalEffectRollbackStatus.NOT_SUPPORTED
            assert effect.state_mutation
            assert effect.information_flow
            assert effect.provider_metadata['classification_fallback'] == 'post_call_failure'
        finally:
            runtime.close()

    def test_transport_exception_returns_error_but_keeps_effect_unknown(self) -> None:
        runtime = Runtime.open('local')
        provider = _FailingJsonRpcProvider()
        runtime.jsonrpc.provider = provider
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='transport failure classification')
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _manifest(
                    'transport-failure',
                    'http://127.0.0.1:9/rpc',
                    with_header=False,
                    state_mutation=True,
                    rollback_class='irreversible',
                ),
                actor='cli',
                require_capability=False,
            )
            runtime.capability.grant(
                pid,
                'jsonrpc:transport-failure:echo',
                [CapabilityRight.READ],
                issued_by='test',
            )

            result = runtime.jsonrpc.call(pid, 'transport-failure', 'echo', {'x': 1})

            assert result.status.value == 'transport_error'
            assert 'transport-secret' not in str(result.error)
            assert set(result.error or {}) == {'code', 'error_type', 'correlation_id'}
            effect = runtime.store.list_external_effects(pid=pid)[0]
            assert effect.transaction_state == 'unknown'
            assert effect.state_mutation
            assert effect.provider_metadata['outcome'] == 'unknown_transport_failure'
            assert 'transport-secret' not in str(effect.provider_metadata)
        finally:
            runtime.close()

    def test_cli_register_list_inspect_and_call(self, monkeypatch: MonkeyPatch) -> None:
        with _jsonrpc_server() as server, tempfile.TemporaryDirectory() as temp_dir:
            monkeypatch.setenv('AGENT_LIBOS_JSONRPC_TEST_TOKEN', 'cli-token')
            db = str(Path(temp_dir) / 'runtime.sqlite')
            manifest = Path(temp_dir) / 'endpoint.yaml'
            manifest.write_text(_manifest('cli-demo', server.url), encoding='utf-8')
            spawned = _run_cli_json(['--db', db, 'spawn', '--goal', 'cli jsonrpc'])
            registered = _run_cli_json(['--db', db, 'jsonrpc', 'register', str(manifest)])
            listed = _run_cli_json(['--db', db, 'jsonrpc', 'list'])
            inspected = _run_cli_json(['--db', db, 'jsonrpc', 'inspect', 'cli-demo'])
            _run_cli_json(['--db', db, 'capabilities', 'grant', spawned['pid'], 'jsonrpc:cli-demo:echo', '--rights', 'read'])
            called = _run_cli_json(['--db', db, 'jsonrpc', 'call', spawned['pid'], 'cli-demo', 'echo', '--params-json', '{"q": "ok"}'])
            assert registered['endpoint_id'] == 'cli-demo'
            assert registered['url'] is None
            assert listed[0]['endpoint_id'] == 'cli-demo'
            assert listed[0]['url'] is None
            assert inspected['url'] == server.url
            assert called['ok']

    def _run(self, awaitable: Any) -> Any:
        import asyncio
        return asyncio.run(awaitable)

class _JsonRpcServer:

    def __init__(self) -> None:
        self.mode = 'ok'
        self.requests: list[dict[str, Any]] = []
        self.httpd: HTTPServer | None = None
        self.thread: threading.Thread | None = None
        self.url = ''

    def __enter__(self) -> '_JsonRpcServer':
        owner = self

        class Handler(BaseHTTPRequestHandler):

            def do_POST(self) -> None:
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode('utf-8'))
                owner.requests.append({'path': self.path, 'authorization': self.headers.get('Authorization'), 'body': body})
                if owner.mode == 'http_500':
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'server failure')
                    return
                response: dict[str, Any]
                if owner.mode == 'jsonrpc_error':
                    response = {'jsonrpc': '2.0', 'id': body.get('id'), 'error': {'code': -32000, 'message': 'remote failure'}}
                else:
                    response = {'jsonrpc': '2.0', 'id': body.get('id'), 'result': {'method': body.get('method'), 'params': body.get('params')}}
                payload = json.dumps(response).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format: str, *_args: Any) -> None:
                return
        self.httpd = HTTPServer(('127.0.0.1', 0), Handler)
        port = int(self.httpd.server_address[1])
        self.url = f'http://127.0.0.1:{port}/rpc'
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)

def _jsonrpc_server() -> _JsonRpcServer:
    return _JsonRpcServer()

class _ProviderWithoutClassifier:

    def call(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('provider call should not run without an external-effect classifier')

class _RecordingJsonRpcProvider:

    def __init__(self) -> None:
        self.calls: list[bytes] = []
        self.kwargs: list[dict[str, Any]] = []

    def call(self, _endpoint: Any, _method: Any, request_body: bytes, **_kwargs: Any) -> JsonRpcTransportResult:
        self.calls.append(request_body)
        self.kwargs.append(dict(_kwargs))
        payload = json.loads(request_body.decode('utf-8'))
        return JsonRpcTransportResult(
            status_code=200,
            body=json.dumps({'jsonrpc': '2.0', 'id': payload.get('id'), 'result': {'ok': True}}).encode('utf-8'),
            elapsed_s=0.01,
            response_bytes=64,
        )

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={'operation': operation, 'endpoint_id': context.get('endpoint_id'), 'status': result.get('status') if isinstance(result, dict) else None},
        )


class _EnvelopeJsonRpcProvider(_RecordingJsonRpcProvider):
    def __init__(self, *, error: Any) -> None:
        super().__init__()
        self.error = error

    def call(
        self,
        _endpoint: Any,
        _method: Any,
        request_body: bytes,
        **_kwargs: Any,
    ) -> JsonRpcTransportResult:
        self.calls.append(request_body)
        payload = json.loads(request_body.decode("utf-8"))
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": self.error,
            }
        ).encode("utf-8")
        return JsonRpcTransportResult(
            status_code=200,
            body=body,
            elapsed_s=0.01,
            response_bytes=len(body),
        )


class _EnvironmentMutatingJsonRpcProvider(HttpJsonRpcProvider):

    def __init__(self, env_name: str) -> None:
        self.env_name = env_name
        self.resolved_headers: dict[str, str] = {}
        self.snapshot_was_immutable = False

    def call(self, *args: Any, **kwargs: Any) -> JsonRpcTransportResult:
        resolved_headers = kwargs.get('resolved_headers')
        self.resolved_headers = dict(resolved_headers or {})
        try:
            resolved_headers['Authorization'] = 'Bearer forged-token'
        except TypeError:
            self.snapshot_was_immutable = True
        os.environ[self.env_name] = 'attacker-token\r\nX-Injected: yes'
        return super().call(*args, **kwargs)

class _ExplodingJsonRpcTransportResult(JsonRpcTransportResult):
    def __init__(self, *, explode_field: str, secret: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_explode_field", explode_field)
        object.__setattr__(self, "_secret", secret)

    def __getattribute__(self, name: str) -> Any:
        if name not in {"_explode_field", "_secret"}:
            explode_field = object.__getattribute__(self, "_explode_field")
            if name == explode_field:
                raise RuntimeError(object.__getattribute__(self, "_secret"))
        return object.__getattribute__(self, name)


class _MalformedJsonRpcProvider(_RecordingJsonRpcProvider):
    def __init__(self, secret: str) -> None:
        super().__init__()
        self.secret = secret

    def call(
        self,
        _endpoint: Any,
        _method: Any,
        request_body: bytes,
        **_kwargs: Any,
    ) -> JsonRpcTransportResult:
        self.calls.append(request_body)
        payload = json.loads(request_body.decode("utf-8"))
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {"ok": True},
            }
        ).encode("utf-8")
        return _ExplodingJsonRpcTransportResult(
            explode_field="error",
            secret=self.secret,
            status_code=200,
            body=body,
            elapsed_s=0.01,
            response_bytes=len(body),
        )


class _NotStartedJsonRpcProvider(_RecordingJsonRpcProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def call(self, _endpoint: Any, _method: Any, request_body: bytes, **_kwargs: Any) -> JsonRpcTransportResult:
        self.calls += 1
        raise ProviderEffectNotStarted('jsonrpc failed before transport')

class _PostCallClassifierFailsProvider(_RecordingJsonRpcProvider):

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def call(self, _endpoint: Any, _method: Any, request_body: bytes, **_kwargs: Any) -> JsonRpcTransportResult:
        self.calls += 1
        payload = json.loads(request_body.decode('utf-8'))
        body = json.dumps({'jsonrpc': '2.0', 'id': payload.get('id'), 'result': {'ok': True}}).encode('utf-8')
        return JsonRpcTransportResult(status_code=200, body=body, elapsed_s=0.01, response_bytes=len(body))

    def classify_external_effect(self, operation: str, context: dict[str, Any], result: Any) -> ExternalEffectClassification:
        if isinstance(result, dict) and result.get('preflight'):
            return super().classify_external_effect(operation, context, result)
        raise RuntimeError('classifier failed after provider call')


class _FailingJsonRpcProvider(_RecordingJsonRpcProvider):
    def call(self, _endpoint: Any, _method: Any, request_body: bytes, **_kwargs: Any) -> JsonRpcTransportResult:
        self.calls.append(request_body)
        raise RuntimeError('transport-secret')

def _manifest_mapping(endpoint_id: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "endpoint_id": endpoint_id,
        "url": "https://api.example.test/jsonrpc",
        "headers": {},
        "methods": [
            {
                "method_id": "echo",
                "rpc_method": "demo.echo",
                "right": "read",
                "rollback_class": "no_rollback_required",
                "state_mutation": False,
                "information_flow": True,
                "params_schema": {},
                "metadata": {},
            }
        ],
        "metadata": {},
    }


def _manifest(endpoint_id: str, url: str, *, rollback_class: str | None='no_rollback_required', rollback_status: str | None=None, state_mutation: bool | str=False, with_header: bool=True, literal_header: bool=False, duplicate_method: bool=False, header_prefix: str='Bearer ', header_suffix: str='', rpc_method: str='demo.echo', params_schema: str='', timeout_s: str='5', max_request_bytes: str='65536', max_response_bytes: str='1048576') -> str:
    header = ''
    if literal_header:
        header = '\nheaders:\n  Authorization: "Bearer literal-secret"\n'
    elif with_header:
        header = f'\nheaders:\n  Authorization:\n    env: AGENT_LIBOS_JSONRPC_TEST_TOKEN\n    prefix: "{header_prefix}"\n'
        if header_suffix:
            header += f'    suffix: "{header_suffix}"\n'
    rollback = f'    rollback_class: {rollback_class}\n' if rollback_class is not None else ''
    rollback += f'    rollback_status: {rollback_status}\n' if rollback_status is not None else ''
    second = '\n  - method_id: echo\n    rpc_method: demo.echo2\n    right: read\n    rollback_class: no_rollback_required\n    state_mutation: false\n    information_flow: true\n' if duplicate_method else ''
    state = 'true' if state_mutation is True else 'false' if state_mutation is False else state_mutation
    schema = f'    params_schema:{params_schema}' if params_schema else ''
    return f'\nschema_version: 1\nendpoint_id: {endpoint_id}\nurl: {url}\n{header}methods:\n  - method_id: echo\n    rpc_method: {rpc_method}\n    right: read\n{rollback}    state_mutation: {state}\n    information_flow: true\n{schema}{second}timeout_s: {timeout_s}\nmax_request_bytes: {max_request_bytes}\nmax_response_bytes: {max_response_bytes}\n'.lstrip()

def _run_cli_json(argv: list[str]) -> Any:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli_main(argv)
    return json.loads(buffer.getvalue())
