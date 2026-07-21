from __future__ import annotations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest
import json
import sqlite3
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.llm.context_memory import context_object_name
from agent_libos.llm.executor import LLMProcessExecutor
from agent_libos.models.exceptions import ValidationError
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    DataFlowContext,
    DataLabels,
    EventType,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectMetadata,
    ObjectPatch,
    ObjectRight,
    ObjectType,
    PROMPT_MODE_LIBOS_DEFAULT,
    ProcessMessageKind,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
    SinkTrustLevel,
    SinkTrustRule,
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


def _new_llm_executor(runtime: Runtime) -> LLMProcessExecutor:
    return LLMProcessExecutor(
        unit_of_work=runtime.uow,
        process=runtime.process,
        operations=runtime.operations,
        data_flow=runtime.data_flow,
        tools=runtime.tools,
        resources=runtime.resources,
        llms=runtime.llms,
        memory=runtime.memory,
        audit=runtime.audit,
        events=runtime.events,
        images=runtime.images,
        messages=runtime.messages,
        human=runtime.human,
        skills=runtime.skills,
        protected_operations=runtime.protected_operations,
        authority_manifests=runtime.authority_manifests,
        capabilities=runtime.capability,
        config=runtime.config,
    )


class TestLLMContextMemory:

    def test_event_cursor_update_serializes_with_child_hierarchical_charge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            event = runtime.events.emit(
                EventType.PROCESS_MESSAGE_NOTICE,
                source=child,
                target=parent,
                payload={'kind': 'race probe'},
            )
            charge_started = threading.Event()
            charge_finished = threading.Event()
            worker: threading.Thread | None = None
            original_get = runtime.llm._process.get

            def charge_child() -> None:
                charge_started.set()
                runtime.resources.charge(
                    child,
                    ResourceUsage(tool_calls=1),
                    source='test.concurrent_child_charge',
                )
                charge_finished.set()

            def get_while_child_charge_waits(pid: str) -> Any:
                nonlocal worker
                current = original_get(pid)
                worker = threading.Thread(target=charge_child, daemon=True)
                worker.start()
                assert charge_started.wait(timeout=2)
                assert not charge_finished.wait(timeout=0.05)
                return current

            monkeypatch.setattr(runtime.llm._process, 'get', get_while_child_charge_waits)

            runtime.llm._advance_event_cursor(parent, event.event_id)
            monkeypatch.setattr(runtime.llm._process, 'get', original_get)

            assert worker is not None
            worker.join(timeout=2)
            assert not worker.is_alive()
            assert charge_finished.is_set()
            assert runtime.process.get(parent).event_cursor == event.event_id
            assert runtime.process.get(parent).resource_usage.tool_calls == 1
        finally:
            runtime.close()

    def test_context_root_update_serializes_with_child_hierarchical_charge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            process = runtime.process.get(parent)
            handle = runtime.llm.context_memory.ensure(
                parent,
                runtime.images[process.image_id],
                process,
                runtime.tools.model_visible_tools(parent),
            )
            charge_started = threading.Event()
            charge_finished = threading.Event()
            worker: threading.Thread | None = None
            original_require = runtime.llm.context_memory._require_process

            def charge_child() -> None:
                charge_started.set()
                runtime.resources.charge(
                    child,
                    ResourceUsage(tool_calls=1),
                    source='test.concurrent_child_charge',
                )
                charge_finished.set()

            def require_while_child_charge_waits(pid: str) -> Any:
                nonlocal worker
                current = original_require(pid)
                worker = threading.Thread(target=charge_child, daemon=True)
                worker.start()
                assert charge_started.wait(timeout=2)
                assert not charge_finished.wait(timeout=0.05)
                return current

            monkeypatch.setattr(
                runtime.llm.context_memory,
                '_require_process',
                require_while_child_charge_waits,
            )

            runtime.llm.context_memory._add_handle_to_view(parent, handle)

            assert worker is not None
            worker.join(timeout=2)
            assert not worker.is_alive()
            assert charge_finished.is_set()
            roots = runtime.process.get(parent).memory_view.roots
            assert handle.oid in {root.oid for root in roots}
            assert runtime.process.get(parent).resource_usage.tool_calls == 1
        finally:
            runtime.close()

    def test_llm_quantum_reads_a_store_bounded_event_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm_context=replace(DEFAULT_CONFIG.llm_context, recent_event_limit=3),
        )
        runtime = Runtime.open('local', config=config)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='bounded events')
            runtime.llm.client = RecordingActionClient([
                {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 2}},
            ])
            original_list_events = runtime.store.list_events
            calls: list[tuple[str | None, int | None, str | None]] = []

            def tracked_list_events(
                target: str | None = None,
                limit: int | None = None,
                before_event_id: str | None = None,
                after_event_id: str | None = None,
            ) -> list[Any]:
                calls.append((target, limit, after_event_id))
                return original_list_events(
                    target=target,
                    limit=limit,
                    before_event_id=before_event_id,
                    after_event_id=after_event_id,
                )

            monkeypatch.setattr(runtime.store, 'list_events', tracked_list_events)

            runtime.run_next_process_once()
            first_cursor = runtime.process.get(pid).event_cursor
            assert first_cursor is not None
            runtime.events.emit(
                EventType.EXTERNAL_WRITE,
                source='event-cursor-test',
                target=pid,
                payload={'step': 2},
            )
            runtime.run_next_process_once()

            assert (pid, 3, None) in calls
            assert (pid, 3, first_cursor) in calls
            assert runtime.process.get(pid).event_cursor != first_cursor
        finally:
            runtime.close()

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

    def test_llm_context_updates_and_compaction_preserve_highest_historical_labels(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient([
                {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 2}},
            ])
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    tenants=('tenant-a',),
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test',
                require_capability=False,
            )
            pid = runtime.process.spawn(image='base-agent:v0', goal='preserve context labels')
            tenant_a = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'tenant-a'},
                metadata=ObjectMetadata(sensitivity='secret', tenant='tenant-a'),
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(pid, [tenant_a], mode=ViewMode.READ_ONLY)
            runtime.store.update_process(process)

            first = runtime.run_next_process_once()
            assert first['ok'], first
            context = runtime.store.get_object_by_name(
                context_object_name(pid),
                namespace=runtime.memory.resolve_namespace(pid),
            )
            assert context is not None
            assert context.metadata.sensitivity == 'secret'
            assert context.metadata.tenant == 'tenant-a'
            assert context.payload['label_history']['sensitivity'] == 'secret'

            runtime.llm.context_memory.replace_with_compacted_summary(
                pid,
                context_oid=context.oid,
                expected_version=context.version,
                summary=_compact_summary('retain classified context'),
                compaction_method='test_compaction',
                preserve_recent_entries=1,
                source_tokens=1000,
                target_tokens=512,
                compressor_pids=[],
            )
            compacted = runtime.store.get_object(context.oid)
            assert compacted is not None
            assert compacted.metadata.sensitivity == 'secret'
            assert compacted.metadata.tenant == 'tenant-a'
            assert compacted.payload['label_history']['sensitivity'] == 'secret'

            later_normal = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'value': 'later normal context'},
                metadata=ObjectMetadata(sensitivity='normal', tenant='tenant-a'),
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(pid, [later_normal], mode=ViewMode.READ_ONLY)
            runtime.store.update_process(process)

            second = runtime.run_next_process_once()
            assert second['ok'], second
            updated = runtime.store.get_object(context.oid)
            assert updated is not None
            assert updated.metadata.sensitivity == 'secret'
            assert updated.metadata.tenant == 'tenant-a'
            assert updated.payload['label_history']['tenant'] == 'tenant-a'
        finally:
            runtime.close()

    def test_llm_message_event_labels_gate_provider_egress(self) -> None:
        runtime = Runtime.open('local')
        try:
            client = RecordingActionClient([
                {
                    'action': 'create_memory_object',
                    'type': 'observation',
                    'payload': {'received': True},
                }
            ])
            runtime.llm.client = client
            denied_pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='deny a secret message event',
            )
            denied_source = runtime.memory.create_object(
                denied_pid,
                ObjectType.EVIDENCE,
                {'secret': 'MESSAGE_EVENT_SECRET_SENTINEL'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            runtime.messages.post(
                sender='human:test',
                recipient_pid=denied_pid,
                subject='MESSAGE_EVENT_SECRET_SENTINEL',
                source_oids=[denied_source.oid],
            )

            denied = runtime.run_process_once(denied_pid)

            assert not denied['ok']
            assert 'data-flow denied egress' in denied['error']
            assert client.user_prompts == []
            denied_decisions = runtime.store.list_data_flow_decisions(
                pid=denied_pid,
                outcome='deny',
            )
            assert denied_decisions
            assert denied_decisions[-1].labels.sensitivity.value == 'secret'
            assert denied_decisions[-1].sink == 'llm:default'
            assert any(
                record.action == 'data_flow.egress'
                and record.target == 'llm:default'
                and record.decision.get('outcome') == 'deny'
                for record in runtime.audit.trace()
            )
            assert any(
                event.type == EventType.DATA_FLOW_DECISION
                and event.payload.get('outcome') == 'deny'
                for event in runtime.events.list(
                    target='data_flow_sink:llm:default',
                )
            )

            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='secret',
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test',
                require_capability=False,
            )
            allowed_pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='allow a secret message event',
            )
            allowed_source = runtime.memory.create_object(
                allowed_pid,
                ObjectType.EVIDENCE,
                {'secret': 'MESSAGE_EVENT_SECRET_SENTINEL'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            runtime.messages.post(
                sender='human:test',
                recipient_pid=allowed_pid,
                subject='MESSAGE_EVENT_SECRET_SENTINEL',
                source_oids=[allowed_source.oid],
            )

            allowed = runtime.run_process_once(allowed_pid)

            assert allowed['ok'], allowed
            assert len(client.user_prompts) == 1
            assert 'MESSAGE_EVENT_SECRET_SENTINEL' in client.user_prompts[0]
            context = runtime.store.get_object_by_name(
                context_object_name(allowed_pid),
                namespace=runtime.memory.resolve_namespace(allowed_pid),
            )
            assert context is not None
            assert context.metadata.sensitivity == 'secret'
            assert context.payload['label_history']['sensitivity'] == 'secret'
        finally:
            runtime.close()

    @pytest.mark.parametrize("publisher", ["create", "update", "append"])
    def test_secret_off_view_object_event_labels_gate_provider_egress(
        self,
        publisher: str,
    ) -> None:
        runtime = Runtime.open("local")
        try:
            client = RecordingActionClient(
                [
                    {
                        "action": "create_memory_object",
                        "type": "observation",
                        "payload": {"should_not_run": True},
                    }
                ]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal=f"deny a secret {publisher} Object event",
            )
            name = f"secret.{publisher}.event.marker"
            handle = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {"entries": []},
                metadata=ObjectMetadata(
                    sensitivity="secret",
                    trust_level="untrusted",
                    tenant="tenant-a",
                    principal="principal-a",
                ),
                name=name,
                immutable=False,
            )
            if publisher != "create":
                process = runtime.process.get(pid)
                process.event_cursor = runtime.store.list_events(target=pid)[-1].event_id
                runtime.store.update_process(process)
                if publisher == "update":
                    runtime.memory.update_object(
                        pid,
                        handle,
                        ObjectPatch(name=f"{name}.updated"),
                    )
                else:
                    runtime.memory.append_object_by_name(
                        pid,
                        name,
                        {"marker": "classified append"},
                    )

            object_event = runtime.store.list_events(target=pid)[-1]
            assert object_event.type in {EventType.OBJECT_CREATED, EventType.OBJECT_UPDATED}
            assert object_event.payload["data_labels"] == {
                "sensitivity": "secret",
                "trust_level": "untrusted",
                "integrity": "unknown",
                "origin": "local",
                "tenant": "tenant-a",
                "principal": "principal-a",
                "declassification_authority": None,
            }

            denied = runtime.run_process_once(pid)

            assert not denied["ok"]
            assert "data-flow denied egress" in denied["error"]
            assert client.user_prompts == []
            decisions = runtime.store.list_data_flow_decisions(pid=pid, outcome="deny")
            assert decisions
            assert decisions[-1].labels.sensitivity.value == "secret"
            assert decisions[-1].labels.tenant == "tenant-a"
            assert decisions[-1].labels.principal == "principal-a"
        finally:
            runtime.close()

    def test_normal_off_view_object_event_still_reaches_provider(self) -> None:
        runtime = Runtime.open("local")
        try:
            client = RecordingActionClient(
                [{"action": "process_exit", "payload": {"done": True}}]
            )
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="allow a normal Object event",
            )
            runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {"value": "normal"},
                metadata=ObjectMetadata(sensitivity="normal"),
                name="normal.object.event.marker",
            )

            allowed = runtime.run_process_once(pid)

            assert allowed["ok"], allowed
            assert len(client.user_prompts) == 1
            assert "normal.object.event.marker" in client.user_prompts[0]
        finally:
            runtime.close()

    def test_llm_context_label_high_water_survives_persistent_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = RecordingActionClient([
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                ])
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern='llm:default',
                        trust_level=SinkTrustLevel.TRUSTED,
                        max_sensitivity='secret',
                        tenants=('tenant-a',),
                        principals=('analyst-a',),
                        identity_sha256=runtime.llms.profile_identity_sha256('default'),
                    ),
                    actor='test',
                    require_capability=False,
                )
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='persist context label high water',
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {'secret': 'persistent-context-sentinel'},
                    metadata=ObjectMetadata(
                        sensitivity='secret',
                        tenant='tenant-a',
                        principal='analyst-a',
                    ),
                )
                process = runtime.process.get(pid)
                process.memory_view = runtime.memory.create_view(
                    pid,
                    [source],
                    mode=ViewMode.READ_ONLY,
                )
                runtime.store.update_process(process)

                first = runtime.run_next_process_once()
                assert first['ok'], first
                before = runtime.store.get_object_by_name(
                    context_object_name(pid),
                    namespace=runtime.memory.resolve_namespace(pid),
                )
                assert before is not None
                assert before.metadata.sensitivity == 'secret'
                assert before.metadata.tenant == 'tenant-a'
                assert before.metadata.principal == 'analyst-a'
                assert before.payload['label_history']['sensitivity'] == 'secret'
                before_oid = before.oid
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                process = reopened.process.get(pid)
                handle = reopened.llm.context_memory.ensure(
                    pid,
                    reopened.images[process.image_id],
                    process,
                    reopened.tools.visible_tools(pid),
                )
                restored = reopened.memory.get_object(pid, handle)

                assert restored.oid != before_oid
                assert restored.metadata.sensitivity == 'secret'
                assert restored.metadata.tenant == 'tenant-a'
                assert restored.metadata.principal == 'analyst-a'
                assert restored.payload['label_history']['sensitivity'] == 'secret'
                assert restored.payload['label_history']['tenant'] == 'tenant-a'
                assert restored.payload['label_history']['principal'] == 'analyst-a'

                later_normal = reopened.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {'value': 'later normal context'},
                    metadata=ObjectMetadata(
                        sensitivity='normal',
                        tenant='tenant-a',
                        principal='analyst-a',
                    ),
                )
                process = reopened.process.get(pid)
                process.memory_view = reopened.memory.create_view(
                    pid,
                    [later_normal],
                    mode=ViewMode.READ_ONLY,
                )
                reopened.store.update_process(process)
                reopened.llm.client = RecordingActionClient([
                    {
                        'action': 'create_memory_object',
                        'type': 'observation',
                        'payload': {'step': 2},
                    },
                ])

                second = reopened.run_process_once(pid)
                assert second['ok'], second
                updated = reopened.store.get_object(restored.oid)
                assert updated is not None
                assert updated.metadata.sensitivity == 'secret'
                assert updated.metadata.tenant == 'tenant-a'
                assert updated.metadata.principal == 'analyst-a'
            finally:
                reopened.close()


    def test_context_label_history_reopen_does_not_rewrite_existing_high_water(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='keep watermark update stable',
                )
                runtime.memory.create_object(
                    pid,
                    ObjectType.PROCESS_STATE,
                    {'kind': 'llm_context', 'entries': []},
                    metadata=ObjectMetadata(sensitivity='secret'),
                    name=context_object_name(pid),
                    immutable=False,
                )
                runtime.store.merge_llm_context_label_history(
                    pid,
                    DataLabels(sensitivity='secret'),
                )
                before = runtime.store.select_table_rows(
                    'llm_context_generations',
                    'pid = ?',
                    (pid,),
                )[0]['updated_at']
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                after = reopened.store.select_table_rows(
                    'llm_context_generations',
                    'pid = ?',
                    (pid,),
                )[0]['updated_at']
                assert after == before
            finally:
                reopened.close()

    def test_conditional_llm_release_replays_exact_prepared_request_once(self) -> None:
        runtime = Runtime.open('local')
        try:
            client = RecordingActionClient([
                {'action': 'process_exit', 'payload': {'done': True}},
            ])
            runtime.llm.client = client
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.CONDITIONAL,
                    max_sensitivity='secret',
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test.host',
                require_capability=False,
            )
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='resume exact conditional LLM request',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'CONDITIONAL_LLM_RELEASE_SENTINEL'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(
                pid,
                [source],
                mode=ViewMode.READ_ONLY,
            )
            runtime.store.update_process(process)

            waiting = runtime.run_next_process_once()

            assert waiting['waiting_human']
            assert client.user_prompts == []
            durable = runtime.store.get_llm_pending_action(pid)
            assert durable is not None
            assert durable['wait_type'] == 'llm_release'
            request_id = waiting['request_id']
            runtime.human.drain_terminal_queue(auto_approve=True)

            replacement = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'value': 'MUST_NOT_REBUILD_APPROVED_LLM_REQUEST'},
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(
                pid,
                [replacement],
                mode=ViewMode.READ_ONLY,
            )
            runtime.store.update_process(process)

            resumed = runtime.run_next_process_once()

            assert resumed['ok'], resumed
            assert resumed['resumed_after_human']
            assert resumed['action']['action'] == 'process_exit'
            assert len(client.user_prompts) == 1
            assert 'MUST_NOT_REBUILD_APPROVED_LLM_REQUEST' not in client.user_prompts[0]
            assert runtime.human.get(request_id).status.value == 'approved'
            assert runtime.human.pending() == []
            completed = runtime.store.get_llm_pending_action(pid)
            assert completed is not None and completed['status'] == 'completed'
            assert len([
                record
                for record in runtime.audit.trace()
                if record.action == 'llm.release_waiting_human'
            ]) == 1
        finally:
            runtime.close()

    def test_rejected_conditional_llm_release_pauses_without_reprompting(self) -> None:
        runtime = Runtime.open('local')
        try:
            client = RecordingActionClient([
                {'action': 'process_exit', 'payload': {'must_not_run': True}},
            ])
            runtime.llm.client = client
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.CONDITIONAL,
                    max_sensitivity='secret',
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test.host',
                require_capability=False,
            )
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='do not recreate a rejected conditional LLM request',
            )
            source = runtime.memory.create_object(
                pid,
                ObjectType.EVIDENCE,
                {'secret': 'REJECTED_LLM_RELEASE_SENTINEL'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            process = runtime.process.get(pid)
            process.memory_view = runtime.memory.create_view(
                pid,
                [source],
                mode=ViewMode.READ_ONLY,
            )
            runtime.store.update_process(process)

            waiting = runtime.run_next_process_once()
            request_id = waiting['request_id']
            runtime.human.reject(request_id, {'approved': False, 'reason': 'test rejection'})

            rejected = runtime.run_next_process_once()

            assert rejected['llm_release_rejected'] is True
            assert rejected['request_id'] == request_id
            assert runtime.process.get(pid).status == ProcessStatus.PAUSED
            assert runtime.run_next_process_once() is None
            assert runtime.human.pending() == []
            assert client.user_prompts == []
            assert len([
                record
                for record in runtime.audit.trace()
                if record.action == 'llm.release_waiting_human'
            ]) == 1
        finally:
            runtime.close()

    def test_parent_cannot_resume_child_after_conditional_llm_release_rejection(self) -> None:
        runtime = Runtime.open('local')
        try:
            client = RecordingActionClient([
                {'action': 'process_exit', 'payload': {'must_not_run': True}},
            ])
            runtime.llm.client = client
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.CONDITIONAL,
                    max_sensitivity='secret',
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test.host',
                require_capability=False,
            )
            parent = runtime.process.spawn(image='base-agent:v0', goal='manage child')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(
                parent,
                'require Host resume after rejected LLM release',
            )
            source = runtime.memory.create_object(
                child,
                ObjectType.EVIDENCE,
                {'secret': 'REJECTED_CHILD_LLM_RELEASE_SENTINEL'},
                metadata=ObjectMetadata(sensitivity='secret'),
            )
            process = runtime.process.get(child)
            process.memory_view = runtime.memory.create_view(
                child,
                [source],
                mode=ViewMode.READ_ONLY,
            )
            runtime.store.update_process(process)

            waiting = runtime.run_process_once(child)
            request_id = waiting['request_id']
            runtime.human.reject(request_id, {'approved': False, 'reason': 'test rejection'})
            rejected = runtime.run_process_once(child)

            assert rejected['llm_release_rejected'] is True
            assert runtime.process.get(child).status == ProcessStatus.PAUSED

            gated_message = runtime.process.get(child).status_message
            model_pause = runtime.tools.call(
                parent,
                'signal_child_process',
                {
                    'child_pid': child,
                    'signal': 'pause',
                    'reason': 'model must not replace the Host-only gate',
                },
            )

            assert model_pause.ok, model_pause.error
            assert runtime.process.get(child).status_message == gated_message

            model_resume = runtime.tools.call(
                parent,
                'signal_child_process',
                {'child_pid': child, 'signal': 'resume'},
            )

            assert not model_resume.ok
            assert 'Host resume' in (model_resume.error or '')
            assert runtime.process.get(child).status == ProcessStatus.PAUSED
            assert runtime.run_process_once(child)['skipped'] is True
            assert runtime.human.pending() == []
            assert client.user_prompts == []
            assert len([
                record
                for record in runtime.audit.trace()
                if record.action == 'llm.release_waiting_human'
            ]) == 1

            runtime.process.resume(child)
            renewed = runtime.run_process_once(child)

            assert renewed['waiting_human'] is True
            assert renewed['request_id'] != request_id
            assert len([
                record
                for record in runtime.audit.trace()
                if record.action == 'llm.release_waiting_human'
            ]) == 2
        finally:
            runtime.close()

    def test_conditional_llm_release_opt_out_does_not_persist_prepared_prompt(self) -> None:
        sentinel = 'CONDITIONAL_LLM_RELEASE_NO_RETENTION_SENTINEL'
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(
                DEFAULT_CONFIG,
                llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
            )
            runtime = Runtime.open(db, config=config)
            try:
                client = RecordingActionClient([
                    {'action': 'process_exit', 'payload': {'done': True}},
                ])
                runtime.llm.client = client
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern='llm:default',
                        trust_level=SinkTrustLevel.CONDITIONAL,
                        max_sensitivity='secret',
                        identity_sha256=runtime.llms.profile_identity_sha256('default'),
                    ),
                    actor='test.host',
                    require_capability=False,
                )
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='do not retain a conditional LLM prompt',
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {'secret': sentinel},
                    metadata=ObjectMetadata(sensitivity='secret'),
                )
                process = runtime.process.get(pid)
                process.memory_view = runtime.memory.create_view(
                    pid,
                    [source],
                    mode=ViewMode.READ_ONLY,
                )
                runtime.store.update_process(process)

                waiting = runtime.run_next_process_once()

                assert waiting['waiting_human']
                assert client.user_prompts == []
                rows = runtime.store.select_table_rows(
                    'llm_pending_actions',
                    'pid = ?',
                    (pid,),
                )
                assert len(rows) == 1
                assert sentinel not in rows[0]['action_json']
                durable = runtime.store.get_llm_pending_action(pid)
                assert durable is not None
                assert durable['action']['kind'] == 'llm_release_request_redacted'
                assert durable['action']['prepared_request_sha256']
                assert 'request_messages' not in durable['action']
                assert 'egress_payload' not in durable['action']

                runtime.human.drain_terminal_queue(auto_approve=True)
                resumed = runtime.run_next_process_once()

                assert resumed['ok'], resumed
                assert resumed['resumed_after_human']
                assert len(client.user_prompts) == 1
                assert sentinel in client.user_prompts[0]
                completed = runtime.store.get_llm_pending_action(pid)
                assert completed is not None and completed['status'] == 'completed'
                assert sentinel not in json.dumps(completed['action'], sort_keys=True)
            finally:
                runtime.close()

    def test_redacted_conditional_llm_release_fails_closed_after_reopen(self) -> None:
        sentinel = 'CONDITIONAL_LLM_RELEASE_REOPEN_SENTINEL'
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(
                DEFAULT_CONFIG,
                llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
            )
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = RecordingActionClient([
                    {'action': 'process_exit', 'payload': {'must_not_run': True}},
                ])
                runtime.data_flow.register_sink_trust(
                    SinkTrustRule(
                        pattern='llm:default',
                        trust_level=SinkTrustLevel.CONDITIONAL,
                        max_sensitivity='secret',
                        identity_sha256=runtime.llms.profile_identity_sha256('default'),
                    ),
                    actor='test.host',
                    require_capability=False,
                )
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='fail a redacted release closed after reopen',
                )
                source = runtime.memory.create_object(
                    pid,
                    ObjectType.EVIDENCE,
                    {'secret': sentinel},
                    metadata=ObjectMetadata(sensitivity='secret'),
                )
                process = runtime.process.get(pid)
                process.memory_view = runtime.memory.create_view(
                    pid,
                    [source],
                    mode=ViewMode.READ_ONLY,
                )
                runtime.store.update_process(process)
                waiting = runtime.run_next_process_once()
                request_id = waiting['request_id']
            finally:
                runtime.close()

            reopened = Runtime.open(db, config=config)
            try:
                client = RecordingActionClient([
                    {'action': 'process_exit', 'payload': {'must_not_run': True}},
                ])
                reopened.llm.client = client

                assert client.user_prompts == []
                assert reopened.process.get(pid).status == ProcessStatus.FAILED
                assert reopened.human.get(request_id).status.value == 'cancelled'
                pending = reopened.store.get_llm_pending_action(pid)
                assert pending is not None and pending['status'] == 'resuming'
                assert any(
                    record.action == 'llm.release_resume_payload_unavailable'
                    and record.target == f'human_request:{request_id}'
                    and record.decision.get('replayed') is False
                    for record in reopened.audit.trace()
                )
                assert any(
                    event.type == EventType.PROCESS_EXITED
                    and event.source == pid
                    and event.payload.get('status') == ProcessStatus.FAILED.value
                    for event in reopened.events.list()
                )
            finally:
                reopened.close()

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

    def test_parallel_tool_calls_disabled_uses_existing_single_action_path(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                    {'action': 'process_exit', 'payload': {'done': True}},
                ]
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='parallel disabled')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['action']['action'] == 'process_exit'
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'create_memory_object'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_parallel_tool_calls_execute_batch_in_order_and_record_observability(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 1}},
                    {'action': 'process_exit', 'payload': {'done': True}},
                ]
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='parallel enabled')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['parallel_tool_calls'] is True
            assert result['executed_count'] == 2
            assert [action['action'] for action in result['actions']] == ['create_memory_object', 'process_exit']
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            tool_calls = [
                record.decision.get('tool')
                for record in runtime.audit.trace()
                if record.action == 'tool.call'
            ]
            assert tool_calls[:2] == ['create_memory_object', 'process_exit']
            llm_call = runtime.store.list_llm_calls(pid)[0]
            assert llm_call.request_options['openai_parallel_tool_calls_enabled'] is True
            assert llm_call.tool_calls[0]['name'] == 'create_memory_object'
            assert llm_call.observability['tool_calls']['sha256']
            llm_operation = next(
                operation
                for operation in runtime.store.list_operations(pid=pid)
                if operation.name == 'llm.action_selection'
            )
            tool_operations = [
                operation
                for operation in runtime.store.list_operations(root_operation_id=llm_operation.operation_id)
                if operation.kind.value == 'tool_call'
            ]
            assert {operation.name for operation in tool_operations} == {
                'tool.create_memory_object',
                'tool.process_exit',
            }
            assert {operation.parent_operation_id for operation in tool_operations} == {llm_operation.operation_id}
            batches = [record for record in runtime.audit.trace() if record.action == 'llm.action_batch']
            assert len(batches) == 1
            assert batches[0].decision['executed_count'] == 2
            assert batches[0].decision['stop_reason'] == 'process_terminal'
        finally:
            runtime.close()

    def test_parallel_tool_calls_invalid_batch_repairs_before_dispatch(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True, action_repair_attempts=2),
        )
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'should_not_run': True}},
                    {'action': 'missing_tool', 'payload': {'bad': True}},
                ],
                [{'action': 'process_exit', 'payload': {'done': True}}],
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='repair invalid batch')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['action']['action'] == 'process_exit'
            assert len(runtime.llm.client.user_prompts) == 2
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'create_memory_object'
                for record in runtime.audit.trace()
            )
            repairs = [record for record in runtime.audit.trace() if record.action == 'llm.action_repair_requested']
            assert len(repairs) == 1
            assert 'missing_tool' in repairs[0].decision['error']
        finally:
            runtime.close()

    def test_parallel_tool_calls_stop_after_process_exit(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'process_exit', 'payload': {'done': True}},
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'should_not_run': True}},
                ]
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='stop after exit')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['executed_count'] == 1
            assert result['action']['action'] == 'process_exit'
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'create_memory_object'
                for record in runtime.audit.trace()
            )
            batch = [record for record in runtime.audit.trace() if record.action == 'llm.action_batch'][0]
            assert batch.decision['stop_reason'] == 'process_terminal'
        finally:
            runtime.close()

    def test_parallel_tool_calls_interrupt_notice_is_not_counted_as_executed(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'get_current_time', 'timezone': 'UTC'},
                    {'action': 'process_exit', 'payload': {'should_not_run': True}},
                ]
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='parallel interrupt')
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                kind=ProcessMessageKind.INTERRUPT,
                subject='urgent',
            )

            result = runtime.run_process_once(pid)

            assert result['ok']
            assert result['parallel_tool_calls'] is True
            assert result['executed_count'] == 0
            assert result['actions'] == []
            assert result['results'] == []
            assert result['stop_reason'] == 'interrupted_by_message'
            assert result['action']['action'] == 'get_current_time'
            assert result['result']['interrupted_by_message']
            assert result['stopped_action']['action'] == 'get_current_time'
            assert result['stopped_result']['interrupted_by_message']
            assert not any(record.action == 'primitive.clock.now' for record in runtime.audit.trace())
            assert not any(record.action == 'tool.call' for record in runtime.audit.trace())
            batch = [record for record in runtime.audit.trace() if record.action == 'llm.action_batch'][0]
            assert batch.decision['executed_count'] == 0
            assert batch.decision['stop_reason'] == 'interrupted_by_message'
        finally:
            runtime.close()

    def test_parallel_tool_calls_tool_failure_keeps_quantum_ok(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'get_current_time', 'timezone': 'Mars/Olympus'},
                    {'action': 'process_exit', 'payload': {'should_not_run': True}},
                ]
            ])
            pid = runtime.process.spawn(image='base-agent:v0', goal='parallel tool failure')

            result = runtime.run_process_once(pid)

            assert result['ok']
            assert result['parallel_tool_calls'] is True
            assert result['executed_count'] == 1
            assert result['action']['action'] == 'get_current_time'
            assert result['result']['ok'] is False
            assert 'unknown timezone' in (result['result']['error'] or '')
            assert result['stop_reason'] == 'tool_failed'
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_parallel_tool_calls_message_wait_stops_batch_and_resumes_pending_action(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 'before-wait'}},
                    {'action': 'receive_process_messages', 'channel': 'control', 'correlation_id': 'job-1'},
                    {'action': 'process_exit', 'payload': {'should_not_run': True}},
                ]
            ])
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'parallel message wait')

            waiting = runtime.run_process_once(child)

            assert waiting['waiting_message']
            assert waiting['parallel_tool_calls'] is True
            assert waiting['executed_count'] == 1
            assert waiting['completed_actions'][0]['action'] == 'create_memory_object'
            assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
            pending = runtime.store.get_llm_pending_action(child)
            assert pending is not None
            waiting_llm_operation = pending['llm_operation_id']
            waiting_tool_operation = pending['tool_operation_id']
            assert waiting_llm_operation
            assert waiting_tool_operation
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )

            matching = runtime.messages.send_from_process(
                parent,
                child,
                channel='control',
                correlation_id='job-1',
                subject='resume',
                payload={'ready': True},
            )
            resumed = runtime.run_process_once(child)

            assert resumed['ok']
            assert resumed['resumed_after_message']
            assert resumed['action']['action'] == 'receive_process_messages'
            assert resumed['result']['payload']['messages'][0]['message_id'] == matching.message_id
            assert len(runtime.llm.client.user_prompts) == 1
            assert runtime.store.get_operation(waiting_llm_operation).outcome.value == 'succeeded'
            assert runtime.store.get_operation(waiting_tool_operation).outcome.value == 'succeeded'
            assert len([
                operation
                for operation in runtime.store.list_operations(pid=child)
                if operation.name == 'tool.receive_process_messages'
            ]) == 1
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_parallel_tool_calls_child_wait_stops_batch_and_resumes_pending_action(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parallel child wait')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'still running')
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 'before-wait'}},
                    {'action': 'wait_child_process', 'child_pid': child},
                    {'action': 'process_exit', 'payload': {'should_not_run': True}},
                ]
            ])

            waiting = runtime.run_process_once(parent)

            assert waiting['waiting_event']
            assert waiting['parallel_tool_calls'] is True
            assert waiting['executed_count'] == 1
            assert waiting['completed_actions'][0]['action'] == 'create_memory_object'
            assert runtime.process.get(parent).status == ProcessStatus.WAITING_EVENT
            pending = runtime.store.get_llm_pending_action(parent)
            assert pending is not None
            waiting_llm_operation = pending['llm_operation_id']
            waiting_tool_operation = pending['tool_operation_id']
            assert waiting_llm_operation
            assert waiting_tool_operation
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )

            runtime.process.exit(child, message='done')
            resumed = runtime.run_process_once(parent)

            assert resumed['ok']
            assert resumed['action']['action'] == 'wait_child_process'
            assert len(runtime.llm.client.user_prompts) == 1
            assert runtime.store.get_operation(waiting_llm_operation).outcome.value == 'succeeded'
            assert runtime.store.get_operation(waiting_tool_operation).outcome.value == 'succeeded'
            assert len([
                operation
                for operation in runtime.store.list_operations(pid=parent)
                if operation.name == 'tool.wait_child_process'
            ]) == 1
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_parallel_tool_calls_human_wait_stops_batch_and_resumes_pending_action(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, parallel_tool_calls=True))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = MultiToolActionClient([
                [
                    {'action': 'create_memory_object', 'type': 'observation', 'payload': {'step': 'before-human'}},
                    {'action': 'ask_human', 'question': 'Continue?'},
                    {'action': 'process_exit', 'payload': {'should_not_run': True}},
                ]
            ])
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='parallel human wait',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'human:owner', 'rights': ['write']},
                    ],
                },
            )

            waiting = runtime.run_process_once(pid)

            assert waiting['waiting_human']
            assert waiting['parallel_tool_calls'] is True
            assert waiting['executed_count'] == 1
            assert waiting['completed_actions'][0]['action'] == 'create_memory_object'
            assert runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )

            request_id = waiting['request_id']
            runtime.human.drain_terminal_queue(auto_answer='yes')
            resumed = runtime.run_process_once(pid)

            assert resumed['ok']
            assert resumed['resumed_after_human']
            assert resumed['action']['action'] == 'ask_human'
            assert resumed['result']['payload']['request_id'] == request_id
            assert resumed['result']['payload']['answer'] == 'yes'
            assert len(runtime.llm.client.user_prompts) == 1
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'process_exit'
                for record in runtime.audit.trace()
            )
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

    def test_llm_call_records_persist_full_io_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = MetadataActionClient()
                pid = runtime.process.spawn(image='base-agent:v0', goal='persist full llm calls by default')
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
                assert call.messages[1]['content']
                assert 'persist full llm calls by default' in call.messages[1]['content']
                assert any((tool['function']['name'] == 'process_exit' for tool in call.tools))
                assert call.tool_calls[0]['name'] == 'process_exit'
                assert call.tool_calls[0]['arguments'] == json.dumps({'payload': {'done': True}})
                assert call.raw_response['id'] == 'raw_resp'
                assert call.reasoning == {'summary': 'selected process_exit'}
                assert call.observability['response_content']['sha256']
                serialized = json.dumps(call.__dict__, sort_keys=True)
                assert 'persist full llm calls by default' in serialized
            finally:
                runtime.close()
            reopened = Runtime.open(db)
            try:
                persisted = reopened.store.list_llm_calls()
                assert len(persisted) == 1
                assert persisted[0].usage['prompt_tokens'] == 13
                assert 'persist full llm calls by default' in persisted[0].messages[1]['content']
                assert persisted[0].raw_response['provider'] == 'fake'
                assert persisted[0].observability['messages']['bytes'] > 0
            finally:
                reopened.close()

    def test_llm_call_records_can_opt_out_of_full_io_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False))
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = MetadataActionClient()
                pid = runtime.process.spawn(image='base-agent:v0', goal='do not persist full llm calls')
                runtime.run_next_process_once()

                call = runtime.store.list_llm_calls(pid)[0]
                assert call.response_content == 'visible assistant text'
                assert call.messages['sha256']
                assert call.tools['sha256']
                assert call.tool_calls['sha256']
                assert call.raw_response['sha256']
                assert call.reasoning['sha256']
                assert call.observability['messages']['sha256']
                serialized = json.dumps(call.__dict__, sort_keys=True)
                assert 'do not persist full llm calls' not in serialized
                assert '"payload": {"done": true}' not in serialized
            finally:
                runtime.close()

            reopened = Runtime.open(db, config=config)
            try:
                persisted = reopened.store.list_llm_calls(pid)[0]
                assert persisted.messages['sha256']
                assert persisted.raw_response['sha256']
            finally:
                reopened.close()

    def test_openai_responses_state_chaining_is_opt_in_and_observable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(
                DEFAULT_CONFIG,
                llm=replace(
                    DEFAULT_CONFIG.llm,
                    store=True,
                    responses_previous_response_id=True,
                    safety_identifier='safe-session',
                    prompt_cache_key='cache-secret',
                    prompt_cache_retention='24h',
                ),
            )
            client = LLMClient(
                model='gpt-test',
                api_key='key',
                api_mode='responses',
                store=True,
                responses_previous_response_id=True,
                safety_identifier='safe-session',
                prompt_cache_key='cache-secret',
                prompt_cache_retention='24h',
                defaults=config.llm,
            )
            fake = FakeAsyncOpenAIResponses(
                [
                    _responses_tool_call(
                        'resp_first',
                        'create_memory_object',
                        {'type': 'note', 'name': 'step', 'payload': {'ok': True}},
                    ),
                    _responses_tool_call('resp_second', 'process_exit', {'payload': {'done': True}}),
                ]
            )
            client._async_client = fake
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = client
                pid = runtime.process.spawn(image='base-agent:v0', goal='chain responses state')
                first = runtime.run_next_process_once()
                second = runtime.run_next_process_once()

                assert first['action']['action'] == 'create_memory_object'
                assert second['action']['action'] == 'process_exit'
                assert 'previous_response_id' not in fake.responses.payloads[0]
                assert fake.responses.payloads[1]['previous_response_id'] == 'resp_first'
                chained_output = next(
                    item for item in fake.responses.payloads[1]['input']
                    if item.get('type') == 'function_call_output'
                )
                assert chained_output['call_id'] == 'call_resp_first'
                assert json.loads(chained_output['output']) == first['result']
                assert fake.responses.payloads[0]['safety_identifier'] == 'safe-session'
                assert fake.responses.payloads[0]['prompt_cache_key'] == 'cache-secret'
                assert fake.responses.payloads[0]['prompt_cache_retention'] == '24h'

                calls = runtime.store.list_llm_calls(pid)
                assert [call.response_id for call in calls] == ['resp_first', 'resp_second']
                assert calls[0].request_options['openai_responses_previous_response_id_enabled'] is True
                assert calls[0].request_options['openai_previous_response_id'] is None
                assert calls[1].request_options['openai_previous_response_id'] == 'resp_first'
                assert calls[0].request_options['openai_prompt_cache_key_configured'] is True
                assert calls[0].request_options['openai_safety_identifier_configured'] is True
                assert calls[0].request_options['openai_prompt_cache_retention'] == '24h'
                assert calls[0].request_options['openai_tool_schema']['strict'] > 0
                serialized_options = json.dumps([call.request_options for call in calls], sort_keys=True)
                assert 'cache-secret' not in serialized_options
                assert 'safe-session' not in serialized_options
            finally:
                runtime.close()

    def test_openai_responses_state_chain_stays_stateless_without_credential_fingerprint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AGENT_LIBOS_TEST_MISSING_PROVIDER_KEY", raising=False)
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, store=True, responses_previous_response_id=True),
        )
        client = LLMClient(
            model="gpt-test",
            api_key=None,
            api_key_env="AGENT_LIBOS_TEST_MISSING_PROVIDER_KEY",
            api_mode="responses",
            store=True,
            responses_previous_response_id=True,
            defaults=config.llm,
        )
        fake = FakeAsyncOpenAIResponses(
            [
                _responses_tool_call(
                    "resp_unverified_first",
                    "create_memory_object",
                    {"type": "note", "name": "step", "payload": {"ok": True}},
                ),
                _responses_tool_call(
                    "resp_unverified_second",
                    "process_exit",
                    {"payload": {"done": True}},
                ),
            ]
        )
        client._async_client = fake
        runtime = Runtime.open("local", config=config)
        try:
            runtime.llm.client = client
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="keep unverified provider response state stateless",
            )

            first = runtime.run_next_process_once()
            second = runtime.run_next_process_once()

            assert first["action"]["action"] == "create_memory_object"
            assert second["action"]["action"] == "process_exit"
            assert len(fake.responses.payloads) == 2
            assert all(
                "previous_response_id" not in payload
                for payload in fake.responses.payloads
            )
            assert not any(
                item.get("type") == "function_call_output"
                for item in fake.responses.payloads[1]["input"]
            )
            calls = runtime.store.list_llm_calls(pid)
            assert [
                call.request_options["openai_provider_chain_eligible"]
                for call in calls
            ] == [False, False]
            assert all(
                call.request_options["openai_provider_chain_fingerprint"] is None
                for call in calls
            )
            assert calls[1].request_options["openai_previous_response_id"] is None
        finally:
            runtime.close()

    def test_openai_responses_state_chain_resets_when_sink_registry_changes_before_dispatch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, store=True, responses_previous_response_id=True),
        )
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            responses_previous_response_id=True,
            defaults=config.llm,
        )
        fake = FakeAsyncOpenAIResponses(
            [
                _responses_tool_call(
                    'resp_before_registry_change',
                    'create_memory_object',
                    {'type': 'note', 'name': 'registry-step', 'payload': {'ok': True}},
                ),
                _responses_tool_call(
                    'resp_after_registry_change',
                    'process_exit',
                    {'payload': {'done': True}},
                ),
            ]
        )
        client._async_client = fake
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = client
            runtime.data_flow.register_sink_trust(
                SinkTrustRule(
                    pattern='llm:default',
                    trust_level=SinkTrustLevel.TRUSTED,
                    max_sensitivity='normal',
                    identity_sha256=runtime.llms.profile_identity_sha256('default'),
                ),
                actor='test',
                require_capability=False,
            )
            pid = runtime.process.spawn(image='base-agent:v0', goal='reset raced response chain')
            first = runtime.run_next_process_once()
            assert first['action']['action'] == 'create_memory_object'

            original_start = runtime.protected_operations.start
            replaced = False

            def replace_registry_before_start(
                contract: Any,
                invocation: Any,
                *,
                provider: Any,
            ) -> Any:
                nonlocal replaced
                contract_name = contract if isinstance(contract, str) else contract.name
                if contract_name == 'primitive.llm.complete' and not replaced:
                    active = runtime.data_flow.inspect_sink_trust('llm:default')
                    assert active is not None
                    runtime.data_flow.register_sink_trust(
                        active.rule,
                        actor='test',
                        replace=True,
                        require_capability=False,
                    )
                    replaced = True
                return original_start(contract, invocation, provider=provider)

            monkeypatch.setattr(runtime.protected_operations, 'start', replace_registry_before_start)
            second = runtime.run_next_process_once()

            assert replaced
            assert second['action']['action'] == 'process_exit'
            assert len(fake.responses.payloads) == 2
            assert 'previous_response_id' not in fake.responses.payloads[1]
            assert not any(
                item.get('type') == 'function_call_output'
                for item in fake.responses.payloads[1]['input']
            )
            calls = runtime.store.list_llm_calls(pid)
            assert [call.status for call in calls] == ['ok', 'ok']
            assert calls[1].request_options['openai_previous_response_id'] is None
            assert (
                calls[0].request_options['data_flow_provider_chain_fingerprint']
                != calls[1].request_options['data_flow_provider_chain_fingerprint']
            )
        finally:
            runtime.close()

    def test_openai_responses_state_chain_persists_parallel_tool_outputs(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(
                DEFAULT_CONFIG.llm,
                store=True,
                responses_previous_response_id=True,
                parallel_tool_calls=True,
            ),
        )
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            responses_previous_response_id=True,
            parallel_tool_calls=True,
            defaults=config.llm,
        )
        fake = FakeAsyncOpenAIResponses(
            [
                _responses_parallel_tool_calls(
                    'resp_parallel',
                    [
                        ('read_process_messages', {'ack': False}),
                        ('read_process_messages', {'ack': False}),
                    ],
                ),
                _responses_tool_call('resp_after_parallel', 'process_exit', {'payload': {'done': True}}),
            ]
        )
        client._async_client = fake
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = client
            pid = runtime.process.spawn(image='base-agent:v0', goal='persist every parallel output')

            first = runtime.run_next_process_once()
            second = runtime.run_next_process_once()

            assert first['parallel_tool_calls']
            assert first['executed_count'] == 2
            assert second['action']['action'] == 'process_exit'
            assert fake.responses.payloads[1]['previous_response_id'] == 'resp_parallel'
            native_outputs = [
                item for item in fake.responses.payloads[1]['input']
                if item.get('type') == 'function_call_output'
            ]
            assert [item['call_id'] for item in native_outputs] == [
                'call_resp_parallel_1',
                'call_resp_parallel_2',
            ]
            assert [json.loads(item['output']) for item in native_outputs] == first['results']
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ("second_model", "second_api_key"),
        [("gpt-next", "key"), ("gpt-test", "different-key")],
    )
    def test_openai_responses_state_chain_resets_when_provider_identity_changes(
        self,
        second_model: str,
        second_api_key: str,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, store=True, responses_previous_response_id=True),
        )
        first_client = LLMClient(
            model="gpt-test",
            api_key="key",
            api_mode="responses",
            store=True,
            responses_previous_response_id=True,
            defaults=config.llm,
        )
        first_fake = FakeAsyncOpenAIResponses(
            [
                _responses_tool_call(
                    "resp_provider_a",
                    "create_memory_object",
                    {"type": "note", "name": "provider-step", "payload": {"ok": True}},
                )
            ]
        )
        first_client._async_client = first_fake
        runtime = Runtime.open("local", config=config)
        try:
            runtime.llm.client = first_client
            pid = runtime.process.spawn(image="base-agent:v0", goal="reset provider response chain")
            first = runtime.run_next_process_once()
            assert first["action"]["action"] == "create_memory_object"

            second_client = LLMClient(
                model=second_model,
                api_key=second_api_key,
                api_mode="responses",
                store=True,
                responses_previous_response_id=True,
                defaults=config.llm,
            )
            second_fake = FakeAsyncOpenAIResponses(
                [_responses_tool_call("resp_provider_b", "process_exit", {"payload": {"done": True}})]
            )
            second_client._async_client = second_fake
            runtime.llm.client = second_client

            second = runtime.run_next_process_once()

            assert second["action"]["action"] == "process_exit"
            assert "previous_response_id" not in second_fake.responses.payloads[0]
            assert not any(
                item.get("type") == "function_call_output"
                for item in second_fake.responses.payloads[0]["input"]
            )
            calls = runtime.store.list_llm_calls(pid)
            assert (
                calls[0].request_options["openai_provider_chain_fingerprint"]
                != calls[1].request_options["openai_provider_chain_fingerprint"]
            )
            serialized_options = json.dumps([call.request_options for call in calls], sort_keys=True)
            assert '"key"' not in serialized_options
            assert '"different-key"' not in serialized_options
        finally:
            runtime.close()

    def test_openai_provider_fingerprint_preserves_case_sensitive_base_url_path(self) -> None:
        upper = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            base_url='https://gateway.example/v1/TenantA',
            store=True,
            responses_previous_response_id=True,
            allow_custom_base_url=True,
            defaults=DEFAULT_CONFIG.llm,
        )
        lower = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            base_url='https://gateway.example/v1/tenanta',
            store=True,
            responses_previous_response_id=True,
            allow_custom_base_url=True,
            defaults=DEFAULT_CONFIG.llm,
        )

        assert (
            LLMProcessExecutor._openai_provider_chain_fingerprint(upper)
            != LLMProcessExecutor._openai_provider_chain_fingerprint(lower)
        )

    def test_openai_responses_state_chain_resets_when_parallel_output_is_incomplete(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(
                DEFAULT_CONFIG.llm,
                store=True,
                responses_previous_response_id=True,
                parallel_tool_calls=True,
            ),
        )
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            responses_previous_response_id=True,
            parallel_tool_calls=True,
            defaults=config.llm,
        )
        fake = FakeAsyncOpenAIResponses(
            [
                _responses_parallel_tool_calls(
                    'resp_incomplete',
                    [
                        ('read_process_messages', {'ack': False}),
                        ('create_memory_object', {'type': 'not-a-real-object-type', 'payload': {}}),
                        ('read_process_messages', {'ack': False}),
                    ],
                ),
                _responses_tool_call('resp_after_incomplete', 'process_exit', {'payload': {'done': True}}),
            ]
        )
        client._async_client = fake
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = client
            pid = runtime.process.spawn(image='base-agent:v0', goal='reset incomplete parallel response chain')

            first = runtime.run_next_process_once()
            second = runtime.run_next_process_once()

            assert first['parallel_tool_calls']
            assert first['stop_reason'] == 'tool_failed'
            assert first['executed_count'] == 2
            assert second['action']['action'] == 'process_exit'
            assert 'previous_response_id' not in fake.responses.payloads[1]
            assert not any(
                item.get('type') == 'function_call_output'
                for item in fake.responses.payloads[1]['input']
            )
            outputs = runtime.store.list_llm_tool_outputs(pid=pid, response_id='resp_incomplete')
            assert [output['call_id'] for output in outputs] == [
                'call_resp_incomplete_1',
                'call_resp_incomplete_2',
            ]
        finally:
            runtime.close()

    def test_openai_responses_state_chain_resets_when_full_io_persistence_is_disabled(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(
                DEFAULT_CONFIG.llm,
                store=True,
                responses_previous_response_id=True,
                persist_full_io=False,
            ),
        )
        client = LLMClient(
            model='gpt-test',
            api_key='key',
            api_mode='responses',
            store=True,
            responses_previous_response_id=True,
            defaults=config.llm,
        )
        fake = FakeAsyncOpenAIResponses(
            [
                _responses_tool_call('resp_redacted', 'read_process_messages', {'ack': False}),
                _responses_tool_call('resp_after_redacted', 'process_exit', {'payload': {'done': True}}),
            ]
        )
        client._async_client = fake
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = client
            pid = runtime.process.spawn(image='base-agent:v0', goal='do not persist output in redacted mode')

            first = runtime.run_next_process_once()
            second = runtime.run_next_process_once()

            assert first['result']['ok']
            assert second['action']['action'] == 'process_exit'
            assert 'previous_response_id' not in fake.responses.payloads[1]
            assert runtime.store.list_llm_tool_outputs(pid=pid, response_id='resp_redacted') == []
        finally:
            runtime.close()

    def test_openai_responses_wait_resume_output_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(
                DEFAULT_CONFIG,
                llm=replace(DEFAULT_CONFIG.llm, store=True, responses_previous_response_id=True),
            )
            first_client = LLMClient(
                model='gpt-test',
                api_key='key',
                api_mode='responses',
                store=True,
                responses_previous_response_id=True,
                defaults=config.llm,
            )
            first_fake = FakeAsyncOpenAIResponses(
                [_responses_tool_call('resp_wait', 'receive_process_messages', {'channel': 'resume-chain'})]
            )
            first_client._async_client = first_fake
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = first_client
                pid = runtime.process.spawn(image='base-agent:v0', goal='resume response chain after reopen')
                waiting = runtime.run_next_process_once()
                assert waiting['waiting_message']
            finally:
                runtime.close()

            second_client = LLMClient(
                model='gpt-test',
                api_key='key',
                api_mode='responses',
                store=True,
                responses_previous_response_id=True,
                defaults=config.llm,
            )
            second_fake = FakeAsyncOpenAIResponses(
                [_responses_tool_call('resp_after_wait', 'process_exit', {'payload': {'done': True}})]
            )
            second_client._async_client = second_fake
            reopened = Runtime.open(db, config=config)
            try:
                reopened.llm.client = second_client
                message = reopened.human.send_process_message(
                    pid,
                    'resume response chain',
                    subject='resume',
                    channel='resume-chain',
                )
                resumed = reopened.run_next_process_once()
                completed = reopened.run_next_process_once()

                assert resumed['resumed_after_message']
                assert resumed['result']['payload']['messages'][0]['message_id'] == message.message_id
                assert completed['action']['action'] == 'process_exit'
                assert second_fake.responses.payloads[0]['previous_response_id'] == 'resp_wait'
                output = next(
                    item for item in second_fake.responses.payloads[0]['input']
                    if item.get('type') == 'function_call_output'
                )
                assert output['call_id'] == 'call_resp_wait'
                assert json.loads(output['output']) == resumed['result']
            finally:
                reopened.close()

    def test_pending_message_resume_is_claimed_once_across_executor_instances(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='claim pending action once')
            runtime.store.upsert_llm_pending_action(
                pid,
                {
                    'wait_type': 'message',
                    'filters': {'channel': 'claim-once'},
                    'action': {'action': 'receive_process_messages', 'channel': 'claim-once'},
                    'data_flow_context': DataFlowContext().to_dict(),
                    'content_preview': '',
                    'tool_call_count': 1,
                    'status': 'pending',
                },
            )
            first = _new_llm_executor(runtime)
            second = _new_llm_executor(runtime)
            runtime.human.send_process_message(pid, 'ready', subject='resume', channel='claim-once')
            dispatch_count = 0

            async def fake_dispatch(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
                nonlocal dispatch_count
                dispatch_count += 1
                await asyncio.sleep(0)
                return {'ok': True, 'tool_id': 'receive', 'result_oid': None, 'payload': {}, 'error': None}

            first.adispatch = fake_dispatch  # type: ignore[method-assign]
            second.adispatch = fake_dispatch  # type: ignore[method-assign]

            async def resume_both() -> list[dict[str, Any]]:
                return list(
                    await asyncio.gather(
                        first._resume_pending_message_action(pid),
                        second._resume_pending_message_action(pid),
                    )
                )

            results = asyncio.run(resume_both())

            assert dispatch_count == 1
            assert sum(bool(result.get('resumed_after_message')) for result in results) == 1
            assert sum(bool(result.get('pending_action_resuming')) for result in results) == 1
            assert runtime.store.get_llm_pending_action(pid)['status'] == 'completed'
        finally:
            runtime.close()

    def test_pending_claim_token_prevents_aba_claim_of_new_wait_generation(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='pending claim generation')
            first = {
                'wait_type': 'human',
                'request_id': 'human_first',
                'resume_token': 'wait_first',
                'action': {'action': 'ask_human', 'question': 'first?'},
                'data_flow_context': DataFlowContext().to_dict(),
                'content_preview': '',
                'tool_call_count': 1,
                'status': 'pending',
            }
            runtime.store.upsert_llm_pending_action(pid, first)
            assert runtime.store.claim_llm_pending_action(pid, resume_token='wait_first') is not None

            second = {
                **first,
                'request_id': 'human_second',
                'resume_token': 'wait_second',
                'action': {'action': 'ask_human', 'question': 'second?'},
            }
            runtime.store.upsert_llm_pending_action(pid, second)

            assert runtime.store.claim_llm_pending_action(pid, resume_token='wait_first') is None
            claimed = runtime.store.claim_llm_pending_action(pid, resume_token='wait_second')
            assert claimed is not None
            assert claimed['request_id'] == 'human_second'
            assert claimed['resume_token'] == 'wait_second'
        finally:
            runtime.close()

    def test_checkpoint_restore_rehydrates_restored_pending_action_in_current_executor(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.llm.client = RecordingActionClient(
                [{'action': 'receive_process_messages', 'channel': 'restore-pending'}]
            )
            pid = runtime.process.spawn(image='base-agent:v0', goal='restore pending LLM action')
            waiting = runtime.run_next_process_once()
            assert waiting['waiting_message']
            checkpoint_id = runtime.checkpoint.create(pid, 'pending message action', actor=pid)

            runtime.human.send_process_message(
                pid,
                'first delivery',
                subject='resume',
                channel='restore-pending',
            )
            first_resume = runtime.run_next_process_once()
            assert first_resume['resumed_after_message']
            assert runtime.store.get_llm_pending_action(pid)['status'] == 'completed'

            runtime.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            runtime.human.send_process_message(
                pid,
                'delivery after restore',
                subject='resume',
                channel='restore-pending',
            )
            restored_resume = runtime.run_next_process_once()

            assert restored_resume['resumed_after_message']
            assert restored_resume['result']['payload']['messages'][0]['body'] == 'delivery after restore'
        finally:
            runtime.close()

    def test_reopen_fails_process_closed_for_interrupted_pending_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='do not replay interrupted action')
                runtime.store.upsert_llm_pending_action(
                    pid,
                    {
                        'wait_type': 'human',
                        'request_id': 'human_interrupted',
                        'action': {'action': 'write_text_file', 'path': 'agent_outputs/no-replay.txt', 'content': 'x'},
                        'data_flow_context': DataFlowContext().to_dict(),
                        'content_preview': '',
                        'tool_call_count': 1,
                        'status': 'resuming',
                    },
                )
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                assert reopened.process.get(pid).status == ProcessStatus.FAILED
                assert reopened.store.get_llm_pending_action(pid)['status'] == 'resuming'
                assert any(
                    record.action == 'llm.pending_action_resume_interrupted'
                    and record.target == f'process:{pid}'
                    for record in reopened.audit.trace()
                )
                assert not reopened.llm.pending.has_memory(pid, "human")
            finally:
                reopened.close()

    def test_pending_resume_claim_prevents_replay_after_effect_before_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            try:
                runtime.llm.client = RecordingActionClient(
                    [{'action': 'receive_process_messages', 'channel': 'crash-window'}]
                )
                pid = runtime.process.spawn(image='base-agent:v0', goal='close replay crash window')
                waiting = runtime.run_next_process_once()
                assert waiting['waiting_message']
                message = runtime.human.send_process_message(
                    pid,
                    'effect happens once',
                    subject='resume',
                    channel='crash-window',
                )

                def crash_before_clear(_pid: str, _resume_token: str) -> None:
                    raise RuntimeError('simulated crash after tool effect')

                runtime.llm._clear_pending_action = crash_before_clear  # type: ignore[method-assign]
                with pytest.raises(RuntimeError, match='simulated crash'):
                    runtime.run_process_once(pid)

                pending = runtime.store.get_llm_pending_action(pid)
                assert pending['status'] == 'resuming'
                assert runtime.store.get_process_message(message.message_id).status.value == 'acked'
                assert runtime.process.get(pid).status == ProcessStatus.FAILED
                assert any(
                    record.action == 'llm.pending_action_resume_interrupted'
                    and record.target == f'process:{pid}'
                    for record in runtime.audit.trace()
                )
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                assert reopened.process.get(pid).status == ProcessStatus.FAILED
                assert reopened.store.get_llm_pending_action(pid)['status'] == 'resuming'
                assert not reopened.llm.pending.has_memory(pid, "message")
            finally:
                reopened.close()

    def test_openai_responses_state_chain_resets_after_context_compaction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            config = replace(
                DEFAULT_CONFIG,
                llm=replace(DEFAULT_CONFIG.llm, store=True, responses_previous_response_id=True),
            )
            client = LLMClient(
                model='gpt-test',
                api_key='key',
                api_mode='responses',
                store=True,
                responses_previous_response_id=True,
                defaults=config.llm,
            )
            fake = FakeAsyncOpenAIResponses(
                [
                    _responses_tool_call(
                        'resp_before_compact',
                        'create_memory_object',
                        {'type': 'note', 'name': 'step', 'payload': {'ok': True}},
                    ),
                    _responses_tool_call('resp_after_compact', 'process_exit', {'payload': {'done': True}}),
                ]
            )
            client._async_client = fake
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = client
                pid = runtime.process.spawn(image='base-agent:v0', goal='chain reset after compaction')
                first = runtime.run_next_process_once()
                context_obj = runtime.store.get_object_by_name(
                    context_object_name(pid),
                    namespace=runtime.memory.resolve_namespace(pid),
                )
                assert first['action']['action'] == 'create_memory_object'
                assert context_obj is not None

                runtime.llm.context_memory.replace_with_compacted_summary(
                    pid,
                    context_oid=context_obj.oid,
                    expected_version=context_obj.version,
                    summary=_compact_summary('chain reset after compaction'),
                    compaction_method='test_compaction',
                    preserve_recent_entries=0,
                    source_tokens=100,
                    target_tokens=10,
                    compressor_pids=[pid],
                )
                second = runtime.run_next_process_once()

                assert second['action']['action'] == 'process_exit'
                assert 'previous_response_id' not in fake.responses.payloads[0]
                assert 'previous_response_id' not in fake.responses.payloads[1]
                calls = runtime.store.list_llm_calls(pid)
                assert calls[0].request_options['openai_previous_response_id'] is None
                assert calls[1].request_options['openai_previous_response_id'] is None
                assert (
                    calls[0].request_options['openai_response_scope_fingerprint']
                    != calls[1].request_options['openai_response_scope_fingerprint']
                )
            finally:
                runtime.close()

    def test_empty_tool_calls_without_auto_wait_still_fail_action_selection(self) -> None:
        config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, action_repair_attempts=1))
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = TextOnlyActionClient(['I will wait for more instructions.'])
            pid = runtime.process.spawn(image='base-agent:v0', goal='empty tool calls disabled')

            result = runtime.run_next_process_once()

            assert not result['ok']
            assert 'no valid tool call or fallback JSON action found' in result['error']
            assert runtime.process.get(pid).status == ProcessStatus.FAILED
            assert not any(record.action == 'llm.empty_tool_calls_auto_wait' for record in runtime.audit.trace())
            call = runtime.store.list_llm_calls(pid)[0]
            assert call.tool_calls == []
            assert call.request_options['agent_libos_auto_wait_on_empty_tool_calls_enabled'] is False
        finally:
            runtime.close()

    def test_empty_tool_calls_auto_waits_for_any_process_message(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, auto_wait_on_empty_tool_calls=True, action_repair_attempts=1),
        )
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = TextOnlyActionClient(['Waiting for the next message.'])
            pid = runtime.process.spawn(image='base-agent:v0', goal='empty tool calls auto wait')

            result = runtime.run_next_process_once()

            assert result['waiting_message']
            assert result['filters'] == {
                'kind': None,
                'sender': None,
                'channel': None,
                'correlation_id': None,
                'reply_to': None,
                'message_ids': None,
            }
            assert runtime.process.get(pid).status == ProcessStatus.WAITING_EVENT
            pending = runtime.store.get_llm_pending_action(pid)
            assert pending['wait_type'] == 'message'
            assert pending['action'] == {'action': 'receive_process_messages'}
            call = runtime.store.list_llm_calls(pid)[0]
            assert call.tool_calls == []
            assert call.request_options['agent_libos_auto_wait_on_empty_tool_calls_enabled'] is True
            auto_waits = [record for record in runtime.audit.trace() if record.action == 'llm.empty_tool_calls_auto_wait']
            assert auto_waits[0].decision['action'] == {'action': 'receive_process_messages'}
        finally:
            runtime.close()

    def test_empty_tool_calls_auto_wait_reads_existing_unread_message(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, auto_wait_on_empty_tool_calls=True, action_repair_attempts=1),
        )
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = TextOnlyActionClient(['Waiting for any message.'])
            pid = runtime.process.spawn(image='base-agent:v0', goal='read queued message')
            queued = runtime.human.send_process_message(pid, 'queued input', subject='queued')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['action']['action'] == 'receive_process_messages'
            assert result['result']['payload']['messages'][0]['message_id'] == queued.message_id
            assert result['result']['payload']['acked_message_ids'] == [queued.message_id]
            assert runtime.messages.unread(pid) == []
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_empty_tool_calls_auto_wait_preserves_json_action_fallback(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, auto_wait_on_empty_tool_calls=True, action_repair_attempts=1),
        )
        runtime = Runtime.open('local', config=config)
        try:
            runtime.llm.client = TextOnlyActionClient(['{"action":"process_exit","payload":{"done":true}}'])
            pid = runtime.process.spawn(image='base-agent:v0', goal='json fallback')

            result = runtime.run_next_process_once()

            assert result['ok']
            assert result['action']['action'] == 'process_exit'
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            assert not any(record.action == 'llm.empty_tool_calls_auto_wait' for record in runtime.audit.trace())
        finally:
            runtime.close()

    def test_empty_tool_calls_auto_wait_does_not_bypass_tool_table(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, auto_wait_on_empty_tool_calls=True, action_repair_attempts=1),
        )
        runtime = Runtime.open('local', config=config)
        try:
            runtime.register_image(
                AgentImage(
                    image_id='no-message-wait:v0',
                    name='no-message-wait',
                    system_prompt='Use only the configured tools.',
                    default_tools=['process_exit'],
                ),
                actor='test',
            )
            runtime.llm.client = TextOnlyActionClient(['Standing by.'])
            pid = runtime.process.spawn(image='no-message-wait:v0', goal='no receive tool')

            result = runtime.run_next_process_once()

            assert not result['ok']
            assert 'receive_process_messages' in result['error']
            assert runtime.process.get(pid).status == ProcessStatus.FAILED
            assert any(record.action == 'llm.empty_tool_calls_auto_wait' for record in runtime.audit.trace())
            assert not any(
                record.action == 'tool.call' and record.decision.get('tool') == 'receive_process_messages'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_empty_tool_calls_auto_wait_pending_action_survives_runtime_reopen(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            llm=replace(DEFAULT_CONFIG.llm, auto_wait_on_empty_tool_calls=True, action_repair_attempts=1),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db, config=config)
            try:
                runtime.llm.client = TextOnlyActionClient(['Waiting across restart.'])
                pid = runtime.process.spawn(image='base-agent:v0', goal='persist auto wait')
                waiting = runtime.run_next_process_once()
                assert waiting['waiting_message']
                assert runtime.store.get_llm_pending_action(pid)['wait_type'] == 'message'
            finally:
                runtime.close()

            reopened = Runtime.open(db, config=config)
            try:
                reopened.llm.client = ExplodingClient()
                message = reopened.human.send_process_message(pid, 'resume now', subject='resume')
                resumed = reopened.run_next_process_once()

                assert resumed['resumed_after_message']
                assert resumed['action']['action'] == 'receive_process_messages'
                assert resumed['result']['payload']['messages'][0]['message_id'] == message.message_id
                assert reopened.store.get_llm_pending_action(pid)['status'] == 'completed'
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
                runtime.tools.activate_tool_group(pid, 'filesystem')
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

    def test_replace_with_compacted_summary_matching_version_commits(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='compact matching context version')
            context = _seed_context_entries(runtime, pid, count=3)

            result = runtime.llm.context_memory.replace_with_compacted_summary(
                pid,
                context_oid=context.oid,
                expected_version=context.version,
                summary=_compact_summary('matching version summary'),
                compaction_method='test_compaction',
                preserve_recent_entries=1,
                source_tokens=1000,
                target_tokens=512,
                compressor_pids=[],
            )

            after = runtime.store.get_object(context.oid)
            assert after is not None
            assert result['old_version'] == context.version
            assert result['new_version'] == context.version + 1
            assert after.version == context.version + 1
            assert after.payload['entries'][0]['kind'] == 'context_compacted'
            assert after.payload['entries'][0]['summary']['goal'] == 'matching version summary'
            assert after.payload['entries'][-1] == {'kind': 'seed_entry', 'index': 2}
            runtime.llm.client = RecordingActionClient([{'action': 'process_exit', 'payload': {'done': True}}])
            assert runtime.run_process_once(pid)['ok']
            manifest = runtime.store.list_context_materialization_manifests(pid=pid)[0]
            context_entry = next(item for item in manifest.objects if item['oid'] == after.oid)
            assert context_entry['version'] == manifest.context_version
            assert context_entry['version'] > after.version
            assert context_entry['transform'] == 'compacted'
            assert manifest.compaction['transform'] == 'compacted'
        finally:
            runtime.close()

    def test_replace_with_compacted_summary_final_version_cas_preserves_concurrent_append(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='race final context version check')
            context = _seed_context_entries(runtime, pid, count=3)
            original_build = runtime.llm.context_memory._build_compacted_payload
            entry_version_checked = threading.Barrier(2)
            allow_replace = threading.Barrier(2)

            def blocked_build(**kwargs: Any) -> tuple[dict[str, Any], int, int]:
                entry_version_checked.wait(timeout=5)
                allow_replace.wait(timeout=5)
                return original_build(**kwargs)

            monkeypatch.setattr(runtime.llm.context_memory, '_build_compacted_payload', blocked_build)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    runtime.llm.context_memory.replace_with_compacted_summary,
                    pid,
                    context_oid=context.oid,
                    expected_version=context.version,
                    summary=_compact_summary('stale summary after entry check'),
                    compaction_method='test_compaction',
                    preserve_recent_entries=0,
                    source_tokens=1000,
                    target_tokens=512,
                    compressor_pids=[],
                )
                entry_version_checked.wait(timeout=5)
                try:
                    runtime.memory.append_object_by_name(
                        pid,
                        context_object_name(pid),
                        {'kind': 'external_update', 'value': 'must survive final CAS'},
                    )
                finally:
                    allow_replace.wait(timeout=5)
                with pytest.raises(ValidationError, match='changed during compaction'):
                    future.result(timeout=5)

            after = runtime.store.get_object(context.oid)
            assert after is not None
            assert after.version == context.version + 1
            assert after.payload['entries'][-1] == {
                'kind': 'external_update',
                'value': 'must survive final CAS',
            }
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
            config = replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False))
            runtime = Runtime.open(db, config=config)
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
                result_oid = child_exit['result']['payload']['result_oid']
                assert runtime.store.get_llm_pending_action(pid)['status'] == 'pending'
            finally:
                runtime.close()

            conn = sqlite3.connect(db)
            try:
                conn.execute(
                    "UPDATE objects SET payload_json = ? WHERE oid = ?",
                    (json.dumps({"storage": "runtime_memory", "present": True}), result_oid),
                )
                conn.commit()
            finally:
                conn.close()

            reopened = Runtime.open(db, config=config)
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


class TextOnlyActionClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = list(contents)
        self.user_prompts: list[str] = []
        self.tool_batches: list[list[dict[str, Any]]] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        if not self.contents:
            raise AssertionError('no text-only response remains')
        self.user_prompts.append(str(messages[-1]['content']))
        self.tool_batches.append(tools)
        index = len(self.user_prompts)
        return LLMCompletion(
            content=self.contents.pop(0),
            tool_calls=[],
            api='chat',
            response_id=f'text_only_resp_{index}',
            request_id=f'text_only_req_{index}',
            model='text-only-test-model',
            usage={'prompt_tokens': 5, 'completion_tokens': 3, 'total_tokens': 8},
        )


class MultiToolActionClient:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = list(batches)
        self.user_prompts: list[str] = []
        self.tool_batches: list[list[dict[str, Any]]] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        self.user_prompts.append(str(messages[-1]["content"]))
        self.tool_batches.append(tools)
        batch = self.batches.pop(0)
        tool_calls = []
        for index, action in enumerate(batch, start=1):
            name = str(action["action"])
            args = {key: value for key, value in action.items() if key != "action"}
            tool_calls.append(
                {
                    "id": f"parallel_{len(self.user_prompts)}_{index}",
                    "name": name,
                    "arguments": json.dumps(args),
                }
            )
        return LLMCompletion(
            content="",
            tool_calls=tool_calls,
            api="chat",
            response_id=f"resp_parallel_{len(self.user_prompts)}",
            request_id=f"req_parallel_{len(self.user_prompts)}",
            model="test-model",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )


class ExplodingClient:

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        raise AssertionError('LLM client should not be called when the process image is missing')


class FakeAsyncOpenAIResponses:
    def __init__(self, responses: list[Any]):
        self.responses = SequencedResponses(responses)


class SequencedResponses:
    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []

    async def create(self, **payload: Any) -> Any:
        self.payloads.append(payload)
        return self.responses.pop(0)


def _responses_tool_call(response_id: str, name: str, arguments: dict[str, Any]) -> Any:
    return SimpleNamespace(
        id=response_id,
        _request_id=f'req_{response_id}',
        model='gpt-test',
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, total_tokens=7),
        output_text='',
        output=[
            SimpleNamespace(
                type='function_call',
                id=f'fc_{response_id}',
                call_id=f'call_{response_id}',
                name=name,
                arguments=json.dumps(arguments),
            )
        ],
    )


def _responses_parallel_tool_calls(
    response_id: str,
    calls: list[tuple[str, dict[str, Any]]],
) -> Any:
    return SimpleNamespace(
        id=response_id,
        _request_id=f'req_{response_id}',
        model='gpt-test',
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, total_tokens=7),
        output_text='',
        output=[
            SimpleNamespace(
                type='function_call',
                id=f'fc_{response_id}_{index}',
                call_id=f'call_{response_id}_{index}',
                name=name,
                arguments=json.dumps(arguments),
            )
            for index, (name, arguments) in enumerate(calls, start=1)
        ],
    )


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
