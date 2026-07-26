from __future__ import annotations
from dataclasses import replace
import pytest
import json
import asyncio
import tempfile
import threading
import time
from typing import Any
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    EventType,
    ObjectType,
    ProcessMessage,
    ProcessMessageKind,
    ProcessMessageStatus,
    ProcessSignal,
    ProcessStatus,
)
from agent_libos.models.exceptions import ProcessError, ProcessMessageWaitRequired, ValidationError
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.skills import get_builtin_skill_catalog
from tests.support.public_errors import assert_public_error_message


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')


def _activate_action(skill_id: str) -> dict[str, str]:
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    return {
        'action': 'activate_skill',
        'skill_id': skill_id,
        'expected_package_sha256': package.package_sha256,
    }


class TestProcessMessage:

    def test_process_message_preserves_legacy_positional_status_arguments(self) -> None:
        message = ProcessMessage(
            'legacy-message',
            'sender',
            'recipient',
            ProcessMessageKind.NORMAL,
            'subject',
            'body',
            'default',
            None,
            None,
            {'legacy': True},
            ProcessMessageStatus.ACKED,
            'created',
            'updated',
            'acked',
        )

        assert message.payload == {'legacy': True}
        assert message.status == ProcessMessageStatus.ACKED
        assert message.created_at == 'created'
        assert message.updated_at == 'updated'
        assert message.acked_at == 'acked'
        assert message.metadata == {}

    def test_post_event_failure_rolls_back_message_and_waiter_wakeup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='wait for message')
            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(pid, block=True, channel='control')
            original_emit = runtime.events.emit

            def fail_post_event(event_type, *args, **kwargs):
                from agent_libos.models import EventType

                if event_type == EventType.PROCESS_MESSAGE_POSTED:
                    raise RuntimeError('injected process message event failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_post_event)
            with pytest.raises(RuntimeError, match='injected process message event failure'):
                runtime.messages.post(
                    sender='test',
                    recipient_pid=pid,
                    channel='control',
                    subject='must roll back',
                )

            assert runtime.messages.unread(pid) == []
            assert runtime.process.get(pid).status == ProcessStatus.WAITING_EVENT

            monkeypatch.setattr(runtime.events, 'emit', original_emit)
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                channel='control',
                subject='committed',
            )
            assert [message.subject for message in runtime.messages.unread(pid)] == ['committed']
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_process_message_tools_send_read_and_ack_related_processes(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            sent = runtime.tools.call(parent, 'send_process_message', {'recipient_pid': child, 'kind': 'normal', 'subject': 'status', 'body': 'send a status update', 'payload': {'priority': 1}})
            assert sent.ok, sent.error
            assert len(runtime.messages.unread(child)) == 1
            read = runtime.tools.call(child, 'read_process_messages', {})
            assert read.ok, read.error
            assert read.payload['messages'][0]['subject'] == 'status'
            assert read.payload['messages'][0]['payload'] == {'priority': 1}
            assert read.payload['messages'][0]['status'] == 'acked'
            assert read.payload['acked_message_ids'] == [sent.payload['message_id']]
            assert runtime.messages.unread(child) == []
        finally:
            runtime.close()

    def test_unrelated_process_cannot_send_process_message(self) -> None:
        runtime = Runtime.open('local')
        try:
            first = runtime.process.spawn(image='base-agent:v0', goal='first')
            second = runtime.process.spawn(image='base-agent:v0', goal='second')
            denied = runtime.tools.call(first, 'send_process_message', {'recipient_pid': second, 'body': 'no'})
            assert not denied.ok
            assert_public_error_message(
                denied.error,
                code='execution_error',
                error_type='ProcessError',
                forbidden=('can only message', second),
            )
            assert runtime.messages.unread(second) == []
        finally:
            runtime.close()

    def test_human_can_send_normal_and_interrupt_process_messages(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='listen to human')
            normal = runtime.human.send_process_message(pid, 'please check progress', subject='status')
            interrupt = runtime.human.send_process_message(pid, 'stop current work and inspect this', kind=ProcessMessageKind.INTERRUPT)
            unread = runtime.messages.unread(pid)
            assert [message.message_id for message in unread] == [normal.message_id, interrupt.message_id]
            assert unread[0].sender == 'human:owner'
            assert unread[0].channel == 'human'
            assert unread[0].payload['source'] == 'human_input'
            assert unread[1].kind == ProcessMessageKind.INTERRUPT
            assert 'human.message' in _audit_actions(runtime)
        finally:
            runtime.close()

    def test_process_message_syscalls_send_read_and_ack(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            parent_session = LibOSSyscallSession(runtime, parent)
            child_session = LibOSSyscallSession(runtime, child)
            sent = asyncio.run(parent_session.handle('process.send_message', {'recipient_pid': child, 'kind': 'normal', 'subject': 'via syscall', 'body': 'hello'}))
            read = asyncio.run(child_session.handle('process.read_messages', {}))
            assert sent['subject'] == 'via syscall'
            assert read['messages'][0]['message_id'] == sent['message_id']
            assert read['messages'][0]['status'] == 'acked'
            assert runtime.messages.unread(child) == []
        finally:
            runtime.close()

    def test_terminated_process_syscall_session_cannot_send_messages(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            parent_session = LibOSSyscallSession(runtime, parent)

            runtime.process.signal_child(parent, child, ProcessSignal.TERMINATE)
            runtime.process.exit(parent, message='done')

            with pytest.raises(ProcessError, match='cannot issue syscalls'):
                asyncio.run(
                    parent_session.handle(
                        'process.send_message',
                        {'recipient_pid': child, 'kind': 'normal', 'subject': 'after exit', 'body': 'late'},
                    )
                )
            assert runtime.messages.unread(child) == []
        finally:
            runtime.close()

    def test_process_message_filters_channel_correlation_reply_and_ids(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            first = runtime.messages.send_from_process(parent, child, channel='control', correlation_id='job-1', subject='request', body='start')
            runtime.messages.send_from_process(parent, child, channel='noise', correlation_id='job-1', subject='ignore')
            reply = runtime.messages.send_from_process(child, parent, channel='control', correlation_id='job-1', reply_to=first.message_id, subject='reply')
            selected = runtime.tools.call(child, 'read_process_messages', {'channel': 'control', 'correlation_id': 'job-1', 'ack': False})
            reply_selected = runtime.tools.call(parent, 'read_process_messages', {'reply_to': first.message_id, 'message_ids': [reply.message_id]})
            assert selected.ok, selected.error
            assert [message['message_id'] for message in selected.payload['messages']] == [first.message_id]
            assert selected.payload['messages'][0]['channel'] == 'control'
            assert selected.payload['messages'][0]['correlation_id'] == 'job-1'
            assert selected.payload['acked_message_ids'] == []
            assert len(runtime.messages.unread(child)) == 2
            assert reply_selected.ok, reply_selected.error
            assert reply_selected.payload['messages'][0]['reply_to'] == first.message_id
            assert reply_selected.payload['acked_message_ids'] == [reply.message_id]
            assert runtime.messages.unread(parent) == []
        finally:
            runtime.close()

    def test_process_message_payload_limits_and_read_limit_are_enforced(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='message limits')
            too_long_subject = 's' * (runtime.config.tools.message_subject_max_chars + 1)
            too_long_body = 'b' * (runtime.config.tools.message_body_max_chars + 1)
            oversized_payload = {'blob': 'x' * runtime.config.tools.message_payload_max_bytes}
            too_long_id = 'i' * (runtime.config.tools.message_id_max_chars + 1)

            with pytest.raises(ValidationError):
                runtime.messages.post(sender='test', recipient_pid=pid, subject=too_long_subject)
            with pytest.raises(ValidationError):
                runtime.messages.post(sender='test', recipient_pid=pid, body=too_long_body)
            with pytest.raises(ValidationError):
                runtime.messages.post(sender='test', recipient_pid=pid, payload=oversized_payload)
            with pytest.raises(ValidationError):
                runtime.messages.post(sender='test', recipient_pid=pid, correlation_id=too_long_id)
            with pytest.raises(ValidationError):
                runtime.messages.post(sender='test', recipient_pid=pid, reply_to=too_long_id)

            for index in range(runtime.config.tools.message_read_limit + 5):
                runtime.messages.post(sender='test', recipient_pid=pid, subject=f'msg-{index}')

            assert len(runtime.messages.list(pid)) == runtime.config.tools.message_read_limit
            with pytest.raises(ValidationError):
                runtime.messages.list(pid, limit=runtime.config.tools.message_read_hard_limit + 1)
            with pytest.raises(ValidationError):
                runtime.messages.list(
                    pid,
                    message_ids=[f'pmsg_{index}' for index in range(runtime.config.tools.message_filter_ids_hard_limit + 1)],
                )
            with pytest.raises(ValidationError):
                runtime.messages.receive(pid, message_ids=[too_long_id])
        finally:
            runtime.close()

    def test_blocking_receive_rejects_oversized_wait_filters_before_status_write(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='oversized wait filters')
            ids = [
                f'pmsg_{index:04d}_' + ('x' * (runtime.config.tools.message_id_max_chars - 10))
                for index in range(runtime.config.tools.message_filter_ids_hard_limit)
            ]

            with pytest.raises(ValidationError):
                runtime.messages.receive(pid, block=True, message_ids=ids)

            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(pid).status_message is None
        finally:
            runtime.close()

    def test_read_process_messages_acks_entire_requested_window(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='ack many messages')
            count = runtime.config.tools.message_read_limit + 5
            for index in range(count):
                runtime.messages.post(sender='test', recipient_pid=pid, subject=f'msg-{index}')

            result = runtime.tools.call(pid, 'read_process_messages', {'limit': count})

            assert result.ok, result.error
            assert len(result.payload['messages']) == count
            assert len(result.payload['acked_message_ids']) == count
            assert runtime.messages.unread(pid) == []
        finally:
            runtime.close()

    def test_default_read_limit_reports_full_matching_mailbox_tail(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='page complete mailbox evidence')
            count = runtime.config.tools.message_read_limit + 5
            for index in range(count):
                runtime.messages.post(sender='test', recipient_pid=pid, subject=f'msg-{index}')

            result = runtime.tools.call(pid, 'read_process_messages', {})

            assert result.ok, result.error
            assert len(result.payload['messages']) == runtime.config.tools.message_read_limit
            assert result.payload['has_more'] is True
            assert result.payload['omitted_count'] == 5
            assert len(runtime.messages.unread(pid)) == 5
        finally:
            runtime.close()

    def test_message_ack_waits_for_tool_result_persistence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='persist before consuming mailbox')
            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='must remain recoverable',
                body='body visible only in the failed result',
            )
            original_create_object = runtime.memory.create_object

            def fail_tool_result(*args, **kwargs):
                if kwargs.get('object_type') == ObjectType.TOOL_RESULT:
                    raise RuntimeError('injected ToolResult persistence failure')
                return original_create_object(*args, **kwargs)

            monkeypatch.setattr(runtime.memory, 'create_object', fail_tool_result)
            with pytest.raises(RuntimeError, match='injected ToolResult persistence failure'):
                runtime.tools.call(pid, 'read_process_messages', {})

            stored = runtime.store.get_process_message(message.message_id)
            assert stored is not None
            assert stored.status == ProcessMessageStatus.UNREAD
            assert stored.acked_at is None
            assert not any(
                event.type == EventType.PROCESS_MESSAGE_ACKED
                and message.message_id in event.payload.get('message_ids', [])
                for event in runtime.events.list(target=pid)
            )
            assert not any(
                record.action == 'process.message.ack'
                and message.message_id in record.decision.get('message_ids', [])
                for record in runtime.audit.trace(target=f'process:{pid}')
            )
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'tool_name',
        ['read_process_messages', 'receive_process_messages'],
    )
    def test_process_message_tool_only_acks_the_budgeted_model_page(
        self,
        tool_name: str,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            tools=replace(
                DEFAULT_CONFIG.tools,
                tool_result_payload_hard_limit_bytes=30_000,
            ),
        )
        runtime = Runtime.open('local', config=config)
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='read a bounded mailbox page')
            posted = [
                runtime.messages.post(
                    sender='test',
                    recipient_pid=pid,
                    kind=ProcessMessageKind.INTERRUPT if index == 0 else ProcessMessageKind.NORMAL,
                    channel='control',
                    correlation_id='job-1',
                    subject=f'bounded-{index}',
                    body=f'body-{index}-' + ('x' * 9_000),
                    payload={'index': index},
                    metadata={'durable_only': f'evidence-{index}'},
                )
                for index in range(4)
            ]

            args: dict[str, Any] = {'limit': len(posted)}
            if tool_name == 'receive_process_messages':
                args['block'] = False
            result = runtime.tools.call(pid, tool_name, args)

            assert result.ok, result.error
            assert 'result_omitted' not in result.payload
            returned_ids = [message['message_id'] for message in result.payload['messages']]
            assert 0 < len(returned_ids) < len(posted)
            assert result.payload['has_more'] is True
            assert result.payload['omitted_count'] == len(posted) - len(returned_ids)
            assert result.payload['acked_message_ids'] == returned_ids
            assert result.payload['continuation']['tool'] == tool_name
            assert result.payload['continuation']['same_filters'] is True
            assert 'remaining_message_ids' not in result.payload

            projected = result.payload['messages'][0]
            assert projected['kind'] == 'interrupt'
            assert projected['channel'] == 'control'
            assert projected['correlation_id'] == 'job-1'
            assert projected['body'].startswith('body-0-')
            assert projected['payload'] == {'index': 0}
            assert 'recipient_pid' not in projected
            assert 'created_at' not in projected
            assert 'acked_at' not in projected
            assert 'metadata' not in projected

            persisted_statuses = {
                message.message_id: runtime.store.get_process_message(message.message_id).status.value
                for message in posted
            }
            remaining_ids = [
                message.message_id
                for message in posted
                if message.message_id not in returned_ids
            ]
            assert all(persisted_statuses[message_id] == 'acked' for message_id in returned_ids)
            assert all(persisted_statuses[message_id] == 'unread' for message_id in remaining_ids)
            serialized_page = json.dumps(result.payload, sort_keys=True)
            assert all(message_id not in serialized_page for message_id in remaining_ids)
            # Projection does not mutate or replace the durable mailbox record.
            durable = runtime.store.get_process_message(posted[0].message_id)
            assert durable is not None
            assert durable.recipient_pid == pid
            assert durable.created_at
            assert durable.metadata['durable_only'] == 'evidence-0'
            assert len(durable.body) > 9_000

            result_object = runtime.store.get_object(result.result_handle.oid)
            assert result_object is not None
            assert result_object.payload.get('content', '') == ''
            durable_result = result_object.payload['result']
            assert durable_result['messages'][0]['message_id'] == returned_ids[0]
            assert durable_result['messages'][0]['recipient_pid'] == pid
            assert durable_result['messages'][0]['created_at']
            assert 'recipient_pid' not in result.payload['messages'][0]
            assert result_object.payload['metadata'].get('result_omitted') is not True

            acknowledged = set(returned_ids)
            page = result
            while page.payload['has_more']:
                next_args: dict[str, Any] = {'limit': len(posted)}
                if tool_name == 'receive_process_messages':
                    next_args['block'] = False
                page = runtime.tools.call(pid, tool_name, next_args)
                assert page.ok, page.error
                page_ids = [message['message_id'] for message in page.payload['messages']]
                assert page_ids
                assert 'result_omitted' not in page.payload
                acknowledged.update(page_ids)

            assert acknowledged == {message.message_id for message in posted}
            assert runtime.messages.unread(pid) == []
        finally:
            runtime.close()

    def test_blocking_receive_rejects_zero_limit(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='zero limit receive')

            with pytest.raises(ValidationError):
                runtime.messages.receive(pid, block=True, limit=0)

            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_ack_with_empty_message_id_filter_acks_nothing(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='empty id ack')
            runtime.messages.post(sender='test', recipient_pid=pid, subject='keep unread')

            assert runtime.messages.ack(pid, []) == []
            assert len(runtime.messages.unread(pid)) == 1
        finally:
            runtime.close()

    def test_ack_audit_failure_rolls_back_message_and_acked_event(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='atomic message ack evidence')
            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='must remain unread',
            )
            original_record = runtime.audit.record

            def fail_ack_audit(*args, **kwargs):
                action = kwargs.get('action')
                if action is None and len(args) > 1:
                    action = args[1]
                if action == 'process.message.ack':
                    raise RuntimeError('injected process message ack audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_ack_audit)
            with pytest.raises(
                RuntimeError,
                match='injected process message ack audit failure',
            ):
                runtime.messages.ack(pid, [message.message_id])

            stored = runtime.store.get_process_message(message.message_id)
            assert stored is not None
            assert stored.status == ProcessMessageStatus.UNREAD
            assert stored.acked_at is None
            assert [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.PROCESS_MESSAGE_ACKED
            ] == []
            assert [
                record
                for record in runtime.audit.trace(target=f'process:{pid}')
                if record.action == 'process.message.ack'
            ] == []
        finally:
            runtime.close()

    def test_multi_message_ack_cas_failure_rolls_back_entire_batch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='all or none message ack')
            messages = [
                runtime.messages.post(
                    sender='test',
                    recipient_pid=pid,
                    subject=f'message {index}',
                )
                for index in range(2)
            ]
            original_ack = runtime.messages.store.ack_process_message
            calls = 0

            def fail_second_cas(message_id: str, **kwargs) -> bool:
                nonlocal calls
                calls += 1
                if calls == 2:
                    return False
                return original_ack(message_id, **kwargs)

            monkeypatch.setattr(
                runtime.messages.store,
                'ack_process_message',
                fail_second_cas,
            )
            with pytest.raises(ProcessError, match='changed while acknowledging'):
                runtime.messages.ack(
                    pid,
                    [message.message_id for message in messages],
                )

            assert calls == 2
            stored = [runtime.store.get_process_message(message.message_id) for message in messages]
            assert all(message is not None for message in stored)
            assert all(message.status == ProcessMessageStatus.UNREAD for message in stored if message)
            assert all(message.acked_at is None for message in stored if message)
            assert [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.PROCESS_MESSAGE_ACKED
            ] == []
            assert [
                record
                for record in runtime.audit.trace(target=f'process:{pid}')
                if record.action == 'process.message.ack'
            ] == []
        finally:
            runtime.close()

    def test_concurrent_double_ack_only_one_call_claims_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        first_selected = threading.Event()
        second_started = threading.Event()
        second_finished = threading.Event()
        results: dict[str, list[str]] = {}
        errors: list[BaseException] = []
        threads: list[threading.Thread] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='double message ack')
            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='claim once',
            )
            original_list = runtime.messages.list

            def coordinated_list(*args, **kwargs):
                selected = original_list(*args, **kwargs)
                if threading.current_thread().name == 'first-message-ack' and selected:
                    first_selected.set()
                    assert second_started.wait(timeout=5)
                    # The fixed ACK holds the store transaction here, so the
                    # second call cannot select until the first commits.
                    second_finished.wait(timeout=0.2)
                return selected

            monkeypatch.setattr(runtime.messages, 'list', coordinated_list)

            def ack_message(name: str) -> None:
                try:
                    if name == 'second':
                        second_started.set()
                    acked = runtime.messages.ack(pid, [message.message_id])
                    results[name] = [item.message_id for item in acked]
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    if name == 'second':
                        second_finished.set()

            first = threading.Thread(
                target=ack_message,
                args=('first',),
                name='first-message-ack',
                daemon=True,
            )
            threads.append(first)
            first.start()
            assert first_selected.wait(timeout=5)
            second = threading.Thread(
                target=ack_message,
                args=('second',),
                name='second-message-ack',
                daemon=True,
            )
            threads.append(second)
            second.start()
            for thread in threads:
                thread.join(timeout=5)

            assert all(not thread.is_alive() for thread in threads)
            assert errors == []
            assert sorted(results.values(), key=len) == [[], [message.message_id]]
            acked_events = [
                event
                for event in runtime.events.list(target=pid)
                if event.type == EventType.PROCESS_MESSAGE_ACKED
            ]
            ack_audits = [
                record
                for record in runtime.audit.trace(target=f'process:{pid}')
                if record.action == 'process.message.ack'
            ]
            assert len(acked_events) == 1
            assert acked_events[0].payload['message_ids'] == [message.message_id]
            assert len(ack_audits) == 1
            assert ack_audits[0].decision['message_ids'] == [message.message_id]
        finally:
            for thread in threads:
                thread.join(timeout=5)
            runtime.close()

    def test_blocking_receive_rejects_empty_message_id_filter(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='empty id receive')

            with pytest.raises(ValidationError):
                runtime.messages.receive(pid, block=True, message_ids=[])

            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_wait_audit_failure_rolls_back_wait_registration_and_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='rollback message wait evidence',
            )
            before_process = runtime.process.get(pid)
            before_audit = runtime.store.list_audit()
            before_events = runtime.store.list_events()
            original_record = runtime.messages.audit.record

            def fail_after_wait_audit(*args: object, **kwargs: object) -> object:
                record = original_record(*args, **kwargs)
                if kwargs.get('action') == 'process.message.wait':
                    raise RuntimeError('injected process message wait audit failure')
                return record

            monkeypatch.setattr(
                runtime.messages.audit,
                'record',
                fail_after_wait_audit,
            )

            with pytest.raises(
                RuntimeError,
                match='injected process message wait audit failure',
            ):
                runtime.messages.receive(pid, block=True, channel='control')

            assert runtime.process.get(pid) == before_process
            assert runtime.store.list_audit() == before_audit
            assert runtime.store.list_events() == before_events
        finally:
            runtime.close()

    def test_post_racing_empty_receive_cannot_be_lost_before_wait_registration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        checked_empty = threading.Event()
        poster_attempted = threading.Event()
        posted = threading.Event()
        poster_errors: list[BaseException] = []
        poster: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='race message wait')
            original_list = runtime.messages.list

            def coordinated_list(*args, **kwargs):
                result = original_list(*args, **kwargs)
                if not result and not checked_empty.is_set():
                    checked_empty.set()
                    assert poster_attempted.wait(timeout=5)
                    # On the old register-after-check path, post completes in
                    # this window and observes a RUNNABLE process. On the fixed
                    # path it blocks on the store lock until WAITING_EVENT is
                    # registered, then wakes the process.
                    time.sleep(0.05)
                return result

            monkeypatch.setattr(runtime.messages, 'list', coordinated_list)

            def post_message() -> None:
                try:
                    assert checked_empty.wait(timeout=5)
                    poster_attempted.set()
                    runtime.messages.post(sender='test', recipient_pid=pid, channel='control', subject='racing post')
                    posted.set()
                except BaseException as exc:
                    poster_errors.append(exc)

            poster = threading.Thread(target=post_message)
            poster.start()
            with pytest.raises(ProcessMessageWaitRequired):
                runtime.messages.receive(pid, block=True, channel='control')

            assert posted.wait(timeout=5)
            poster.join(timeout=5)
            assert not poster.is_alive()
            assert poster_errors == []
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert [message.subject for message in runtime.messages.unread(pid)] == ['racing post']
        finally:
            if poster is not None:
                poster.join(timeout=5)
            runtime.close()

    def test_observe_labels_racing_ack_preserves_acked_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        carrier_created = threading.Event()
        ack_started = threading.Event()
        ack_finished = threading.Event()
        ack_errors: list[BaseException] = []
        ack_thread: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='observe labels while acking')
            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='labeled message',
            )
            stale_messages = runtime.messages.list(pid)
            original_create_object = runtime.memory.create_object

            def coordinated_create_object(*args, **kwargs):
                handle = original_create_object(*args, **kwargs)
                metadata = kwargs.get('metadata')
                if metadata is not None and 'label_carrier' in metadata.tags:
                    carrier_created.set()
                    assert ack_started.wait(timeout=5)
                    # The unfixed path releases the store lock after carrier
                    # creation, allowing ACK to commit before its stale
                    # full-row message update. The fixed path keeps ACK
                    # blocked until observation commits.
                    ack_finished.wait(timeout=0.1)
                return handle

            monkeypatch.setattr(runtime.memory, 'create_object', coordinated_create_object)

            def ack_message() -> None:
                try:
                    assert carrier_created.wait(timeout=5)
                    ack_started.set()
                    runtime.messages.ack(pid, [message.message_id])
                except BaseException as exc:
                    ack_errors.append(exc)
                finally:
                    ack_finished.set()

            ack_thread = threading.Thread(target=ack_message, daemon=True)
            ack_thread.start()
            observed = runtime.messages.observe_labels(pid, stale_messages)
            ack_thread.join(timeout=5)

            assert not ack_thread.is_alive()
            assert ack_errors == []
            assert len(observed) == 1
            stored = runtime.store.get_process_message(message.message_id)
            assert stored is not None
            assert stored.status == ProcessMessageStatus.ACKED
            assert stored.acked_at is not None
            assert stored.metadata['label_carrier_oid'] == observed[0]
            assert runtime.messages.unread(pid) == []
        finally:
            if ack_thread is not None:
                ack_thread.join(timeout=5)
            runtime.close()

    def test_ack_racing_observe_labels_preserves_carrier_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        ack_selected = threading.Event()
        observer_started = threading.Event()
        observer_finished = threading.Event()
        observer_errors: list[BaseException] = []
        observed: list[str] = []
        observer: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='ack while observing labels')
            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='preserve carrier',
            )
            stale_messages = runtime.messages.list(pid)
            original_list = runtime.messages.list

            def coordinated_list(*args, **kwargs):
                selected = original_list(*args, **kwargs)
                if selected and not ack_selected.is_set():
                    ack_selected.set()
                    assert observer_started.wait(timeout=5)
                    # A transactionally selected ACK keeps label observation
                    # out of this window. The column-scoped CAS also cannot
                    # overwrite metadata once observation commits.
                    observer_finished.wait(timeout=0.2)
                return selected

            monkeypatch.setattr(runtime.messages, 'list', coordinated_list)

            def observe() -> None:
                try:
                    assert ack_selected.wait(timeout=5)
                    observer_started.set()
                    observed.extend(runtime.messages.observe_labels(pid, stale_messages))
                except BaseException as exc:
                    observer_errors.append(exc)
                finally:
                    observer_finished.set()

            observer = threading.Thread(target=observe, daemon=True)
            observer.start()
            acked = runtime.messages.ack(pid, [message.message_id])
            observer.join(timeout=5)

            assert not observer.is_alive()
            assert observer_errors == []
            assert [item.message_id for item in acked] == [message.message_id]
            assert len(observed) == 1
            stored = runtime.store.get_process_message(message.message_id)
            assert stored is not None
            assert stored.status == ProcessMessageStatus.ACKED
            assert stored.acked_at is not None
            assert stored.metadata['label_carrier_oid'] == observed[0]
        finally:
            if observer is not None:
                observer.join(timeout=5)
            runtime.close()

    def test_observe_labels_racing_exit_cannot_revive_terminal_process(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        carrier_created = threading.Event()
        exit_started = threading.Event()
        exit_finished = threading.Event()
        exit_errors: list[BaseException] = []
        exit_thread: threading.Thread | None = None
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='observe labels while exiting')
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='labeled message',
            )
            stale_messages = runtime.messages.list(pid)
            original_create_object = runtime.memory.create_object

            def coordinated_create_object(*args, **kwargs):
                handle = original_create_object(*args, **kwargs)
                metadata = kwargs.get('metadata')
                if metadata is not None and 'label_carrier' in metadata.tags:
                    carrier_created.set()
                    assert exit_started.wait(timeout=5)
                    exit_finished.wait(timeout=0.1)
                return handle

            monkeypatch.setattr(runtime.memory, 'create_object', coordinated_create_object)

            def exit_process() -> None:
                try:
                    assert carrier_created.wait(timeout=5)
                    exit_started.set()
                    runtime.process.exit(pid)
                except BaseException as exc:
                    exit_errors.append(exc)
                finally:
                    exit_finished.set()

            exit_thread = threading.Thread(target=exit_process, daemon=True)
            exit_thread.start()
            runtime.messages.observe_labels(pid, stale_messages)
            exit_thread.join(timeout=5)

            assert not exit_thread.is_alive()
            assert exit_errors == []
            assert runtime.process.get(pid).status == ProcessStatus.EXITED
            assert runtime.store.claim_runnable_process(pid) is None
            with pytest.raises(ProcessError, match='terminal process'):
                runtime.messages.observe_labels(pid, stale_messages)
        finally:
            if exit_thread is not None:
                exit_thread.join(timeout=5)
            runtime.close()

    def test_double_observe_labels_reuses_one_carrier(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        first_carrier_created = threading.Event()
        second_carrier_created = threading.Event()
        start = threading.Barrier(3)
        creation_lock = threading.Lock()
        creation_oids: list[str] = []
        observer_threads: list[threading.Thread] = []
        observer_errors: list[BaseException] = []
        observer_results: list[list[str]] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='observe labels twice')
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                subject='labeled message',
            )
            stale_copies = [runtime.messages.list(pid), runtime.messages.list(pid)]
            original_create_object = runtime.memory.create_object

            def coordinated_create_object(*args, **kwargs):
                handle = original_create_object(*args, **kwargs)
                metadata = kwargs.get('metadata')
                if metadata is not None and 'label_carrier' in metadata.tags:
                    with creation_lock:
                        creation_oids.append(handle.oid)
                        ordinal = len(creation_oids)
                    if ordinal == 1:
                        first_carrier_created.set()
                        # On the unfixed path the second observer also sees
                        # stale metadata and creates a second carrier. Under
                        # the fixed transaction it cannot enter this window.
                        second_carrier_created.wait(timeout=0.2)
                    else:
                        second_carrier_created.set()
                return handle

            monkeypatch.setattr(runtime.memory, 'create_object', coordinated_create_object)

            def observe(messages: list[ProcessMessage]) -> None:
                try:
                    start.wait(timeout=5)
                    observer_results.append(runtime.messages.observe_labels(pid, messages))
                except BaseException as exc:
                    observer_errors.append(exc)

            observer_threads = [
                threading.Thread(target=observe, args=(messages,), daemon=True)
                for messages in stale_copies
            ]
            for thread in observer_threads:
                thread.start()
            start.wait(timeout=5)
            for thread in observer_threads:
                thread.join(timeout=5)

            assert all(not thread.is_alive() for thread in observer_threads)
            assert observer_errors == []
            assert first_carrier_created.is_set()
            assert len(creation_oids) == 1
            assert len(observer_results) == 2
            assert observer_results[0] == observer_results[1]
            stored = runtime.store.get_process_message(stale_copies[0][0].message_id)
            assert stored is not None
            assert stored.metadata['label_carrier_oid'] == observer_results[0][0]
        finally:
            for thread in observer_threads:
                thread.join(timeout=5)
            runtime.close()

    def test_receive_process_messages_blocks_until_matching_message_then_resumes(self) -> None:
        client = PlannedActionClient([{'action': 'receive_process_messages', 'channel': 'control', 'correlation_id': 'job-1'}])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'wait for control message')
            runtime.activate_skill(child, 'agent-libos-child-processes')
            waiting = runtime.run_process_once(child)
            assert waiting['waiting_message']
            assert waiting['filters']['channel'] == 'control'
            assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
            assert len(client.user_prompts) == 1
            runtime.messages.send_from_process(parent, child, channel='noise', correlation_id='job-1', subject='not yet')
            assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
            skipped = runtime.run_process_once(child)
            assert skipped['skipped']
            assert len(client.user_prompts) == 1
            matching = runtime.messages.send_from_process(parent, child, channel='control', correlation_id='job-1', subject='resume', payload={'ready': True})
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            resumed = runtime.run_process_once(child)
            assert resumed['ok']
            assert resumed['resumed_after_message']
            assert resumed['action']['action'] == 'receive_process_messages'
            assert resumed['result']['payload']['messages'][0]['message_id'] == matching.message_id
            assert resumed['result']['payload']['messages'][0]['payload'] == {'ready': True}
            assert resumed['result']['payload']['acked_message_ids'] == [matching.message_id]
            assert len(client.user_prompts) == 1
            assert [message.subject for message in runtime.messages.unread(child)] == ['not yet']
            assert 'process.message.wait_wake' in _audit_actions(runtime)
        finally:
            runtime.close()

    def test_receive_message_syscall_waits_inside_single_syscall_until_matching_message(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.scheduler.poll_interval_s = 0.001
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'wait via syscall')
            child_session = LibOSSyscallSession(runtime, child)

            async def scenario() -> dict[str, Any]:
                task = asyncio.create_task(child_session.handle('process.receive_messages', {'block': True, 'channel': 'control', 'correlation_id': 'job-1'}))
                await asyncio.sleep(0.01)
                assert not task.done()
                assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
                runtime.messages.send_from_process(parent, child, channel='noise', correlation_id='job-1', subject='not matching')
                await asyncio.sleep(0.01)
                assert not task.done()
                assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
                matching = runtime.messages.send_from_process(parent, child, channel='control', correlation_id='job-1', subject='matching')
                result = await asyncio.wait_for(task, timeout=1.0)
                result['expected_message_id'] = matching.message_id
                return result
            result = asyncio.run(scenario())
            assert result['ready']
            assert result['messages'][0]['message_id'] == result['expected_message_id']
            assert result['messages'][0]['status'] == 'acked'
            assert result['acked_message_ids'] == [result['expected_message_id']]
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_receive_message_syscall_blocks_by_default(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.scheduler.poll_interval_s = 0.001
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'default receive')
            child_session = LibOSSyscallSession(runtime, child)

            async def scenario() -> dict[str, Any]:
                task = asyncio.create_task(child_session.handle('process.receive_messages', {'channel': 'control'}))
                await asyncio.sleep(0.01)
                assert not task.done()
                assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
                matching = runtime.messages.send_from_process(parent, child, channel='control')
                result = await asyncio.wait_for(task, timeout=1.0)
                result['expected_message_id'] = matching.message_id
                return result
            result = asyncio.run(scenario())
            assert result['ready']
            assert result['messages'][0]['message_id'] == result['expected_message_id']
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_interrupted_message_syscall_restores_runnable_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.scheduler.poll_interval_s = 0.001
            child = runtime.process.spawn(image='base-agent:v0', goal='wait then interrupt')
            session = LibOSSyscallSession(runtime, child)

            async def scenario() -> None:
                task = asyncio.create_task(session.handle('process.receive_messages', {'block': True, 'channel': 'control'}))
                await asyncio.sleep(0.01)
                assert runtime.process.get(child).status == ProcessStatus.WAITING_EVENT
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

            asyncio.run(scenario())
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(child).status_message is None
            assert 'syscall.wait_interrupted' in _audit_actions(runtime)
        finally:
            runtime.close()

    def test_process_messages_are_durable_in_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db)
            pid = runtime.process.spawn(image='base-agent:v0', goal='persist queue')
            message = runtime.messages.post(sender='test', recipient_pid=pid, subject='persisted')
            runtime.close()
            reopened = Runtime.open(db)
            try:
                unread = reopened.messages.unread(pid)
                assert [item.message_id for item in unread] == [message.message_id]
                assert unread[0].subject == 'persisted'
            finally:
                reopened.close()

    def test_interrupt_message_preempts_tool_call_until_read(self) -> None:
        client = PlannedActionClient([{'action': 'get_current_time', 'timezone': 'UTC'}, {'action': 'read_process_messages'}])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='handle interrupts')
            runtime.activate_skill(pid, 'agent-libos-runtime-session')
            runtime.activate_skill(pid, 'agent-libos-child-processes')
            runtime.messages.post(sender='test', recipient_pid=pid, kind=ProcessMessageKind.INTERRUPT, subject='urgent', body='inspect this before other work')
            interrupted = runtime.run_process_once(pid)
            assert interrupted['result']['interrupted_by_message']
            assert interrupted['result']['message_notice']['phase'] == 'before_tool_call'
            assert 'primitive.clock.now' not in _audit_actions(runtime)
            assert 'process_message_notice' in client.user_prompts[0]
            read = runtime.run_process_once(pid)
            assert read['action']['action'] == 'read_process_messages'
            assert read['result']['payload']['messages'][0]['kind'] == 'interrupt'
            assert runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT) == []
        finally:
            runtime.close()

    @pytest.mark.parametrize('message_action', ['read_process_messages', 'receive_process_messages'])
    def test_interrupt_allows_message_skill_discovery_and_activation_before_ack(self, message_action: str) -> None:
        client = PlannedActionClient([
            {'action': 'discover_skills', 'text': 'messages', 'limit': 4},
            _activate_action('agent-libos-child-processes'),
            {'action': message_action},
        ])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='activate interrupt handling')
            before = runtime.process.get(pid)
            assert 'activate_skill' in before.model_tool_table
            assert 'read_process_messages' not in before.model_tool_table
            assert 'receive_process_messages' not in before.model_tool_table

            message = runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                kind=ProcessMessageKind.INTERRUPT,
                subject='urgent',
                body='activate message handling before acknowledging this',
            )

            discovered = runtime.run_process_once(pid)
            assert discovered['action'] == {
                'action': 'discover_skills',
                'text': 'messages',
                'limit': 4,
            }
            assert discovered['result']['ok']
            assert [
                item['skill_id']
                for item in discovered['result']['payload']['skills']
            ] == ['agent-libos-child-processes']
            assert message.message_id in {
                item.message_id
                for item in runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT)
            }

            activated = runtime.run_process_once(pid)
            assert activated['action'] == _activate_action('agent-libos-child-processes')
            assert activated['result']['ok']
            assert not activated['result'].get('interrupted_by_message', False)
            assert message.message_id in {
                item.message_id
                for item in runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT)
            }
            after_activation = runtime.process.get(pid)
            assert 'read_process_messages' in after_activation.model_tool_table
            assert 'receive_process_messages' in after_activation.model_tool_table
            assert 'agent-libos-child-processes' not in client.user_prompts[0]
            assert 'discover_skills' in client.user_prompts[0]
            assert 'agent-libos-child-processes' in client.user_prompts[1]

            handled = runtime.run_process_once(pid)
            assert handled['action']['action'] == message_action
            assert handled['result']['ok']
            assert handled['result']['payload']['messages'][0]['message_id'] == message.message_id
            assert handled['result']['payload']['acked_message_ids'] == [message.message_id]
            assert runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT) == []
        finally:
            runtime.close()

    def test_interrupt_keeps_unrelated_skill_activation_blocked(self) -> None:
        client = PlannedActionClient([
            _activate_action('agent-libos-runtime-session'),
        ])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='do not bypass interrupt handling')
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                kind=ProcessMessageKind.INTERRUPT,
                subject='urgent',
            )

            blocked = runtime.run_process_once(pid)
            assert blocked['result']['interrupted_by_message']
            assert 'discover_skills' in blocked['result']['error']
            assert "text 'messages'" in blocked['result']['error']
            assert 'agent-libos-runtime-session' not in runtime.process.get(pid).loaded_skills
            assert len(runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT)) == 1
        finally:
            runtime.close()

    def test_interrupt_keeps_unrelated_skill_discovery_blocked(self) -> None:
        client = PlannedActionClient([
            {'action': 'discover_skills', 'text': 'workspace editing', 'limit': 4},
        ])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='read the interrupt first')
            runtime.messages.post(
                sender='test',
                recipient_pid=pid,
                kind=ProcessMessageKind.INTERRUPT,
                subject='urgent',
            )

            blocked = runtime.run_process_once(pid)

            assert blocked['result']['interrupted_by_message']
            assert 'discover_skills' in blocked['result']['error']
            assert runtime.process.get(pid).loaded_skills == {}
            assert len(runtime.messages.unread(pid, kind=ProcessMessageKind.INTERRUPT)) == 1
        finally:
            runtime.close()

    def test_normal_message_notifies_after_tool_call_without_preempting(self) -> None:
        client = PlannedActionClient([{'action': 'get_current_time', 'timezone': 'UTC'}])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='handle normal messages')
            runtime.activate_skill(pid, 'agent-libos-runtime-session')
            runtime.capability.grant(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
            runtime.messages.post(sender='test', recipient_pid=pid, kind=ProcessMessageKind.NORMAL, subject='later', body='read after current tool')
            result = runtime.run_process_once(pid)
            assert result['action']['action'] == 'get_current_time'
            assert 'primitive.clock.now' in _audit_actions(runtime)
            assert result['result']['message_notice']['phase'] == 'after_tool_call'
            assert result['result']['message_notice']['kind'] == 'normal'
            assert len(runtime.messages.unread(pid, kind=ProcessMessageKind.NORMAL)) == 1
        finally:
            runtime.close()

    def test_normal_message_notice_requires_mediated_read_without_copying_body_to_prompt(self) -> None:
        client = PlannedActionClient([
            {'action': 'get_current_time', 'timezone': 'UTC'},
            {'action': 'discover_skills', 'text': 'messages', 'limit': 4},
            _activate_action('agent-libos-child-processes'),
            {'action': 'read_process_messages'},
        ])
        runtime = Runtime.open('local')
        runtime.llm.client = client
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='apply queued user update')
            runtime.activate_skill(pid, 'agent-libos-runtime-session')
            runtime.capability.grant(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
            message = runtime.human.send_process_message(
                pid,
                'explicit user acceptance criterion that must not leak before read',
            )

            first = runtime.run_process_once(pid)
            assert first['action']['action'] == 'get_current_time'
            assert first['result']['message_notice']['message_ids'] == [message.message_id]

            discovered = runtime.run_process_once(pid)
            assert discovered['action']['action'] == 'discover_skills'
            assert discovered['result']['ok']

            activated = runtime.run_process_once(pid)
            assert activated['action'] == _activate_action('agent-libos-child-processes')
            activation_prompt = client.user_prompts[2]
            assert message.body not in activation_prompt
            assert message.message_id not in activation_prompt
            directive = activation_prompt.rsplit(
                'Pending explicit process input (mandatory control action):',
                maxsplit=1,
            )[1]
            assert message.message_id not in directive
            assert 'cumulative acceptance checklist' in directive
            assert 'older goal' not in activation_prompt
            assert 'it changes only the requirements it states' in directive
            assert activation_prompt.rstrip().endswith(
                'The message body remains behind the mediated message-read boundary '
                'and is not copied into prompt context.'
            )

            handled = runtime.run_process_once(pid)
            assert handled['action']['action'] == 'read_process_messages'
            assert handled['result']['payload']['messages'][0]['body'] == message.body
            assert handled['result']['payload']['acked_message_ids'] == [message.message_id]
            assert runtime.messages.unread(pid) == []
        finally:
            runtime.close()

    def test_model_message_read_normalizes_reversibly_stringified_json_arguments(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='read explicit input')
            runtime.activate_skill(pid, 'agent-libos-child-processes')
            message = runtime.human.send_process_message(
                pid,
                'handle this exact interrupt',
                kind=ProcessMessageKind.INTERRUPT,
            )
            runtime.llm.client = PlannedActionClient([
                {
                    'action': 'read_process_messages',
                    'include_acked': 'false',
                    'kind': 'None',
                    'sender': 'null',
                    'channel': 'None',
                    'correlation_id': 'null',
                    'reply_to': 'None',
                    'message_ids': f'["{message.message_id}"]',
                    'limit': '100',
                    'ack': 'true',
                }
            ])

            result = runtime.run_process_once(pid)

            assert result['ok']
            assert result['action']['message_ids'] == [message.message_id]
            assert result['action']['limit'] == 100
            assert result['action']['kind'] is None
            assert result['result']['payload']['messages'][0]['body'] == message.body
            assert result['result']['payload']['acked_message_ids'] == [message.message_id]
            normalized = next(
                record
                for record in runtime.audit.trace(actor=pid)
                if record.action == 'llm.process_message_arguments_normalized'
            )
            assert normalized.decision['normalized_fields'] == [
                'ack',
                'channel',
                'correlation_id',
                'include_acked',
                'kind',
                'limit',
                'message_ids',
                'reply_to',
                'sender',
            ]
        finally:
            runtime.close()

class PlannedActionClient:

    def __init__(self, actions: list[dict[str, Any]]):
        self.actions = list(actions)
        self.user_prompts: list[str] = []

    def complete_action(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> LLMCompletion:
        if not self.actions:
            raise AssertionError('no planned action remains')
        self.user_prompts.append(str(messages[-1]['content']))
        action = self.actions.pop(0)
        name = str(action['action'])
        args = {key: value for key, value in action.items() if key != 'action'}
        return LLMCompletion(content='', tool_calls=[{'id': f'message_{len(self.user_prompts)}', 'name': name, 'arguments': json.dumps(args)}])

def _audit_actions(runtime: Runtime) -> set[str]:
    return {record.action for record in runtime.audit.trace()}
