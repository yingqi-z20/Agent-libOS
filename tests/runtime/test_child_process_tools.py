from __future__ import annotations
import pytest
import asyncio
import json
import re
import threading
from typing import Any
from agent_libos import Runtime
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    EventType,
    ForkMode,
    KilledProcessOutcome,
    MergeResult,
    ObjectPatch,
    ObjectOwnerKind,
    ObjectType,
    ProcessMessageKind,
    ProcessSignal,
    ProcessStatus,
    ResourceBudget,
    process_outcome_to_mapping,
    process_wait_state_to_mapping,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ProcessError, ProcessWaitRequired
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.skills import get_builtin_skill_catalog
from scripts.llm_context_probe import last_tool_result
from tests.support.public_errors import assert_public_error_message


def _grant_process_spawn(runtime: Runtime, pid: str) -> None:
    runtime.capability.grant(pid, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')


def _skill_package_sha256(skill_id: str) -> str:
    package = get_builtin_skill_catalog().get(skill_id)
    assert package is not None
    return package.package_sha256


def _grant_image_read(runtime: Runtime, pid: str, image_id: str) -> None:
    runtime.capability.grant(pid, runtime.image_registry.resource_for(image_id), [CapabilityRight.READ], issued_by='test')


class TestChildProcessTool:

    def test_copy_fork_shares_writable_object_identity_instead_of_cloning(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(
                image='base-agent:v0',
                goal='share an explicitly mutable object',
            )
            _grant_process_spawn(runtime, parent)
            shared = runtime.memory.create_object(
                parent,
                ObjectType.ARTIFACT,
                {'version': 1},
                immutable=False,
                name='copy.mode.shared',
            )
            parent_process = runtime.process.get(parent)
            assert parent_process.memory_view is not None
            parent_process.memory_view.roots.append(shared)
            runtime.store.update_process(parent_process)

            child = runtime.process.fork(
                parent,
                goal='mutate the shared object',
                mode=ForkMode.COPY,
            )
            child_process = runtime.process.get(child)
            assert child_process.memory_view is not None
            child_handle = next(
                handle
                for handle in child_process.memory_view.roots
                if handle.oid == shared.oid
            )

            assert CapabilityRight.WRITE.value in child_handle.rights
            runtime.memory.update_object(
                child,
                child_handle,
                ObjectPatch(payload={'version': 2}),
            )

            assert runtime.memory.get_object(parent, shared).payload == {'version': 2}
        finally:
            runtime.close()

    def test_interrupt_signal_is_rejected_and_durable_interrupt_message_remains_supported(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='interrupt child')
            child = runtime.process.fork(parent, goal='receive an interrupt')
            before_events = [
                event.event_id
                for event in runtime.events.list(target=child)
                if event.type == EventType.PROCESS_SIGNAL
            ]

            with pytest.raises(ProcessError, match='durable interrupt process message'):
                runtime.process.signal_child(parent, child, ProcessSignal.INTERRUPT, reason='read this first')

            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert [
                event.event_id
                for event in runtime.events.list(target=child)
                if event.type == EventType.PROCESS_SIGNAL
            ] == before_events

            message = runtime.messages.send_from_process(
                parent,
                child,
                kind=ProcessMessageKind.INTERRUPT,
                subject='Parent interrupt',
                body='read this first',
            )

            assert runtime.messages.unread(child, kind=ProcessMessageKind.INTERRUPT) == [message]
        finally:
            runtime.close()

    def test_fork_wait_tool_blocks_parent_until_child_exits_and_exposes_result(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.scheduler.poll_interval_s = 1.0
            client = ParentChildClient()
            runtime.llm.client = client
            parent = runtime.process.spawn(image='base-agent:v0', goal='fork child and wait')
            _grant_process_spawn(runtime, parent)
            results = asyncio.run(runtime.arun_until_idle(max_quanta=8))
            parent_process = runtime.process.get(parent)
            assert parent_process.status == ProcessStatus.EXITED, repr(
                parent_process.outcome
            )
            assert client.child_pid is not None
            assert client.child_pid is not None
            assert runtime.process.get(client.child_pid).status == ProcessStatus.EXITED
            assert any((isinstance(result, dict) and result.get('waiting_event') for result in results))
            wait_result = next((result for result in results if _action_name(result) == 'wait_child_process'))
            result_oid = wait_result['result']['payload']['result_oid']
            parent_view = runtime.process.get(parent).memory_view
            assert parent_view is not None
            assert parent_view is not None
            assert result_oid in [handle.oid for handle in parent_view.roots]
            assert runtime.store.get_object(result_oid) is None
            assert not runtime.capability.check(parent, f'object:{result_oid}', CapabilityRight.READ)
            assert 'process.wait_wake' in [record.action for record in runtime.audit.trace()]
        finally:
            runtime.close()

    def test_parent_exit_releases_waited_child_result_memory(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='wait child result')
            child = runtime.process.fork(parent, goal='produce waited result')
            result = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'waited': True},
                name='waited.child.result',
            )
            runtime.process.exit(child, result=result)

            waited = runtime.process.wait(parent, child)

            assert waited.result is not None
            result_obj = runtime.store.get_object(result.oid)
            assert result_obj is not None
            assert result_obj.owner_kind == ObjectOwnerKind.PROCESS_RESULT
            assert result_obj.owner_id == child
            assert runtime.capability.check(parent, f'object:{result.oid}', CapabilityRight.READ)

            runtime.process.exit(parent)

            assert runtime.store.get_object(result.oid) is None
            assert not runtime.capability.check(parent, f'object:{result.oid}', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_parent_exit_rejects_live_descendants_without_orphaning_them(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='settle descendants before exit')
            child = runtime.process.fork(parent, goal='remain live until explicitly settled')

            with pytest.raises(ProcessError, match='descendants are nonterminal'):
                runtime.process.exit(parent, message='must not commit')

            blocked = runtime.tools.call(
                parent,
                'process_exit',
                {'payload': {'must_not_commit': True}},
            )
            assert not blocked.ok

            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert child in runtime.scheduler.runnable_pids()

            runtime.process.signal_child(parent, child, ProcessSignal.TERMINATE)
            runtime.process.exit(parent, message='now safe')
            assert runtime.process.get(parent).status == ProcessStatus.EXITED
            assert runtime.process.get(child).status == ProcessStatus.KILLED
        finally:
            runtime.close()

    def test_child_list_signal_and_budget_are_enforced(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='manage one child', resource_budget=ResourceBudget(max_child_processes=1))
            _grant_process_spawn(runtime, parent)
            other = runtime.process.spawn(image='base-agent:v0', goal='not a child')
            forked = runtime.tools.call(parent, 'fork_child_process', {'goal': 'child', 'include_parent_roots': False})
            assert forked.ok, forked.error
            child = forked.payload['child_pid']
            listed = runtime.tools.call(parent, 'list_child_processes', {})
            assert listed.ok, listed.error
            assert [entry['pid'] for entry in listed.payload['children']] == [child]
            assert listed.payload['children'][0]['working_directory'] == '.'
            paused = runtime.tools.call(parent, 'signal_child_process', {'child_pid': child, 'signal': 'pause'})
            assert paused.ok, paused.error
            assert paused.payload['status'] == 'paused'
            resumed = runtime.tools.call(parent, 'signal_child_process', {'child_pid': child, 'signal': 'resume'})
            assert resumed.ok, resumed.error
            assert resumed.payload['status'] == 'runnable'
            denied_signal = runtime.tools.call(parent, 'signal_child_process', {'child_pid': other, 'signal': 'pause'})
            assert not denied_signal.ok
            assert_public_error_message(
                denied_signal.error,
                code='execution_error',
                error_type='ProcessError',
                forbidden=('not a child', other),
            )
            denied_fork = runtime.tools.call(parent, 'fork_child_process', {'goal': 'second child'})
            assert not denied_fork.ok
            assert_public_error_message(
                denied_fork.error,
                code='execution_error',
                error_type='ResourceLimitExceeded',
                forbidden=('exhausted child process budget',),
            )
        finally:
            runtime.close()

    def test_nonblocking_wait_child_process_does_not_suspend_parent(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='poll child')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'still running')

            waited = runtime.tools.call(parent, 'wait_child_process', {'child_pid': child, 'block': False})

            assert waited.ok, waited.error
            assert waited.payload['ready'] is False
            child_state = runtime.process.get(child)
            assert waited.payload['wait_state'] is None
            assert waited.payload['outcome'] is None
            assert waited.payload['state_generation'] == child_state.state_generation
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status_message is None
        finally:
            runtime.close()

    def test_child_wait_boundaries_publish_typed_terminal_outcome(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='observe typed child outcome')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'cancelled child')
            runtime.process.cancel(child, 'typed cancellation reason')
            terminal = runtime.process.get(child)
            assert isinstance(terminal.outcome, KilledProcessOutcome)
            expected_outcome = process_outcome_to_mapping(terminal.outcome)

            direct = runtime.process.wait(parent, child)
            assert direct.wait_state is None
            assert direct.outcome == terminal.outcome
            assert direct.state_generation == terminal.state_generation

            session = LibOSSyscallSession(runtime, parent)
            listed = asyncio.run(session.handle('process.list_children', {}))
            listed_child = next(item for item in listed['children'] if item['pid'] == child)
            assert listed_child['wait_state'] is None
            assert listed_child['outcome'] == expected_outcome
            assert listed_child['state_generation'] == terminal.state_generation

            syscall_wait = asyncio.run(
                session.handle(
                    'process.wait',
                    {'child_pid': child, 'block': False},
                )
            )
            assert syscall_wait['ready'] is True
            assert syscall_wait['wait_state'] is None
            assert syscall_wait['outcome'] == expected_outcome
            assert syscall_wait['state_generation'] == terminal.state_generation

            tool_wait = runtime.tools.call(
                parent,
                'wait_child_process',
                {'child_pid': child, 'block': False},
            )
            assert tool_wait.ok, tool_wait.error
            assert tool_wait.payload['ready'] is True
            assert tool_wait.payload['wait_state'] is None
            assert tool_wait.payload['outcome'] == expected_outcome
            assert tool_wait.payload['state_generation'] == terminal.state_generation
        finally:
            runtime.close()

    def test_child_signal_boundaries_publish_canonical_typed_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='typed signal parent')
            _grant_process_spawn(runtime, parent)

            tool_child = runtime.spawn_child_process(parent, 'tool signal child')
            tool_paused = runtime.tools.call(
                parent,
                'signal_child_process',
                {'child_pid': tool_child, 'signal': ProcessSignal.PAUSE.value},
            )
            assert tool_paused.ok, tool_paused.error
            paused = runtime.process.get(tool_child)
            assert tool_paused.payload['status'] == ProcessStatus.PAUSED.value
            assert tool_paused.payload['wait_state'] == process_wait_state_to_mapping(paused.wait_state)
            assert tool_paused.payload['outcome'] is None
            assert tool_paused.payload['state_generation'] == paused.state_generation

            tool_terminated = runtime.tools.call(
                parent,
                'signal_child_process',
                {'child_pid': tool_child, 'signal': ProcessSignal.TERMINATE.value},
            )
            assert tool_terminated.ok, tool_terminated.error
            terminated = runtime.process.get(tool_child)
            assert tool_terminated.payload['status'] == ProcessStatus.KILLED.value
            assert tool_terminated.payload['wait_state'] is None
            assert tool_terminated.payload['outcome'] == process_outcome_to_mapping(terminated.outcome)
            assert tool_terminated.payload['state_generation'] == terminated.state_generation

            syscall_child = runtime.spawn_child_process(parent, 'syscall signal child')
            session = LibOSSyscallSession(runtime, parent)
            syscall_paused = asyncio.run(
                session.handle(
                    'process.signal',
                    {'child_pid': syscall_child, 'signal': ProcessSignal.PAUSE.value},
                )
            )
            paused = runtime.process.get(syscall_child)
            assert syscall_paused['status'] == ProcessStatus.PAUSED.value
            assert syscall_paused['wait_state'] == process_wait_state_to_mapping(paused.wait_state)
            assert syscall_paused['outcome'] is None
            assert syscall_paused['state_generation'] == paused.state_generation

            syscall_terminated = asyncio.run(
                session.handle(
                    'process.signal',
                    {'child_pid': syscall_child, 'signal': ProcessSignal.TERMINATE.value},
                )
            )
            terminated = runtime.process.get(syscall_child)
            assert syscall_terminated['status'] == ProcessStatus.KILLED.value
            assert syscall_terminated['wait_state'] is None
            assert syscall_terminated['outcome'] == process_outcome_to_mapping(terminated.outcome)
            assert syscall_terminated['state_generation'] == terminated.state_generation
        finally:
            runtime.close()

    def test_wait_child_process_rechecks_child_after_wait_state_write(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='wait race parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'wait race child')
            process_repository = runtime.process.transitions.store
            original_transition = process_repository.apply_process_state_transition
            triggered = {'value': False}

            def racing_transition(pid, status, *args, **kwargs):
                if (
                    pid == parent
                    and ProcessStatus(status) == ProcessStatus.WAITING_EVENT
                    and not triggered['value']
                ):
                    triggered['value'] = True
                    runtime.process.exit(child, message='done before parent wait persisted')
                return original_transition(pid, status, *args, **kwargs)

            monkeypatch.setattr(
                process_repository,
                'apply_process_state_transition',
                racing_transition,
            )
            waited = runtime.process.wait(parent, child)

            assert triggered['value']
            assert waited.status == ProcessStatus.EXITED
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status_message is None
        finally:
            runtime.close()

    def test_terminal_process_cannot_be_resumed_by_signal(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='exit once')
            runtime.process.exit(pid, message='done')

            with pytest.raises(ProcessError, match='cannot signal terminal process'):
                runtime.process.resume(pid)

            assert runtime.process.get(pid).status == ProcessStatus.EXITED
        finally:
            runtime.close()

    def test_waiting_process_cannot_be_resumed_without_wait_condition(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='wait parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'wait child')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)

            with pytest.raises(ProcessError, match='cannot resume waiting process'):
                runtime.process.resume(parent)

            assert runtime.process.get(parent).status == ProcessStatus.WAITING_EVENT
            assert runtime.process.get(parent).status_message == f'waiting for {child}'
        finally:
            runtime.close()

    def test_terminal_process_cannot_exit_again_or_overwrite_status(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='exit once')
            runtime.process.exit(pid, message='done')

            with pytest.raises(ProcessError, match='cannot exit terminal process'):
                runtime.process.exit(pid, message='late overwrite')

            process = runtime.process.get(pid)
            assert process.status == ProcessStatus.EXITED
            assert process.status_message is not None
            assert process.status_message.startswith('result_oid:')
            result = runtime.store.get_object(process.status_message.split(':', 1)[1])
            assert result is not None
            assert result.payload == {'message': 'done'}
            assert 'late overwrite' not in str(result.payload)
        finally:
            runtime.close()

    def test_resource_kill_wakes_parent_waiting_on_child(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='wait for killed child')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'will be killed')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            assert runtime.process.get(parent).status == ProcessStatus.WAITING_EVENT

            runtime.resources.kill_if_exceeded(child, reason='test budget exhausted')

            assert runtime.process.get(child).status == ProcessStatus.KILLED
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status_message is None
        finally:
            runtime.close()

    def test_resource_kill_uses_terminal_cleanup_for_root_process_memory(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='root killed')
            owned = runtime.memory.create_object(
                pid,
                ObjectType.OBSERVATION,
                {'released': True},
                name='root.kill.released',
            )

            runtime.resources.kill_if_exceeded(pid, reason='test budget exhausted')

            assert runtime.process.get(pid).status == ProcessStatus.KILLED
            assert runtime.store.get_object(owned.oid) is None
            assert any(
                event.type == EventType.PROCESS_EXITED
                and event.source == pid
                and event.payload.get('status') == ProcessStatus.KILLED.value
                for event in runtime.events.list()
            )
        finally:
            runtime.close()

    def test_failed_spawn_child_launch_does_not_leave_runnable_or_budget_residue(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(
                image='base-agent:v0',
                goal='parent',
                resource_budget=ResourceBudget(max_child_processes=1),
            )
            _grant_process_spawn(runtime, parent)
            failed_pid = {'value': None}

            def fail_child_launch(pid: str, image_id: str) -> None:
                process = runtime.process.get(pid)
                if process.parent_pid == parent:
                    failed_pid['value'] = pid
                    raise RuntimeError('child boot failed')

            runtime.process.add_after_spawn_hook(fail_child_launch)

            with pytest.raises(RuntimeError, match='child boot failed'):
                runtime.spawn_child_process(parent, 'child fails during boot')

            assert failed_pid['value'] is not None
            assert runtime.store.get_process(failed_pid['value']) is None
            assert runtime.process.list_children(parent) == []
            assert runtime.process.get(parent).resource_usage.child_processes == 0
            assert runtime.store.get_namespace(runtime.memory.process_namespace(failed_pid['value'])) is None
            assert runtime.capability.capabilities_for(failed_pid['value']) == []
        finally:
            runtime.close()

    def test_spawn_child_process_creates_fresh_child_without_parent_memory_or_default_caps(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='review-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            _grant_image_read(runtime, parent, 'coding-agent:v0')
            parent_note = runtime.memory.create_object(pid=parent, object_type='observation', name='parent.note', payload={'visible_to_parent': True})
            spawned = runtime.tools.call(parent, 'spawn_child_process', {'goal': 'fresh child', 'image': 'coding-agent:v0'})
            assert spawned.ok, spawned.error
            child = runtime.process.get(spawned.payload['child_pid'])
            assert child.parent_pid == parent
            assert child.image_id == 'coding-agent:v0'
            assert 'read_text_file' in child.tool_table
            assert parent_note.oid not in [handle.oid for handle in child.memory_view.roots]
            assert [handle.oid for handle in child.memory_view.roots] == [child.goal_oid]
            read_resource = runtime.filesystem.resource_for_path('README.md')
            assert not runtime.capability.check(child.pid, read_resource, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_fork_child_process_tool_allows_authorized_cross_image_fork(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='review-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            _grant_image_read(runtime, parent, 'coding-agent:v0')
            parent_note = runtime.memory.create_object(
                pid=parent,
                object_type='observation',
                name='parent.fork.note',
                payload={'visible_to_parent': True},
            )

            forked = runtime.tools.call(
                parent,
                'fork_child_process',
                {'goal': 'cross image fork', 'image': 'coding-agent:v0', 'include_parent_roots': False},
            )

            assert forked.ok, forked.error
            child = runtime.process.get(forked.payload['child_pid'])
            assert child.parent_pid == parent
            assert child.image_id == 'coding-agent:v0'
            assert 'read_text_file' in child.tool_table
            assert parent_note.oid not in [handle.oid for handle in child.memory_view.roots]
            assert [handle.oid for handle in child.memory_view.roots] == [child.goal_oid]
            read_resource = runtime.filesystem.resource_for_path('README.md')
            assert not runtime.capability.check(child.pid, read_resource, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_fork_child_process_tool_requires_cross_image_read_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='review-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            before = len(runtime.process.list())

            denied = runtime.tools.call(
                parent,
                'fork_child_process',
                {'goal': 'denied cross image fork', 'image': 'coding-agent:v0'},
            )

            assert not denied.ok
            assert_public_error_message(
                denied.error,
                code='permission_denied',
                error_type='CapabilityDenied',
                forbidden=('image:coding-agent:v0',),
            )
            assert len(runtime.process.list()) == before
            assert runtime.process.list_children(parent) == []
        finally:
            runtime.close()

    def test_spawn_child_process_requires_process_spawn_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            before = len(runtime.process.list())

            with pytest.raises(CapabilityDenied, match='process:spawn'):
                runtime.spawn_child_process(parent, 'denied child')

            assert len(runtime.process.list()) == before
            assert runtime.process.list_children(parent) == []
        finally:
            runtime.close()

    def test_spawn_child_process_rule_can_bind_authority_to_one_image(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(
                image='base-agent:v0',
                goal='bounded maintenance parent',
                authority_manifest={
                    'authorized_capabilities': [
                        {
                            'resource': 'process:spawn',
                            'rights': ['write'],
                            'constraints': {
                                'authority_rules': [
                                    {
                                        'rule_id': 'test.context-maintenance.spawn',
                                        'operation': 'process.spawn_child',
                                        'effect': 'allow',
                                        'risk': 'low',
                                        'conditions': {
                                            'image_id': 'context-compressor:v0',
                                        },
                                    }
                                ]
                            },
                        },
                        {
                            'resource': 'image:context-compressor:v0',
                            'rights': ['read'],
                        },
                    ],
                },
            )

            child = runtime.spawn_child_process(
                parent,
                'compact bounded context',
                image='context-compressor:v0',
            )
            assert runtime.process.get(child).image_id == 'context-compressor:v0'
            assert any(
                record.action == 'process.spawn_child'
                and record.actor == parent
                and record.target == f'process:{child}'
                for record in runtime.audit.trace(actor=parent)
            )
            assert any(
                event.type == EventType.PROCESS_CREATED
                and event.source == parent
                and event.target == child
                for event in runtime.events.list()
            )

            with pytest.raises(CapabilityDenied, match='constraints rejected'):
                runtime.spawn_child_process(parent, 'forbidden same-image child')
            with pytest.raises(CapabilityDenied, match='constraints rejected'):
                runtime.fork_child_process(parent, 'forbidden fork')
            assert [process.pid for process in runtime.process.list_children(parent)] == [child]
        finally:
            runtime.close()

    def test_fork_syscall_requires_process_spawn_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            session = LibOSSyscallSession(runtime, parent)
            before = len(runtime.process.list())

            with pytest.raises(CapabilityDenied, match='process:spawn'):
                asyncio.run(session.handle('process.fork', {'goal': 'denied child'}))

            assert len(runtime.process.list()) == before
            assert runtime.process.list_children(parent) == []
        finally:
            runtime.close()

    def test_cross_image_spawn_and_exec_require_image_read_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            before = len(runtime.process.list())

            with pytest.raises(CapabilityDenied, match='image:coding-agent:v0'):
                runtime.spawn_child_process(parent, 'denied child', image='coding-agent:v0')

            assert len(runtime.process.list()) == before
            assert runtime.process.list_children(parent) == []
            _grant_image_read(runtime, parent, 'coding-agent:v0')
            child = runtime.spawn_child_process(parent, 'allowed child', image='coding-agent:v0')
            assert runtime.process.get(child).image_id == 'coding-agent:v0'

            target = runtime.process.spawn(image='base-agent:v0', goal='exec target')
            original = runtime.process.get(target)
            with pytest.raises(CapabilityDenied, match='image:coding-agent:v0'):
                runtime.exec_process(target, 'coding-agent:v0', goal='denied exec')

            unchanged = runtime.process.get(target)
            assert unchanged.image_id == 'base-agent:v0'
            assert unchanged.goal_oid == original.goal_oid
            _grant_image_read(runtime, target, 'coding-agent:v0')
            runtime.exec_process(target, 'coding-agent:v0', goal='allowed exec', preserve_capabilities=False, preserve_memory=False)
            assert runtime.process.get(target).image_id == 'coding-agent:v0'
        finally:
            runtime.close()

    def test_spawn_child_process_inherits_only_explicit_capabilities(self) -> None:
        runtime = Runtime.open('local')
        try:
            readme_resource = runtime.filesystem.resource_for_path('README.md')
            parent = runtime.process.spawn(
                image='review-agent:v0',
                goal='parent',
                authority_manifest={
                    'authorized_capabilities': [
                        {'resource': 'process:spawn', 'rights': ['write']},
                        {'resource': readme_resource, 'rights': ['read'], 'delegable': True},
                    ],
                },
            )
            spawned = runtime.tools.call(parent, 'spawn_child_process', {'goal': 'read one file', 'inherit_read_files': ['README.md']})
            assert spawned.ok, spawned.error
            child = runtime.process.get(spawned.payload['child_pid'])
            allowed = readme_resource
            other = runtime.filesystem.resource_for_path('pyproject.toml')
            assert runtime.capability.check(child.pid, allowed, CapabilityRight.READ)
            assert not runtime.capability.check(child.pid, other, CapabilityRight.READ)
        finally:
            runtime.close()

    def test_exec_process_swaps_image_without_granting_target_image_capabilities(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='become coding agent')
            workspace_capability = runtime.filesystem.grant_workspace(
                pid,
                [CapabilityRight.READ],
                issued_by='test',
            )
            _grant_image_read(runtime, pid, 'coding-agent:v0')
            executed = runtime.tools.call(pid, 'exec_process', {'image': 'coding-agent:v0', 'goal': 'inspect without automatic capability lift', 'preserve_capabilities': False, 'preserve_memory': False})
            assert executed.ok, executed.error
            process = runtime.process.get(pid)
            assert process.image_id == 'coding-agent:v0'
            assert 'read_text_file' in process.tool_table
            assert 'spawn_child_process' in process.tool_table
            read_resource = runtime.filesystem.resource_for_path('README.md')
            assert not runtime.capability.check(pid, read_resource, CapabilityRight.READ)
            persisted_workspace = runtime.store.get_capability(workspace_capability.cap_id)
            assert persisted_workspace is not None and persisted_workspace.revoked
            assert '_agent_libos_exec_rollback_token' not in persisted_workspace.metadata
            assert [handle.oid for handle in process.memory_view.roots] == [process.goal_oid]
            assert 'process.exec' in [record.action for record in runtime.audit.trace()]
        finally:
            runtime.close()

    def test_failed_exec_process_rolls_back_to_previous_process_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_image(
                {
                    'image_id': 'failing-exec:v0',
                    'name': 'failing-exec',
                },
                actor='cli',
            )
            pid = runtime.process.spawn(image='base-agent:v0', goal='stay on base')
            _grant_image_read(runtime, pid, 'failing-exec:v0')
            other = runtime.process.spawn(image='base-agent:v0', goal='unrelated')
            before = runtime.process.get(pid)
            before_tools = dict(before.tool_table)
            original_configure_skills = runtime.image_boot._configure_skills

            def fail_after_unrelated_mutation(
                target_pid: str,
                image_id: str,
                assigned_by: str,
                *,
                publication_id: str | None = None,
                **_boot_context: object,
            ) -> None:
                other_process = runtime.process.get(other)
                runtime.store.patch_process_control(
                    other,
                    {'status_message': 'must survive scoped rollback'},
                    expected_revision=other_process.revision,
                    allowed_statuses={ProcessStatus.RUNNABLE},
                    reason='inject unrelated Host mutation during exec rollback test',
                )
                raise RuntimeError('skill boot failed')

            runtime.image_boot._configure_skills = fail_after_unrelated_mutation

            with pytest.raises(RuntimeError):
                runtime.exec_process(pid, 'failing-exec:v0', goal='should not apply')

            after = runtime.process.get(pid)
            assert after.status == ProcessStatus.RUNNABLE
            assert after.image_id == 'base-agent:v0'
            assert after.goal_oid == before.goal_oid
            assert after.tool_table == before_tools
            assert runtime.process.get(other).status_message == 'must survive scoped rollback'
            runtime.image_boot._configure_skills = original_configure_skills
        finally:
            runtime.close()

    def test_exec_event_failure_rolls_back_image_tools_skills_and_capabilities(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='stay on base')
            _grant_image_read(runtime, pid, 'coding-agent:v0')
            workspace_capability = runtime.filesystem.grant_workspace(
                pid,
                [CapabilityRight.READ],
                issued_by='test',
            )
            before = runtime.process.get(pid)
            before_capabilities = {cap.cap_id for cap in runtime.capability.capabilities_for(pid)}
            original_emit = runtime.events.emit

            def fail_exec_event(event_type, *args, **kwargs):
                if event_type == EventType.PROCESS_EXEC:
                    raise RuntimeError('injected process exec event failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_exec_event)
            with pytest.raises(RuntimeError, match='injected process exec event failure'):
                runtime.exec_process(
                    pid,
                    'coding-agent:v0',
                    goal='must roll back',
                    preserve_capabilities=False,
                    preserve_memory=False,
                )

            after = runtime.process.get(pid)
            assert after.image_id == before.image_id
            assert after.goal_oid == before.goal_oid
            assert after.tool_table == before.tool_table
            assert after.loaded_skills == before.loaded_skills
            assert {cap.cap_id for cap in runtime.capability.capabilities_for(pid)} == before_capabilities
            restored_workspace = runtime.store.get_capability(workspace_capability.cap_id)
            assert restored_workspace is not None and restored_workspace.active
            assert runtime.capability.check(
                pid,
                runtime.filesystem.workspace_resource(),
                CapabilityRight.READ,
            )
        finally:
            runtime.close()

    def test_exit_event_failure_rolls_back_terminal_state_and_parent_wakeup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='parent')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'child')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            original_emit = runtime.events.emit

            def fail_exit_event(event_type, *args, **kwargs):
                if event_type == EventType.PROCESS_EXITED:
                    raise RuntimeError('injected process exit event failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_exit_event)
            with pytest.raises(RuntimeError, match='injected process exit event failure'):
                runtime.process.exit(child)

            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status == ProcessStatus.WAITING_EVENT

            monkeypatch.setattr(runtime.events, 'emit', original_emit)
            runtime.process.exit(child)
            assert runtime.process.get(child).status == ProcessStatus.EXITED
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_process_exit_generated_result_rolls_back_with_terminal_transition(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='wait for atomic exit')
            _grant_process_spawn(runtime, parent)
            child = runtime.spawn_child_process(parent, 'publish one atomic final result')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            payload = {'sentinel': 'PROCESS_EXIT_ATOMIC_RESULT_SENTINEL'}
            original_emit = runtime.events.emit

            def fail_exit_event(event_type: EventType, *args: Any, **kwargs: Any) -> Any:
                if event_type == EventType.PROCESS_EXITED:
                    raise RuntimeError('injected generated-result exit failure')
                return original_emit(event_type, *args, **kwargs)

            monkeypatch.setattr(runtime.events, 'emit', fail_exit_event)
            failed = runtime.tools.call(child, 'process_exit', {'payload': payload})

            assert not failed.ok
            assert runtime.process.get(child).status == ProcessStatus.RUNNABLE
            assert runtime.process.get(parent).status == ProcessStatus.WAITING_EVENT
            assert [
                obj.oid
                for obj in runtime.store.list_objects()
                if obj.type == ObjectType.SUMMARY and obj.payload == payload
            ] == []

            monkeypatch.setattr(runtime.events, 'emit', original_emit)
            exited = runtime.tools.call(child, 'process_exit', {'payload': payload})

            assert exited.ok, exited.error
            result_oid = exited.payload['result_oid']
            result = runtime.store.get_object(result_oid)
            assert result is not None
            assert result.payload == payload
            assert result.provenance.created_from_action == 'process.exit'
            assert runtime.process.get(child).status == ProcessStatus.EXITED
            assert runtime.process.get(parent).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_merge_child_memory_tool_adds_child_view_objects_to_parent(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='merge child')
            child = runtime.process.fork(parent, goal='produce result')
            created = runtime.tools.call(child, 'create_memory_object', {'name': 'child.result', 'type': 'summary', 'payload': {'merged': True}})
            assert created.ok, created.error
            result_oid = created.payload['oid']
            runtime.tools.call(child, 'process_exit', {'result_oid': result_oid})
            merged = runtime.tools.call(parent, 'merge_child_memory', {'child_pid': child})
            assert merged.ok, merged.error
            assert result_oid in merged.payload['merged_oids']
            parent_view = runtime.process.get(parent).memory_view
            assert parent_view is not None
            assert parent_view is not None
            assert result_oid in [handle.oid for handle in parent_view.roots]
        finally:
            runtime.close()

    def test_merge_after_child_exit_preserves_non_result_child_created_memory_until_parent_exit(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='merge child scratch')
            child = runtime.process.fork(parent, goal='produce scratch and result')
            scratch = runtime.tools.call(
                child,
                'create_memory_object',
                {'name': 'child.scratch', 'type': 'evidence', 'payload': {'scratch': True}},
            )
            result = runtime.tools.call(
                child,
                'create_memory_object',
                {'name': 'child.final', 'type': 'summary', 'payload': {'result': True}},
            )
            assert scratch.ok, scratch.error
            assert result.ok, result.error
            scratch_oid = scratch.payload['oid']
            result_oid = result.payload['oid']

            exited = runtime.tools.call(child, 'process_exit', {'result_oid': result_oid})
            assert exited.ok, exited.error
            assert runtime.store.get_object(scratch_oid) is not None

            merged = runtime.tools.call(parent, 'merge_child_memory', {'child_pid': child})

            assert merged.ok, merged.error
            assert scratch_oid in merged.payload['merged_oids']
            assert result_oid in merged.payload['merged_oids']
            scratch_obj = runtime.store.get_object(scratch_oid)
            assert scratch_obj.owner_kind == ObjectOwnerKind.PROCESS
            assert scratch_obj.owner_id == parent
            runtime.process.exit(parent)
            assert runtime.store.get_object(scratch_oid) is None
            assert runtime.store.get_object(result_oid) is None
        finally:
            runtime.close()

    def test_merge_child_memory_rolls_back_view_ownership_release_and_audit_together(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='merge rollback parent')
            child = runtime.process.fork(parent, goal='merge rollback child')
            adopted = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'adopt': True},
                name='child.merge.rollback.adopt',
            )
            released = runtime.memory.create_object(
                child,
                ObjectType.EVIDENCE,
                {'release': True},
                name='child.merge.rollback.release',
            )
            runtime.capability.revoke_resource_trusted(
                f'object:{released.oid}',
                revoked_by='test',
                reason='force merge skip',
            )
            runtime.process.exit(child, result=adopted)

            before_parent = runtime.process.get(parent)
            assert before_parent.memory_view is not None
            before_parent_roots = [handle.oid for handle in before_parent.memory_view.roots]
            before_adopted = runtime.store.get_object(adopted.oid)
            before_released = runtime.store.get_object(released.oid)
            assert before_adopted is not None
            assert before_released is not None
            before_parent_caps = {
                cap.cap_id for cap in runtime.store.list_capabilities(subject=parent)
            }
            before_audit_ids = {record.record_id for record in runtime.audit.trace()}
            original_record = runtime.audit.record

            def fail_final_merge_audit(*args: Any, **kwargs: Any):
                if kwargs.get('action') == 'process.merge_child_memory':
                    raise RuntimeError('injected child merge audit failure')
                return original_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_final_merge_audit)

            with pytest.raises(RuntimeError, match='injected child merge audit failure'):
                runtime.process.merge_child_memory(parent, child)

            after_parent = runtime.process.get(parent)
            assert after_parent.memory_view is not None
            assert [handle.oid for handle in after_parent.memory_view.roots] == before_parent_roots
            after_adopted = runtime.store.get_object(adopted.oid)
            after_released = runtime.store.get_object(released.oid)
            assert after_adopted is not None
            assert after_released is not None
            assert (after_adopted.owner_kind, after_adopted.owner_id, after_adopted.version) == (
                before_adopted.owner_kind,
                before_adopted.owner_id,
                before_adopted.version,
            )
            assert (after_released.owner_kind, after_released.owner_id, after_released.version) == (
                before_released.owner_kind,
                before_released.owner_id,
                before_released.version,
            )
            assert {
                cap.cap_id for cap in runtime.store.list_capabilities(subject=parent)
            } == before_parent_caps
            assert {record.record_id for record in runtime.audit.trace()} == before_audit_ids
        finally:
            runtime.close()

    def test_terminal_parent_cannot_adopt_terminal_child_memory(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='terminal merge parent')
            child = runtime.process.fork(parent, goal='terminal merge child')
            child_object = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'terminal': True},
                name='child.terminal.parent.merge',
            )
            runtime.process.exit(child, result=child_object)
            monkeypatch.setattr(
                runtime.process,
                '_complete_terminal_cleanup',
                lambda *_args, **_kwargs: {},
            )
            runtime.process.exit(parent)
            assert runtime.process.get(parent).status == ProcessStatus.EXITED

            with pytest.raises(ProcessError, match='terminated process'):
                runtime.process.merge_child_memory(parent, child)

            retained = runtime.store.get_object(child_object.oid)
            assert retained is not None
            assert retained.owner_kind == ObjectOwnerKind.PROCESS
            assert retained.owner_id == child
            parent_view = runtime.process.get(parent).memory_view
            assert parent_view is not None
            assert child_object.oid not in {handle.oid for handle in parent_view.roots}
            runtime.process.retry_terminal_cleanup(parent)
        finally:
            runtime.close()

    def test_merge_child_memory_replay_is_a_durable_noop(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='idempotent merge parent')
            child = runtime.process.fork(parent, goal='idempotent merge child')
            child_object = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'once': True},
                name='child.merge.once',
            )
            runtime.process.exit(child, result=child_object)

            first = runtime.process.merge_child_memory(parent, child)
            second = runtime.process.merge_child_memory(parent, child)

            assert child_object.oid in first.merged_oids
            assert second == type(second)(merged_oids=[], skipped_oids=[])
            active_parent_caps = [
                cap
                for cap in runtime.store.list_capabilities(subject=parent)
                if cap.resource == f'object:{child_object.oid}' and cap.active and not cap.revoked
            ]
            assert len(active_parent_caps) == 1
            assert sum(
                record.action == 'process.merge_child_memory'
                for record in runtime.audit.trace()
            ) == 1
        finally:
            runtime.close()

    def test_concurrent_child_memory_merge_has_one_durable_winner(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='concurrent merge parent')
            child = runtime.process.fork(parent, goal='concurrent merge child')
            child_object = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'once': 'concurrent'},
                name='child.merge.concurrent.once',
            )
            runtime.process.exit(child, result=child_object)
            barrier = threading.Barrier(3)
            results: list[MergeResult] = []
            errors: list[BaseException] = []

            def merge() -> None:
                try:
                    barrier.wait(timeout=5)
                    results.append(runtime.process.merge_child_memory(parent, child))
                except BaseException as exc:
                    errors.append(exc)

            workers = [threading.Thread(target=merge, daemon=True) for _ in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait(timeout=5)
            for worker in workers:
                worker.join(timeout=5)

            assert all(not worker.is_alive() for worker in workers)
            assert errors == []
            assert sum(not result.merged_oids for result in results) == 1
            assert sum(
                child_object.oid in result.merged_oids for result in results
            ) == 1
            assert sum(
                record.action == 'process.merge_child_memory'
                for record in runtime.audit.trace()
            ) == 1
            assert len(
                [
                    cap
                    for cap in runtime.store.list_capabilities(subject=parent)
                    if cap.resource == f'object:{child_object.oid}'
                    and cap.active
                    and not cap.revoked
                ]
            ) == 1
        finally:
            runtime.close()

    def test_parent_exit_releases_unmerged_terminal_child_memory(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='discard child memory')
            child = runtime.process.fork(parent, goal='produce unmerged scratch')
            scratch = runtime.tools.call(
                child,
                'create_memory_object',
                {'name': 'child.unmerged', 'type': 'evidence', 'payload': {'temporary': True}},
            )
            assert scratch.ok, scratch.error
            scratch_oid = scratch.payload['oid']
            runtime.tools.call(child, 'process_exit', {'payload': {'done': True}})
            assert runtime.store.get_object(scratch_oid) is not None

            runtime.process.exit(parent)

            assert runtime.store.get_object(scratch_oid) is None
        finally:
            runtime.close()

    def test_fork_root_oids_do_not_upgrade_read_only_objects_to_materialize(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            parent = runtime.process.spawn(image='base-agent:v0', goal='fork read-only root')
            secret = runtime.memory.create_object(
                pid=owner,
                object_type=ObjectType.EVIDENCE,
                payload={'secret': 'child must not materialize this'},
                name='read.only.secret',
            )
            runtime.capability.grant(parent, f'object:{secret.oid}', [CapabilityRight.READ], issued_by='test')
            _grant_process_spawn(runtime, parent)

            forked = runtime.tools.call(
                parent,
                'fork_child_process',
                {'goal': 'child', 'include_parent_roots': False, 'root_oids': [secret.oid]},
            )

            assert forked.ok, forked.error
            child = runtime.process.get(forked.payload['child_pid'])
            root = next(handle for handle in child.memory_view.roots if handle.oid == secret.oid)
            assert root.rights == {'read'}
            context = runtime.memory.materialize_context(child.pid, child.memory_view)
            assert secret.oid in context.omitted_objects
            assert 'child must not materialize this' not in context.text
        finally:
            runtime.close()

    def test_process_exit_result_oid_requires_object_read_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            process = runtime.process.spawn(image='base-agent:v0', goal='try unauthorized result oid')
            secret = runtime.memory.create_object(
                pid=owner,
                object_type=ObjectType.EVIDENCE,
                payload={'secret': 'not a result'},
                name='private.result',
            )

            exited = runtime.tools.call(process, 'process_exit', {'result_oid': secret.oid})

            assert not exited.ok
            assert_public_error_message(
                exited.error,
                code='permission_denied',
                error_type='CapabilityDenied',
                forbidden=('lacks read', secret.oid),
            )
            assert runtime.process.get(process).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    def test_fork_does_not_resurrect_revoked_image_default_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='coding-agent:v0', goal='fork after revoke')
            _grant_process_spawn(runtime, parent)
            path = 'README.md'
            for cap in list(runtime.capability.capabilities_for(parent)):
                if cap.resource == 'filesystem:workspace:*' and CapabilityRight.READ.value in cap.rights:
                    runtime.capability.revoke(cap.cap_id, revoked_by=parent, reason='revoked before fork')
            forked = runtime.tools.call(parent, 'fork_child_process', {'goal': 'try reading'})
            assert forked.ok, forked.error
            child = forked.payload['child_pid']
            denied = runtime.tools.call(child, 'read_text_file', {'path': path})
            assert not denied.ok
            assert_public_error_message(
                denied.error,
                code='permission_denied',
                error_type='CapabilityDenied',
                forbidden=('lacks read',),
            )
        finally:
            runtime.close()

class ParentChildClient:

    def __init__(self) -> None:
        self.parent_pid: str | None = None
        self.child_pid: str | None = None
        self.parent_step = 0
        self.child_step = 0
        self.parent_completion_payload: dict[str, Any] | None = None
        self.calls = 0

    async def acomplete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        return self.complete_action(messages, tools)

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        del tools
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        is_child = 'return value 42' in rendered
        review = _find_completion_review(messages)
        if review is not None:
            payload = (
                {'value': 42}
                if is_child
                else self.parent_completion_payload
            )
            assert payload is not None
            return self._completion(
                'process_exit',
                _completion_claim(review, payload),
            )
        if is_child and self.child_step == 0:
            self.child_step = 1
            return self._completion('read_process_messages', {})
        if is_child and self.child_step == 1:
            self.child_step = 2
            return self._completion('process_exit', {'payload': {'value': 42}})
        if is_child:
            raise AssertionError('child action plan is complete')
        if self.parent_step == 0:
            self.parent_step = 1
            return self._completion(
                'activate_skill',
                {
                    'skill_id': 'agent-libos-child-processes',
                    'expected_package_sha256': _skill_package_sha256(
                        'agent-libos-child-processes'
                    ),
                },
            )
        if self.parent_step == 1:
            self.parent_step = 2
            return self._completion('fork_child_process', {'goal': 'return value 42', 'mode': 'worker', 'include_parent_roots': False})
        if self.parent_step == 2:
            self.child_pid = _last_tool_result(messages, 'fork_child_process')['child_pid']
            self.parent_step = 3
            return self._completion('wait_child_process', {'child_pid': self.child_pid})
        if self.parent_step == 3:
            wait_result = _last_tool_result(messages, 'wait_child_process')
            self.parent_step = 4
            self.parent_completion_payload = {
                'waited': wait_result['ready'],
                'child_pid': wait_result['child_pid'],
            }
            return self._completion(
                'process_exit',
                {'payload': self.parent_completion_payload},
            )
        raise AssertionError('parent action plan is complete')

    def _completion(self, name: str, args: dict[str, Any]) -> LLMCompletion:
        return LLMCompletion(content='', tool_calls=[{'id': f'child_process_{self.calls}', 'name': name, 'arguments': json.dumps(args)}])


def _find_completion_review(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            return _find_completion_review(json.loads(value))
        except json.JSONDecodeError:
            for line in value.splitlines():
                candidate = line.strip()
                if not candidate.startswith('{'):
                    continue
                try:
                    found = _find_completion_review(json.loads(candidate))
                except json.JSONDecodeError:
                    continue
                if found is not None:
                    return found
            return None
    if isinstance(value, list):
        for item in value:
            found = _find_completion_review(item)
            if found is not None:
                return found
        return None
    if not isinstance(value, dict):
        return None
    review = value.get('completion_review')
    if isinstance(review, dict) and isinstance(review.get('review_token'), str):
        return review
    for item in value.values():
        found = _find_completion_review(item)
        if found is not None:
            return found
    return None


def _completion_claim(
    review: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    available = [
        str(tool)
        for tool in review.get('available_evidence_tools', [])
        if isinstance(tool, str) and tool != 'process_exit'
    ]
    assert available
    requirements = review.get('requirements')
    assert isinstance(requirements, list) and requirements
    checks = []
    for requirement in requirements:
        eligible = (
            requirement.get('eligible_evidence_tools')
            if isinstance(requirement, dict)
            else None
        )
        candidates = [tool for tool in available if not eligible or tool in eligible]
        assert candidates
        checks.append(
            {
                'status': 'completed',
                'evidence_tool_calls': candidates[:1],
                'evidence_summary': 'Verified by the cited successful runtime tool call.',
            }
        )
    return {
        'payload': dict(payload),
        'review_token': review['review_token'],
        'completion_evidence': {
            'acceptance_checks': checks,
            'final_verification': available[:1],
        },
    }


def _last_tool_result(messages: list[dict[str, str]], tool_name: str) -> dict[str, Any]:
    result = last_tool_result(messages, tool_name)
    if result is not None:
        return result
    raise AssertionError(f'no visible result for {tool_name}')

def _action_name(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    action = result.get('action')
    if isinstance(action, dict):
        return action.get('action')
    return None
