from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.models import (
    CapabilityEffect,
    CapabilityRight,
    ChildProcessWait,
    DataFlowContext,
    DataLabels,
    EventType,
    ExitedProcessOutcome,
    HostResumeProcessWait,
    ObjectOwnerKind,
    ObjectRight,
    ObjectType,
    PausedProcessWait,
    ProcessStatus,
    ResourceBudget,
)
from agent_libos.models.exceptions import CapabilityDenied, ProcessError, ProcessWaitRequired, ResourceLimitExceeded
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.checkpoint import (
    ForkCheckpointArgs,
    ForkCheckpointOutput,
    ForkCheckpointTool,
)


class TestCheckpointFork:

    def test_fork_tool_preserves_runtime_commit_status_maps_and_warnings(self) -> None:
        payload = {
            'checkpoint_id': 'ckpt_test',
            'source_pid': 'pid_source',
            'fork_root_pid': 'pid_fork',
            'pid_map': {'pid_source': 'pid_fork'},
            'object_map': {'oid_source': 'oid_fork'},
            'tool_map': {'tool_source': 'tool_fork'},
            'status': 'forked_with_warnings',
            'main_state_committed': True,
            'post_commit_failures': [
                {'phase': 'fork_event_emission', 'error': 'injected failure'},
            ],
        }

        def fork_from_checkpoint(
            actor: str,
            checkpoint_id: str,
            parent_pid: str | None,
        ) -> dict[str, object]:
            assert actor == 'pid_caller'
            assert checkpoint_id == 'ckpt_test'
            assert parent_pid == 'pid_parent'
            return dict(payload)

        checkpoint = SimpleNamespace(fork_from_checkpoint=fork_from_checkpoint)
        result = ForkCheckpointTool().run(
            ForkCheckpointArgs(
                checkpoint_id='ckpt_test',
                parent_pid='pid_parent',
            ),
            ToolContext(
                trace_id='trace_test',
                call_id='call_test',
                pid='pid_caller',
                runtime=SimpleNamespace(checkpoint=checkpoint),
            ),
        )

        projected = result.model_dump()
        assert {key: projected[key] for key in payload} == payload
        assert projected['pid_map_page'] == {
            'count': 1,
            'returned_count': 1,
            'truncated': False,
            'next_cursor': None,
        }
        assert projected['object_map_page']['count'] == 1
        assert projected['tool_map_page']['count'] == 1
        assert projected['post_commit_failures_page']['count'] == 1

    def test_fork_output_additions_keep_legacy_construction_compatible(self) -> None:
        output = ForkCheckpointOutput(
            checkpoint_id='ckpt_legacy',
            source_pid='pid_source',
            fork_root_pid='pid_fork',
            pid_map={'pid_source': 'pid_fork'},
            object_map={},
        )

        assert output.tool_map == {}
        assert output.status == 'forked'
        assert output.main_state_committed is True
        assert output.post_commit_failures == []

    @pytest.mark.parametrize(
        ('sink', 'phase'),
        [
            ('event', 'fork_event_emission'),
            ('audit', 'fork_audit_recording'),
        ],
    )
    def test_fork_reports_event_and_audit_failures_after_main_state_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sink: str,
        phase: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'{sink} failure after fork')
            checkpoint_id = runtime.checkpoint.create(pid, f'before {sink} failure', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            if sink == 'event':
                original_emit = runtime.events.emit

                def fail_fork_event(event_type, *args, **kwargs):
                    if event_type == EventType.PROCESS_FORKED:
                        raise RuntimeError('injected fork event failure')
                    return original_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', fail_fork_event)
            else:
                original_record = runtime.audit.record

                def fail_fork_audit(*args, **kwargs):
                    if kwargs.get('action') == 'checkpoint.fork':
                        raise RuntimeError('injected fork audit failure')
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', fail_fork_audit)

            result = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert result['status'] == 'forked_with_warnings'
            assert result['main_state_committed'] is True
            assert phase in [failure['phase'] for failure in result['post_commit_failures']]
            assert runtime.store.get_process(result['fork_root_pid']) is not None
        finally:
            runtime.close()

    def test_fork_from_checkpoint_remaps_process_namespace_objects_and_capabilities(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork')
            original = runtime.memory.create_object(pid, ObjectType.SUMMARY, {'value': 7}, name='state')
            runtime.capability.grant(pid, 'filesystem:workspace:README.md', [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'fork point', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_pid = forked['fork_root_pid']
            fork_obj = runtime.memory.get_object_by_name(fork_pid, 'state')
            assert fork_pid != pid
            assert fork_obj.oid != original.oid
            assert fork_obj.namespace == runtime.memory.process_namespace(fork_pid)
            assert fork_obj.payload == {'value': 7}
            assert runtime.capability.check(fork_pid, 'filesystem:workspace:README.md', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_fork_from_checkpoint_remaps_observed_message_label_carrier(self) -> None:
        runtime = Runtime.open('local')
        try:
            sender = runtime.process.spawn(image='base-agent:v0', goal='classified sender')
            receiver = runtime.process.spawn_child(sender, 'classified receiver')
            message = runtime.messages.send_from_process(
                sender,
                receiver,
                body='classified checkpoint payload',
                source_context=DataFlowContext(
                    labels=DataLabels(sensitivity='secret'),
                ),
            )
            original_carriers = runtime.messages.observe_labels(receiver, [message])
            assert len(original_carriers) == 1
            original_carrier = original_carriers[0]

            checkpoint_id = runtime.checkpoint.create(
                receiver,
                'fork observed classified message',
                actor=receiver,
            )
            runtime.capability.grant(
                receiver,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            forked = runtime.checkpoint.fork_from_checkpoint(receiver, checkpoint_id)
            fork_pid = forked['fork_root_pid']
            fork_messages = [
                item
                for item in runtime.messages.unread(fork_pid)
                if item.body == message.body
            ]
            assert len(fork_messages) == 1
            fork_carrier = forked['object_map'][original_carrier]

            assert runtime.messages.observe_labels(fork_pid, fork_messages) == [fork_carrier]
            persisted_fork_message = runtime.store.get_process_message(
                fork_messages[0].message_id
            )
            assert persisted_fork_message is not None
            assert persisted_fork_message.metadata['label_carrier_oid'] == fork_carrier
            cloned_carrier = runtime.store.get_object(fork_carrier)
            assert cloned_carrier is not None
            assert cloned_carrier.metadata.sensitivity == 'secret'
            assert cloned_carrier.metadata.tenant is None

            persisted_original = runtime.store.get_process_message(message.message_id)
            assert persisted_original is not None
            assert persisted_original.metadata['label_carrier_oid'] == original_carrier
            assert runtime.messages.observe_labels(receiver, [persisted_original]) == [
                original_carrier
            ]
        finally:
            runtime.close()

    def test_fork_from_checkpoint_does_not_clone_finite_use_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork finite authority')
            resource = 'test:one-shot-fork-authority'
            finite = runtime.capability.grant_once(
                pid,
                resource,
                [CapabilityRight.READ],
                issued_by='test',
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'finite authority fork point', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_pid = forked['fork_root_pid']

            assert runtime.store.get_capability(finite.cap_id).uses_remaining == 1
            assert not runtime.capability.check(fork_pid, resource, CapabilityRight.READ)
            assert resource not in [cap.resource for cap in runtime.capability.list_subject(fork_pid)]
        finally:
            runtime.close()

    def test_fork_from_checkpoint_remaps_object_task_result_owner_to_forked_process_result(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork object task result')
            result = runtime.memory.create_object(pid, ObjectType.SUMMARY, {'value': 7}, name='task-result')
            runtime.memory.transfer_owner(
                ObjectOwnerKind.PROCESS,
                pid,
                ObjectOwnerKind.OBJECT_TASK,
                'otask_original',
                [result.oid],
                actor='test',
                reason='simulate_object_task_result',
            )
            creator_handle = runtime.capability.handle_for_object(
                pid,
                result.oid,
                [ObjectRight.READ.value, ObjectRight.MATERIALIZE.value, ObjectRight.LINK.value],
                issued_by='object_task:otask_original',
            )
            runtime._add_handle_to_process_view(pid, creator_handle)
            checkpoint_id = runtime.checkpoint.create(pid, 'fork object task result', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')

            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            forked_obj = runtime.store.get_object(forked['object_map'][result.oid])
            assert forked_obj is not None
            assert forked_obj.owner_kind == ObjectOwnerKind.PROCESS_RESULT
            assert forked_obj.owner_id == forked['pid_map'][pid]
        finally:
            runtime.close()

    def test_fork_from_checkpoint_does_not_resurrect_revoked_capability(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork revoked capability')
            resource = runtime.filesystem.resource_for_path('secret.txt')
            cap = runtime.capability.grant(pid, resource, [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'before revoke', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.capability.revoke(cap.cap_id, revoked_by=pid, reason='holder gave up authority')
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_root = forked['fork_root_pid']
            assert not runtime.capability.check(pid, resource, CapabilityRight.READ)
            assert not runtime.capability.check(fork_root, resource, CapabilityRight.READ)
            assert resource not in [capability.resource for capability in runtime.capability.list_subject(fork_root)]
        finally:
            runtime.close()

    def test_fork_from_checkpoint_revalidates_capability_after_concurrent_revoke(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        filter_reached = threading.Event()
        revoke_done = threading.Event()
        errors: list[BaseException] = []
        forked_results: list[dict[str, object]] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork concurrent revoke')
            cap = runtime.capability.grant(pid, 'test:fork-race', [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'before fork revoke race', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            original_filter = runtime.checkpoint._fork_capability_rows

            def pause_after_filter(rows):
                filtered = original_filter(rows)
                filter_reached.set()
                assert revoke_done.wait(timeout=2)
                return filtered

            monkeypatch.setattr(runtime.checkpoint, '_fork_capability_rows', pause_after_filter)

            def fork() -> None:
                try:
                    forked_results.append(runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id))
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            fork_thread = threading.Thread(target=fork)
            fork_thread.start()
            assert filter_reached.wait(timeout=2)
            runtime.capability.revoke(cap.cap_id, revoked_by=pid, reason='concurrent fork revoke wins')
            revoke_done.set()
            fork_thread.join(timeout=3)

            assert not fork_thread.is_alive()
            assert errors == []
            fork_pid = str(forked_results[0]['fork_root_pid'])
            assert not runtime.capability.check(fork_pid, cap.resource, CapabilityRight.READ)
            assert cap.resource not in [item.resource for item in runtime.capability.list_subject(fork_pid)]
        finally:
            runtime.close()

    def test_fork_revalidates_actor_checkpoint_execute_inside_publish_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='fork actor revoke race')
            checkpoint_id = runtime.checkpoint.create(actor, 'actor authority race', actor=actor)
            execute = runtime.capability.grant(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}
            original_restore_jit = runtime.checkpoint._restore_jit_sources

            def revoke_after_preflight(remapped):
                original_restore_jit(remapped)
                runtime.capability.revoke(
                    execute.cap_id,
                    revoked_by=actor,
                    reason='revoke checkpoint execute after fork preflight',
                )

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', revoke_after_preflight)

            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            runtime.close()

    def test_fork_consumes_one_shot_checkpoint_execute_only_on_publish(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='fork with one-shot execute')
            checkpoint_id = runtime.checkpoint.create(actor, 'one-shot execute fork', actor=actor)
            execute = runtime.capability.grant_once(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            original_insert = runtime.checkpoint._insert_row

            def fail_first_process_publish(cur, table, row):
                if table == 'processes':
                    raise RuntimeError('injected fork publish failure')
                return original_insert(cur, table, row)

            monkeypatch.setattr(runtime.checkpoint, '_insert_row', fail_first_process_publish)

            with pytest.raises(RuntimeError, match='injected fork publish failure'):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert runtime.store.get_capability(execute.cap_id).uses_remaining == 1
            monkeypatch.setattr(runtime.checkpoint, '_insert_row', original_insert)

            forked = runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert runtime.store.get_process(forked['fork_root_pid']) is not None
            assert runtime.store.get_capability(execute.cap_id).uses_remaining == 0
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)
        finally:
            runtime.close()

    def test_fork_from_checkpoint_does_not_clone_external_ref_by_default(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='external ref fork')
            external_owner = runtime.process.spawn(image='base-agent:v0', goal='borrowed external ref owner')
            external = runtime.memory.create_object(
                pid,
                ObjectType.EXTERNAL_REF,
                {'provider': 'remote', 'handle': 'opaque'},
                name='external.ref',
            )
            borrowed_external = runtime.memory.create_object(
                external_owner,
                ObjectType.EXTERNAL_REF,
                {'provider': 'remote', 'handle': 'borrowed'},
                name='borrowed.external.ref',
            )
            borrowed_external_handle = runtime.capability.handle_for_object(
                pid,
                borrowed_external.oid,
                [CapabilityRight.READ],
                issued_by='test.borrowed.external',
            )
            runtime._add_handle_to_process_view(pid, external)
            runtime._add_handle_to_process_view(pid, borrowed_external_handle)
            checkpoint_id = runtime.checkpoint.create(pid, 'external ref checkpoint', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')

            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_pid = forked['fork_root_pid']

            assert external.oid not in forked['object_map']
            assert all(
                obj.type != ObjectType.EXTERNAL_REF
                for obj in runtime.store.list_objects_owned_by(ObjectOwnerKind.PROCESS, fork_pid)
            )
            assert not runtime.capability.check(fork_pid, f'object:{external.oid}', CapabilityRight.READ)
            assert not runtime.capability.check(
                fork_pid,
                f'object:{borrowed_external.oid}',
                CapabilityRight.READ,
            )
            fork_roots = {handle.oid for handle in runtime.process.get(fork_pid).memory_view.roots}
            assert external.oid not in fork_roots
            assert borrowed_external.oid not in fork_roots
        finally:
            runtime.close()

    def test_fork_from_checkpoint_respects_post_checkpoint_deny_policy(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork denied capability')
            secret = runtime.filesystem.resource_for_path('secret.txt')
            runtime.capability.grant(pid, 'filesystem:workspace:*', [CapabilityRight.READ], issued_by='test')
            checkpoint_id = runtime.checkpoint.create(pid, 'before deny policy', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.capability.issue_trusted(pid, secret, [CapabilityRight.READ], issued_by='test', effect=CapabilityEffect.DENY)
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_root = forked['fork_root_pid']
            assert not runtime.capability.check(pid, secret, CapabilityRight.READ)
            assert not runtime.capability.check(fork_root, secret, CapabilityRight.READ)
            assert not runtime.capability.check(fork_root, 'filesystem:workspace:public.txt', CapabilityRight.READ)
        finally:
            runtime.close()

    def test_fork_from_checkpoint_normalizes_waiting_process_state(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='waiting parent')
            runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            child = runtime.spawn_child_process(parent, 'unfinished child')
            with pytest.raises(ProcessWaitRequired):
                runtime.process.wait(parent, child)
            waiting = runtime.process.get(parent)
            assert waiting.status == ProcessStatus.WAITING_EVENT
            assert waiting.wait_state == ChildProcessWait(child_pid=child)
            checkpoint_id = runtime.checkpoint.create(parent, 'waiting fork point', actor=parent)
            runtime.capability.grant(parent, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(parent, checkpoint_id)
            fork_root = runtime.process.get(forked['fork_root_pid'])
            assert fork_root.status == ProcessStatus.RUNNABLE
            assert fork_root.status_message is None
            assert fork_root.wait_state is None
            assert fork_root.outcome is None
            assert fork_root.state_generation == 0
        finally:
            runtime.close()

    def test_fork_remaps_terminal_child_outcome_to_cloned_result(self) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='fork terminal child')
            runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            child = runtime.spawn_child_process(parent, 'terminal child')
            result = runtime.memory.create_object(
                child,
                ObjectType.SUMMARY,
                {'answer': 42},
                name='terminal-result',
            )
            runtime.process.exit(child, result)
            source_child = runtime.process.get(child)
            assert source_child.outcome == ExitedProcessOutcome(result_oid=result.oid)

            checkpoint_id = runtime.checkpoint.create(parent, 'terminal child outcome', actor=parent)
            runtime.capability.grant(
                parent,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            forked = runtime.checkpoint.fork_from_checkpoint(parent, checkpoint_id)

            fork_child = runtime.process.get(forked['pid_map'][child])
            cloned_result_oid = forked['object_map'][result.oid]
            assert fork_child.outcome == ExitedProcessOutcome(result_oid=cloned_result_oid)
            assert fork_child.status_message == f'result_oid:{cloned_result_oid}'
            assert cloned_result_oid != result.oid
            assert runtime.store.get_object(cloned_result_oid) is not None
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('pause_method', 'expected_wait_type'),
        [
            ('pause', PausedProcessWait),
            ('pause_for_host_resume', HostResumeProcessWait),
        ],
    )
    def test_fork_remaps_paused_reason_to_cloned_object(
        self,
        pause_method: str,
        expected_wait_type: type[PausedProcessWait] | type[HostResumeProcessWait],
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal=f'fork {pause_method}')
            getattr(runtime.process, pause_method)(pid, 'carry this reason into the fork')
            source = runtime.process.get(pid)
            assert isinstance(source.wait_state, expected_wait_type)
            reason_oid = source.wait_state.reason_oid
            assert reason_oid is not None

            checkpoint_id = runtime.checkpoint.create(pid, f'{pause_method} state', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            fork_process = runtime.process.get(forked['fork_root_pid'])
            cloned_reason_oid = forked['object_map'][reason_oid]
            assert isinstance(fork_process.wait_state, expected_wait_type)
            assert fork_process.wait_state.reason_oid == cloned_reason_oid
            assert cloned_reason_oid != reason_oid
            assert runtime.store.get_object(cloned_reason_oid) is not None
        finally:
            runtime.close()

    def test_fork_from_checkpoint_rolls_back_rows_and_payloads_when_insert_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork rollback')
            original = runtime.memory.create_object(pid, ObjectType.SUMMARY, {'value': 7}, name='state')
            checkpoint_id = runtime.checkpoint.create(pid, 'fork rollback point', actor=pid)
            runtime.capability.grant(pid, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            before_pids = {process.pid for process in runtime.process.list()}
            original_insert = runtime.checkpoint._insert_row

            def fail_on_process_insert(cur, table, row):
                if table == 'processes':
                    raise RuntimeError('injected fork failure')
                return original_insert(cur, table, row)

            monkeypatch.setattr(runtime.checkpoint, '_insert_row', fail_on_process_insert)
            with pytest.raises(RuntimeError, match='injected fork failure'):
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
            assert runtime.memory.get_object(pid, original).payload == {'value': 7}
        finally:
            runtime.close()

    def test_fork_from_checkpoint_does_not_replace_current_image_without_image_write(self) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-fork-image:v0'
        try:
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-fork-image', system_prompt='snapshot prompt'),
                actor='test',
            )
            source = runtime.process.spawn(image=image_id, goal='checkpoint image source')
            checkpoint_id = runtime.checkpoint.create(source, 'image fork point', actor=source)
            actor = runtime.process.spawn(image='base-agent:v0', goal='checkpoint executor')
            runtime.capability.grant(actor, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-fork-image', system_prompt='current prompt'),
                actor='test',
                replace=True,
            )

            forked = runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert runtime.process.get(forked['fork_root_pid']).image_id == image_id
            assert not runtime.capability.check(actor, runtime.image_registry.resource_for(image_id), CapabilityRight.WRITE)
            assert runtime.get_image(image_id).system_prompt == 'current prompt'
            stored = runtime.store.get_image(image_id)
            assert stored is not None
            assert stored[0].system_prompt == 'current prompt'
        finally:
            runtime.close()

    def test_fork_from_checkpoint_requires_image_write_to_restore_missing_image(self) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-fork-missing-image:v0'
        try:
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-fork-missing-image', system_prompt='snapshot prompt'),
                actor='test',
            )
            source = runtime.process.spawn(image=image_id, goal='checkpoint missing image source')
            checkpoint_id = runtime.checkpoint.create(source, 'missing image fork point', actor=source)
            actor = runtime.process.spawn(image='base-agent:v0', goal='checkpoint executor')
            runtime.capability.grant(actor, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.images.pop(image_id)
            runtime.store.delete_image(image_id)
            before_pids = {process.pid for process in runtime.process.list()}

            with pytest.raises(CapabilityDenied, match=f'image:{image_id}'):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
            runtime.image_registry.grant_register(actor, image_id, issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert runtime.process.get(forked['fork_root_pid']).image_id == image_id
            assert runtime.get_image(image_id).system_prompt == 'snapshot prompt'
            stored = runtime.store.get_image(image_id)
            assert stored is not None
            assert stored[0].system_prompt == 'snapshot prompt'
        finally:
            runtime.close()

    def test_fork_revalidates_missing_snapshot_image_write_inside_publish_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-fork-image-write-race:v0'
        try:
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-fork-image-write-race'),
                actor='test',
            )
            source = runtime.process.spawn(image=image_id, goal='snapshot image source')
            checkpoint_id = runtime.checkpoint.create(source, 'image write race', actor=source)
            actor = runtime.process.spawn(image='base-agent:v0', goal='image restore actor')
            runtime.capability.grant(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            image_write = runtime.image_registry.grant_register(actor, image_id, issued_by='test')
            runtime.images.pop(image_id)
            runtime.store.delete_image(image_id)
            before_pids = {process.pid for process in runtime.process.list()}
            original_restore_jit = runtime.checkpoint._restore_jit_sources

            def revoke_after_preflight(remapped):
                original_restore_jit(remapped)
                runtime.capability.revoke(
                    image_write.cap_id,
                    revoked_by=actor,
                    reason='revoke image write after fork preflight',
                )

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', revoke_after_preflight)

            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None
        finally:
            runtime.close()

    def test_checkpoint_fork_parent_attachment_requires_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            other = runtime.process.spawn(image='base-agent:v0', goal='other')
            checkpoint_id = runtime.checkpoint.create(owner, 'fork parent boundary', actor=owner)
            runtime.capability.grant(owner, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id, parent_pid=other)
            runtime.capability.grant(owner, runtime.checkpoint.process_resource(other), [CapabilityRight.ADMIN], issued_by='test')
            forked = runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id, parent_pid=other)
            assert runtime.process.get(forked['fork_root_pid']).parent_pid == other
        finally:
            runtime.close()

    def test_fork_revalidates_parent_admin_inside_publish_transaction(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='fork parent admin race')
            parent = runtime.process.spawn(image='base-agent:v0', goal='fork target parent')
            checkpoint_id = runtime.checkpoint.create(actor, 'parent admin race', actor=actor)
            runtime.capability.grant(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            parent_admin = runtime.capability.grant(
                actor,
                runtime.checkpoint.process_resource(parent),
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}
            original_restore_jit = runtime.checkpoint._restore_jit_sources

            def revoke_after_preflight(remapped):
                original_restore_jit(remapped)
                runtime.capability.revoke(
                    parent_admin.cap_id,
                    revoked_by=actor,
                    reason='revoke parent admin after fork preflight',
                )

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', revoke_after_preflight)

            with pytest.raises(CapabilityDenied):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id, parent_pid=parent)

            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            runtime.close()

    def test_fork_rejects_parent_that_becomes_terminal_after_preflight(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            actor = runtime.process.spawn(image='base-agent:v0', goal='fork terminal parent race')
            parent = runtime.process.spawn(image='base-agent:v0', goal='fork target parent')
            checkpoint_id = runtime.checkpoint.create(actor, 'terminal parent race', actor=actor)
            runtime.capability.grant(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            runtime.capability.grant(
                actor,
                runtime.checkpoint.process_resource(parent),
                [CapabilityRight.ADMIN],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}
            original_restore_jit = runtime.checkpoint._restore_jit_sources

            def exit_parent_after_preflight(remapped):
                original_restore_jit(remapped)
                runtime.process.exit(parent, message='terminal before fork publish')

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', exit_parent_after_preflight)

            with pytest.raises(ProcessError, match='terminal process'):
                runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id, parent_pid=parent)

            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            runtime.close()

    def test_checkpoint_fork_child_root_attaches_to_requested_parent(self) -> None:
        runtime = Runtime.open('local')
        try:
            source_parent = runtime.process.spawn(image='base-agent:v0', goal='source parent')
            runtime.capability.grant(source_parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
            source_child = runtime.spawn_child_process(source_parent, 'source child')
            target_parent = runtime.process.spawn(image='base-agent:v0', goal='target parent')
            checkpoint_id = runtime.checkpoint.create(source_child, 'child root fork', actor=source_child)
            runtime.capability.grant(source_child, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.capability.grant(source_child, runtime.checkpoint.process_resource(target_parent), [CapabilityRight.ADMIN], issued_by='test')

            forked = runtime.checkpoint.fork_from_checkpoint(source_child, checkpoint_id, parent_pid=target_parent)

            assert runtime.process.get(forked['fork_root_pid']).parent_pid == target_parent
        finally:
            runtime.close()

    def test_checkpoint_fork_parent_child_budget_exhaustion_rolls_back(self) -> None:
        runtime = Runtime.open('local')
        try:
            owner = runtime.process.spawn(image='base-agent:v0', goal='owner')
            exhausted_parent = runtime.process.spawn(
                image='base-agent:v0',
                goal='exhausted parent',
                resource_budget=ResourceBudget(max_child_processes=0),
            )
            checkpoint_id = runtime.checkpoint.create(owner, 'budgeted fork', actor=owner)
            runtime.capability.grant(owner, f'checkpoint:{checkpoint_id}', [CapabilityRight.EXECUTE], issued_by='test')
            runtime.capability.grant(owner, runtime.checkpoint.process_resource(exhausted_parent), [CapabilityRight.ADMIN], issued_by='test')
            before_pids = {process.pid for process in runtime.process.list()}

            with pytest.raises(ResourceLimitExceeded):
                runtime.checkpoint.fork_from_checkpoint(owner, checkpoint_id, parent_pid=exhausted_parent)

            assert {process.pid for process in runtime.process.list()} == before_pids
            assert runtime.process.get(exhausted_parent).resource_usage.child_processes == 0
        finally:
            runtime.close()
