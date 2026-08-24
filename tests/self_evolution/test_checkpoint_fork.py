from __future__ import annotations

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_libos import AgentImage, Runtime, TaskRunSpecV1
from agent_libos.config import DEFAULT_CONFIG
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
    OperationOutcome,
    OperationState,
    PausedProcessWait,
    ProcessStatus,
    ResourceBudget,
    ToolCandidate,
    ToolCandidateStatus,
    ToolSpec,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    ProcessError,
    ProcessWaitRequired,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.checkpoint import (
    ForkCheckpointArgs,
    ForkCheckpointOutput,
    ForkCheckpointTool,
)
from agent_libos.utils.serde import dumps


class _CommitThenRaiseConnection:
    def __init__(self, connection: object) -> None:
        self._connection = connection
        self._fault_pending = True
        self.sql_calls_after_fault: list[str] = []

    def __getattr__(self, name: str):
        return getattr(self._connection, name)

    def commit(self) -> None:
        self._connection.commit()
        if self._fault_pending:
            self._fault_pending = False
            raise RuntimeError('injected checkpoint fork commit acknowledgement loss')

    def execute(self, *args, **kwargs):
        if not self._fault_pending:
            self.sql_calls_after_fault.append('execute')
            raise AssertionError('poisoned checkpoint connection was read')
        return self._connection.execute(*args, **kwargs)

    def cursor(self, *args, **kwargs):
        if not self._fault_pending:
            self.sql_calls_after_fault.append('cursor')
            raise AssertionError('poisoned checkpoint connection was read')
        return self._connection.cursor(*args, **kwargs)


def _fork_operation(runtime: Runtime, pid: str):
    operations = [
        operation
        for operation in runtime.store.list_operations(pid=pid)
        if operation.name == 'checkpoint.fork'
    ]
    assert len(operations) == 1
    return operations[0]


def _assert_only_checkpoint_authorization_audit(
    runtime: Runtime,
    *,
    before_record_ids: set[str],
    actor: str,
    checkpoint_id: str,
    capability_id: str,
) -> None:
    new_records = [
        record
        for record in runtime.store.list_audit()
        if record.record_id not in before_record_ids
    ]
    assert len(new_records) == 1
    decision = new_records[0]
    assert decision.actor == actor
    assert decision.action == 'capability.authorize'
    assert decision.target == f'checkpoint:{checkpoint_id}'
    assert decision.capability_refs == [capability_id]
    assert decision.decision['allowed'] is True
    assert decision.decision['right'] == CapabilityRight.EXECUTE.value


def _insert_tool_candidate(
    runtime: Runtime,
    *,
    candidate_id: str,
    pid: str,
    requested_capabilities: list[dict[str, object]],
) -> str:
    runtime.store.insert_tool_candidate(
        ToolCandidate(
            candidate_id=candidate_id,
            pid=pid,
            spec=ToolSpec(
                name=candidate_id.replace('_', '-'),
                description='checkpoint resource remap fixture',
            ),
            source_code='export function run() { return {}; }',
            tests=[],
            requested_capabilities=requested_capabilities,
            status=ToolCandidateStatus.PROPOSED,
            validation=None,
            created_at='2040-01-01T00:00:00+00:00',
            updated_at='2040-01-01T00:00:00+00:00',
        )
    )
    descriptor = runtime.memory.create_object(
        pid,
        ObjectType.TOOL_CANDIDATE,
        {
            'candidate_id': candidate_id,
            'requested_capabilities': requested_capabilities,
        },
        name=f'{candidate_id}.descriptor',
    )
    return descriptor.oid


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
                {
                    'phase': 'fork_event_emission',
                    'error_type': 'RuntimeError',
                    'message': 'injected failure',
                },
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
        assert output.reconciliation_pending is False
        assert output.post_commit_failures == []

    def test_fork_remap_does_not_clone_durable_task_run_binding(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='do not clone durable execution identity',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'task run binding boundary',
                actor=pid,
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _checkpoint, snapshot = found
            source_row = next(
                row
                for row in snapshot['rows']['processes']
                if row['pid'] == pid
            )
            source_row.update(
                {
                    'task_run_id': 'trun_source',
                    'task_run_epoch': 7,
                    'task_run_role': 'root',
                }
            )

            remapped = runtime.checkpoint._remap_snapshot(
                snapshot,
                parent_pid=None,
                root_pid=pid,
            )
            fork_pid = remapped['pid_map'][pid]
            fork_row = next(
                row
                for row in remapped['rows']['processes']
                if row['pid'] == fork_pid
            )

            assert fork_row['task_run_id'] is None
            assert fork_row['task_run_epoch'] is None
            assert fork_row['task_run_role'] is None
        finally:
            runtime.close()

    def test_fork_does_not_clone_durable_task_run_goal_marker(self) -> None:
        runtime = Runtime.open('local')
        try:
            marker = {
                '$task_run_ref': {
                    'run_id': 'trun_source',
                    'payload_sha256': 'a' * 64,
                    'schema_version': 1,
                }
            }
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal=marker,
            )
            source = runtime.process.get(pid)
            assert source.goal_oid is not None
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'task run marker boundary',
                actor=pid,
            )
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            result = runtime.checkpoint.fork_from_checkpoint(
                pid,
                checkpoint_id,
            )
            forked = runtime.process.get(result['fork_root_pid'])

            assert source.goal_oid not in result['object_map']
            assert forked.goal_oid is None
            assert forked.memory_view is not None
            assert all(
                runtime.store.get_object(handle.oid).payload != marker
                for handle in forked.memory_view.roots
            )
        finally:
            runtime.close()

    def test_fork_rejects_nested_outer_store_transaction(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='nested fork transaction')
            checkpoint_id = runtime.checkpoint.create(pid, 'nested fork', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}

            with runtime.uow.transaction():
                with pytest.raises(
                    ValidationError,
                    match='cannot run inside an existing store transaction',
                ):
                    runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('mutation', 'message'),
        [
            ('missing_module_id', 'snapshot modules'),
            ('missing_source_sha256', 'snapshot modules'),
            ('malformed_source_sha256', 'snapshot modules'),
            ('mismatched_source_sha256', 'checkpoint requires startup modules'),
        ],
    )
    def test_fork_rejects_invalid_module_identity_without_publication(
        self,
        mutation: str,
        message: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject an incomplete checkpoint module identity during fork',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'incomplete module identity',
                actor=pid,
            )
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            assert snapshot['modules']
            module = snapshot['modules'][0]
            if mutation == 'missing_module_id':
                module.pop('module_id')
            elif mutation == 'missing_source_sha256':
                module.pop('source_sha256')
            elif mutation == 'malformed_source_sha256':
                module['source_sha256'] = 'not-a-sha256'
            else:
                module['source_sha256'] = 'f' * 64
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )
            before_pids = {process.pid for process in runtime.process.list()}

            with pytest.raises(ValidationError, match=message):
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            runtime.close()

    def test_fork_rejects_malformed_capability_row_without_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject capability type laundering during fork',
            )
            capability = runtime.capability.grant(
                pid,
                'test:typed-snapshot-fork',
                [CapabilityRight.READ],
                issued_by='test',
                delegable=False,
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'malformed capability row',
                actor=pid,
            )
            fork_authority = runtime.capability.grant_once(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            row = next(
                item
                for item in snapshot['rows']['capabilities']
                if item['cap_id'] == capability.cap_id
            )
            row['delegable'] = 'false'
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            cache_calls: list[str] = []

            def unexpected_cache_or_publication(*_args: object, **_kwargs: object) -> None:
                cache_calls.append('called')
                raise AssertionError('malformed snapshot reached cache or publication')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_images',
                unexpected_cache_or_publication,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_jit_sources',
                unexpected_cache_or_publication,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_insert_fork_rows',
                unexpected_cache_or_publication,
            )
            before_processes = runtime.store.list_processes()
            before_capabilities = runtime.store.list_capabilities()
            before_audit_ids = {
                record.record_id for record in runtime.store.list_audit()
            }
            before_events = runtime.events.list()
            before_publications = runtime.store.list_runtime_publications()

            with pytest.raises(
                ValidationError,
                match=r'rows\.capabilities\[\d+\]\.delegable must be a boolean',
            ):
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert runtime.store.list_processes() == before_processes
            assert runtime.store.list_capabilities() == before_capabilities
            # Authorization decisions are append-only evidence even when
            # downstream snapshot validation rejects the fork. No fork
            # publication or finite-use settlement may survive the failure.
            _assert_only_checkpoint_authorization_audit(
                runtime,
                before_record_ids=before_audit_ids,
                actor=pid,
                checkpoint_id=checkpoint_id,
                capability_id=fork_authority.cap_id,
            )
            assert runtime.events.list() == before_events
            assert runtime.store.list_runtime_publications() == before_publications
            assert cache_calls == []
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'carrier',
        ['capability_row', 'tool_candidate', 'tool_candidate_object'],
        ids=['capability-row', 'tool-candidate', 'tool-candidate-object'],
    )
    def test_fork_rejects_malformed_capability_resource_before_publication(
        self,
        monkeypatch: pytest.MonkeyPatch,
        carrier: str,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject malformed resource during fork',
            )
            obj = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'value': 1},
                name='malformed-resource-fixture',
            )
            capability = runtime.capability.grant(
                pid,
                f'object:{obj.oid}/*',
                [CapabilityRight.READ],
                issued_by='test',
            )
            descriptor_oid = _insert_tool_candidate(
                runtime,
                candidate_id='candidate_malformed_resource',
                pid=pid,
                requested_capabilities=[
                    {
                        'resource': f'object:{obj.oid}:*',
                        'rights': ['read'],
                    }
                ],
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'malformed capability resource',
                actor=pid,
            )
            fork_authority = runtime.capability.grant_once(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            malformed_resource = f'object:{obj.oid}/*/escape'
            if carrier == 'capability_row':
                capability_row = next(
                    row
                    for row in snapshot['rows']['capabilities']
                    if row['cap_id'] == capability.cap_id
                )
                capability_row['resource'] = malformed_resource
            elif carrier == 'tool_candidate':
                candidate_row = next(
                    row
                    for row in snapshot['rows']['tool_candidates']
                    if row['candidate_id'] == 'candidate_malformed_resource'
                )
                candidate_row['requested_capabilities_json'] = dumps(
                    [
                        {
                            'resource': malformed_resource,
                            'rights': ['read'],
                        }
                    ]
                )
            else:
                snapshot['object_payloads'][descriptor_oid][
                    'requested_capabilities'
                ] = [
                    {
                        'resource': malformed_resource,
                        'rights': ['read'],
                    }
                ]
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            boundary_calls: list[str] = []

            def unexpected_cache_or_publication(
                *_args: object,
                **_kwargs: object,
            ) -> None:
                boundary_calls.append('called')
                raise AssertionError(
                    'malformed resource reached cache or publication'
                )

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_jit_sources',
                unexpected_cache_or_publication,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_insert_fork_rows',
                unexpected_cache_or_publication,
            )
            before_processes = runtime.store.list_processes()
            before_capabilities = runtime.store.list_capabilities()
            before_audit_ids = {
                record.record_id for record in runtime.store.list_audit()
            }
            before_events = runtime.events.list()
            before_publications = runtime.store.list_runtime_publications()

            with pytest.raises(
                ValidationError,
                match='not a valid capability resource',
            ):
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert runtime.store.list_processes() == before_processes
            assert runtime.store.list_capabilities() == before_capabilities
            assert runtime.store.get_capability(
                fork_authority.cap_id
            ).uses_remaining == 1
            new_audit_records = [
                record
                for record in runtime.store.list_audit()
                if record.record_id not in before_audit_ids
            ]
            assert any(
                record.actor == pid
                and record.action == 'capability.authorize'
                and record.target == f'checkpoint:{checkpoint_id}'
                and record.capability_refs == [fork_authority.cap_id]
                for record in new_audit_records
            )
            assert all(
                record.action != 'checkpoint.fork'
                for record in new_audit_records
            )
            assert runtime.events.list() == before_events
            assert runtime.store.list_runtime_publications() == before_publications
            assert boundary_calls == []
        finally:
            runtime.close()

    def test_fork_rejects_active_exhausted_capability_without_side_effects(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='reject active exhausted capability during fork',
            )
            capability = runtime.capability.grant(
                pid,
                'test:active-exhausted-snapshot-fork',
                [CapabilityRight.READ],
                issued_by='test',
            )
            checkpoint_id = runtime.checkpoint.create(
                pid,
                'active exhausted capability row',
                actor=pid,
            )
            fork_authority = runtime.capability.grant_once(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
            assert found is not None
            _, snapshot = found
            row = next(
                item
                for item in snapshot['rows']['capabilities']
                if item['cap_id'] == capability.cap_id
            )
            row.update(
                {
                    'effect': CapabilityEffect.DENY.value,
                    'uses_remaining': 0,
                    'status': 'active',
                }
            )
            runtime.store._execute(
                'UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?',
                (dumps(snapshot), checkpoint_id),
            )

            cache_calls: list[str] = []

            def unexpected_cache_or_publication(*_args: object, **_kwargs: object) -> None:
                cache_calls.append('called')
                raise AssertionError('malformed snapshot reached cache or publication')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_images',
                unexpected_cache_or_publication,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_restore_jit_sources',
                unexpected_cache_or_publication,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_insert_fork_rows',
                unexpected_cache_or_publication,
            )
            before_processes = runtime.store.list_processes()
            before_capabilities = runtime.store.list_capabilities()
            before_audit_ids = {
                record.record_id for record in runtime.store.list_audit()
            }
            before_events = runtime.events.list()
            before_publications = runtime.store.list_runtime_publications()

            with pytest.raises(
                ValidationError,
                match='active capability uses_remaining must be positive',
            ):
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert runtime.store.list_processes() == before_processes
            assert runtime.store.list_capabilities() == before_capabilities
            _assert_only_checkpoint_authorization_audit(
                runtime,
                before_record_ids=before_audit_ids,
                actor=pid,
                checkpoint_id=checkpoint_id,
                capability_id=fork_authority.cap_id,
            )
            assert runtime.events.list() == before_events
            assert runtime.store.list_runtime_publications() == before_publications
            assert cache_calls == []
        finally:
            runtime.close()

    def test_checkpoint_fork_operation_rejects_forged_receipt(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='forged fork receipt')
            error = RuntimeError('forged receipt')
            error.checkpoint_fork_receipt = {
                'checkpoint_id': 'ckpt_forged',
                'source_pid': pid,
                'fork_root_pid': 'pid_forged',
                'pid_map': {pid: 'pid_forged'},
                'object_map': {},
                'tool_map': {},
                'status': 'forked',
                'main_state_committed': True,
                'reconciliation_pending': False,
                'post_commit_failures': [],
                'untrusted_extra': {'secret': 'must-not-persist'},
            }

            with pytest.raises(RuntimeError, match='forged receipt'):
                with runtime.operations.scope(
                    kind='runtime',
                    name='checkpoint.fork',
                    actor=pid,
                    pid=pid,
                ) as operation:
                    operation_id = operation.operation_id
                    raise error

            persisted = runtime.store.get_operation(operation_id)
            assert persisted is not None
            assert persisted.outcome == OperationOutcome.FAILED
            assert 'checkpoint_fork_receipt' not in persisted.metadata
        finally:
            runtime.close()

    def test_fork_in_memory_commit_ack_loss_fences_without_cache_rollback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        fault_connections: list[_CommitThenRaiseConnection] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork payload ack loss')
            original = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'value': 41},
                name='state',
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'payload ack loss', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            original_insert = runtime.checkpoint._insert_fork_rows
            original_publish = (
                runtime.uow.snapshots.publish_checkpoint_fork_process_rows
            )
            quarantined_statuses: list[ProcessStatus] = []

            def insert_with_commit_fault(*args, **kwargs):
                fault_connection = _CommitThenRaiseConnection(runtime.store.conn)
                fault_connections.append(fault_connection)
                runtime.store.conn = fault_connection
                return original_insert(*args, **kwargs)

            def observe_quarantine(rows):
                quarantined_statuses.extend(
                    runtime.store.get_process(str(row['pid'])).status
                    for row in rows.processes
                    if str(row['status']) not in {
                        ProcessStatus.EXITED.value,
                        ProcessStatus.FAILED.value,
                        ProcessStatus.KILLED.value,
                    }
                )
                return original_publish(rows)

            monkeypatch.setattr(
                runtime.checkpoint,
                '_insert_fork_rows',
                insert_with_commit_fault,
            )
            monkeypatch.setattr(
                runtime.uow.snapshots,
                'publish_checkpoint_fork_process_rows',
                observe_quarantine,
            )

            with pytest.raises(
                ValidationError,
                match='unusable after uncertain transaction commit',
            ) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['status'] == 'fork_outcome_unknown'
            assert receipt['main_state_committed'] is None
            assert receipt['reconciliation_pending'] is True
            assert receipt['outcome_diagnostic']['phase'] == 'fork_commit_confirmation'
            assert receipt['outcome_diagnostic']['lifecycle_fenced'] is True
            assert runtime.lifecycle.state == 'close_failed'

            # A :memory: database has no independent observer after the only
            # connection is poisoned and closed. The runtime must neither read
            # that connection nor publish a guessed result. Keep the post-commit
            # payload image for explicit recovery instead of rolling back only
            # the volatile cache and manufacturing mixed state.
            assert len(fault_connections) == 1
            assert fault_connections[0].sql_calls_after_fault == []
            assert quarantined_statuses == []
            fork_oid = receipt['object_map'][original.oid]
            assert runtime.store._object_payloads[fork_oid] == {'value': 41}
            with pytest.raises(
                ValidationError,
                match='unusable after uncertain transaction commit',
            ):
                runtime.store.get_object(fork_oid)
            with pytest.raises(
                ValidationError,
                match='unusable after uncertain transaction commit',
            ):
                runtime.store.object_payload(fork_oid)
        finally:
            runtime.close()

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

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_fork_precommit_base_exception_cleans_prepared_runtime_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        cleanup_calls: list[str] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork interrupted before commit')
            checkpoint_id = runtime.checkpoint.create(pid, 'precommit interruption', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            before_pids = {process.pid for process in runtime.process.list()}
            original_discard_jit = runtime.checkpoint._discard_remapped_jit_sources
            original_discard_images = runtime.checkpoint._discard_uncommitted_fork_images

            def interrupt_prepare(_remapped):
                raise fault_type('injected precommit fork interruption')

            def observe_discard_jit(remapped):
                cleanup_calls.append('jit')
                original_discard_jit(remapped)

            def observe_discard_images(image_ids):
                cleanup_calls.append('images')
                original_discard_images(image_ids)

            monkeypatch.setattr(runtime.checkpoint, '_restore_jit_sources', interrupt_prepare)
            monkeypatch.setattr(runtime.checkpoint, '_discard_remapped_jit_sources', observe_discard_jit)
            monkeypatch.setattr(runtime.checkpoint, '_discard_uncommitted_fork_images', observe_discard_images)

            with pytest.raises(fault_type) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert str(caught.value) == 'injected precommit fork interruption'
            assert cleanup_calls == ['jit', 'images']
            assert {process.pid for process in runtime.process.list()} == before_pids
            assert not hasattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('sink', 'fault_type'),
        [
            ('event', KeyboardInterrupt),
            ('event', asyncio.CancelledError),
            ('audit', KeyboardInterrupt),
            ('audit', asyncio.CancelledError),
        ],
        ids=[
            'event-keyboard-interrupt',
            'event-cancelled-error',
            'audit-keyboard-interrupt',
            'audit-cancelled-error',
        ],
    )
    def test_fork_postcommit_base_exception_preserves_state_and_attaches_receipt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sink: str,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        cleanup_calls: list[str] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork interrupted after commit')
            checkpoint_id = runtime.checkpoint.create(pid, 'postcommit interruption', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            original_discard_jit = runtime.checkpoint._discard_remapped_jit_sources
            original_discard_images = runtime.checkpoint._discard_uncommitted_fork_images

            def observe_discard_jit(remapped):
                cleanup_calls.append('jit')
                original_discard_jit(remapped)

            def observe_discard_images(image_ids):
                cleanup_calls.append('images')
                original_discard_images(image_ids)

            monkeypatch.setattr(runtime.checkpoint, '_discard_remapped_jit_sources', observe_discard_jit)
            monkeypatch.setattr(runtime.checkpoint, '_discard_uncommitted_fork_images', observe_discard_images)
            interruption = fault_type(f'injected fork {sink} interruption')
            if sink == 'event':
                original_emit = runtime.events.emit

                def interrupt_fork_event(event_type, *args, **kwargs):
                    if event_type == EventType.PROCESS_FORKED:
                        raise interruption
                    return original_emit(event_type, *args, **kwargs)

                monkeypatch.setattr(runtime.events, 'emit', interrupt_fork_event)
            else:
                original_record = runtime.audit.record

                def interrupt_fork_audit(*args, **kwargs):
                    if kwargs.get('action') == 'checkpoint.fork':
                        raise interruption
                    return original_record(*args, **kwargs)

                monkeypatch.setattr(runtime.audit, 'record', interrupt_fork_audit)

            with pytest.raises(fault_type) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert caught.value is interruption
            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['checkpoint_id'] == checkpoint_id
            assert receipt['source_pid'] == pid
            assert receipt['status'] == 'forked_with_warnings'
            assert receipt['main_state_committed'] is True
            assert receipt['pid_map'][pid] == receipt['fork_root_pid']
            expected_phase = (
                'fork_audit_recording' if sink == 'audit' else 'fork_event_emission'
            )
            assert receipt['post_commit_failures'][-1]['phase'] == expected_phase
            assert runtime.store.get_process(receipt['fork_root_pid']) is not None
            assert cleanup_calls == []
            assert any('checkpoint fork main state committed' in note for note in caught.value.__notes__)
            operation = _fork_operation(runtime, pid)
            assert operation.state == OperationState.TERMINAL
            assert operation.outcome == OperationOutcome.SUCCEEDED
            assert operation.metadata['checkpoint_fork_receipt'] == receipt
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_fork_commit_ack_interruption_uses_committed_receipt_without_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        cleanup_calls: list[str] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork commit acknowledgement')
            checkpoint_id = runtime.checkpoint.create(pid, 'commit acknowledgement', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            original_insert = runtime.checkpoint._insert_fork_rows

            def interrupt_after_commit(*args, **kwargs):
                original_insert(*args, **kwargs)
                raise fault_type('injected fork commit acknowledgement interruption')

            monkeypatch.setattr(runtime.checkpoint, '_insert_fork_rows', interrupt_after_commit)
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_remapped_jit_sources',
                lambda _remapped: cleanup_calls.append('jit'),
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_uncommitted_fork_images',
                lambda _image_ids: cleanup_calls.append('images'),
            )

            with pytest.raises(fault_type) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['main_state_committed'] is True
            assert len(receipt['post_commit_failures']) == 1
            failure = receipt['post_commit_failures'][0]
            assert failure['phase'] == 'fork_commit_acknowledgement'
            assert failure['error_type'] == fault_type.__name__
            assert failure['code'] == 'checkpoint_fork_post_commit_failed'
            assert failure['correlation_id'] in failure['message']
            assert len(failure['internal_error']['exception_text']['sha256']) == 64
            assert 'injected fork commit acknowledgement interruption' not in str(failure)
            assert runtime.store.get_process(receipt['fork_root_pid']) is not None
            assert cleanup_calls == []
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        'fault_type',
        [KeyboardInterrupt, asyncio.CancelledError],
        ids=['keyboard_interrupt', 'cancelled_error'],
    )
    def test_fork_durable_root_confirms_commit_before_in_memory_ack(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fault_type: type[BaseException],
    ) -> None:
        runtime = Runtime.open('local')
        cleanup_calls: list[str] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork durable commit confirmation')
            checkpoint_id = runtime.checkpoint.create(pid, 'durable commit confirmation', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            interruption = fault_type('injected interruption before in-memory fork ack')

            def interrupt_ack(_publication_state):
                raise interruption

            monkeypatch.setattr(
                runtime.checkpoint,
                '_acknowledge_fork_main_state_commit',
                interrupt_ack,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_remapped_jit_sources',
                lambda _remapped: cleanup_calls.append('jit'),
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_uncommitted_fork_images',
                lambda _image_ids: cleanup_calls.append('images'),
            )

            with pytest.raises(fault_type) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert caught.value is interruption
            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['main_state_committed'] is True
            assert receipt['post_commit_failures'][0]['phase'] == 'fork_commit_acknowledgement'
            assert runtime.store.get_process(receipt['fork_root_pid']) is not None
            assert cleanup_calls == []
        finally:
            runtime.close()

    def test_fork_commit_diagnostic_failure_fences_runtime_and_marks_unknown(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        cleanup_calls: list[str] = []
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork unknown commit outcome')
            checkpoint_id = runtime.checkpoint.create(pid, 'unknown commit outcome', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            interruption = asyncio.CancelledError('injected fork publication interruption')

            original_insert = runtime.checkpoint._insert_fork_rows

            def interrupt_after_durable_quarantine(*args, **kwargs):
                original_insert(*args, **kwargs)
                kwargs['publication_state']['main_state_committed'] = False
                raise interruption

            def fail_durable_confirmation(_root_pid):
                raise RuntimeError('injected durable fork confirmation failure')

            monkeypatch.setattr(
                runtime.checkpoint,
                '_insert_fork_rows',
                interrupt_after_durable_quarantine,
            )
            monkeypatch.setattr(runtime.checkpoint, '_fork_root_is_persisted', fail_durable_confirmation)
            monkeypatch.setattr(
                runtime.checkpoint,
                '_secure_checkpoint_fork_subtree',
                lambda **_kwargs: None,
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_remapped_jit_sources',
                lambda _remapped: cleanup_calls.append('jit'),
            )
            monkeypatch.setattr(
                runtime.checkpoint,
                '_discard_uncommitted_fork_images',
                lambda _image_ids: cleanup_calls.append('images'),
            )

            with pytest.raises(asyncio.CancelledError) as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            assert caught.value is interruption
            assert isinstance(caught.value.__cause__, RuntimeError)
            assert str(caught.value.__cause__) == 'injected durable fork confirmation failure'
            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['status'] == 'fork_outcome_unknown'
            assert receipt['main_state_committed'] is None
            assert receipt['reconciliation_pending'] is True
            assert receipt['outcome_diagnostic']['phase'] == 'fork_commit_confirmation'
            assert receipt['outcome_diagnostic']['diagnostic_error_type'] == 'RuntimeError'
            assert receipt['outcome_diagnostic']['lifecycle_fenced'] is True
            assert cleanup_calls == []
            assert runtime.lifecycle.state == 'close_failed'
            operation = _fork_operation(runtime, pid)
            assert operation.outcome == OperationOutcome.UNKNOWN
            assert operation.metadata['checkpoint_fork_receipt']['status'] == 'fork_outcome_unknown'
            with pytest.raises(RuntimeError):
                runtime.process.spawn(image='base-agent:v0', goal='must remain fenced')
        finally:
            runtime.close()

    def test_fork_payload_rehydration_failure_terminalizes_subtree(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='fork rehydrate failure')
            runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'value': 9},
                name='state',
            )
            checkpoint_id = runtime.checkpoint.create(pid, 'rehydrate failure', actor=pid)
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            def fail_rehydration(*_args, **_kwargs):
                raise RuntimeError('injected fork payload rehydration failure')

            monkeypatch.setattr(
                runtime.uow.snapshots,
                'rehydrate_checkpoint_fork_object_payloads',
                fail_rehydration,
            )

            with pytest.raises(RuntimeError, match='payload rehydration failure') as caught:
                runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)

            receipt = getattr(
                caught.value,
                runtime.checkpoint.FORK_COMMITTED_RECEIPT_ATTRIBUTE,
            )
            assert receipt['status'] == 'fork_recovery_required'
            assert receipt['main_state_committed'] is True
            assert receipt['reconciliation_pending'] is True
            assert receipt['outcome_diagnostic']['fork_subtree_quarantined'] is True
            assert all(
                runtime.store.get_process(fork_pid).status == ProcessStatus.FAILED
                for fork_pid in receipt['pid_map'].values()
            )
            operation = _fork_operation(runtime, pid)
            assert operation.outcome == OperationOutcome.UNKNOWN
            assert operation.metadata['checkpoint_fork_receipt'] == receipt
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

    def test_fork_remaps_scoped_object_authority_and_drops_non_clonable_scopes(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            task_marker = {
                '$task_run_ref': {
                    'run_id': 'trun_scoped_authority_source',
                    'payload_sha256': 'a' * 64,
                    'schema_version': 1,
                }
            }
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal=task_marker,
            )
            task_reference_oid = runtime.process.get(pid).goal_oid
            assert task_reference_oid is not None
            clonable = runtime.memory.create_object(
                pid,
                ObjectType.SUMMARY,
                {'value': 7},
                name='clonable.scoped',
            )
            external = runtime.memory.create_object(
                pid,
                ObjectType.EXTERNAL_REF,
                {'provider': 'remote', 'handle': 'opaque'},
                name='external.scoped',
            )
            external_owner = runtime.process.spawn(
                image='base-agent:v0',
                goal='unobserved external owner',
            )
            unobserved_external = runtime.memory.create_object(
                external_owner,
                ObjectType.EXTERNAL_REF,
                {'provider': 'remote', 'handle': 'capability-only'},
                name='external.capability-only',
            )
            task_reference_owner = runtime.process.spawn(
                image='base-agent:v0',
                goal={
                    '$task_run_ref': {
                        'run_id': 'trun_capability_only_source',
                        'payload_sha256': 'b' * 64,
                        'schema_version': 1,
                    }
                },
            )
            unobserved_task_reference_oid = runtime.process.get(
                task_reference_owner
            ).goal_oid
            assert unobserved_task_reference_oid is not None
            clone_resources = {
                f'object:{clonable.oid}',
                f'object:{clonable.oid}/*',
                f'object:{clonable.oid}:*',
            }
            non_clonable_resources = {
                f'object:{external.oid}',
                f'object:{external.oid}/*',
                f'object:{external.oid}:*',
                f'object:{task_reference_oid}',
                f'object:{task_reference_oid}/*',
                f'object:{task_reference_oid}:*',
                f'object:{unobserved_external.oid}',
                f'object:{unobserved_external.oid}/*',
                f'object:{unobserved_external.oid}:*',
                f'object:{unobserved_task_reference_oid}',
                f'object:{unobserved_task_reference_oid}/*',
                f'object:{unobserved_task_reference_oid}:*',
            }
            global_object_resource = 'object:*'
            explicit_source_capabilities = [
                runtime.capability.grant(
                    pid,
                    resource,
                    [CapabilityRight.READ],
                    issued_by='test.scoped-object-fork',
                )
                for resource in sorted(
                    clone_resources
                    | non_clonable_resources
                    | {global_object_resource}
                )
            ]
            filesystem_resource = 'filesystem:workspace:README.md'
            runtime.capability.grant(
                pid,
                filesystem_resource,
                [CapabilityRight.READ],
                issued_by='test.scoped-object-fork-control',
            )
            requested_resources = sorted(
                clone_resources
                | non_clonable_resources
                | {global_object_resource, filesystem_resource}
            )
            descriptor_oid = _insert_tool_candidate(
                runtime,
                candidate_id='candidate_scoped_object_fork',
                pid=pid,
                requested_capabilities=[
                    {'resource': resource, 'rights': ['read']}
                    for resource in requested_resources
                ],
            )

            checkpoint_id = runtime.checkpoint.create(
                pid,
                'scoped object authority fork',
                actor=pid,
            )
            runtime.capability.grant(
                pid,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )

            forked = runtime.checkpoint.fork_from_checkpoint(pid, checkpoint_id)
            fork_pid = forked['fork_root_pid']
            cloned_oid = forked['object_map'][clonable.oid]
            remapped_clone_resources = {
                f'object:{cloned_oid}',
                f'object:{cloned_oid}/*',
                f'object:{cloned_oid}:*',
            }
            fork_resources = {
                capability.resource
                for capability in runtime.capability.list_subject(fork_pid)
            }

            assert remapped_clone_resources <= fork_resources
            assert clone_resources.isdisjoint(fork_resources)
            assert non_clonable_resources.isdisjoint(fork_resources)
            assert global_object_resource not in fork_resources
            assert filesystem_resource in fork_resources
            for source_capability in explicit_source_capabilities:
                persisted = runtime.store.get_capability(source_capability.cap_id)
                assert persisted is not None
                assert persisted.active
                assert persisted.resource == source_capability.resource

            candidate_rows = runtime.store._query(
                'SELECT candidate_id FROM tool_candidates WHERE pid = ?',
                (fork_pid,),
            )
            assert len(candidate_rows) == 1
            fork_candidate = runtime.store.get_tool_candidate(
                str(candidate_rows[0]['candidate_id'])
            )
            assert fork_candidate is not None
            fork_requested_resources = {
                str(spec['resource'])
                for spec in fork_candidate.requested_capabilities
            }
            assert fork_requested_resources == {
                *remapped_clone_resources,
                filesystem_resource,
            }
            fork_descriptor = runtime.store.get_object(
                forked['object_map'][descriptor_oid]
            )
            assert fork_descriptor is not None
            descriptor_requested_resources = {
                str(spec['resource'])
                for spec in fork_descriptor.payload['requested_capabilities']
            }
            assert descriptor_requested_resources == fork_requested_resources

            assert any(
                event.type == EventType.PROCESS_FORKED
                and event.target == fork_pid
                and event.payload['checkpoint_id'] == checkpoint_id
                for event in runtime.events.list()
            )
            assert any(
                record.action == 'checkpoint.fork'
                and record.target == f'checkpoint:{checkpoint_id}'
                and record.decision['fork_root_pid'] == fork_pid
                for record in runtime.store.list_audit()
            )
        finally:
            runtime.close()

    def test_fork_drops_released_unmapped_object_authority_before_later_restore(
        self,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='drop dormant object authority during fork',
            )
            external = runtime.memory.create_object(
                pid,
                ObjectType.EXTERNAL_REF,
                {'provider': 'remote', 'handle': 'dormant'},
                name='released.external.ref',
            )
            dormant_resources = {
                f'object:{external.oid}',
                f'object:{external.oid}/*',
                f'object:{external.oid}:*',
            }
            capabilities = {
                capability.resource: capability
                for capability in (
                    runtime.capability.grant(
                        pid,
                        resource,
                        [CapabilityRight.READ],
                        issued_by='test.released-object-fork',
                    )
                    for resource in sorted(dormant_resources)
                )
            }
            restorable_checkpoint_id = runtime.checkpoint.create(
                pid,
                'external ref remains live',
                actor=pid,
            )

            assert runtime.memory.delete_object_trusted(
                'test',
                external.oid,
                reason='exercise dormant scoped authority',
            )
            assert runtime.store.get_object(external.oid) is None
            assert runtime.store.get_capability(
                capabilities[f'object:{external.oid}/*'].cap_id
            ).active
            assert runtime.store.get_capability(
                capabilities[f'object:{external.oid}:*'].cap_id
            ).active

            descriptor_oid = _insert_tool_candidate(
                runtime,
                candidate_id='candidate_released_object_authority',
                pid=pid,
                requested_capabilities=[
                    {'resource': resource, 'rights': ['read']}
                    for resource in sorted(dormant_resources)
                ],
            )
            dormant_checkpoint_id = runtime.checkpoint.create(
                pid,
                'external ref is released',
                actor=pid,
            )

            forked = runtime.checkpoint.fork_from_checkpoint(
                pid,
                dormant_checkpoint_id,
                require_capability=False,
            )
            fork_pid = forked['fork_root_pid']
            fork_resources = {
                capability.resource
                for capability in runtime.capability.list_subject(fork_pid)
            }
            assert dormant_resources.isdisjoint(fork_resources)
            assert external.oid not in forked['object_map']
            assert external.oid not in {
                root.oid for root in runtime.process.get(fork_pid).memory_view.roots
            }
            assert runtime.store.get_capability(
                capabilities[f'object:{external.oid}/*'].cap_id
            ).active
            assert runtime.store.get_capability(
                capabilities[f'object:{external.oid}:*'].cap_id
            ).active

            candidate_rows = runtime.store._query(
                'SELECT candidate_id FROM tool_candidates WHERE pid = ?',
                (fork_pid,),
            )
            assert len(candidate_rows) == 1
            fork_candidate = runtime.store.get_tool_candidate(
                str(candidate_rows[0]['candidate_id'])
            )
            assert fork_candidate is not None
            assert fork_candidate.requested_capabilities == []
            fork_descriptor = runtime.store.get_object(
                forked['object_map'][descriptor_oid]
            )
            assert fork_descriptor is not None
            assert fork_descriptor.payload['requested_capabilities'] == []

            runtime.checkpoint.restore(
                pid,
                restorable_checkpoint_id,
                require_capability=False,
            )
            restored_external = runtime.store.get_object(external.oid)
            assert restored_external is not None
            assert restored_external.type == ObjectType.EXTERNAL_REF
            assert not runtime.capability.check(
                fork_pid,
                f'object:{external.oid}',
                CapabilityRight.READ,
            )
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

    def test_fork_outer_failure_never_publishes_restored_image_to_concurrent_readers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        image_id = 'checkpoint-fork-staged-image:v0'
        row_insert_entered = threading.Event()
        release_row_insert = threading.Event()
        outcomes: list[BaseException] = []
        try:
            runtime.register_image(
                AgentImage(image_id=image_id, name='checkpoint-fork-staged-image'),
                actor='test',
            )
            source = runtime.process.spawn(image=image_id, goal='snapshot staged image')
            checkpoint_id = runtime.checkpoint.create(
                source,
                'staged image fork point',
                actor=source,
            )
            actor = runtime.process.spawn(image='base-agent:v0', goal='fork staged image')
            runtime.capability.grant(
                actor,
                f'checkpoint:{checkpoint_id}',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            runtime.image_registry.grant_register(actor, image_id, issued_by='test')
            runtime.images.pop(image_id)
            runtime.store.delete_image(image_id)
            before_pids = {process.pid for process in runtime.process.list()}

            def fail_late_row_insert(*_args: object, **_kwargs: object) -> None:
                row_insert_entered.set()
                if not release_row_insert.wait(timeout=5):
                    raise RuntimeError('timed out waiting to fail checkpoint fork rows')
                raise RuntimeError('injected late checkpoint fork row failure')

            monkeypatch.setattr(
                runtime.uow.snapshots,
                'insert_checkpoint_fork_rows',
                fail_late_row_insert,
            )

            def fork() -> None:
                try:
                    runtime.checkpoint.fork_from_checkpoint(actor, checkpoint_id)
                except BaseException as exc:
                    outcomes.append(exc)

            thread = threading.Thread(target=fork, daemon=True)
            thread.start()
            assert row_insert_entered.wait(timeout=5)

            assert image_id not in runtime.images
            assert runtime.llm._images.get(image_id) is None
            with pytest.raises(NotFound):
                runtime.launch.require_image(image_id)

            release_row_insert.set()
            thread.join(timeout=10)

            assert not thread.is_alive()
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], RuntimeError)
            assert 'injected late checkpoint fork row failure' in str(outcomes[0])
            assert image_id not in runtime.images
            assert runtime.store.get_image(image_id) is None
            assert {process.pid for process in runtime.process.list()} == before_pids
        finally:
            release_row_insert.set()
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

    def test_checkpoint_fork_rejects_active_task_run_parent_without_runnable_bypass(
        self,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            task_runs=replace(
                DEFAULT_CONFIG.task_runs,
                plaintext_payloads_enabled=True,
            ),
        )
        runtime = Runtime.open('local', config=config)
        try:
            source = runtime.process.spawn(
                image='base-agent:v0',
                goal='ordinary checkpoint fork source',
            )
            checkpoint_id = runtime.checkpoint.create(
                source,
                'ordinary source for TaskRun parent rejection',
                actor=source,
            )
            task_run = runtime.task_runs.create(
                TaskRunSpecV1(
                    goal='remain supervised while a fork is attempted',
                    display_title='fork parent TaskRun guard',
                    image_id='base-agent:v0',
                ),
                client_request_id='fork-active-task-run-parent',
            )
            parent_pid = task_run.root_pid
            assert parent_pid is not None
            parent_before = runtime.process.get(parent_pid)
            pids_before = {process.pid for process in runtime.process.list()}

            with pytest.raises(
                ValidationError,
                match='cannot attach to a Durable TaskRun process',
            ):
                runtime.checkpoint.fork_from_checkpoint(
                    parent_pid,
                    checkpoint_id,
                    parent_pid=parent_pid,
                    require_capability=False,
                )

            assert {process.pid for process in runtime.process.list()} == pids_before
            assert runtime.process.get(parent_pid) == parent_before
            assert runtime.task_runs.get(task_run.run_id) == task_run
            assert runtime.scheduler.next_runnable(pids=[parent_pid]) is None
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
