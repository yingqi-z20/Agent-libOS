from __future__ import annotations

import tempfile

import pytest

from agent_libos import Runtime
from agent_libos.models import (
    EventType,
    KilledProcessOutcome,
    ObjectMetadata,
    ObjectType,
    ProcessStatus,
    ResourceBudget,
    ResourceUsage,
)
from agent_libos.models.exceptions import ResourceLimitExceeded, ValidationError
from tests.support.public_errors import assert_public_error_message


class TestResourceManager:
    def test_invalid_resource_usage_does_not_echo_unknown_field_names(self) -> None:
        runtime = Runtime.open("local")
        sentinel = "secret-unknown-resource-field"
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="resource validation")

            with pytest.raises(ValidationError) as caught:
                runtime.resources.preflight(
                    pid,
                    {sentinel: 1},
                    source="test",
                )

            assert str(caught.value) == "resource usage contains unknown fields"
            assert sentinel not in str(caught.value)
        finally:
            runtime.close()

    def test_hierarchical_charge_rolls_back_every_process_when_parent_update_fails(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(image="base-agent:v0", goal="parent")
            child = runtime.process.spawn_child(parent, goal="child")
            initial_charge_records = {
                record.record_id for record in runtime.audit.trace() if record.action == "resource.charge"
            }
            original_patch = runtime.resources.store.patch_process
            original_control_patch = runtime.resources.store.patch_process_control
            updated: list[str] = []

            def record_child_patch(pid: str, *args: object, **kwargs: object) -> object:
                updated.append(pid)
                return original_patch(pid, *args, **kwargs)

            def fail_parent_patch(pid: str, *args: object, **kwargs: object) -> object:
                updated.append(pid)
                raise RuntimeError("injected parent update failure")

            runtime.resources.store.patch_process = record_child_patch  # type: ignore[method-assign]
            runtime.resources.store.patch_process_control = fail_parent_patch  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match="injected parent update failure"):
                runtime.resources.charge(child, ResourceUsage(tool_calls=1), source="test")
            runtime.resources.store.patch_process = original_patch  # type: ignore[method-assign]
            runtime.resources.store.patch_process_control = original_control_patch  # type: ignore[method-assign]

            assert updated == [child, parent]
            assert runtime.process.get(child).resource_usage.tool_calls == 0
            assert runtime.process.get(parent).resource_usage.tool_calls == 0
            assert {
                record.record_id for record in runtime.audit.trace() if record.action == "resource.charge"
            } == initial_charge_records
        finally:
            runtime.close()

    def test_resource_kill_notifies_terminal_hooks_after_releasing_store_lock(self) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='terminal lock order')
            original = runtime.resources._object_task_terminal_notifier
            lock_states: list[bool] = []

            def notifier(selected_pid: str) -> None:
                is_owned = getattr(runtime.store._lock, '_is_owned', lambda: False)
                lock_states.append(bool(is_owned()))
                if original is not None:
                    original(selected_pid)

            runtime.resources.bind_object_task_terminal_notifier(notifier)
            runtime.resources.kill_if_exceeded(pid, reason='test terminal lock order')

            assert lock_states == [False]
        finally:
            runtime.close()

    def test_resource_kill_finalizes_remaining_processes_after_one_cleanup_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            parent = runtime.process.spawn(image='base-agent:v0', goal='failed parent cleanup')
            child = runtime.process.spawn_child(parent, goal='remaining child cleanup')
            original_finalize = runtime.process._finalize_terminal_process
            finalized: list[str] = []
            fail_parent = True
            failure_text = 'injected parent cleanup failure'

            def fail_parent_once(process: object, preserve_oids: set[str]) -> None:
                nonlocal fail_parent
                selected_pid = str(getattr(process, 'pid'))
                finalized.append(selected_pid)
                if selected_pid == parent and fail_parent:
                    raise RuntimeError(failure_text)
                original_finalize(process, preserve_oids)

            monkeypatch.setattr(runtime.process, '_finalize_terminal_process', fail_parent_once)
            runtime.resources.bind_process_kill_finalizer(runtime.process.finalize_killed_processes)

            runtime.resources.kill_if_exceeded(parent, reason='test best-effort descendant cleanup')

            assert finalized == [parent, child]
            parent_cleanup = runtime.process.terminal_cleanup_state(parent)
            child_cleanup = runtime.process.terminal_cleanup_state(child)
            assert parent_cleanup['state'] == 'failed'
            assert parent_cleanup['failed_phase'] == 'process_finalize'
            assert parent_cleanup['completed_phases'] == ['terminal_notify']
            assert parent_cleanup['last_error']['error_type'] == 'RuntimeError'
            assert len(parent_cleanup['last_error']['exception_text']['sha256']) == 64
            assert child_cleanup['state'] == 'completed'
            warnings = [
                record
                for record in runtime.audit.trace()
                if record.action == 'resource.limit_finalize_failed'
            ]
            assert len(warnings) == 1
            warning = warnings[0].decision['errors'][0]
            assert warning['phase'] == 'process_finalize'
            assert warning['code'] == 'killed_process_finalization_failed'
            assert warning['correlation_id'].startswith('corr_')
            assert warning['error_type'] == 'RuntimeError'
            assert len(warning['exception_text']['sha256']) == 64
            assert failure_text not in str(warnings[0].decision)

            fail_parent = False
            repaired = runtime.process.retry_terminal_cleanup(parent)
            assert repaired['state'] == 'completed'
            assert repaired['attempt_count'] == 2
            assert finalized == [parent, child, parent]
            assert not any(
                record.action == 'resource.limit_finalize_failed'
                and failure_text in str(record.decision)
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_killed_process_cleanup_attempts_event_and_notifier_after_finalizer_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pid = runtime.process.spawn(image='base-agent:v0', goal='independent killed cleanup phases')
            process = runtime.process.get(pid)
            runtime.process_transitions.transition(
                pid,
                ProcessStatus.KILLED,
                expected_revision=process.revision,
                outcome=KilledProcessOutcome(code="test_fixture"),
            )
            notified: list[str] = []
            runtime.process.bind_object_task_terminal_notifier(notified.append)
            original_finalize = runtime.process._finalize_terminal_process
            failure_text = 'injected killed finalizer failure'

            def fail_finalize(process: object, preserve_oids: set[str]) -> None:
                raise RuntimeError(failure_text)

            monkeypatch.setattr(runtime.process, '_finalize_terminal_process', fail_finalize)

            with pytest.raises(RuntimeError) as caught:
                runtime.process.finalize_killed_processes([pid], reason='test independent phases')

            correlation_id = assert_public_error_message(
                str(caught.value),
                code='killed_process_finalization_failed',
                error_type='RuntimeError',
                forbidden=[failure_text],
            )
            failures = caught.value.finalization_failures  # type: ignore[attr-defined]
            assert len(failures) == 1
            assert failures[0]['pid'] == pid
            assert failures[0]['phase'] == 'terminal_cleanup'
            assert failures[0]['error_type'] == 'ProcessTerminalCleanupRequired'
            assert failures[0]['correlation_id'] == correlation_id
            assert failure_text not in str(failures)
            assert notified == [pid]
            matching_events = [
                event.type == EventType.PROCESS_EXITED
                and event.source == pid
                and event.payload.get('reason') == 'test independent phases'
                for event in runtime.events.list()
            ]
            assert sum(matching_events) == 1
            cleanup = runtime.process.terminal_cleanup_state(pid)
            assert cleanup['state'] == 'failed'
            assert cleanup['failed_phase'] == 'process_finalize'
            assert cleanup['completed_phases'] == ['terminal_notify']
            assert cleanup['last_error']['error_type'] == 'RuntimeError'
            assert len(cleanup['last_error']['exception_text']['sha256']) == 64

            monkeypatch.setattr(
                runtime.process,
                '_finalize_terminal_process',
                original_finalize,
            )
            repaired = runtime.process.retry_terminal_cleanup(pid)
            assert repaired['state'] == 'completed'
            assert repaired['attempt_count'] == 2
            assert notified == [pid]

            runtime.process.finalize_killed_processes(
                [pid],
                reason='test independent phases',
            )
            runtime.process.finalize_killed_processes(
                [pid],
                reason='test independent phases',
            )
            assert sum(
                event.type == EventType.PROCESS_EXITED
                and event.source == pid
                and event.payload.get('reason') == 'test independent phases'
                for event in runtime.events.list()
            ) == 1

            with pytest.raises(RuntimeError) as collision:
                runtime.process.finalize_killed_processes(
                    [pid],
                    reason='conflicting terminal identity',
                )
            assert_public_error_message(
                str(collision.value),
                code='killed_process_finalization_failed',
                error_type='RuntimeError',
                forbidden=['conflicting terminal identity'],
            )
            collision_failures = collision.value.finalization_failures  # type: ignore[attr-defined]
            assert collision_failures[0]['phase'] == 'exit_event'
            assert collision_failures[0]['error_type'] == 'ValidationError'
            assert sum(
                event.type == EventType.PROCESS_EXITED
                and event.source == pid
                for event in runtime.events.list()
            ) == 1
        finally:
            runtime.close()

    def test_killed_process_finalization_sanitizes_base_exceptions_and_continues(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime = Runtime.open('local')
        try:
            pids = [
                runtime.process.spawn(goal='interrupted killed cleanup'),
                runtime.process.spawn(goal='remaining killed cleanup'),
            ]
            for pid in pids:
                process = runtime.process.get(pid)
                runtime.process_transitions.transition(
                    pid,
                    ProcessStatus.KILLED,
                    expected_revision=process.revision,
                    outcome=KilledProcessOutcome(code='test_fixture'),
                )

            original_emit = runtime.process._emit_killed_process_exit_event
            secret = 'private:///terminal-interrupt-token'

            def interrupt_first(process: object, *, reason: str) -> None:
                if getattr(process, 'pid') == pids[0]:
                    raise KeyboardInterrupt(secret)
                original_emit(process, reason=reason)  # type: ignore[arg-type]

            monkeypatch.setattr(
                runtime.process,
                '_emit_killed_process_exit_event',
                interrupt_first,
            )

            with pytest.raises(BaseExceptionGroup) as caught:
                runtime.process.finalize_killed_processes(
                    pids,
                    reason='base exception cleanup',
                )

            group = caught.value
            failures = group.finalization_failures  # type: ignore[attr-defined]
            assert failures[0]['pid'] == pids[0]
            assert failures[0]['phase'] == 'exit_event'
            assert failures[0]['error_type'] == 'KeyboardInterrupt'
            assert secret not in repr(group)
            assert secret not in repr(group.exceptions)
            assert secret not in repr(failures)
            assert all(
                runtime.process.terminal_cleanup_state(pid)['state'] == 'completed'
                for pid in pids
            )
            assert any(
                event.type == EventType.PROCESS_EXITED
                and event.source == pids[1]
                for event in runtime.events.list()
            )

            monkeypatch.setattr(
                runtime.process,
                '_emit_killed_process_exit_event',
                original_emit,
            )
            runtime.process.finalize_killed_processes(
                pids,
                reason='base exception cleanup',
            )
            assert all(
                sum(
                    event.type == EventType.PROCESS_EXITED
                    and event.source == pid
                    for event in runtime.events.list()
                ) == 1
                for pid in pids
            )
        finally:
            runtime.close()

    def test_resource_models_reject_non_finite_numbers(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ResourceBudget(max_tool_calls=float("inf"))
        with pytest.raises(ValueError, match="finite"):
            ResourceUsage(tool_calls=float("nan"))

    def test_resource_models_reject_fractional_discrete_counters(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            ResourceBudget(max_tool_calls=0.5)
        with pytest.raises(ValueError, match="integer"):
            ResourceUsage(tool_calls=0.5)

        budget = ResourceBudget(max_runtime_seconds=0.25, max_subprocess_cpu_seconds=0.5)
        usage = ResourceUsage(runtime_seconds=0.25, subprocess_cpu_seconds=0.5)
        assert budget.max_runtime_seconds == 0.25
        assert budget.max_subprocess_cpu_seconds == 0.5
        assert usage.runtime_seconds == 0.25
        assert usage.subprocess_cpu_seconds == 0.5

    def test_resource_manager_rejects_mutated_fractional_discrete_usage(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="integral resource usage")
            usage = ResourceUsage(tool_calls=1)
            usage.tool_calls = 0.5

            with pytest.raises(
                ValidationError,
                match="resource usage values are invalid",
            ):
                runtime.resources.charge(pid, usage, source="test")
        finally:
            runtime.close()

    def test_resource_manager_rejects_mutated_non_finite_usage(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(image="base-agent:v0", goal="finite resource usage")
            usage = ResourceUsage(tool_calls=1)
            usage.tool_calls = float("nan")

            with pytest.raises(
                ValidationError,
                match="resource usage values are invalid",
            ):
                runtime.resources.charge(pid, usage, source="test")
        finally:
            runtime.close()

    def test_hierarchical_charge_updates_child_and_parent_usage(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_tool_calls=4, max_llm_total_tokens=100),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="child",
                resource_budget=ResourceBudget(max_tool_calls=2, max_llm_total_tokens=50),
            )

            runtime.resources.charge(
                child,
                ResourceUsage(tool_calls=1, llm_total_tokens=7),
                source="test",
            )

            assert runtime.process.get(child).resource_usage.tool_calls == 1
            assert runtime.process.get(parent).resource_usage.tool_calls == 1
            assert runtime.process.get(child).resource_usage.llm_total_tokens == 7
            assert runtime.process.get(parent).resource_usage.llm_total_tokens == 7
        finally:
            runtime.close()

    def test_remaining_budgets_batches_hierarchy_and_child_reservations(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="batched parent budget",
                resource_budget=ResourceBudget(max_tool_calls=10, max_llm_total_tokens=100),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="batched child budget",
                resource_budget=ResourceBudget(max_tool_calls=4, max_llm_total_tokens=30),
            )

            before = runtime.resources.remaining_budgets([parent, child])

            assert before[parent].max_tool_calls == 6
            assert before[child].max_tool_calls == 4
            assert before[parent].max_llm_total_tokens == 70
            assert before[child].max_llm_total_tokens == 30

            runtime.resources.charge(
                child,
                ResourceUsage(tool_calls=1, llm_total_tokens=7),
                source="test",
            )
            after = runtime.resources.remaining_budgets([child, parent])

            assert after[parent].max_tool_calls == 6
            assert after[child].max_tool_calls == 3
            assert after[parent].max_llm_total_tokens == 70
            assert after[child].max_llm_total_tokens == 23
        finally:
            runtime.close()

    def test_preflight_fails_without_consuming_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="budget",
                resource_budget=ResourceBudget(max_tool_calls=1),
            )
            runtime.resources.charge(pid, ResourceUsage(tool_calls=1), source="test")

            with pytest.raises(ResourceLimitExceeded):
                runtime.resources.preflight(pid, ResourceUsage(tool_calls=1), source="test")

            assert runtime.process.get(pid).resource_usage.tool_calls == 1
        finally:
            runtime.close()

    def test_resource_usage_is_persisted_with_process_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db = f"{temp_dir}/runtime.sqlite"
            runtime = Runtime.open(db)
            try:
                pid = runtime.process.spawn(image="base-agent:v0", goal="persist")
                runtime.resources.charge(pid, ResourceUsage(llm_calls=1, llm_total_tokens=9), source="test")
            finally:
                runtime.close()

            reopened = Runtime.open(db)
            try:
                process = reopened.process.get(pid)
                assert process.resource_usage.llm_calls == 1
                assert process.resource_usage.llm_total_tokens == 9
            finally:
                reopened.close()

    def test_child_budget_cannot_exceed_parent_remaining_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_tool_calls=2),
            )
            runtime.resources.charge(parent, ResourceUsage(tool_calls=1), source="test")

            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(
                    parent,
                    goal="oversized child",
                    resource_budget=ResourceBudget(max_tool_calls=2),
                )
        finally:
            runtime.close()

    def test_child_budget_reservations_prevent_sibling_overcommit(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_tool_calls=5, max_child_processes=None),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="child",
                resource_budget=ResourceBudget(max_tool_calls=3, max_child_processes=None),
            )

            assert runtime.resources.remaining_budget(parent).max_tool_calls == 2
            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(
                    parent,
                    goal="oversized sibling",
                    resource_budget=ResourceBudget(max_tool_calls=3, max_child_processes=None),
                )

            runtime.resources.charge(child, ResourceUsage(tool_calls=2), source="test")
            assert runtime.resources.remaining_budget(parent).max_tool_calls == 2
            runtime.process.spawn_child(
                parent,
                goal="sibling",
                resource_budget=ResourceBudget(max_tool_calls=2, max_child_processes=None),
            )
        finally:
            runtime.close()

    def test_child_exit_releases_unused_reserved_budget(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_tool_calls=3, max_child_processes=None),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="child",
                resource_budget=ResourceBudget(max_tool_calls=3, max_child_processes=None),
            )
            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(
                    parent,
                    goal="blocked",
                    resource_budget=ResourceBudget(max_tool_calls=1, max_child_processes=None),
                )

            runtime.process.exit(child)
            runtime.process.spawn_child(
                parent,
                goal="after release",
                resource_budget=ResourceBudget(max_tool_calls=3, max_child_processes=None),
            )
        finally:
            runtime.close()

    def test_child_count_budget_denial_leaves_no_partial_child(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_child_processes=0),
            )

            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(parent, goal="blocked child")

            assert runtime.process.list_children(parent) == []
            assert runtime.process.get(parent).resource_usage.child_processes == 0
            assert runtime.store.list_resource_reservations(parent_pid=parent) == []
        finally:
            runtime.close()

    def test_context_materialization_total_tokens_are_cumulative(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="context budget",
            )
            first = runtime.memory.create_object(
                pid=pid,
                object_type=ObjectType.OBSERVATION,
                payload={"text": "first"},
                metadata=ObjectMetadata(token_estimate=3),
            )
            second = runtime.memory.create_object(
                pid=pid,
                object_type=ObjectType.OBSERVATION,
                payload={"text": "second"},
                metadata=ObjectMetadata(token_estimate=3),
            )
            first_preview = runtime.memory.materialize_context(
                pid,
                runtime.memory.create_view(pid, [first]),
                charge_resources=False,
            )
            second_preview = runtime.memory.materialize_context(
                pid,
                runtime.memory.create_view(pid, [second]),
                charge_resources=False,
            )
            process = runtime.process.get(pid)
            process.resource_budget = ResourceBudget(
                max_context_materialization_tokens=max(first_preview.token_count, second_preview.token_count) + 1,
                max_context_materialization_total_tokens=first_preview.token_count,
            )
            runtime.store.update_process(process)

            context = runtime.memory.materialize_context(pid, runtime.memory.create_view(pid, [first]))
            assert context.token_count == first_preview.token_count
            assert context.token_count > 3
            assert runtime.process.get(pid).resource_usage.context_materialized_tokens == first_preview.token_count

            exhausted = runtime.memory.materialize_context(pid, runtime.memory.create_view(pid, [second]))
            assert exhausted.token_count == 0
            assert second.oid in exhausted.omitted_objects
            assert runtime.process.get(pid).resource_usage.context_materialized_tokens == first_preview.token_count
        finally:
            runtime.close()

    def test_jsonrpc_remaining_budget_counts_request_and_response_together(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="jsonrpc budget",
                resource_budget=ResourceBudget(max_jsonrpc_bytes=100),
            )
            runtime.resources.charge(
                pid,
                ResourceUsage(jsonrpc_request_bytes=40, jsonrpc_response_bytes=40),
                source="test",
            )

            assert runtime.resources.remaining_budget(pid).max_jsonrpc_bytes == 20
            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(
                    pid,
                    goal="oversized jsonrpc child",
                    resource_budget=ResourceBudget(max_jsonrpc_bytes=30),
                )
        finally:
            runtime.close()

    def test_mcp_remaining_budget_counts_request_and_response_together(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="mcp budget",
                resource_budget=ResourceBudget(max_mcp_bytes=100),
            )
            runtime.resources.charge(
                pid,
                ResourceUsage(mcp_request_bytes=30, mcp_response_bytes=50),
                source="test",
            )

            assert runtime.resources.remaining_budget(pid).max_mcp_bytes == 20
            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(
                    pid,
                    goal="oversized mcp child",
                    resource_budget=ResourceBudget(max_mcp_bytes=30),
                )
        finally:
            runtime.close()

    def test_child_process_creation_is_hierarchical_usage(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_child_processes=1),
            )
            child = runtime.process.spawn_child(parent, goal="child")

            assert runtime.process.get(parent).resource_usage.child_processes == 1
            with pytest.raises(ResourceLimitExceeded):
                runtime.process.spawn_child(child, goal="grandchild")
            assert runtime.process.get(parent).resource_usage.child_processes == 1
        finally:
            runtime.close()

    def test_parent_budget_exceedance_kills_parent_subtree(self) -> None:
        runtime = Runtime.open("local")
        try:
            parent = runtime.process.spawn(
                image="base-agent:v0",
                goal="parent",
                resource_budget=ResourceBudget(max_llm_total_tokens=10),
            )
            child = runtime.process.spawn_child(
                parent,
                goal="child",
                resource_budget=ResourceBudget(max_llm_total_tokens=None),
            )

            with pytest.raises(ResourceLimitExceeded):
                runtime.resources.charge(
                    child,
                    ResourceUsage(llm_total_tokens=11),
                    source="test",
                    allow_overage=True,
                    kill_on_exceed=True,
                )

            assert runtime.process.get(parent).status.value == "killed"
            assert runtime.process.get(child).status.value == "killed"
        finally:
            runtime.close()
