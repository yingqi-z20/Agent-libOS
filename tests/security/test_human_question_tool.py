from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import pytest
import asyncio
import json
import tempfile
import threading
import time
from agent_libos import Runtime
from agent_libos.models.exceptions import HumanResponseRequired, ProcessError, ValidationError
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectRollbackStatus,
    HumanRequestStatus,
    ProcessSignal,
    ProcessStatus,
)
from agent_libos.substrate import ProviderEffectNotStarted

HUMAN_COLLABORATION_SKILL = 'agent-libos-human-collaboration'


class TestHumanQuestionTool:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')
        self.human_output: list[str] = []
        self.runtime.substrate.human.output_sink = self.human_output.append

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_human_process_interrupt_signal_rejects_message_semantics(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='reject ambiguous interrupt signal')

        with pytest.raises(ProcessError, match='durable interrupt process message'):
            self.runtime.human.interrupt(pid, ProcessSignal.INTERRUPT, {'reason': 'read this first'})

        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert self.runtime.messages.unread(pid) == []

    def test_ask_human_tool_waits_and_returns_answer_after_queue_processing(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='ask a human')
        self.runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
        prompts: list[str] = []
        with pytest.raises(HumanResponseRequired) as raised:
            self.runtime.tools.call(pid, 'ask_human', {'question': 'Which color should I use?', 'context': {'artifact': 'draft'}})
        pending = self.runtime.human.pending()[0]
        self.runtime.substrate.human.input_reader = lambda prompt: prompts.append(prompt) or 'blue'
        processed = self.runtime.human.drain_terminal_queue()
        result = self.runtime.tools.call(pid, 'ask_human', {'question': 'Which color should I use?', 'context': {'artifact': 'draft'}})
        assert raised.value.request_id == pending.request_id
        assert pending.payload['type'] == 'question'
        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        assert processed[0].status == HumanRequestStatus.APPROVED
        assert processed[0].decision['answer'] == 'blue'
        assert 'artifact' in prompts[0]
        assert result.ok, result.error
        assert result.payload['answer'] == 'blue'
        assert result.payload['request_id'] == pending.request_id
        effects = [
            effect
            for effect in self.runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human' and effect.operation == 'read'
        ]
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].information_flow
        persisted_metadata = json.dumps(effects[0].provider_metadata, sort_keys=True)
        assert 'Which color should I use?' not in persisted_metadata
        assert 'blue' not in persisted_metadata

    def test_auto_answer_write_is_recorded_without_persisting_prompt_or_answer(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='auto answer ledger')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Sensitive deployment secret?'},
            blocking=True,
        )

        processed = self.runtime.human.drain_terminal_queue(auto_answer='private-answer')

        assert processed[0].request_id == request_id
        assert processed[0].status == HumanRequestStatus.APPROVED
        assert processed[0].decision['answer'] == 'private-answer'
        effects = [
            effect
            for effect in self.runtime.store.list_external_effects(pid=pid)
            if effect.provider == 'human' and effect.operation == 'write'
        ]
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        metadata = json.dumps(effects[0].provider_metadata, sort_keys=True)
        assert 'Sensitive deployment secret?' not in metadata
        assert 'private-answer' not in metadata

    def test_auto_answer_classifier_failure_uses_conservative_fallback_without_reprompt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='auto answer sink failure')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Prompt exactly once?'},
            blocking=True,
        )

        def fail_classifier(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError('classifier unavailable')

        monkeypatch.setattr(self.runtime.human.provider, 'classify_external_effect', fail_classifier)
        processed = self.runtime.human.drain_terminal_queue(auto_answer='yes')

        assert processed[0].request_id == request_id
        assert processed[0].decision['answer'] == 'yes'
        assert self.runtime.human.drain_terminal_queue(auto_answer='yes') == []
        assert len(self.human_output) == 1
        effects = [effect for effect in self.runtime.store.list_external_effects(pid=pid) if effect.provider == 'human']
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].provider_metadata['classification_fallback'] == 'post_effect_failure'

    @pytest.mark.parametrize('certified_not_started', [False, True])
    def test_terminal_read_failure_is_retryable_only_when_provider_certifies_not_started(
        self,
        certified_not_started: bool,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='terminal read failure')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Do not persist this prompt'},
            blocking=True,
        )

        provider_calls = 0

        def fail_read(_prompt: str) -> str:
            nonlocal provider_calls
            provider_calls += 1
            if certified_not_started:
                raise ProviderEffectNotStarted('read did not start')
            raise RuntimeError('ambiguous terminal read failure')

        self.runtime.substrate.human.input_reader = fail_read
        expected = ProviderEffectNotStarted if certified_not_started else RuntimeError
        with pytest.raises(expected):
            self.runtime.human.process_next_terminal()

        request = self.runtime.human.get(request_id)
        assert request.status == (
            HumanRequestStatus.PENDING
            if certified_not_started
            else HumanRequestStatus.CANCELLED
        )
        assert provider_calls == 1
        effects = [effect for effect in self.runtime.store.list_external_effects(pid=pid) if effect.provider == 'human']
        if certified_not_started:
            assert effects == []
        else:
            assert request.decision == {
                'provider_outcome': 'unknown',
                'automatic_retry_disabled': True,
                'manual_recovery_required': True,
                'operation': 'read',
                'purpose': 'text_answer',
                'error_type': 'RuntimeError',
            }
            assert self.runtime.process.get(pid).status == ProcessStatus.PAUSED
            assert self.runtime.human.process_next_terminal() is None
            assert provider_calls == 1
            assert len(effects) == 1
            assert effects[0].effect_state == 'finalized'
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            metadata = json.dumps(effects[0].provider_metadata, sort_keys=True)
            assert 'Do not persist this prompt' not in metadata
            assert 'ambiguous terminal read failure' not in metadata

    @pytest.mark.parametrize('certified_not_started', [False, True])
    def test_terminal_write_failure_is_retryable_only_when_provider_certifies_not_started(
        self,
        certified_not_started: bool,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='terminal write failure')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Display this question once'},
            blocking=True,
        )
        provider_calls = 0

        def fail_write(_prompt: str) -> None:
            nonlocal provider_calls
            provider_calls += 1
            if certified_not_started:
                raise ProviderEffectNotStarted('write did not start')
            raise RuntimeError('ambiguous terminal write failure')

        self.runtime.substrate.human.output_sink = fail_write
        expected = ProviderEffectNotStarted if certified_not_started else RuntimeError
        with pytest.raises(expected):
            self.runtime.human.process_next_terminal(auto_answer='yes')

        request = self.runtime.human.get(request_id)
        assert request.status == (
            HumanRequestStatus.PENDING
            if certified_not_started
            else HumanRequestStatus.CANCELLED
        )
        assert provider_calls == 1
        if not certified_not_started:
            assert request.decision is not None
            assert request.decision['provider_outcome'] == 'unknown'
            assert request.decision['automatic_retry_disabled'] is True
            assert self.runtime.human.process_next_terminal(auto_answer='yes') is None
            assert provider_calls == 1

    def test_human_output_not_started_abandons_effect_and_restores_one_time_authority(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='output did not start')
        capability = self.runtime.capability.grant_once(
            pid,
            'human:owner',
            [CapabilityRight.WRITE],
            issued_by='test',
        )

        def fail_write(_message: str) -> None:
            raise ProviderEffectNotStarted('write did not start')

        self.runtime.substrate.human.output_sink = fail_write
        with pytest.raises(ProviderEffectNotStarted):
            self.runtime.human.output(pid, 'retryable output')

        request = self.runtime.human.list(pid=pid)[0]
        assert request.status == HumanRequestStatus.CANCELLED
        assert self.runtime.store.list_external_effects(pid=pid) == []
        restored = self.runtime.store.get_capability(capability.cap_id)
        assert restored is not None and restored.uses_remaining == 1

    def test_one_time_ask_human_capability_is_consumed_after_question_is_queued(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='ask once')
        self.runtime.capability.grant_once(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
        with pytest.raises(HumanResponseRequired):
            self.runtime.tools.call(pid, 'ask_human', {'question': 'Proceed?'})
        pending = self.runtime.human.pending()[0]
        assert not self.runtime.capability.check(pid, 'human:owner', CapabilityRight.WRITE)
        self.runtime.substrate.human.input_reader = lambda _prompt: 'yes'
        self.runtime.human.drain_terminal_queue()
        result = self.runtime.tools.call(pid, 'ask_human', {'question': 'Proceed?'})
        assert result.ok, result.error
        assert result.payload['request_id'] == pending.request_id
        assert result.payload['answer'] == 'yes'

    def test_ask_human_tool_cannot_bypass_human_capability(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='ask without authority')
        denied = self.runtime.tools.call(pid, 'ask_human', {'question': 'May I ask?'})
        assert not denied.ok
        assert 'lacks write on human:owner' in (denied.error or '')
        assert self.runtime.human.pending() == []
        assert 'human.query' not in self._audit_actions()

    def test_async_runtime_resumes_human_question_with_answer(self) -> None:
        self.runtime.llm.client = PlannedActionClient([{'action': 'ask_human', 'question': 'What deployment window should I use?'}, {'action': 'process_exit', 'payload': {'done': True}}])
        pid = self.runtime.process.spawn(
            image='base-agent:v0',
            goal='ask then exit',
            authority_manifest=_human_manifest(),
        )
        self.runtime.skills.activate_skill(
            pid,
            HUMAN_COLLABORATION_SKILL,
            actor=pid,
        )
        results = asyncio.run(self.runtime.arun_until_idle(max_quanta=4, human_auto_answer='Sunday 02:00 UTC'))
        process = self.runtime.process.get(pid)
        assert process.status == ProcessStatus.EXITED, (process.status_message, results)
        assert self.runtime.llm.client.calls == 2
        assert results[0]['waiting_human']
        assert 'action' not in results[0]
        ask_result = next((result for result in results if _action_name(result) == 'ask_human'))
        assert 'answer' in ask_result['result']['payload'], ask_result
        assert ask_result['result']['payload']['answer'] == 'Sunday 02:00 UTC'
        assert self.runtime.human.list(pid)[0].decision['answer'] == 'Sunday 02:00 UTC'

    def test_question_cannot_be_approved_without_a_typed_answer(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='question needs answer')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Which environment?'},
            blocking=True,
        )

        with pytest.raises(ValidationError, match='answer'):
            self.runtime.human.approve(request_id, {'approved': True})
        with pytest.raises(ValidationError, match='answer'):
            self.runtime.human.approve(request_id, {'approved': True, 'answer': '   '})

        assert self.runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN

    def test_multiple_blocking_requests_keep_process_waiting_until_all_are_decided(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='wait for all questions')
        first = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'First?'},
            blocking=True,
        )
        second = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Second?'},
            blocking=True,
        )

        self.runtime.human.approve(first, {'approved': True, 'answer': 'one'})
        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN

        self.runtime.human.approve(second, {'approved': True, 'answer': 'two'})
        assert self.runtime.process.get(pid).status == ProcessStatus.RUNNABLE

    @pytest.mark.parametrize('terminal_action', ['cancel', 'exit'])
    def test_terminal_process_cancels_pending_human_requests(self, terminal_action: str) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='terminal request cleanup')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Too late?'},
            blocking=True,
        )

        if terminal_action == 'cancel':
            self.runtime.process.cancel(pid, 'test cancellation')
        else:
            self.runtime.process.exit(pid, message='test exit')

        request = self.runtime.human.get(request_id)
        assert request.status == HumanRequestStatus.CANCELLED
        assert request.decision is not None
        assert request.decision['reason']
        with pytest.raises(ValidationError, match='not pending'):
            self.runtime.human.approve(request_id, {'approved': True, 'answer': 'late'})

    def test_signal_audit_failure_rolls_back_terminal_state_event_and_human_request(self, monkeypatch) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='atomic terminal signal')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'Still pending after rollback?'},
            blocking=True,
        )
        before_signal_events = [
            event.event_id
            for event in self.runtime.events.list(target=pid)
            if event.type.value == 'process_signal'
        ]
        original_record = self.runtime.audit.record

        def fail_signal_audit(*args, **kwargs):
            if kwargs.get('action') == 'process.signal':
                raise RuntimeError('injected signal audit failure')
            return original_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', fail_signal_audit)
        with pytest.raises(RuntimeError, match='injected signal audit failure'):
            self.runtime.process.cancel(pid, 'rollback the whole signal')

        assert self.runtime.process.get(pid).status == ProcessStatus.WAITING_HUMAN
        assert self.runtime.human.get(request_id).status == HumanRequestStatus.PENDING
        assert [
            event.event_id
            for event in self.runtime.events.list(target=pid)
            if event.type.value == 'process_signal'
        ] == before_signal_events

    def test_terminal_signal_attempts_notifier_when_process_finalizer_fails(self, monkeypatch) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='independent terminal cleanup')
        notified: list[str] = []

        monkeypatch.setattr(
            self.runtime.process,
            '_object_task_terminal_notifier',
            notified.append,
        )
        monkeypatch.setattr(
            self.runtime.process,
            '_finalize_terminal_process',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('injected finalizer failure')),
        )

        self.runtime.process.cancel(pid, 'terminal cleanup failure is post-commit')

        assert self.runtime.process.get(pid).status == ProcessStatus.KILLED
        assert notified == [pid]
        warnings = [
            record
            for record in self.runtime.audit.trace()
            if record.action == 'process.signal_finalize_failed'
        ]
        assert len(warnings) == 1
        assert warnings[0].decision['errors'][0]['phase'] == 'process_finalize'

    def test_process_exit_attempts_notifier_when_process_finalizer_fails(self, monkeypatch) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='independent exit cleanup')
        notified: list[str] = []
        monkeypatch.setattr(self.runtime.process, '_object_task_terminal_notifier', notified.append)
        monkeypatch.setattr(
            self.runtime.process,
            '_finalize_terminal_process',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('injected exit finalizer failure')),
        )

        self.runtime.process.exit(pid, message='exit cleanup failure is post-commit')

        assert self.runtime.process.get(pid).status == ProcessStatus.EXITED
        assert notified == [pid]
        warnings = [
            record
            for record in self.runtime.audit.trace()
            if record.action == 'process.exit_finalize_failed'
        ]
        assert len(warnings) == 1
        assert warnings[0].decision['errors'][0]['phase'] == 'process_finalize'

    def test_process_exit_attempts_finalizer_when_terminal_notifier_fails(self, monkeypatch) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='independent exit notification')
        finalized: list[str] = []
        monkeypatch.setattr(
            self.runtime.process,
            '_object_task_terminal_notifier',
            lambda _pid: (_ for _ in ()).throw(RuntimeError('injected exit notifier failure')),
        )
        monkeypatch.setattr(
            self.runtime.process,
            '_finalize_terminal_process',
            lambda process, **_kwargs: finalized.append(process.pid),
        )

        self.runtime.process.exit(pid, message='exit notifier failure is post-commit')

        assert self.runtime.process.get(pid).status == ProcessStatus.EXITED
        assert finalized == [pid]
        warnings = [
            record
            for record in self.runtime.audit.trace()
            if record.action == 'process.exit_finalize_failed'
        ]
        assert len(warnings) == 1
        assert warnings[0].decision['errors'][0]['phase'] == 'terminal_notify'

    def test_terminal_notifier_attempts_object_tasks_when_human_cancellation_fails(self, monkeypatch) -> None:
        notified: list[str] = []
        monkeypatch.setattr(
            self.runtime.human,
            'cancel_pending_for_process',
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('injected human failure')),
        )
        monkeypatch.setattr(self.runtime.object_tasks, 'notify_process_terminal', notified.append)

        with pytest.raises(RuntimeError, match='terminal process notification failed'):
            self.runtime._notify_process_terminal('agent_test')

        assert notified == ['agent_test']

    def test_terminal_exit_does_not_wait_for_blocked_human_provider_read(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='cancel blocked human read')
        request_id = self.runtime.human.query(
            pid,
            'owner',
            {'type': 'question', 'question': 'May block forever?'},
            blocking=True,
        )
        read_started = threading.Event()
        release_read = threading.Event()
        terminal_errors: list[BaseException] = []
        exit_errors: list[BaseException] = []

        def blocked_read(_prompt: str) -> str:
            read_started.set()
            assert release_read.wait(timeout=2)
            return 'late answer'

        def drain() -> None:
            try:
                self.runtime.human.process_next_terminal()
            except BaseException as exc:
                terminal_errors.append(exc)

        def exit_process() -> None:
            try:
                self.runtime.process.exit(pid, message='cancel pending question')
            except BaseException as exc:
                exit_errors.append(exc)

        self.runtime.substrate.human.input_reader = blocked_read
        terminal = threading.Thread(target=drain)
        terminal.start()
        assert read_started.wait(timeout=1)
        exiting = threading.Thread(target=exit_process)
        exiting.start()
        exiting.join(timeout=0.5)
        try:
            assert not exiting.is_alive()
            assert exit_errors == []
            assert self.runtime.process.get(pid).status == ProcessStatus.EXITED
            assert self.runtime.human.get(request_id).status == HumanRequestStatus.CANCELLED
        finally:
            release_read.set()
            terminal.join(timeout=2)
            exiting.join(timeout=2)
        assert len(terminal_errors) == 1
        assert isinstance(terminal_errors[0], ValidationError)

    def test_terminal_exit_does_not_wait_for_blocked_human_provider_write(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='exit during human output')
        self.runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
        write_started = threading.Event()
        release_write = threading.Event()
        output_errors: list[BaseException] = []
        exit_errors: list[BaseException] = []

        def blocked_write(_message: str) -> None:
            write_started.set()
            assert release_write.wait(timeout=2)

        def deliver() -> None:
            try:
                self.runtime.human.output(pid, 'committed before exit')
            except BaseException as exc:
                output_errors.append(exc)

        def exit_process() -> None:
            try:
                self.runtime.process.exit(pid, message='exit while output provider blocks')
            except BaseException as exc:
                exit_errors.append(exc)

        self.runtime.substrate.human.output_sink = blocked_write
        output = threading.Thread(target=deliver)
        output.start()
        assert write_started.wait(timeout=1)
        exiting = threading.Thread(target=exit_process)
        exiting.start()
        exiting.join(timeout=0.5)
        try:
            assert not exiting.is_alive()
            assert exit_errors == []
            assert self.runtime.process.get(pid).status == ProcessStatus.EXITED
        finally:
            release_write.set()
            output.join(timeout=2)
            exiting.join(timeout=2)
        assert output_errors == []
        assert self.runtime.human.list(pid)[0].status == HumanRequestStatus.DELIVERED

    def test_pending_ask_human_llm_action_survives_runtime_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = f'{temp_dir}/runtime.sqlite'
            runtime = Runtime.open(db_path)
            try:
                runtime.llm.client = PlannedActionClient([{'action': 'ask_human', 'question': 'Continue after reopen?'}])
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='ask then reopen',
                    authority_manifest=_human_manifest(),
                )
                runtime.skills.activate_skill(
                    pid,
                    HUMAN_COLLABORATION_SKILL,
                    actor=pid,
                )
                waiting = runtime.run_next_process_once()
                request_id = waiting['request_id']
                assert waiting['waiting_human']
                persisted_context = runtime.human.get(request_id).payload[
                    '_agent_libos_data_flow_context'
                ]
                source_oids = [
                    item['oid'] for item in persisted_context['source_refs']
                ]
                assert all(runtime.store.get_object(oid) is not None for oid in source_oids)
            finally:
                runtime.close()

            runtime = Runtime.open(db_path)
            try:
                runtime.llm.client = ExplodingClient()
                source_rows = {
                    oid: runtime.store.select_table_rows('objects', 'oid = ?', (oid,))
                    for oid in source_oids
                }
                assert all(source_rows.values()), source_rows
                assert all(
                    rows[0]['lifecycle_state'] in {'live', 'released'}
                    for rows in source_rows.values()
                ), {
                    oid: rows[0]['lifecycle_state']
                    for oid, rows in source_rows.items()
                }
                runtime.human.drain_terminal_queue(auto_answer='yes')

                resumed = runtime.run_next_process_once()

                assert resumed['resumed_after_human']
                assert resumed['action']['action'] == 'ask_human'
                assert resumed['result']['ok']
                assert resumed['result']['payload']['request_id'] == request_id
                assert resumed['result']['payload']['answer'] == 'yes'
                assert runtime.human.pending() == []
                assert [request.request_id for request in runtime.human.list(pid)] == [request_id]
            finally:
                runtime.close()

    def test_concurrent_identical_ask_human_calls_share_pending_request(self) -> None:
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='ask concurrently')
        self.runtime.capability.grant(pid, 'human:owner', [CapabilityRight.WRITE], issued_by='test')
        original_ask = self.runtime.human.ask

        def slow_ask(*args: object, **kwargs: object) -> str:
            time.sleep(0.05)
            return original_ask(*args, **kwargs)

        self.runtime.human.ask = slow_ask  # type: ignore[method-assign]
        barrier = threading.Barrier(2)

        def call() -> str:
            barrier.wait(timeout=2)
            with pytest.raises(HumanResponseRequired) as raised:
                self.runtime.tools.call(pid, 'ask_human', {'question': 'Same question?'})
            return raised.value.request_id

        with ThreadPoolExecutor(max_workers=2) as executor:
            request_ids = list(executor.map(lambda _: call(), range(2)))

        assert request_ids[0] == request_ids[1]
        assert [request.request_id for request in self.runtime.human.pending()] == [request_ids[0]]

    def _audit_actions(self) -> list[str]:
        return [record.action for record in self.runtime.audit.trace()]


def _human_manifest() -> dict[str, object]:
    return {
        'authorized_capabilities': [
            {
                'resource': 'human:owner',
                'rights': [CapabilityRight.WRITE.value],
            }
        ],
        'permitted_effects': ['human.*', 'llm.*'],
    }


class PlannedActionClient:

    def __init__(self, actions: list[dict[str, object]]):
        self.actions = list(actions)
        self.calls = 0

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        if not self.actions:
            raise AssertionError('no planned action remains')
        action = self.actions.pop(0)
        name = str(action['action'])
        args = {key: value for key, value in action.items() if key != 'action'}
        return LLMCompletion(content='', tool_calls=[{'id': f'human_question_{self.calls}', 'name': name, 'arguments': json.dumps(args)}])


class ExplodingClient:
    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        raise AssertionError('model should not be called while resuming a pending human action')

def _action_name(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    action = result.get('action')
    if isinstance(action, dict):
        return action.get('action')
    return None
