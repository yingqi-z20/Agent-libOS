from __future__ import annotations
import asyncio
import pytest
import time
from datetime import datetime
from agent_libos import Runtime
from agent_libos.models import CapabilityRight, EventType, ExternalEffectClassification, ExternalEffectRollbackClass, ExternalEffectRollbackStatus
from agent_libos.models.exceptions import CapabilityDenied, ValidationError
from agent_libos.substrate import ProviderEffectNotStarted

class TestClockTool:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_current_time_tool_uses_clock_primitive(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='check time')
        self.runtime.capability.grant(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
        result = self.runtime.tools.call(pid, 'get_current_time', {'timezone': 'UTC'})
        assert result.ok, result.error
        assert result.payload['timezone'] == 'UTC'
        parsed = datetime.fromisoformat(result.payload['iso8601'])
        assert parsed.tzinfo is not None
        assert 'primitive.clock.now' in self._audit_actions()
        assert any((event.target == 'clock:now' and event.payload.get('operation') == 'now' for event in self.runtime.events.list()))

    def test_clock_post_provider_event_failure_leaves_durable_unknown_effect_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='durable clock effect intent')
        self.runtime.capability.grant(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
        provider = RecordingClockProvider()
        self.runtime.clock.provider = provider
        original_emit = self.runtime.events.emit

        def fail_now_event(event_type: EventType | str, *args: object, **kwargs: object) -> object:
            if EventType(event_type) == EventType.EXTERNAL_READ and kwargs.get('target') == 'clock:now':
                raise RuntimeError('injected clock result event failure')
            return original_emit(event_type, *args, **kwargs)

        monkeypatch.setattr(self.runtime.events, 'emit', fail_now_event)
        with pytest.raises(RuntimeError, match='injected clock result event failure'):
            self.runtime.clock.now(pid, tz='UTC')

        assert provider.now_calls == 1
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effects[0].provider_metadata['effect_state'] == 'pending'

    def test_current_time_tool_supports_iana_timezone(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='check shanghai time')
        self.runtime.capability.grant(pid, 'clock:*', [CapabilityRight.READ], issued_by='test')
        result = self.runtime.tools.call(pid, 'get_current_time', {'timezone': 'Asia/Shanghai'})
        assert result.ok, result.error
        assert result.payload['timezone'] == 'Asia/Shanghai'
        assert '+08:00' in result.payload['iso8601']

    @pytest.mark.parametrize('timezone_name', ['/etc/passwd', '../UTC'])
    def test_clock_primitive_normalizes_invalid_zoneinfo_keys(
        self,
        timezone_name: str,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='invalid timezone')

        with pytest.raises(ValidationError, match='unknown timezone'):
            self.runtime.clock.now(pid, tz=timezone_name)

        assert 'primitive.clock.now' not in self._audit_actions()

    def test_sleep_tool_uses_clock_primitive(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='sleep briefly')
        self.runtime.capability.grant(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        started = time.monotonic()
        result = self.runtime.tools.call(pid, 'sleep', {'seconds': 0.02})
        elapsed = time.monotonic() - started
        assert result.ok, result.error
        assert result.payload['requested_seconds'] == 0.02
        assert result.payload['elapsed_seconds'] >= 0.0
        assert elapsed >= 0.015
        assert 'primitive.clock.sleep' in self._audit_actions()
        assert any((event.target == 'clock:sleep' and event.payload.get('operation') == 'sleep' for event in self.runtime.events.list()))
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].operation == 'sleep'
        assert effects[0].information_flow

    def test_sleep_tool_rejects_unbounded_duration(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='sleep too long')
        result = self.runtime.tools.call(pid, 'sleep', {'seconds': 61})
        assert not result.ok
        assert 'Invalid arguments' in (result.error or '')
        assert 'primitive.clock.sleep' not in self._audit_actions()

    def test_clock_primitive_rejects_non_finite_sleep_duration(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='sleep nan')
        with pytest.raises(ValidationError, match='finite'):
            self.runtime.clock.sleep(pid, float('nan'))
        assert 'primitive.clock.sleep' not in self._audit_actions()

    def test_clock_now_requires_capability_before_provider_read(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock now denied')
        provider = RecordingClockProvider()
        self.runtime.clock.provider = provider

        with pytest.raises(CapabilityDenied, match='clock:now'):
            self.runtime.clock.now(pid, tz='UTC')

        assert provider.now_calls == 0
        assert 'primitive.clock.now' not in self._audit_actions()
        assert not any(event.target == 'clock:now' for event in self.runtime.events.list())

    def test_clock_sleep_requires_capability_before_provider_wait(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock sleep denied')
        provider = RecordingClockProvider()
        self.runtime.clock.provider = provider

        with pytest.raises(CapabilityDenied, match='clock:sleep'):
            self.runtime.clock.sleep(pid, 0.0)
        with pytest.raises(CapabilityDenied, match='clock:sleep'):
            asyncio.run(self.runtime.clock.asleep(pid, 0.0))

        assert provider.monotonic_calls == 0
        assert provider.sleep_calls == []
        assert provider.asleep_calls == []
        assert 'primitive.clock.sleep' not in self._audit_actions()
        assert not any(event.target == 'clock:sleep' for event in self.runtime.events.list())

    def test_clock_tools_cannot_bypass_missing_primitive_capability(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock tools denied')

        now = self.runtime.tools.call(pid, 'get_current_time', {'timezone': 'UTC'})
        slept = self.runtime.tools.call(pid, 'sleep', {'seconds': 0.0})

        assert not now.ok
        assert not slept.ok
        assert now.payload['error']['code'] == 'permission_denied'
        assert slept.payload['error']['code'] == 'permission_denied'
        assert (now.error or '').startswith('permission_denied: CapabilityDenied')
        assert (slept.error or '').startswith('permission_denied: CapabilityDenied')
        assert 'clock:now' not in (now.error or '')
        assert 'clock:sleep' not in (slept.error or '')
        assert 'primitive.clock.now' not in self._audit_actions()
        assert 'primitive.clock.sleep' not in self._audit_actions()

    def test_clock_one_time_capabilities_are_consumed_after_success(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock one-shot')
        self.runtime.capability.grant_once(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
        self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')

        now = self.runtime.clock.now(pid, tz='UTC')
        slept = self.runtime.clock.sleep(pid, 0.0)

        assert now.timezone == 'UTC'
        assert slept.requested_seconds == 0.0
        with pytest.raises(CapabilityDenied, match='clock:now'):
            self.runtime.clock.now(pid, tz='UTC')
        with pytest.raises(CapabilityDenied, match='clock:sleep'):
            self.runtime.clock.sleep(pid, 0.0)

    def test_clock_one_time_now_is_restored_when_provider_fails_before_read(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock failure restore')
        cap = self.runtime.capability.grant_once(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
        provider = RecordingClockProvider(fail_now=True)
        self.runtime.clock.provider = provider

        with pytest.raises(RuntimeError, match='clock provider failed'):
            self.runtime.clock.now(pid, tz='UTC')

        restored = self.runtime.store.get_capability(cap.cap_id)
        assert restored is not None
        assert restored.uses_remaining == 1
        assert self.runtime.capability.check(pid, 'clock:now', CapabilityRight.READ)
        assert 'primitive.clock.now' not in self._audit_actions()

    def test_clock_ambiguous_provider_failure_consumes_one_time_authority_and_records_effect(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock ambiguous failure')
        cap = self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        provider = RecordingClockProvider(fail_sleep_unknown=True)
        self.runtime.clock.provider = provider

        with pytest.raises(RuntimeError, match='sleep may have completed'):
            self.runtime.clock.sleep(pid, 0.0)

        consumed = self.runtime.store.get_capability(cap.cap_id)
        assert consumed is not None
        assert consumed.uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].operation == 'sleep'
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effects[0].provider_metadata['outcome'] == 'unknown_after_provider_exception'

    def test_clock_async_cancellation_commits_reservation_and_records_unknown_effect(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='cancel async sleep')
        cap = self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        provider = BlockingAsyncClockProvider()
        self.runtime.clock.provider = provider

        async def cancel_after_provider_starts() -> None:
            task = asyncio.create_task(self.runtime.clock.asleep(pid, 10.0))
            await provider.started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(cancel_after_provider_starts())

        consumed = self.runtime.store.get_capability(cap.cap_id)
        assert consumed is not None
        assert consumed.uses_remaining == 0
        reservations = self.runtime.store.select_table_rows(
            'capability_use_reservations',
            'cap_id = ?',
            [cap.cap_id],
        )
        assert reservations[0]['status'] == 'committed'
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].operation == 'sleep'
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN

    @pytest.mark.parametrize('async_mode', [False, True])
    def test_post_sleep_measurement_not_started_keeps_unknown_sleep_effect(
        self,
        async_mode: bool,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='post sleep measurement failure')
        cap = self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        provider = PostSleepMeasurementFailureClockProvider()
        self.runtime.clock.provider = provider

        with pytest.raises(ProviderEffectNotStarted, match='measurement did not start'):
            if async_mode:
                asyncio.run(self.runtime.clock.asleep(pid, 0.0))
            else:
                self.runtime.clock.sleep(pid, 0.0)

        consumed = self.runtime.store.get_capability(cap.cap_id)
        assert consumed is not None and consumed.uses_remaining == 0
        assert provider.sleep_calls + provider.asleep_calls == [0.0]
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].operation == 'sleep'
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effects[0].provider_metadata['effect_state'] == 'finalized'

    @pytest.mark.parametrize('async_mode', [False, True])
    @pytest.mark.parametrize('certified_not_started', [False, True])
    def test_initial_sleep_measurement_failure_has_durable_boundary_semantics(
        self,
        async_mode: bool,
        certified_not_started: bool,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='initial sleep measurement failure')
        cap = self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        provider = InitialMeasurementFailureClockProvider(certified_not_started=certified_not_started)
        self.runtime.clock.provider = provider
        expected_error = ProviderEffectNotStarted if certified_not_started else RuntimeError

        with pytest.raises(expected_error, match='initial measurement'):
            if async_mode:
                asyncio.run(self.runtime.clock.asleep(pid, 0.0))
            else:
                self.runtime.clock.sleep(pid, 0.0)

        capability = self.runtime.store.get_capability(cap.cap_id)
        assert capability is not None
        effects = self.runtime.store.list_external_effects(pid=pid)
        if certified_not_started:
            assert capability.uses_remaining == 1
            assert effects == []
        else:
            assert capability.uses_remaining == 0
            assert len(effects) == 1
            assert effects[0].effect_state == 'finalized'
            assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
            assert effects[0].information_flow
        assert provider.sleep_calls == []
        assert provider.asleep_calls == []

    @pytest.mark.parametrize('async_mode', [False, True])
    def test_sleep_provider_not_started_after_initial_measurement_keeps_effect(
        self,
        async_mode: bool,
    ) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='sleep provider not started')
        cap = self.runtime.capability.grant_once(pid, 'clock:sleep', [CapabilityRight.READ], issued_by='test')
        provider = SleepNotStartedClockProvider()
        self.runtime.clock.provider = provider

        with pytest.raises(ProviderEffectNotStarted, match='sleep body did not start'):
            if async_mode:
                asyncio.run(self.runtime.clock.asleep(pid, 0.0))
            else:
                self.runtime.clock.sleep(pid, 0.0)

        capability = self.runtime.store.get_capability(cap.cap_id)
        assert capability is not None and capability.uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effects[0].information_flow

    def test_clock_post_effect_classifier_failure_records_conservative_effect(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='clock classifier failure')
        self.runtime.capability.grant(pid, 'clock:now', [CapabilityRight.READ], issued_by='test')
        self.runtime.clock.provider = RecordingClockProvider(fail_classifier=True)

        result = self.runtime.clock.now(pid, tz='UTC')

        assert result.timezone == 'UTC'
        effect = self.runtime.store.list_external_effects(pid=pid)[0]
        assert effect.rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effect.provider_metadata['classification_fallback'] == 'post_effect_failure'

    def test_clock_tools_are_in_process_tool_table(self) -> None:
        pid = self.runtime.process.spawn(image='base-agent:v0', goal='time tools')
        names = {schema['function']['name'] for schema in self.runtime.tools.openai_tool_schemas(pid)}
        assert 'get_current_time' not in names
        assert 'sleep' not in names

        result = self.runtime.skills.activate_skill(
            pid,
            'agent-libos-runtime-session',
            actor=pid,
        )

        assert result['authority_changed'] is False
        names = {schema['function']['name'] for schema in self.runtime.tools.openai_tool_schemas(pid)}
        assert 'get_current_time' in names
        assert 'sleep' in names

    def _audit_actions(self) -> list[str]:
        return [record.action for record in self.runtime.audit.trace()]


class RecordingClockProvider:
    def __init__(
        self,
        *,
        fail_now: bool = False,
        fail_sleep_unknown: bool = False,
        fail_classifier: bool = False,
    ) -> None:
        self.fail_now = fail_now
        self.fail_sleep_unknown = fail_sleep_unknown
        self.fail_classifier = fail_classifier
        self.now_calls = 0
        self.monotonic_calls = 0
        self.sleep_calls: list[float] = []
        self.asleep_calls: list[float] = []

    def now(self, timezone_):
        self.now_calls += 1
        if self.fail_now:
            raise ProviderEffectNotStarted('clock provider failed')
        return datetime(2040, 1, 2, 3, 4, 5, tzinfo=timezone_)

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        return float(self.monotonic_calls)

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        if self.fail_sleep_unknown:
            raise RuntimeError('sleep may have completed')

    async def asleep(self, seconds: float) -> None:
        self.asleep_calls.append(seconds)

    def classify_external_effect(self, operation: str, context: dict, result) -> ExternalEffectClassification:
        if self.fail_classifier:
            raise RuntimeError('clock classifier unavailable')
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
            rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
            state_mutation=False,
            information_flow=True,
            metadata={'operation': operation},
        )


class PostSleepMeasurementFailureClockProvider(RecordingClockProvider):
    def monotonic(self) -> float:
        self.monotonic_calls += 1
        if self.monotonic_calls == 2:
            raise ProviderEffectNotStarted('post-sleep measurement did not start')
        return float(self.monotonic_calls)


class InitialMeasurementFailureClockProvider(RecordingClockProvider):
    def __init__(self, *, certified_not_started: bool) -> None:
        super().__init__()
        self.certified_not_started = certified_not_started

    def monotonic(self) -> float:
        self.monotonic_calls += 1
        if self.certified_not_started:
            raise ProviderEffectNotStarted('initial measurement did not start')
        raise RuntimeError('initial measurement failed ambiguously')


class SleepNotStartedClockProvider(RecordingClockProvider):
    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        raise ProviderEffectNotStarted('sleep body did not start')

    async def asleep(self, seconds: float) -> None:
        self.asleep_calls.append(seconds)
        raise ProviderEffectNotStarted('sleep body did not start')


class BlockingAsyncClockProvider(RecordingClockProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def asleep(self, seconds: float) -> None:
        self.asleep_calls.append(seconds)
        self.started.set()
        await asyncio.Future()
