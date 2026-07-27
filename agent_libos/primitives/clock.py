from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import ValidationError
from agent_libos.models import (
    CapabilityDecision,
    CapabilityRight,
    EventType,
)
from agent_libos.ports import AuditPort, EventPort
from agent_libos.sdk import (
    ProtectedOperationEvidence,
    ProtectedOperationInvocation,
    ProtectedOperationSDK,
    ProviderPhase,
)
from agent_libos.substrate import ClockProvider, LocalClockProvider

_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
_CLOCK_NOW_RESOURCE = "clock:now"
_CLOCK_SLEEP_RESOURCE = "clock:sleep"


@dataclass(frozen=True)
class ClockNowResult:
    iso8601: str
    unix_seconds: float
    timezone: str


@dataclass(frozen=True)
class SleepResult:
    requested_seconds: float
    elapsed_seconds: float


class ClockPrimitive:
    """Clock primitive used by scheduler-facing tools."""

    FIXED_TIMEZONE_FALLBACKS = {
        "Asia/Shanghai": timezone(timedelta(hours=8), name="Asia/Shanghai"),
    }

    def __init__(
        self,
        capabilities: CapabilityManager,
        audit: AuditPort,
        events: EventPort,
        max_sleep_seconds: float = _TOOL_DEFAULTS.max_sleep_seconds,
        provider: ClockProvider | None = None,
        *,
        protected_operations: ProtectedOperationSDK,
    ):
        self.capabilities = capabilities
        self.audit = audit
        self.events = events
        self.protected_operations = protected_operations
        self.max_sleep_seconds = max_sleep_seconds
        self.provider = provider or LocalClockProvider()

    def now(self, pid: str, tz: str = _TOOL_DEFAULTS.clock_timezone) -> ClockNowResult:
        selected_tz = self._timezone(tz)
        resource = _CLOCK_NOW_RESOURCE
        operation_context = self._authorization_context(
            pid=pid,
            resource=resource,
            primitive="runtime.clock.now",
            operation="now",
            extra={"timezone": tz},
        )
        decision = self.capabilities.require(
            pid,
            resource,
            CapabilityRight.READ,
            operation_context,
            consume=False,
        )
        effect_context = {"timezone": tz, "resource": resource}
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=resource,
            decisions=(decision,),
            canonical_args=operation_context,
            observation=effect_context,
        )
        with self._protected().start("primitive.clock.now", invocation, provider=self.provider) as operation:
            current = operation.call(
                ProviderPhase("now", information_flow=True),
                self.provider.now,
                selected_tz,
            )
            result = ClockNowResult(
                iso8601=current.isoformat(),
                unix_seconds=current.timestamp(),
                timezone=tz,
            )
            return operation.complete(
                result,
                self._evidence(pid, "now", resource, result.__dict__),
                classification_context=effect_context,
                classification_result=result.__dict__,
            )

    def sleep(self, pid: str, seconds: float) -> SleepResult:
        duration = self._validate_sleep_duration(seconds)
        decision = self._authorize_sleep(pid, duration)
        effect_context = {"requested_seconds": duration, "resource": _CLOCK_SLEEP_RESOURCE}
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=_CLOCK_SLEEP_RESOURCE,
            decisions=(decision,),
            canonical_args=self._authorization_context(
                pid=pid,
                resource=_CLOCK_SLEEP_RESOURCE,
                primitive="runtime.clock.sleep",
                operation="sleep",
                extra={"requested_seconds": duration},
            ),
            observation=effect_context,
        )
        with self._protected().start("primitive.clock.sleep", invocation, provider=self.provider) as operation:
            started = operation.call(ProviderPhase("monotonic.start", information_flow=True), self.provider.monotonic)
            operation.call(ProviderPhase("sleep", information_flow=True), self.provider.sleep, duration)
            elapsed = operation.call(
                ProviderPhase("monotonic.end", information_flow=True),
                self.provider.monotonic,
            ) - started
            result = SleepResult(requested_seconds=duration, elapsed_seconds=elapsed)
            return operation.complete(
                result,
                self._evidence(pid, "sleep", _CLOCK_SLEEP_RESOURCE, result.__dict__),
                classification_context=effect_context,
                classification_result=result.__dict__,
            )

    async def asleep(self, pid: str, seconds: float) -> SleepResult:
        duration = self._validate_sleep_duration(seconds)
        decision = self._authorize_sleep(pid, duration)
        effect_context = {"requested_seconds": duration, "resource": _CLOCK_SLEEP_RESOURCE}
        invocation = ProtectedOperationInvocation(
            pid=pid,
            actor=pid,
            target=_CLOCK_SLEEP_RESOURCE,
            decisions=(decision,),
            canonical_args=self._authorization_context(
                pid=pid,
                resource=_CLOCK_SLEEP_RESOURCE,
                primitive="runtime.clock.sleep",
                operation="sleep",
                extra={"requested_seconds": duration},
            ),
            observation=effect_context,
        )
        with self._protected().start("primitive.clock.sleep", invocation, provider=self.provider) as operation:
            started = operation.call(ProviderPhase("monotonic.start", information_flow=True), self.provider.monotonic)
            await operation.acall(ProviderPhase("sleep", information_flow=True), self.provider.asleep, duration)
            elapsed = operation.call(
                ProviderPhase("monotonic.end", information_flow=True),
                self.provider.monotonic,
            ) - started
            result = SleepResult(requested_seconds=duration, elapsed_seconds=elapsed)
            return operation.complete(
                result,
                self._evidence(pid, "sleep", _CLOCK_SLEEP_RESOURCE, result.__dict__),
                classification_context=effect_context,
                classification_result=result.__dict__,
            )

    def _protected(self):
        return self.protected_operations

    def _evidence(
        self,
        pid: str,
        operation: str,
        target: str,
        result: dict[str, Any],
    ) -> ProtectedOperationEvidence:
        payload = {"adapter": "clock", "operation": operation, **result}
        return ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target=target,
            event_payload=payload,
            audit_action=f"primitive.clock.{operation}",
            audit_actor=pid,
            audit_target=target,
            audit_decision=result,
            effect_metadata=result,
        )

    def _validate_sleep_duration(self, seconds: float) -> float:
        duration = float(seconds)
        if not math.isfinite(duration):
            raise ValidationError("sleep seconds must be finite")
        if duration < 0:
            raise ValidationError("sleep seconds must be non-negative")
        if duration > self.max_sleep_seconds:
            raise ValidationError(f"sleep seconds exceeds max_sleep_seconds={self.max_sleep_seconds}")
        return duration

    def _timezone(self, tz: str):
        if tz.upper() == "UTC":
            return timezone.utc
        try:
            return ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            if tz in self.FIXED_TIMEZONE_FALLBACKS:
                return self.FIXED_TIMEZONE_FALLBACKS[tz]
            raise ValidationError(f"unknown timezone: {tz}") from exc

    def _authorize_sleep(self, pid: str, duration: float) -> CapabilityDecision:
        return self.capabilities.require(
            pid,
            _CLOCK_SLEEP_RESOURCE,
            CapabilityRight.READ,
            self._authorization_context(
                pid=pid,
                resource=_CLOCK_SLEEP_RESOURCE,
                primitive="runtime.clock.sleep",
                operation="sleep",
                extra={"requested_seconds": duration},
            ),
            consume=False,
        )

    def _authorization_context(
        self,
        *,
        pid: str,
        resource: str,
        primitive: str,
        operation: str,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "pid": pid,
            "primitive": primitive,
            "operation": operation,
            "resource": resource,
            "right": CapabilityRight.READ.value,
            **extra,
        }
