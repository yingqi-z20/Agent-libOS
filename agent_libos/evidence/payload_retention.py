from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from agent_libos.models.audit import AuditRecord
from agent_libos.models.external_effect import ExternalEffectRecord
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.models.llm import LLMCallRecord
from agent_libos.ports.audit import AuditPort
from agent_libos.utils.serde import bounded_json_loads, dumps, to_jsonable

if TYPE_CHECKING:
    from agent_libos.config.defaults import RuntimeDefaults


_RETENTION_KEY = "$agent_libos_payload_retention"
_RETENTION_SCHEMA_VERSION = 1
_SHA256_HEX_LENGTH = 64
_TERMINAL_EFFECT_TRANSACTION_STATES = frozenset(
    {"committed", "failed", "compensated"}
)
_TERMINAL_LLM_CALL_STATES = frozenset({"ok", "error"})
_LLM_PAYLOAD_FIELDS = (
    "messages",
    "tools",
    "response_content",
    "tool_calls",
    "reasoning",
    "raw_response",
    "error",
)
_IMAGE_ONLY_REQUEST_KEY = "image_only_request"
_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX = "image_only_request:"
_IMAGE_ONLY_REQUEST_SCHEMA_VERSION = 1
_IMAGE_ONLY_TRANSCRIPT_KEY = "image_only_transcript"
_IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION = 2
_TASK_RUN_TOOL_OUTPUT_REDACTION_KEY = (
    "$agent_libos_task_run_tool_output_redaction"
)
_TASK_RUN_TOOL_OUTPUT_REDACTION_SCHEMA_VERSION = 1
_TASK_RUN_HUMAN_REDACTION_KEY = "$agent_libos_task_run_human_redaction"
_TASK_RUN_HUMAN_REDACTION_SCHEMA_VERSION = 1
_TASK_RUN_HUMAN_TYPE_PATTERN = re.compile(r"[A-Za-z0-9_.:-]{1,128}")


def _validate_retention_age(name: str, value: int | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer or None")


def _validate_retention_limits(batch_limit: int, hard_limit: int) -> None:
    for name, value in (("batch_limit", batch_limit), ("hard_limit", hard_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"payload retention {name} must be positive")
    if batch_limit > hard_limit:
        raise ValueError("payload retention batch_limit exceeds hard_limit")


class PayloadRetentionTier(str, Enum):
    """Monotonic tiers for durable provider payloads."""

    FULL = "full"
    SUMMARY = "summary"
    HASH_ONLY = "hash_only"


class PayloadRetentionKind(str, Enum):
    LLM_CALL = "llm_call"
    EXTERNAL_EFFECT = "external_effect"


@dataclass(frozen=True, order=True)
class PayloadRetentionCursor:
    """Stable keyset cursor shared by retention scans."""

    created_at: str
    record_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.created_at, str) or not self.created_at:
            raise ValueError("payload retention cursor created_at must not be empty")
        if not isinstance(self.record_id, str) or not self.record_id:
            raise ValueError("payload retention cursor record_id must not be empty")


_RecordT = TypeVar("_RecordT", LLMCallRecord, ExternalEffectRecord)


@dataclass(frozen=True)
class PayloadRetentionPage(Generic[_RecordT]):
    records: tuple[_RecordT, ...]
    next_cursor: PayloadRetentionCursor | None = None
    # LLM scans identify every bounded candidate that is the actual latest call
    # for its (pid, purpose) chain. Runtime-dependency predicates then decide
    # whether that head carries Responses continuation state, an image_only
    # transcript, or no replay state at all. External-effect scans leave this
    # as ``None``. Keeping the classification beside the page prevents an N+1
    # latest-call lookup in the maintenance service.
    latest_llm_call_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not isinstance(
            self.next_cursor, PayloadRetentionCursor
        ):
            raise ValueError("payload retention next_cursor has an invalid type")
        if self.next_cursor is not None and not self.records:
            raise ValueError("empty payload retention page cannot have a next cursor")
        if self.latest_llm_call_ids is not None:
            if not isinstance(self.latest_llm_call_ids, frozenset) or any(
                not isinstance(call_id, str) or not call_id
                for call_id in self.latest_llm_call_ids
            ):
                raise ValueError("payload retention latest LLM call ids are invalid")
            record_ids = {
                str(record.call_id)
                for record in self.records
                if isinstance(record, LLMCallRecord)
            }
            if not self.latest_llm_call_ids.issubset(record_ids):
                raise ValueError(
                    "payload retention latest LLM call ids must belong to the page"
                )


@dataclass(frozen=True)
class PayloadRetentionPolicy:
    """Explicitly enabled, staged retention policy.

    ``FULL`` rows first move to ``SUMMARY``. A later maintenance pass may move
    an already summarized row to ``HASH_ONLY``. This deliberately prevents a
    single pass from skipping the summary stage.
    """

    enabled: bool = False
    summary_after_seconds: int | None = None
    hash_only_after_seconds: int | None = None
    batch_limit: int = 100
    hard_limit: int = 1_000

    @classmethod
    def from_runtime_defaults(
        cls,
        defaults: RuntimeDefaults,
    ) -> PayloadRetentionPolicy:
        return cls(
            enabled=defaults.payload_retention_enabled,
            summary_after_seconds=(
                defaults.payload_retention_summary_after_seconds
            ),
            hash_only_after_seconds=(
                defaults.payload_retention_hash_only_after_seconds
            ),
            batch_limit=defaults.payload_retention_page_size,
            hard_limit=defaults.payload_retention_page_hard_limit,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("payload retention enabled must be a boolean")
        for name in ("summary_after_seconds", "hash_only_after_seconds"):
            _validate_retention_age(name, getattr(self, name))
        _validate_retention_limits(self.batch_limit, self.hard_limit)
        if self.enabled and self.summary_after_seconds is None:
            raise ValueError(
                "enabled payload retention requires summary_after_seconds"
            )
        if (
            self.hash_only_after_seconds is not None
            and self.summary_after_seconds is None
        ):
            raise ValueError(
                "hash_only_after_seconds requires summary_after_seconds"
            )
        if (
            self.hash_only_after_seconds is not None
            and self.summary_after_seconds is not None
            and self.hash_only_after_seconds < self.summary_after_seconds
        ):
            raise ValueError(
                "hash_only_after_seconds must be at least summary_after_seconds"
            )


@dataclass(frozen=True)
class PayloadRetentionRequest:
    kind: PayloadRetentionKind
    dry_run: bool = True
    cursor: PayloadRetentionCursor | None = None
    limit: int | None = None
    actor: str = "host.retention"
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PayloadRetentionKind):
            raise ValueError("payload retention kind must be a PayloadRetentionKind")
        if not isinstance(self.dry_run, bool):
            raise ValueError("payload retention dry_run must be a boolean")
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ValueError("payload retention actor must not be empty")
        if self.limit is not None and (
            isinstance(self.limit, bool)
            or not isinstance(self.limit, int)
            or self.limit <= 0
        ):
            raise ValueError("payload retention request limit must be positive")


@dataclass(frozen=True)
class PayloadRetentionResult:
    kind: PayloadRetentionKind
    status: str
    dry_run: bool
    scanned: int = 0
    eligible: int = 0
    would_update: int = 0
    updated: int = 0
    conflicts: int = 0
    protected_nonterminal: int = 0
    protected_runtime_dependency: int = 0
    invalid_timestamp: int = 0
    already_retained: int = 0
    next_cursor: PayloadRetentionCursor | None = None
    candidate_set_sha256: str = ""
    audit_record_id: str | None = None


class PayloadRetentionStore(Protocol):
    """Typed backend contract for bounded, CAS-protected maintenance."""

    def transaction(self) -> AbstractContextManager[Any]:
        ...

    def scan_llm_call_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[LLMCallRecord]:
        ...

    def update_llm_call_payload_retention(
        self,
        record: LLMCallRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
    ) -> bool:
        ...

    def scan_external_effect_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[ExternalEffectRecord]:
        ...

    def update_external_effect_payload_retention(
        self,
        record: ExternalEffectRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        ...


class TaskRunLLMRedactionStore(Protocol):
    """Narrow, caller-transactional store boundary for terminal Run purge."""

    def list_task_run_llm_calls(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[LLMCallRecord]:
        ...

    def redact_task_run_llm_call_payload(
        self,
        record: LLMCallRecord,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_tier: str,
    ) -> bool:
        ...

    def list_task_run_llm_tool_outputs(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[dict[str, Any]]:
        ...

    def redact_task_run_llm_tool_output(
        self,
        *,
        run_id: str,
        pid: str,
        response_id: str,
        call_id: str,
        expected_output_sha256: str,
        redacted_output: str,
    ) -> bool:
        ...


class TaskRunExternalEffectRedactionStore(Protocol):
    """Run-linked external-effect boundary for caller-transactional purge.

    Implementations must enforce both the durable ``task_run_links`` evidence
    relationship and Process ``task_run_id`` membership on the read and on
    every CAS.  The caller owns the transaction spanning all returned rows.
    """

    def list_task_run_external_effects(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[ExternalEffectRecord]:
        ...

    def redact_task_run_external_effect_payload(
        self,
        record: ExternalEffectRecord,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_tier: str,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        ...


class TaskRunHumanRedactionStore(Protocol):
    """Run-linked Human I/O boundary for caller-transactional purge."""

    def list_task_run_human_requests(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[HumanRequest]:
        ...

    def redact_task_run_human_request_payload(
        self,
        request: HumanRequest,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_status: str,
    ) -> bool:
        ...

class PayloadRetentionAdmission(Protocol):
    """Lifecycle gate used when maintenance is exposed by a live Runtime."""

    def admit(self, *, read_only: bool = False) -> AbstractContextManager[Any]:
        ...


class PayloadRetentionMaintenance:
    """Run one bounded, auditable payload-retention maintenance page."""

    def __init__(
        self,
        store: PayloadRetentionStore,
        audit: AuditPort,
        policy: PayloadRetentionPolicy | None = None,
        *,
        admission: PayloadRetentionAdmission | None = None,
    ) -> None:
        self._store = store
        self._audit = audit
        self._policy = policy or PayloadRetentionPolicy()
        self._admission = admission

    def run(
        self,
        request: PayloadRetentionRequest,
        *,
        now: datetime | None = None,
    ) -> PayloadRetentionResult:
        lease = (
            self._admission.admit()
            if self._admission is not None
            else nullcontext()
        )
        with lease:
            return self._run(request, now=now)

    def _run(
        self,
        request: PayloadRetentionRequest,
        *,
        now: datetime | None = None,
    ) -> PayloadRetentionResult:
        selected_now = _normalized_datetime(now or datetime.now(timezone.utc))
        if not self._policy.enabled:
            result = PayloadRetentionResult(
                kind=request.kind,
                status="disabled",
                dry_run=request.dry_run,
                candidate_set_sha256=_candidate_set_digest(()),
            )
            return self._audit_result(request, result)

        assert self._policy.summary_after_seconds is not None
        limit = request.limit or self._policy.batch_limit
        if limit > self._policy.hard_limit:
            raise ValueError(
                f"payload retention request limit exceeds hard limit "
                f"{self._policy.hard_limit}"
            )
        older_than = (
            selected_now
            - timedelta(seconds=self._policy.summary_after_seconds)
        ).isoformat()
        if request.kind is PayloadRetentionKind.LLM_CALL:
            page = self._store.scan_llm_call_payloads_for_retention(
                older_than=older_than,
                after=request.cursor,
                limit=limit,
            )
            result, updates = self._plan_llm_page(
                page,
                request=request,
                now=selected_now,
                limit=limit,
            )
        else:
            page = self._store.scan_external_effect_payloads_for_retention(
                older_than=older_than,
                after=request.cursor,
                limit=limit,
            )
            result, updates = self._plan_effect_page(
                page,
                request=request,
                now=selected_now,
                limit=limit,
            )

        if request.dry_run:
            return self._audit_result(request, result)

        with self._store.transaction():
            updated = 0
            conflicts = 0
            for apply_update in updates:
                if apply_update():
                    updated += 1
                else:
                    conflicts += 1
            applied = replace(result, updated=updated, conflicts=conflicts)
            return self._audit_result(request, applied)

    def _plan_llm_page(
        self,
        page: PayloadRetentionPage[LLMCallRecord],
        *,
        request: PayloadRetentionRequest,
        now: datetime,
        limit: int,
    ) -> tuple[PayloadRetentionResult, list[Callable[[], bool]]]:
        _validate_page(
            page,
            limit=limit,
            after=request.cursor,
            id_for=lambda item: item.call_id,
        )
        if page.latest_llm_call_ids is None:
            raise ValueError(
                "payload retention backend did not classify latest LLM calls"
            )
        latest_call_ids = page.latest_llm_call_ids
        updates: list[Callable[[], bool]] = []
        candidate_ids: list[str] = []
        eligible = protected = runtime_dependency = invalid = already = 0
        for record in page.records:
            candidate_ids.append(record.call_id)
            if not _llm_call_is_terminal(record):
                protected += 1
                continue
            provider_chain_head = record.call_id in latest_call_ids
            if llm_call_payload_is_runtime_dependency(
                record,
                provider_chain_head=provider_chain_head,
            ):
                runtime_dependency += 1
                continue
            terminal_at = _parse_timestamp(record.completed_at)
            if terminal_at is None:
                invalid += 1
                continue
            current = llm_call_payload_retention_tier(record)
            target = self._next_tier(current, terminal_at=terminal_at, now=now)
            normalize_legacy = (
                current is PayloadRetentionTier.SUMMARY
                and not _has_retention_marker(record.observability)
                and target is PayloadRetentionTier.SUMMARY
            )
            if target is current and not normalize_legacy:
                already += 1
                continue
            eligible += 1
            expected_sha = llm_call_payload_sha256(record)
            retained = retain_llm_call_payload(
                record,
                target,
                provider_chain_head=provider_chain_head,
            )

            def apply(
                *,
                selected: LLMCallRecord = retained,
                expected_payload_sha256: str = expected_sha,
                expected_tier: PayloadRetentionTier = current,
            ) -> bool:
                return self._store.update_llm_call_payload_retention(
                    selected,
                    expected_payload_sha256=expected_payload_sha256,
                    expected_tier=expected_tier,
                )

            updates.append(apply)
        result = PayloadRetentionResult(
            kind=request.kind,
            status="dry_run" if request.dry_run else "applied",
            dry_run=request.dry_run,
            scanned=len(page.records),
            eligible=eligible,
            would_update=eligible,
            protected_nonterminal=protected,
            protected_runtime_dependency=runtime_dependency,
            invalid_timestamp=invalid,
            already_retained=already,
            next_cursor=page.next_cursor,
            candidate_set_sha256=_candidate_set_digest(candidate_ids),
        )
        return result, updates

    def _plan_effect_page(
        self,
        page: PayloadRetentionPage[ExternalEffectRecord],
        *,
        request: PayloadRetentionRequest,
        now: datetime,
        limit: int,
    ) -> tuple[PayloadRetentionResult, list[Callable[[], bool]]]:
        _validate_page(
            page,
            limit=limit,
            after=request.cursor,
            id_for=lambda item: item.effect_id,
        )
        updates: list[Callable[[], bool]] = []
        candidate_ids: list[str] = []
        eligible = protected = invalid = already = 0
        for record in page.records:
            candidate_ids.append(record.effect_id)
            if not external_effect_payload_is_terminal(record):
                protected += 1
                continue
            terminal_at = _parse_timestamp(record.updated_at or record.created_at)
            if terminal_at is None:
                invalid += 1
                continue
            current = external_effect_payload_retention_tier(record)
            target = self._next_tier(current, terminal_at=terminal_at, now=now)
            if target is current:
                already += 1
                continue
            eligible += 1
            expected_sha = external_effect_payload_sha256(record)
            retained = retain_external_effect_payload(record, target)

            def apply(
                *,
                selected: ExternalEffectRecord = retained,
                expected_payload_sha256: str = expected_sha,
                expected_tier: PayloadRetentionTier = current,
                expected_effect_state: str = record.effect_state,
                expected_transaction_state: str = record.transaction_state,
            ) -> bool:
                return self._store.update_external_effect_payload_retention(
                    selected,
                    expected_payload_sha256=expected_payload_sha256,
                    expected_tier=expected_tier,
                    expected_effect_state=expected_effect_state,
                    expected_transaction_state=expected_transaction_state,
                )

            updates.append(apply)
        result = PayloadRetentionResult(
            kind=request.kind,
            status="dry_run" if request.dry_run else "applied",
            dry_run=request.dry_run,
            scanned=len(page.records),
            eligible=eligible,
            would_update=eligible,
            protected_nonterminal=protected,
            invalid_timestamp=invalid,
            already_retained=already,
            next_cursor=page.next_cursor,
            candidate_set_sha256=_candidate_set_digest(candidate_ids),
        )
        return result, updates

    def _next_tier(
        self,
        current: PayloadRetentionTier,
        *,
        terminal_at: datetime,
        now: datetime,
    ) -> PayloadRetentionTier:
        age_seconds = (now - terminal_at).total_seconds()
        if age_seconds < 0:
            return current
        if current is PayloadRetentionTier.FULL:
            if (
                self._policy.summary_after_seconds is not None
                and age_seconds >= self._policy.summary_after_seconds
            ):
                return PayloadRetentionTier.SUMMARY
            return current
        if current is PayloadRetentionTier.SUMMARY:
            if (
                self._policy.hash_only_after_seconds is not None
                and age_seconds >= self._policy.hash_only_after_seconds
            ):
                return PayloadRetentionTier.HASH_ONLY
        return current

    def _audit_result(
        self,
        request: PayloadRetentionRequest,
        result: PayloadRetentionResult,
    ) -> PayloadRetentionResult:
        cursor_digest = (
            _candidate_set_digest(
                (result.next_cursor.created_at, result.next_cursor.record_id)
            )
            if result.next_cursor is not None
            else None
        )
        record: AuditRecord = self._audit.record(
            actor=request.actor,
            action="evidence.payload_retention.maintenance",
            target=request.kind.value,
            decision={
                "schema_version": _RETENTION_SCHEMA_VERSION,
                "status": result.status,
                "dry_run": result.dry_run,
                "scanned": result.scanned,
                "eligible": result.eligible,
                "would_update": result.would_update,
                "updated": result.updated,
                "conflicts": result.conflicts,
                "protected_nonterminal": result.protected_nonterminal,
                "protected_runtime_dependency": result.protected_runtime_dependency,
                "invalid_timestamp": result.invalid_timestamp,
                "already_retained": result.already_retained,
                "candidate_set_sha256": result.candidate_set_sha256,
                "next_cursor_present": result.next_cursor is not None,
                "next_cursor_sha256": cursor_digest,
                "summary_after_seconds": self._policy.summary_after_seconds,
                "hash_only_after_seconds": self._policy.hash_only_after_seconds,
            },
            correlation_id=request.correlation_id,
        )
        return replace(result, audit_record_id=record.record_id)


def retain_llm_call_payload(
    record: LLMCallRecord,
    target: PayloadRetentionTier,
    *,
    provider_chain_head: bool | None = None,
) -> LLMCallRecord:
    """Return a copy with payload fields monotonically reduced to ``target``."""

    if not isinstance(target, PayloadRetentionTier):
        raise ValueError("LLM payload retention target has an invalid type")
    if not _llm_call_is_terminal(record):
        raise ValueError("nonterminal LLM call payloads cannot be retained")
    if llm_call_payload_is_runtime_dependency(
        record,
        provider_chain_head=provider_chain_head,
    ):
        raise ValueError("runtime-dependent LLM call payloads cannot be retained")
    current = llm_call_payload_retention_tier(record)
    if _tier_rank(target) < _tier_rank(current):
        raise ValueError("LLM payload retention cannot restore a less restrictive tier")
    if target is current and not (
        current is PayloadRetentionTier.SUMMARY
        and not _has_retention_marker(record.observability)
    ):
        return record
    if current is PayloadRetentionTier.HASH_ONLY:
        return record
    if (
        current is PayloadRetentionTier.FULL
        and target is PayloadRetentionTier.HASH_ONLY
    ):
        raise ValueError("LLM payload retention cannot skip the summary tier")
    return _retained_llm_call_payload(record, target=target, source_tier=current)


def redact_terminal_task_run_llm_call(record: LLMCallRecord) -> LLMCallRecord:
    """Return a canonical hash-only copy for a fenced, terminal TaskRun.

    Unlike scheduled retention, terminal Run purge intentionally permits a
    direct ``full`` -> ``hash_only`` transition and may redact the latest
    Provider/image transcript head.  The caller must first fence/finalize the
    owning Run; the store boundary rechecks exact Run membership. Identity,
    usage, Provider receipts, timing and other non-payload evidence remain
    unchanged.
    """

    if not isinstance(record, LLMCallRecord):
        raise ValueError("TaskRun LLM redaction target has an invalid type")
    if not _llm_call_is_terminal(record):
        raise ValueError("nonterminal TaskRun LLM calls cannot be redacted")
    current = llm_call_payload_retention_tier(record)
    if current is PayloadRetentionTier.HASH_ONLY:
        # Validate existing provenance before treating the row as complete.
        llm_call_payload_sha256(record)
        return record

    redacted = _retained_llm_call_payload(
        record,
        target=PayloadRetentionTier.HASH_ONLY,
        source_tier=current,
    )
    if llm_call_payload_sha256(redacted) != llm_call_payload_sha256(record):
        raise RuntimeError("TaskRun LLM redaction changed payload provenance")
    return redacted


def _retained_llm_call_payload(
    record: LLMCallRecord,
    *,
    target: PayloadRetentionTier,
    source_tier: PayloadRetentionTier,
) -> LLMCallRecord:
    """Project all LLM payload fields through one canonical retention path."""

    source_fields = _llm_payload_sources(record, current=source_tier)
    trust_retention_envelopes = _has_retention_marker(record.observability)
    retained_fields: dict[str, Any] = {}
    for field_name in _LLM_PAYLOAD_FIELDS:
        value = source_fields[field_name]
        if value is None and field_name in {"reasoning", "raw_response", "error"}:
            retained_fields[field_name] = None
            continue
        envelope = content_free_payload_envelope(
            value,
            target,
            source_tier=source_tier,
            trust_retention_envelopes=trust_retention_envelopes,
        )
        retained_fields[field_name] = (
            dumps(envelope)
            if field_name in {"response_content", "error"}
            else envelope
        )
    retained_observability = {
        _RETENTION_KEY: {
            "schema_version": _RETENTION_SCHEMA_VERSION,
            "tier": target.value,
            "payload_sha256": _field_digest(
                retained_fields,
                source_tier=target,
            ),
            "source_observability_sha256": _source_observability_sha256(record),
        }
    }
    return replace(
        record,
        messages=retained_fields["messages"],
        tools=retained_fields["tools"],
        response_content=retained_fields["response_content"],
        tool_calls=retained_fields["tool_calls"],
        reasoning=retained_fields["reasoning"],
        raw_response=retained_fields["raw_response"],
        observability=retained_observability,
        error=retained_fields["error"],
    )


def validate_terminal_task_run_llm_redaction(
    current: LLMCallRecord,
    target: LLMCallRecord,
    *,
    expected_payload_sha256: str,
    expected_tier: PayloadRetentionTier,
) -> PayloadRetentionTier:
    """Validate the exceptional direct-to-hash terminal Run transition."""

    if not isinstance(current, LLMCallRecord) or not isinstance(
        target, LLMCallRecord
    ):
        raise ValueError("TaskRun LLM redaction records have invalid types")
    if not isinstance(expected_tier, PayloadRetentionTier):
        raise ValueError("TaskRun LLM redaction expected tier is invalid")
    if not _llm_call_is_terminal(current):
        raise ValueError("nonterminal TaskRun LLM calls cannot be redacted")
    current_tier = llm_call_payload_retention_tier(current)
    if current_tier is not expected_tier:
        raise ValueError("TaskRun LLM redaction source tier changed")
    if current_tier is PayloadRetentionTier.HASH_ONLY:
        raise ValueError("TaskRun LLM call is already hash-only")
    if (
        not _valid_sha256(expected_payload_sha256)
        or llm_call_payload_sha256(current) != expected_payload_sha256
    ):
        raise ValueError("TaskRun LLM redaction source digest changed")
    canonical = redact_terminal_task_run_llm_call(current)
    if _llm_payload_write_projection(target) != _llm_payload_write_projection(
        canonical
    ):
        raise ValueError("TaskRun LLM redaction target is not canonical")
    if llm_call_payload_retention_tier(target) is not PayloadRetentionTier.HASH_ONLY:
        raise ValueError("TaskRun LLM redaction target is not hash-only")
    if llm_call_payload_sha256(target) != expected_payload_sha256:
        raise ValueError("TaskRun LLM redaction target changed payload provenance")
    return PayloadRetentionTier.HASH_ONLY


def redact_task_run_llm_calls(
    store: TaskRunLLMRedactionStore,
    run_id: str,
    linked_pids: Iterable[str],
) -> int:
    """Redact every linked LLM row without owning or committing a transaction.

    A CAS conflict raises so a TaskRun cannot leave ``finalizing`` with only a
    partial purge. The caller owns the surrounding transaction/savepoint and
    the store owns the authoritative Run/PID membership check.
    """

    selected_pids = _validated_task_run_redaction_pids(run_id, linked_pids)
    return _redact_task_run_llm_call_rows(
        store,
        run_id,
        selected_pids,
    ) + _redact_task_run_llm_tool_output_rows(store, run_id, selected_pids)


def _validated_task_run_redaction_pids(
    run_id: str,
    linked_pids: Iterable[str],
) -> tuple[str, ...]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("TaskRun redaction run_id must not be empty")
    if isinstance(linked_pids, (str, bytes, bytearray)):
        raise ValueError("TaskRun redaction linked_pids must be an iterable of ids")
    raw_pids = tuple(linked_pids)
    if any(not isinstance(pid, str) or not pid for pid in raw_pids):
        raise ValueError("TaskRun redaction linked_pids are invalid")
    return tuple(sorted(set(raw_pids)))


def _redact_task_run_llm_call_rows(
    store: TaskRunLLMRedactionStore,
    run_id: str,
    selected_pids: tuple[str, ...],
) -> int:
    records = store.list_task_run_llm_calls(run_id, selected_pids)
    if not isinstance(records, list):
        raise RuntimeError("TaskRun LLM redaction backend returned an invalid page")
    seen_call_ids: set[str] = set()
    updated = 0
    for record in records:
        _validate_task_run_llm_call_row(record, selected_pids, seen_call_ids)
        current_tier = llm_call_payload_retention_tier(record)
        current_sha256 = llm_call_payload_sha256(record)
        target = redact_terminal_task_run_llm_call(record)
        if target is record:
            continue
        if not store.redact_task_run_llm_call_payload(
            target,
            run_id=run_id,
            expected_payload_sha256=current_sha256,
            expected_tier=current_tier.value,
        ):
            raise RuntimeError(
                f"TaskRun LLM redaction conflicted for call: {record.call_id}"
            )
        updated += 1
    return updated


def _validate_task_run_llm_call_row(
    record: Any,
    selected_pids: tuple[str, ...],
    seen_call_ids: set[str],
) -> None:
    if not isinstance(record, LLMCallRecord):
        raise RuntimeError("TaskRun LLM redaction backend returned an invalid row")
    if record.call_id in seen_call_ids:
        raise RuntimeError("TaskRun LLM redaction backend returned duplicate rows")
    seen_call_ids.add(record.call_id)
    if record.pid is None or (
        selected_pids and str(record.pid) not in selected_pids
    ):
        raise RuntimeError("TaskRun LLM redaction backend returned an unlinked row")


def _redact_task_run_llm_tool_output_rows(
    store: TaskRunLLMRedactionStore,
    run_id: str,
    selected_pids: tuple[str, ...],
) -> int:
    output_rows = store.list_task_run_llm_tool_outputs(run_id, selected_pids)
    if not isinstance(output_rows, list):
        raise RuntimeError(
            "TaskRun LLM tool-output redaction backend returned an invalid page"
        )
    seen_output_ids: set[tuple[str, str, str]] = set()
    updated = 0
    for row in output_rows:
        pid, response_id, call_id, output_text = (
            _validated_task_run_llm_tool_output_row(
                row,
                selected_pids,
                seen_output_ids,
            )
        )
        redacted_output, expected_sha256, changed = (
            redact_terminal_task_run_llm_tool_output(output_text)
        )
        if not changed:
            continue
        if not store.redact_task_run_llm_tool_output(
            run_id=run_id,
            pid=pid,
            response_id=response_id,
            call_id=call_id,
            expected_output_sha256=expected_sha256,
            redacted_output=redacted_output,
        ):
            raise RuntimeError(
                "TaskRun LLM tool-output redaction conflicted for "
                f"pid={pid} response_id={response_id} call_id={call_id}"
            )
        updated += 1
    return updated


def _validated_task_run_llm_tool_output_row(
    row: Any,
    selected_pids: tuple[str, ...],
    seen_output_ids: set[tuple[str, str, str]],
) -> tuple[str, str, str, str]:
    if not isinstance(row, dict):
        raise RuntimeError(
            "TaskRun LLM tool-output redaction backend returned an invalid row"
        )
    pid = row.get("pid")
    response_id = row.get("response_id")
    call_id = row.get("call_id")
    output_text = row.get("output_text")
    if any(
        not isinstance(value, str) or not value
        for value in (pid, response_id, call_id)
    ) or not isinstance(output_text, str):
        raise RuntimeError("TaskRun LLM tool-output redaction row is malformed")
    if selected_pids and pid not in selected_pids:
        raise RuntimeError(
            "TaskRun LLM tool-output redaction backend returned an unlinked row"
        )
    output_id = (pid, response_id, call_id)
    if output_id in seen_output_ids:
        raise RuntimeError(
            "TaskRun LLM tool-output redaction backend returned duplicate rows"
        )
    seen_output_ids.add(output_id)
    return pid, response_id, call_id, output_text


def redact_terminal_task_run_llm_tool_output(
    output_text: str,
) -> tuple[str, str, bool]:
    """Return ``(redacted_text, source_sha256, changed)`` for a Tool output."""

    if not isinstance(output_text, str):
        raise ValueError("TaskRun LLM tool output must be text")
    try:
        decoded = bounded_json_loads(output_text, max_bytes=1_048_576)
    except ValueError:
        decoded = None
    if isinstance(decoded, dict) and set(decoded) == {
        _TASK_RUN_TOOL_OUTPUT_REDACTION_KEY
    }:
        marker = decoded[_TASK_RUN_TOOL_OUTPUT_REDACTION_KEY]
        if (
            isinstance(marker, dict)
            and set(marker) == {"schema_version", "sha256", "bytes"}
            and marker.get("schema_version")
            == _TASK_RUN_TOOL_OUTPUT_REDACTION_SCHEMA_VERSION
            and _valid_sha256(marker.get("sha256"))
            and isinstance(marker.get("bytes"), int)
            and not isinstance(marker.get("bytes"), bool)
            and marker["bytes"] >= 0
            and dumps(decoded) == output_text
        ):
            return output_text, str(marker["sha256"]), False
    encoded = output_text.encode("utf-8")
    source_sha256 = hashlib.sha256(encoded).hexdigest()
    redacted = dumps(
        {
            _TASK_RUN_TOOL_OUTPUT_REDACTION_KEY: {
                "schema_version": _TASK_RUN_TOOL_OUTPUT_REDACTION_SCHEMA_VERSION,
                "sha256": source_sha256,
                "bytes": len(encoded),
            }
        }
    )
    return redacted, source_sha256, True


def human_request_payload_sha256(request: HumanRequest) -> str:
    """Hash the original Human prompt and answer without retaining either."""

    if not isinstance(request, HumanRequest):
        raise ValueError("TaskRun Human redaction target has an invalid type")
    if not isinstance(request.payload, dict):
        raise ValueError("TaskRun Human request payload must be an object")
    if request.decision is not None and not isinstance(request.decision, dict):
        raise ValueError("TaskRun Human request decision must be an object or null")
    payload_sha256 = _task_run_human_field_sha256(request.payload, "payload")
    decision_sha256 = _task_run_human_field_sha256(
        request.decision,
        "decision",
    )
    return hashlib.sha256(
        dumps(
            {
                "payload_sha256": payload_sha256,
                "decision_sha256": decision_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()


def redact_terminal_task_run_human_request(
    request: HumanRequest,
) -> HumanRequest:
    """Return a content-free Human request while preserving lifecycle evidence."""

    source_sha256 = human_request_payload_sha256(request)
    payload, payload_changed = _redacted_task_run_human_field(
        request.payload,
        field="payload",
        request_type=_task_run_human_request_type(request.payload),
    )
    decision: dict[str, Any] | None = request.decision
    decision_changed = False
    if decision is not None:
        decision, decision_changed = _redacted_task_run_human_field(
            decision,
            field="decision",
        )
    if not payload_changed and not decision_changed:
        return request
    target = replace(request, payload=payload, decision=decision)
    if human_request_payload_sha256(target) != source_sha256:
        raise RuntimeError("TaskRun Human redaction changed payload provenance")
    return target


def validate_terminal_task_run_human_redaction(
    current: HumanRequest,
    target: HumanRequest,
    *,
    expected_payload_sha256: str,
    expected_status: HumanRequestStatus,
) -> None:
    """Validate one exact Human prompt/answer reduction for a terminal Run."""

    if not isinstance(current, HumanRequest) or not isinstance(target, HumanRequest):
        raise ValueError("TaskRun Human redaction records have invalid types")
    if not isinstance(expected_status, HumanRequestStatus):
        raise ValueError("TaskRun Human redaction expected status is invalid")
    if current.status is not expected_status:
        raise ValueError("TaskRun Human request status changed")
    if (
        not _valid_sha256(expected_payload_sha256)
        or human_request_payload_sha256(current) != expected_payload_sha256
    ):
        raise ValueError("TaskRun Human request payload digest changed")
    canonical = redact_terminal_task_run_human_request(current)
    if canonical is current:
        raise ValueError("TaskRun Human request is already hash-only")
    if target != canonical:
        raise ValueError("TaskRun Human redaction target is not canonical")
    if replace(target, payload=current.payload, decision=current.decision) != current:
        raise ValueError("TaskRun Human redaction changed lifecycle evidence")
    if human_request_payload_sha256(target) != expected_payload_sha256:
        raise ValueError("TaskRun Human redaction changed payload provenance")


def redact_task_run_human_requests(
    store: TaskRunHumanRedactionStore,
    run_id: str,
    linked_pids: Iterable[str],
) -> int:
    """Redact every Run-linked Human prompt/answer in the caller transaction."""

    selected_pids = _validated_task_run_redaction_pids(run_id, linked_pids)
    records = store.list_task_run_human_requests(run_id, selected_pids)
    if not isinstance(records, list):
        raise RuntimeError("TaskRun Human redaction backend returned an invalid page")
    seen_request_ids: set[str] = set()
    updated = 0
    for request in records:
        _validate_task_run_human_request_row(
            request,
            selected_pids,
            seen_request_ids,
        )
        expected_sha256 = human_request_payload_sha256(request)
        target = redact_terminal_task_run_human_request(request)
        if target is request:
            continue
        if not store.redact_task_run_human_request_payload(
            target,
            run_id=run_id,
            expected_payload_sha256=expected_sha256,
            expected_status=request.status.value,
        ):
            raise RuntimeError(
                "TaskRun Human redaction conflicted for request: "
                f"{request.request_id}"
            )
        updated += 1
    return updated


def _validate_task_run_human_request_row(
    request: Any,
    selected_pids: tuple[str, ...],
    seen_request_ids: set[str],
) -> None:
    if not isinstance(request, HumanRequest):
        raise RuntimeError("TaskRun Human redaction backend returned an invalid row")
    request_id = str(request.request_id)
    if not request_id or request_id in seen_request_ids:
        raise RuntimeError(
            "TaskRun Human redaction backend returned a duplicate or invalid row"
        )
    seen_request_ids.add(request_id)
    if selected_pids and str(request.pid) not in selected_pids:
        raise RuntimeError("TaskRun Human redaction backend returned an unlinked row")


def _task_run_human_request_type(payload: Mapping[str, Any]) -> str:
    selected = payload.get("type")
    if isinstance(selected, str) and _TASK_RUN_HUMAN_TYPE_PATTERN.fullmatch(selected):
        return selected
    marker = _task_run_human_redaction_marker(payload, field="payload")
    if marker is not None:
        return str(marker["request_type"])
    return "unknown"


def _task_run_human_field_sha256(value: Any, field: str) -> str:
    marker = _task_run_human_redaction_marker(value, field=field)
    if marker is not None:
        return str(marker["sha256"])
    return hashlib.sha256(dumps(value).encode("utf-8")).hexdigest()


def _redacted_task_run_human_field(
    value: dict[str, Any],
    *,
    field: str,
    request_type: str | None = None,
) -> tuple[dict[str, Any], bool]:
    if _task_run_human_redaction_marker(value, field=field) is not None:
        return value, False
    encoded = dumps(value).encode("utf-8")
    marker: dict[str, Any] = {
        "schema_version": _TASK_RUN_HUMAN_REDACTION_SCHEMA_VERSION,
        "field": field,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
    if field == "payload":
        marker["request_type"] = request_type or "unknown"
    return {_TASK_RUN_HUMAN_REDACTION_KEY: marker}, True


def _task_run_human_redaction_marker(
    value: Any,
    *,
    field: str,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {_TASK_RUN_HUMAN_REDACTION_KEY}:
        return None
    marker = value[_TASK_RUN_HUMAN_REDACTION_KEY]
    expected = {"schema_version", "field", "sha256", "bytes"}
    if field == "payload":
        expected.add("request_type")
    if not isinstance(marker, dict) or set(marker) != expected:
        return None
    if (
        marker.get("schema_version") != _TASK_RUN_HUMAN_REDACTION_SCHEMA_VERSION
        or marker.get("field") != field
        or not _valid_sha256(marker.get("sha256"))
        or not isinstance(marker.get("bytes"), int)
        or isinstance(marker.get("bytes"), bool)
        or marker["bytes"] < 0
    ):
        return None
    if field == "payload" and (
        not isinstance(marker.get("request_type"), str)
        or not _TASK_RUN_HUMAN_TYPE_PATTERN.fullmatch(marker["request_type"])
    ):
        return None
    if dumps(value) != dumps({_TASK_RUN_HUMAN_REDACTION_KEY: marker}):
        return None
    return marker


def redact_task_run_external_effects(
    store: TaskRunExternalEffectRedactionStore,
    run_id: str,
    linked_pids: Iterable[str],
) -> int:
    """Reduce every terminal, linked Run effect to canonical hash-only form.

    A full payload deliberately takes the ordinary two monotonic CAS steps,
    ``full`` -> ``summary`` -> ``hash_only``.  This function never owns a
    transaction: the TaskRun finalizer must wrap the entire purge so any
    conflict rolls back every prior reduction and keeps the Run finalizing.
    The return value counts effect rows changed, not individual CAS writes.
    """

    selected_pids = _validated_task_run_redaction_pids(run_id, linked_pids)
    records = store.list_task_run_external_effects(run_id, selected_pids)
    validated = _validated_task_run_external_effect_rows(
        records,
        selected_pids,
    )
    updated = 0
    for record in validated:
        current = record
        changed = False
        while (
            external_effect_payload_retention_tier(current)
            is not PayloadRetentionTier.HASH_ONLY
        ):
            current_tier = external_effect_payload_retention_tier(current)
            target_tier = (
                PayloadRetentionTier.SUMMARY
                if current_tier is PayloadRetentionTier.FULL
                else PayloadRetentionTier.HASH_ONLY
            )
            source_sha256 = external_effect_payload_sha256(current)
            target = retain_external_effect_payload(current, target_tier)
            if not store.redact_task_run_external_effect_payload(
                target,
                run_id=run_id,
                expected_payload_sha256=source_sha256,
                expected_tier=current_tier.value,
                expected_effect_state=current.effect_state,
                expected_transaction_state=current.transaction_state,
            ):
                raise RuntimeError(
                    "TaskRun external-effect redaction conflicted for effect: "
                    f"{current.effect_id}"
                )
            current = target
            changed = True
        updated += int(changed)
    return updated


def _validated_task_run_external_effect_rows(
    records: Any,
    selected_pids: tuple[str, ...],
) -> tuple[ExternalEffectRecord, ...]:
    if not isinstance(records, list):
        raise RuntimeError(
            "TaskRun external-effect redaction backend returned an invalid page"
        )
    validated: list[ExternalEffectRecord] = []
    seen_effect_ids: set[str] = set()
    for record in records:
        if not isinstance(record, ExternalEffectRecord):
            raise RuntimeError(
                "TaskRun external-effect redaction backend returned an invalid row"
            )
        if record.effect_id in seen_effect_ids:
            raise RuntimeError(
                "TaskRun external-effect redaction backend returned duplicate rows"
            )
        seen_effect_ids.add(record.effect_id)
        if selected_pids and str(record.pid) not in selected_pids:
            raise RuntimeError(
                "TaskRun external-effect redaction backend returned an unlinked row"
            )
        if not external_effect_payload_is_terminal(record):
            raise ValueError(
                "nonterminal TaskRun external effects cannot be redacted"
            )
        external_effect_payload_sha256(record)
        validated.append(record)
    return tuple(validated)


def content_free_llm_call_fields(
    *,
    messages: Any,
    tools: Any,
    response_content: str,
    tool_calls: Any,
    reasoning: Any | None,
    raw_response: Any | None,
    error: str | None,
    source_observability: dict[str, Any],
) -> dict[str, Any]:
    """Build a canonical write-time ``summary`` projection for one LLM call.

    This is the write-time counterpart of :func:`retain_llm_call_payload` for
    ``llm.persist_full_io=false``.  It intentionally does not apply the
    maintenance runtime-dependency checks: disabling full-I/O persistence is a
    Host policy that prevents the raw provider payload from being written in
    the first place.  The resulting marker and field envelopes use the same
    canonical shape and digest rules as retention maintenance, so storage can
    index and validate the row as an actual ``summary`` tier.
    """

    if not isinstance(source_observability, dict):
        raise ValueError("LLM source observability must be a mapping")
    source_fields = {
        "messages": messages,
        "tools": tools,
        "response_content": response_content,
        "tool_calls": tool_calls,
        "reasoning": reasoning,
        "raw_response": raw_response,
        "error": error,
    }
    retained_fields: dict[str, Any] = {}
    for field_name in _LLM_PAYLOAD_FIELDS:
        value = source_fields[field_name]
        if value is None and field_name in {"reasoning", "raw_response", "error"}:
            retained_fields[field_name] = None
            continue
        envelope = content_free_payload_envelope(
            value,
            PayloadRetentionTier.SUMMARY,
            source_tier=PayloadRetentionTier.FULL,
            trust_retention_envelopes=False,
        )
        retained_fields[field_name] = (
            dumps(envelope)
            if field_name in {"response_content", "error"}
            else envelope
        )
    payload_sha256 = _field_digest(
        retained_fields,
        source_tier=PayloadRetentionTier.SUMMARY,
    )
    retained_fields["observability"] = {
        _RETENTION_KEY: {
            "schema_version": _RETENTION_SCHEMA_VERSION,
            "tier": PayloadRetentionTier.SUMMARY.value,
            "payload_sha256": payload_sha256,
            "source_observability_sha256": _json_sha256(source_observability),
        }
    }
    return retained_fields


def retain_external_effect_payload(
    record: ExternalEffectRecord,
    target: PayloadRetentionTier,
) -> ExternalEffectRecord:
    """Reduce terminal provider payloads without removing ledger identity."""

    if not isinstance(target, PayloadRetentionTier):
        raise ValueError("external-effect retention target has an invalid type")
    if not external_effect_payload_is_terminal(record):
        raise ValueError("nonterminal external effect payloads cannot be retained")
    current = external_effect_payload_retention_tier(record)
    source_sha256 = external_effect_payload_sha256(record)
    if _tier_rank(target) < _tier_rank(current):
        raise ValueError("external-effect retention cannot restore a less restrictive tier")
    if target is current:
        return record
    if current is PayloadRetentionTier.HASH_ONLY:
        return record
    if (
        current is PayloadRetentionTier.FULL
        and target is PayloadRetentionTier.HASH_ONLY
    ):
        raise ValueError(
            "external-effect payload retention cannot skip the summary tier"
        )
    return replace(
        record,
        provider_metadata=content_free_payload_envelope(
            record.provider_metadata,
            target,
            source_tier=current,
        ),
        provider_receipt=content_free_payload_envelope(
            record.provider_receipt,
            target,
            source_tier=current,
        ),
        payload_retention_schema_version=_RETENTION_SCHEMA_VERSION,
        payload_retention_tier=target.value,
        payload_retention_sha256=source_sha256,
    )


def content_free_payload_envelope(
    value: Any,
    tier: PayloadRetentionTier,
    *,
    source_tier: PayloadRetentionTier = PayloadRetentionTier.FULL,
    trust_retention_envelopes: bool = True,
) -> dict[str, Any]:
    """Create a summary/hash envelope that never embeds source content.

    ``source_tier`` is record-level provenance. A FULL record is always
    hashed as raw JSON, even when the value happens to look like one of our
    envelopes. ``trust_retention_envelopes`` lets legacy record migrations
    trust their legacy digests without trusting a model-controlled value that
    mimics the newer envelope format.
    """

    if not isinstance(tier, PayloadRetentionTier):
        raise ValueError("payload retention envelope tier has an invalid type")
    if not isinstance(source_tier, PayloadRetentionTier):
        raise ValueError("payload retention source tier has an invalid type")
    if not isinstance(trust_retention_envelopes, bool):
        raise ValueError("payload retention envelope trust must be a boolean")
    if tier is PayloadRetentionTier.FULL:
        raise ValueError("full payloads are not represented by retention envelopes")
    if _tier_rank(tier) < _tier_rank(source_tier):
        raise ValueError("payload retention envelope cannot restore source content")
    source_is_retained = source_tier is not PayloadRetentionTier.FULL
    existing = (
        _retention_metadata_from_value(value)
        if source_is_retained and trust_retention_envelopes
        else None
    )
    legacy = _legacy_observation_metadata(value) if source_is_retained else None
    digest = (
        str(existing["sha256"])
        if existing is not None
        else str(legacy["sha256"])
        if legacy is not None
        else _json_sha256(value)
    )
    metadata: dict[str, Any] = {
        "schema_version": _RETENTION_SCHEMA_VERSION,
        "tier": tier.value,
        "sha256": digest,
    }
    if tier is PayloadRetentionTier.SUMMARY:
        if existing is not None and existing.get("bytes") is not None:
            metadata["bytes"] = int(existing["bytes"])
            metadata["json_kind"] = str(existing.get("json_kind") or "unknown")
            if existing.get("item_count") is not None:
                metadata["item_count"] = int(existing["item_count"])
        elif legacy is not None:
            metadata["bytes"] = int(legacy["bytes"])
            metadata["json_kind"] = "unknown"
        else:
            encoded = dumps(to_jsonable(value)).encode("utf-8")
            metadata["bytes"] = len(encoded)
            metadata["json_kind"] = _json_kind(value)
            item_count = _item_count(value)
            if item_count is not None:
                metadata["item_count"] = item_count
    return {_RETENTION_KEY: metadata}


def llm_call_payload_retention_tier(record: LLMCallRecord) -> PayloadRetentionTier:
    marker = _retention_metadata(record.observability)
    if marker is not None and _valid_sha256(marker.get("payload_sha256")):
        return PayloadRetentionTier(str(marker["tier"]))
    if _looks_like_legacy_llm_summary(record):
        return PayloadRetentionTier.SUMMARY
    return PayloadRetentionTier.FULL


def external_effect_payload_retention_tier(
    record: ExternalEffectRecord,
) -> PayloadRetentionTier:
    if record.payload_retention_schema_version != _RETENTION_SCHEMA_VERSION:
        raise ValueError("external effect payload retention schema is unsupported")
    try:
        return PayloadRetentionTier(record.payload_retention_tier)
    except ValueError as exc:
        raise ValueError("external effect payload retention tier is invalid") from exc


def llm_call_payload_sha256(record: LLMCallRecord) -> str:
    current = llm_call_payload_retention_tier(record)
    fields = _llm_payload_sources(record, current=current)
    has_marker = _has_retention_marker(record.observability)
    if has_marker:
        marker, marker_tier = _strict_llm_retention_marker(record.observability)
        if marker_tier is not current:
            raise ValueError("LLM payload retention marker tier changed")
        nullable_fields = {"reasoning", "raw_response", "error"}
        serialized_fields = {"response_content", "error"}
        for field_name, value in fields.items():
            if value is None:
                if field_name not in nullable_fields:
                    raise ValueError(
                        f"LLM payload retention field cannot be null: {field_name}"
                    )
                continue
            _strict_payload_retention_envelope(
                value,
                tier=current,
                serialized=field_name in serialized_fields,
            )
        calculated = _field_digest(fields, source_tier=current)
        if calculated != marker["payload_sha256"]:
            raise ValueError(
                "LLM payload fields disagree with durable retention provenance"
            )
        return str(marker["payload_sha256"])
    return _field_digest(
        fields,
        source_tier=current,
        trust_retention_envelopes=False,
    )


def validate_llm_call_payload_retention_update(
    current: LLMCallRecord,
    target: LLMCallRecord,
    *,
    expected_payload_sha256: str,
    expected_tier: PayloadRetentionTier,
    provider_chain_head: bool | None = None,
) -> PayloadRetentionTier:
    """Validate one monotonic, content-free LLM payload reduction.

    The durable marker is provenance, not authority to trust arbitrary target
    fields.  Recompute the target digest from strict content-free envelopes and
    bind its source-observability digest to the current row before storage may
    replace any payload-bearing columns.
    """

    if not _llm_call_is_terminal(current):
        raise ValueError("nonterminal LLM call payloads cannot be retained")
    if llm_call_payload_is_runtime_dependency(
        current,
        provider_chain_head=provider_chain_head,
    ):
        raise ValueError("runtime-dependent LLM call payloads cannot be retained")
    current_tier = llm_call_payload_retention_tier(current)
    if current_tier is not expected_tier:
        raise ValueError("LLM payload retention source tier changed")
    if (
        not _valid_sha256(expected_payload_sha256)
        or llm_call_payload_sha256(current) != expected_payload_sha256
    ):
        raise ValueError("LLM payload retention source digest changed")

    marker, target_tier = _strict_llm_retention_marker(target.observability)
    valid_transition = (
        current_tier is PayloadRetentionTier.FULL
        and target_tier is PayloadRetentionTier.SUMMARY
    ) or (
        current_tier is PayloadRetentionTier.SUMMARY
        and target_tier is PayloadRetentionTier.HASH_ONLY
    ) or (
        current_tier is PayloadRetentionTier.SUMMARY
        and target_tier is PayloadRetentionTier.SUMMARY
        and not _has_retention_marker(current.observability)
    )
    if not valid_transition:
        raise ValueError("LLM payload retention transition is not monotonic")

    canonical = retain_llm_call_payload(
        current,
        target_tier,
        provider_chain_head=provider_chain_head,
    )
    if _llm_payload_write_projection(target) != _llm_payload_write_projection(
        canonical
    ):
        raise ValueError("LLM payload retention target is not canonical")
    if marker["payload_sha256"] != expected_payload_sha256:
        raise ValueError("LLM payload retention target disagrees with its provenance")
    return target_tier


def validate_external_effect_payload_retention_update(
    current: ExternalEffectRecord,
    target: ExternalEffectRecord,
    *,
    expected_payload_sha256: str,
    expected_tier: PayloadRetentionTier,
    expected_effect_state: str,
    expected_transaction_state: str,
) -> PayloadRetentionTier:
    """Validate a canonical, content-free external-effect payload reduction."""

    if (
        current.effect_state != expected_effect_state
        or current.transaction_state != expected_transaction_state
        or not external_effect_payload_is_terminal(current)
    ):
        raise ValueError("external-effect retention source state changed")
    current_tier = external_effect_payload_retention_tier(current)
    if current_tier is not expected_tier:
        raise ValueError("external-effect retention source tier changed")
    if (
        not _valid_sha256(expected_payload_sha256)
        or external_effect_payload_sha256(current) != expected_payload_sha256
    ):
        raise ValueError("external-effect retention source digest changed")
    target_tier = external_effect_payload_retention_tier(target)
    valid_transition = (
        current_tier is PayloadRetentionTier.FULL
        and target_tier is PayloadRetentionTier.SUMMARY
    ) or (
        current_tier is PayloadRetentionTier.SUMMARY
        and target_tier is PayloadRetentionTier.HASH_ONLY
    )
    if not valid_transition:
        raise ValueError("external-effect payload retention transition is not monotonic")
    canonical = retain_external_effect_payload(current, target_tier)
    if _external_effect_payload_write_projection(
        target
    ) != _external_effect_payload_write_projection(canonical):
        raise ValueError("external-effect payload retention target is not canonical")
    return target_tier


def external_effect_payload_sha256(record: ExternalEffectRecord) -> str:
    current = external_effect_payload_retention_tier(record)
    fields = {
        "provider_metadata": record.provider_metadata,
        "provider_receipt": record.provider_receipt,
    }
    if current is PayloadRetentionTier.FULL:
        return _field_digest(fields, source_tier=current)
    for value in fields.values():
        _strict_payload_retention_envelope(
            value,
            tier=current,
            serialized=False,
        )
    calculated = _field_digest(fields, source_tier=current)
    expected = record.payload_retention_sha256
    if not _valid_sha256(expected) or calculated != expected:
        raise ValueError(
            "external effect payload disagrees with durable retention provenance"
        )
    return str(expected)


def external_effect_payload_is_terminal(record: ExternalEffectRecord) -> bool:
    return (
        record.effect_state == "finalized"
        and record.transaction_state in _TERMINAL_EFFECT_TRANSACTION_STATES
    )


def llm_call_payload_is_runtime_dependency(
    record: LLMCallRecord,
    *,
    provider_chain_head: bool | None = None,
) -> bool:
    """Protect rows that still carry executable/resume semantics.

    A Responses chain head is selected from the latest call at runtime, and
    context-compressor recovery can reconstruct a missing result
    from a durable ``process_exit`` tool call. Until those semantics have a
    separate durable projection, retention must preserve these payloads. The
    predicate is intentionally conservative: a terminal process may retain an
    otherwise stale anchor, but a live or pending process can never lose it.
    """

    if provider_chain_head is not None and not isinstance(provider_chain_head, bool):
        raise ValueError("provider-chain head classification must be a boolean")
    if llm_call_payload_can_be_image_only_request_anchor(record):
        # A failed first image_only request can be the only durable copy of
        # the original goal after a Runtime reopen. The request-purpose stream
        # gets a content-free success tombstone once a complete transcript is
        # durable, so only its actual latest row remains protected.
        return provider_chain_head is not False
    if llm_call_payload_can_be_image_only_transcript_head(record):
        # Transparent replay is reconstructed from the latest complete call
        # and its paired output rows. Older heads may follow ordinary staged
        # retention after a newer complete head supersedes them.
        return provider_chain_head is not False
    if llm_call_payload_can_be_provider_chain_head(record):
        # Standalone callers without a typed storage classification remain
        # fail-closed. Maintenance pages always pass the actual latest-call
        # decision produced by the backend query.
        return provider_chain_head is not False
    legacy_tool_calls = _legacy_observation_metadata(record.tool_calls)
    if legacy_tool_calls is not None and bool(legacy_tool_calls.get("truncated")):
        return True
    tool_calls = _decode_retained_tool_calls(record.tool_calls)
    if legacy_tool_calls is not None and not isinstance(tool_calls, list):
        return True
    if not isinstance(tool_calls, list):
        return False
    try:
        from agent_libos.llm.tool_protocol import tool_call_to_action
    except ImportError:
        return True
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        try:
            action = tool_call_to_action(tool_call)
        except Exception:
            continue
        if action.get("action") == "process_exit":
            return True
    return False


def llm_call_payload_can_be_provider_chain_head(record: LLMCallRecord) -> bool:
    """Return whether a latest call would carry provider-chain resume state."""

    return bool(
        record.status == "ok"
        and record.pid is not None
        and record.api == "responses"
        and record.response_id
        and record.request_options.get("openai_provider_chain_eligible") is True
    )


def llm_call_payload_can_be_image_only_transcript_head(
    record: LLMCallRecord,
) -> bool:
    marker = record.request_options.get(_IMAGE_ONLY_TRANSCRIPT_KEY)
    return bool(
        record.status == "ok"
        and record.pid is not None
        and isinstance(marker, dict)
        and marker.get("schema_version") == _IMAGE_ONLY_TRANSCRIPT_SCHEMA_VERSION
    )


def llm_call_payload_can_be_image_only_request_anchor(
    record: LLMCallRecord,
) -> bool:
    marker = record.request_options.get(_IMAGE_ONLY_REQUEST_KEY)
    if not (
        record.status == "error"
        and record.pid is not None
        and record.purpose.startswith(_IMAGE_ONLY_REQUEST_PURPOSE_PREFIX)
        and isinstance(record.messages, list)
        and isinstance(marker, dict)
        and marker.get("schema_version") == _IMAGE_ONLY_REQUEST_SCHEMA_VERSION
        and marker.get("canonical_message_count") == 2
        and marker.get("purpose") == record.purpose
    ):
        return False
    for key in (
        "image_id",
        "goal_oid",
        "system_prompt_sha256",
        "llm_context_generation",
    ):
        if not isinstance(marker.get(key), str):
            return False
    if not isinstance(marker.get("labels"), dict):
        return False
    input_oids = marker.get("input_oids")
    return bool(
        isinstance(input_oids, list)
        and all(isinstance(oid, str) and oid for oid in input_oids)
        and len(record.messages) >= 2
        and all(isinstance(message, dict) for message in record.messages[:2])
    )


def llm_call_payload_requires_latest_guard(record: LLMCallRecord) -> bool:
    return bool(
        llm_call_payload_can_be_provider_chain_head(record)
        or llm_call_payload_can_be_image_only_transcript_head(record)
        or llm_call_payload_can_be_image_only_request_anchor(record)
    )


def _llm_call_is_terminal(record: LLMCallRecord) -> bool:
    return (
        record.status in _TERMINAL_LLM_CALL_STATES
        and bool(record.completed_at)
    )


def _decode_retained_tool_calls(value: Any) -> Any:
    if isinstance(value, dict) and isinstance(value.get("preview"), str):
        try:
            from agent_libos.utils.serde import loads

            return loads(value["preview"])
        except (TypeError, ValueError):
            return None
    return value


def _llm_payload_sources(
    record: LLMCallRecord,
    *,
    current: PayloadRetentionTier,
) -> dict[str, Any]:
    sources = {field_name: getattr(record, field_name) for field_name in _LLM_PAYLOAD_FIELDS}
    if current is PayloadRetentionTier.SUMMARY and not _has_retention_marker(
        record.observability
    ):
        for field_name in _LLM_PAYLOAD_FIELDS:
            observed = record.observability.get(field_name)
            if _legacy_observation_metadata(observed) is not None:
                sources[field_name] = observed
    return sources


def _source_observability_sha256(record: LLMCallRecord) -> str:
    marker = _retention_metadata(record.observability)
    if _has_retention_marker(record.observability) and marker is not None and _valid_sha256(
        marker.get("source_observability_sha256")
    ):
        return str(marker["source_observability_sha256"])
    return _json_sha256(record.observability)


def _looks_like_legacy_llm_summary(record: LLMCallRecord) -> bool:
    return all(
        _legacy_observation_metadata(value) is not None
        for value in (record.messages, record.tools, record.tool_calls)
    )


def _field_digest(
    fields: dict[str, Any],
    *,
    source_tier: PayloadRetentionTier = PayloadRetentionTier.FULL,
    trust_retention_envelopes: bool = True,
) -> str:
    field_hashes = {
        name: _payload_value_sha256(
            value,
            source_tier=source_tier,
            trust_retention_envelopes=trust_retention_envelopes,
        )
        for name, value in sorted(fields.items())
    }
    return _json_sha256(field_hashes)


def _llm_payload_write_projection(record: LLMCallRecord) -> tuple[Any, ...]:
    """Return the exact columns changed by the SQL retention update."""

    return (
        dumps(record.messages),
        dumps(record.tools),
        record.response_content,
        dumps(record.tool_calls),
        dumps(record.reasoning) if record.reasoning is not None else None,
        dumps(record.raw_response) if record.raw_response is not None else None,
        dumps(record.observability),
        record.error,
    )


def _external_effect_payload_write_projection(
    record: ExternalEffectRecord,
) -> tuple[Any, ...]:
    """Return the exact columns changed by the SQL retention update."""

    return (
        dumps(record.provider_metadata),
        dumps(record.provider_receipt),
        record.payload_retention_schema_version,
        record.payload_retention_tier,
        record.payload_retention_sha256,
    )


def _payload_value_sha256(
    value: Any,
    *,
    source_tier: PayloadRetentionTier,
    trust_retention_envelopes: bool,
) -> str:
    if source_tier is not PayloadRetentionTier.FULL:
        marker = (
            _retention_metadata_from_value(value)
            if trust_retention_envelopes
            else None
        )
        if marker is not None and _valid_sha256(marker.get("sha256")):
            return str(marker["sha256"])
        legacy = _legacy_observation_metadata(value)
        if legacy is not None:
            return str(legacy["sha256"])
    return _json_sha256(value)


def _retention_metadata_from_value(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            from agent_libos.utils.serde import loads

            decoded = loads(value)
        except (TypeError, ValueError):
            decoded = None
        marker = _retention_metadata(decoded)
        if marker is not None:
            return marker
    return _retention_metadata(value)


def _strict_llm_retention_marker(
    value: Any,
) -> tuple[dict[str, Any], PayloadRetentionTier]:
    marker = _retention_metadata(value)
    required = {
        "schema_version",
        "tier",
        "payload_sha256",
        "source_observability_sha256",
    }
    if marker is None or set(marker) != required:
        raise ValueError("LLM payload retention marker is invalid")
    if not _valid_sha256(marker.get("payload_sha256")) or not _valid_sha256(
        marker.get("source_observability_sha256")
    ):
        raise ValueError("LLM payload retention marker digest is invalid")
    return marker, PayloadRetentionTier(str(marker["tier"]))


def _strict_payload_retention_envelope(
    value: Any,
    *,
    tier: PayloadRetentionTier,
    serialized: bool,
) -> dict[str, Any]:
    if serialized:
        if not isinstance(value, str):
            raise ValueError("serialized LLM payload envelope must be text")
    elif not isinstance(value, dict):
        raise ValueError("structured payload retention envelope must be an object")
    marker = _retention_metadata_from_value(value)
    base_keys = {"schema_version", "tier", "sha256"}
    if marker is None or marker.get("tier") != tier.value:
        raise ValueError("LLM payload retention field envelope is invalid")
    if not _valid_sha256(marker.get("sha256")):
        raise ValueError("LLM payload retention field digest is invalid")
    if tier is PayloadRetentionTier.HASH_ONLY:
        if set(marker) != base_keys:
            raise ValueError("hash-only LLM payload envelope contains extra data")
        return marker
    if tier is not PayloadRetentionTier.SUMMARY:
        raise ValueError("full LLM payloads cannot use retention envelopes")
    required = base_keys | {"bytes", "json_kind"}
    allowed = required | {"item_count"}
    if not required.issubset(marker) or not set(marker).issubset(allowed):
        raise ValueError("summary LLM payload envelope has an invalid shape")
    byte_count = marker["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise ValueError("summary LLM payload envelope bytes are invalid")
    if marker["json_kind"] not in {"null", "object", "array", "scalar", "unknown"}:
        raise ValueError("summary LLM payload envelope kind is invalid")
    if "item_count" in marker:
        item_count = marker["item_count"]
        if (
            isinstance(item_count, bool)
            or not isinstance(item_count, int)
            or item_count < 0
        ):
            raise ValueError("summary LLM payload envelope item count is invalid")
    return marker


def _retention_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if set(value) != {_RETENTION_KEY}:
        return None
    selected = value.get(_RETENTION_KEY)
    if not isinstance(selected, dict):
        return None
    schema_version = selected.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != _RETENTION_SCHEMA_VERSION
    ):
        return None
    try:
        tier = PayloadRetentionTier(str(selected.get("tier")))
    except ValueError:
        return None
    if "sha256" in selected and not _valid_sha256(selected.get("sha256")):
        return None
    if tier is PayloadRetentionTier.FULL:
        return None
    return selected


def _has_retention_marker(value: Any) -> bool:
    selected = _retention_metadata(value)
    return selected is not None and _valid_sha256(selected.get("payload_sha256"))


def _legacy_observation_metadata(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if not _valid_sha256(value.get("sha256")):
        return None
    byte_count = value.get("bytes")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        return None
    if "preview" not in value:
        return None
    return value


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(dumps(to_jsonable(value)).encode("utf-8")).hexdigest()


def _json_kind(value: Any) -> str:
    selected = to_jsonable(value)
    if selected is None:
        return "null"
    if isinstance(selected, dict):
        return "object"
    if isinstance(selected, list):
        return "array"
    return "scalar"


def _item_count(value: Any) -> int | None:
    selected = to_jsonable(value)
    return len(selected) if isinstance(selected, (dict, list)) else None


def _tier_rank(tier: PayloadRetentionTier) -> int:
    return {
        PayloadRetentionTier.FULL: 0,
        PayloadRetentionTier.SUMMARY: 1,
        PayloadRetentionTier.HASH_ONLY: 2,
    }[tier]


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return _normalized_datetime(
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        )
    except (TypeError, ValueError):
        return None


def _normalized_datetime(value: datetime) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("payload retention now must be a datetime")
    if value.tzinfo is None:
        raise ValueError("payload retention timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _candidate_set_digest(record_ids: tuple[str, ...] | list[str]) -> str:
    return _json_sha256(list(record_ids))


def _validate_page(
    page: PayloadRetentionPage[Any],
    *,
    limit: int,
    after: PayloadRetentionCursor | None,
    id_for: Any,
) -> None:
    if len(page.records) > limit:
        raise ValueError("payload retention backend returned more than the requested limit")
    cursors = [
        PayloadRetentionCursor(str(item.created_at), str(id_for(item)))
        for item in page.records
    ]
    if cursors != sorted(cursors):
        raise ValueError("payload retention backend page is not keyset ordered")
    if len(cursors) != len(set(cursors)):
        raise ValueError("payload retention backend page contains duplicate cursors")
    if after is not None and cursors and cursors[0] <= after:
        raise ValueError("payload retention backend did not advance past the cursor")
    if page.next_cursor is not None and page.next_cursor != cursors[-1]:
        raise ValueError("payload retention next cursor must identify the last row")
