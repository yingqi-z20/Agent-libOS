from __future__ import annotations

import json
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.payload_retention import (
    PayloadRetentionCursor,
    PayloadRetentionKind,
    PayloadRetentionMaintenance,
    PayloadRetentionPage,
    PayloadRetentionPolicy,
    PayloadRetentionRequest,
    PayloadRetentionTier,
    content_free_payload_envelope,
    external_effect_payload_retention_tier,
    external_effect_payload_sha256,
    human_request_payload_sha256,
    llm_call_payload_is_runtime_dependency,
    llm_call_payload_retention_tier,
    llm_call_payload_sha256,
    redact_task_run_external_effects,
    redact_task_run_human_requests,
    redact_terminal_task_run_human_request,
    retain_external_effect_payload,
    retain_llm_call_payload,
    validate_external_effect_payload_retention_update,
    validate_terminal_task_run_human_redaction,
)
from agent_libos.llm.records import observable_llm_call_fields
from agent_libos.models.audit import AuditRecord
from agent_libos.models.external_effect import (
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
)
from agent_libos.models.human import HumanRequest, HumanRequestStatus
from agent_libos.models.llm import LLMCallRecord


_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
_OLD = "2026-01-01T00:00:00+00:00"
_SENTINEL = "RETENTION_PAYLOAD_MUST_NOT_SURVIVE"


def _llm_call(
    call_id: str = "call-1",
    *,
    status: str = "ok",
    completed_at: str | None = _OLD,
    tool_calls: Any = None,
    api: str | None = None,
    response_id: str | None = None,
    request_options: dict[str, Any] | None = None,
) -> LLMCallRecord:
    return LLMCallRecord(
        call_id=call_id,
        pid="pid-retention",
        image_id="image:v1",
        purpose="action_selection",
        status=status,
        api=api,
        response_id=response_id,
        messages=[{"role": "user", "content": _SENTINEL}],
        tools=[{"name": "secret-tool", "description": _SENTINEL}],
        request_options=request_options or {"llm_profile_id": "default"},
        response_content=f"assistant {_SENTINEL}",
        tool_calls=(
            [{"name": "noop", "arguments": {"secret": _SENTINEL}}]
            if tool_calls is None
            else tool_calls
        ),
        reasoning={"trace": _SENTINEL},
        usage={"total_tokens": 12},
        raw_response={"raw": _SENTINEL},
        observability={"legacy_preview": _SENTINEL},
        error=f"provider error {_SENTINEL}" if status == "error" else None,
        created_at=_OLD,
        completed_at=completed_at,
    )


def _effect(
    effect_id: str,
    *,
    effect_state: str = "finalized",
    transaction_state: str = "committed",
) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id="audit-source",
        event_id="event-source",
        pid="pid-retention",
        provider="remote",
        operation="write",
        target="opaque-target",
        rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
        rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
        state_mutation=True,
        information_flow=True,
        provider_metadata={"request": _SENTINEL},
        provider_receipt={"response": _SENTINEL},
        canonical_args_hash="a" * 64,
        idempotency_key="stable-idempotency-key",
        effect_state=effect_state,
        transaction_state=transaction_state,
        created_at=_OLD,
        updated_at=_OLD,
    )


def _human_request(request_id: str = "human-run-terminal") -> HumanRequest:
    return HumanRequest(
        request_id=request_id,
        pid="pid-retention",
        human="host",
        payload={
            "type": "question",
            "question": f"prompt {_SENTINEL}",
            "context": {"secret": _SENTINEL},
        },
        status=HumanRequestStatus.APPROVED,
        decision={"answer": f"answer {_SENTINEL}"},
        blocking=True,
        created_at=_OLD,
        updated_at=_OLD,
    )


class _Audit:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.records: list[AuditRecord] = []

    def record(self, actor: str, action: str, **kwargs: Any) -> AuditRecord:
        if self.fail:
            raise RuntimeError("injected audit failure")
        record = AuditRecord(
            record_id=f"audit-{len(self.records) + 1}",
            timestamp=_NOW.isoformat(),
            actor=actor,
            action=action,
            target=kwargs.get("target"),
            input_refs=list(kwargs.get("input_refs") or []),
            output_refs=list(kwargs.get("output_refs") or []),
            capability_refs=list(kwargs.get("capability_refs") or []),
            decision=kwargs.get("decision"),
            correlation_id=kwargs.get("correlation_id"),
        )
        self.records.append(record)
        return record


class _Store:
    def __init__(
        self,
        *,
        llm_calls: list[LLMCallRecord] | None = None,
        effects: list[ExternalEffectRecord] | None = None,
        conflict_ids: set[str] | None = None,
    ) -> None:
        self.llm_calls = {record.call_id: record for record in llm_calls or []}
        self.effects = {record.effect_id: record for record in effects or []}
        self.conflict_ids = set(conflict_ids or set())
        self.scan_count = 0
        self.update_count = 0

    @contextmanager
    def transaction(self) -> Iterator[None]:
        before = (deepcopy(self.llm_calls), deepcopy(self.effects), self.update_count)
        try:
            yield
        except BaseException:
            self.llm_calls, self.effects, self.update_count = before
            raise

    def scan_llm_call_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[LLMCallRecord]:
        self.scan_count += 1
        selected = sorted(
            (
                record
                for record in self.llm_calls.values()
                if record.created_at <= older_than
                and (
                    after is None
                    or (record.created_at, record.call_id)
                    > (after.created_at, after.record_id)
                )
            ),
            key=lambda record: (record.created_at, record.call_id),
        )
        records = tuple(selected[:limit])
        latest_by_chain: dict[tuple[str, str], LLMCallRecord] = {}
        for candidate in self.llm_calls.values():
            if candidate.pid is None:
                continue
            key = (candidate.pid, candidate.purpose)
            current = latest_by_chain.get(key)
            if current is None or (candidate.created_at, candidate.call_id) > (
                current.created_at,
                current.call_id,
            ):
                latest_by_chain[key] = candidate
        latest_ids = frozenset(
            record.call_id
            for record in records
            if record.pid is not None
            and latest_by_chain.get((record.pid, record.purpose)) is record
        )
        next_cursor = (
            PayloadRetentionCursor(records[-1].created_at, records[-1].call_id)
            if len(selected) > limit and records
            else None
        )
        return PayloadRetentionPage(
            records,
            next_cursor,
            latest_llm_call_ids=latest_ids,
        )

    def update_llm_call_payload_retention(
        self,
        record: LLMCallRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
    ) -> bool:
        current = self.llm_calls.get(record.call_id)
        if (
            current is None
            or record.call_id in self.conflict_ids
            or llm_call_payload_sha256(current) != expected_payload_sha256
            or llm_call_payload_retention_tier(current) is not expected_tier
        ):
            return False
        self.llm_calls[record.call_id] = record
        self.update_count += 1
        return True

    def scan_external_effect_payloads_for_retention(
        self,
        *,
        older_than: str,
        after: PayloadRetentionCursor | None,
        limit: int,
    ) -> PayloadRetentionPage[ExternalEffectRecord]:
        self.scan_count += 1
        selected = sorted(
            (
                record
                for record in self.effects.values()
                if record.created_at <= older_than
                and (
                    after is None
                    or (record.created_at, record.effect_id)
                    > (after.created_at, after.record_id)
                )
            ),
            key=lambda record: (record.created_at, record.effect_id),
        )
        records = tuple(selected[:limit])
        next_cursor = (
            PayloadRetentionCursor(records[-1].created_at, records[-1].effect_id)
            if len(selected) > limit and records
            else None
        )
        return PayloadRetentionPage(records, next_cursor)

    def update_external_effect_payload_retention(
        self,
        record: ExternalEffectRecord,
        *,
        expected_payload_sha256: str,
        expected_tier: PayloadRetentionTier,
        expected_effect_state: str,
        expected_transaction_state: str,
    ) -> bool:
        current = self.effects.get(record.effect_id)
        if (
            current is None
            or record.effect_id in self.conflict_ids
            or current.effect_state != expected_effect_state
            or current.transaction_state != expected_transaction_state
            or external_effect_payload_sha256(current) != expected_payload_sha256
            or external_effect_payload_retention_tier(current) is not expected_tier
        ):
            return False
        self.effects[record.effect_id] = record
        self.update_count += 1
        return True


class _TaskRunEffectStore:
    def __init__(
        self,
        effects: list[ExternalEffectRecord],
        *,
        conflict_at: int | None = None,
    ) -> None:
        self.run_id = "run-retention"
        self.owned_pids = {"pid-retention"}
        self.links = {record.effect_id for record in effects}
        self.effects = {record.effect_id: record for record in effects}
        self.conflict_at = conflict_at
        self.cas_calls: list[tuple[str, str, str]] = []
        self.listed_override: list[Any] | None = None

    def list_task_run_external_effects(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[ExternalEffectRecord]:
        if run_id != self.run_id:
            raise ValueError("unknown TaskRun")
        selected = tuple(linked_pids)
        if selected and not set(selected).issubset(self.owned_pids):
            raise ValueError("unowned TaskRun PID")
        if self.listed_override is not None:
            return list(self.listed_override)
        scope = set(selected) or self.owned_pids
        return [
            record
            for effect_id, record in sorted(self.effects.items())
            if effect_id in self.links and record.pid in scope
        ]

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
        current = self.effects.get(record.effect_id)
        if (
            run_id != self.run_id
            or current is None
            or current.pid not in self.owned_pids
            or record.effect_id not in self.links
        ):
            return False
        self.cas_calls.append(
            (record.effect_id, expected_tier, record.payload_retention_tier)
        )
        if self.conflict_at == len(self.cas_calls):
            return False
        try:
            validate_external_effect_payload_retention_update(
                current,
                record,
                expected_payload_sha256=expected_payload_sha256,
                expected_tier=PayloadRetentionTier(expected_tier),
                expected_effect_state=expected_effect_state,
                expected_transaction_state=expected_transaction_state,
            )
        except (TypeError, ValueError):
            return False
        self.effects[record.effect_id] = record
        return True


class _TaskRunHumanStore:
    def __init__(
        self,
        requests: list[HumanRequest],
        *,
        conflict: bool = False,
    ) -> None:
        self.run_id = "run-retention"
        self.owned_pids = {"pid-retention"}
        self.links = {request.request_id for request in requests}
        self.requests = {request.request_id: request for request in requests}
        self.conflict = conflict
        self.cas_calls = 0
        self.listed_override: list[Any] | None = None

    def list_task_run_human_requests(
        self,
        run_id: str,
        linked_pids: Iterable[str] = (),
    ) -> list[HumanRequest]:
        if run_id != self.run_id:
            raise ValueError("unknown TaskRun")
        selected = tuple(linked_pids)
        if selected and not set(selected).issubset(self.owned_pids):
            raise ValueError("unowned TaskRun PID")
        if self.listed_override is not None:
            return list(self.listed_override)
        scope = set(selected) or self.owned_pids
        return [
            request
            for request_id, request in sorted(self.requests.items())
            if request_id in self.links and request.pid in scope
        ]

    def redact_task_run_human_request_payload(
        self,
        request: HumanRequest,
        *,
        run_id: str,
        expected_payload_sha256: str,
        expected_status: str,
    ) -> bool:
        current = self.requests.get(request.request_id)
        self.cas_calls += 1
        if (
            self.conflict
            or run_id != self.run_id
            or current is None
            or current.pid not in self.owned_pids
            or request.request_id not in self.links
        ):
            return False
        try:
            validate_terminal_task_run_human_redaction(
                current,
                request,
                expected_payload_sha256=expected_payload_sha256,
                expected_status=HumanRequestStatus(expected_status),
            )
        except (TypeError, ValueError):
            return False
        self.requests[request.request_id] = request
        return True


def _policy() -> PayloadRetentionPolicy:
    return PayloadRetentionPolicy(
        enabled=True,
        summary_after_seconds=0,
        hash_only_after_seconds=0,
        batch_limit=10,
        hard_limit=20,
    )


def test_content_free_summary_and_hash_only_envelopes_preserve_original_digest() -> None:
    source = {"secret": _SENTINEL, "nested": [1, 2, 3]}

    summary = content_free_payload_envelope(source, PayloadRetentionTier.SUMMARY)
    hash_only = content_free_payload_envelope(
        summary,
        PayloadRetentionTier.HASH_ONLY,
        source_tier=PayloadRetentionTier.SUMMARY,
    )

    serialized_summary = json.dumps(summary, sort_keys=True)
    serialized_hash = json.dumps(hash_only, sort_keys=True)
    assert _SENTINEL not in serialized_summary
    assert "preview" not in serialized_summary
    assert summary["$agent_libos_payload_retention"]["json_kind"] == "object"
    assert summary["$agent_libos_payload_retention"]["item_count"] == 2
    assert (
        summary["$agent_libos_payload_retention"]["sha256"]
        == hash_only["$agent_libos_payload_retention"]["sha256"]
    )
    assert "bytes" not in serialized_hash


def test_full_llm_response_does_not_trust_embedded_retention_envelope() -> None:
    forged_sha256 = "f" * 64
    forged_envelope = {
        "$agent_libos_payload_retention": {
            "schema_version": 1,
            "tier": "summary",
            "sha256": forged_sha256,
            "bytes": 1,
            "json_kind": "scalar",
        }
    }
    original = replace(
        _llm_call(tool_calls=[]),
        response_content=json.dumps(forged_envelope, sort_keys=True),
    )
    original_sha256 = llm_call_payload_sha256(original)

    summary = retain_llm_call_payload(original, PayloadRetentionTier.SUMMARY)
    hash_only = retain_llm_call_payload(summary, PayloadRetentionTier.HASH_ONLY)

    response_summary = json.loads(summary.response_content or "null")
    assert llm_call_payload_retention_tier(original) is PayloadRetentionTier.FULL
    assert (
        response_summary["$agent_libos_payload_retention"]["sha256"]
        != forged_sha256
    )
    assert llm_call_payload_sha256(summary) == original_sha256
    assert llm_call_payload_sha256(hash_only) == original_sha256


@pytest.mark.parametrize("field_name", ["provider_metadata", "provider_receipt"])
def test_full_effect_does_not_trust_embedded_retention_envelope(
    field_name: str,
) -> None:
    forged_sha256 = "e" * 64
    forged_envelope = {
        "$agent_libos_payload_retention": {
            "schema_version": 1,
            "tier": "summary",
            "sha256": forged_sha256,
            "bytes": 1,
            "json_kind": "object",
        }
    }
    original = replace(
        _effect(f"effect-forged-{field_name}"),
        **{field_name: forged_envelope},
    )
    original_sha256 = external_effect_payload_sha256(original)

    summary = retain_external_effect_payload(original, PayloadRetentionTier.SUMMARY)
    hash_only = retain_external_effect_payload(
        summary,
        PayloadRetentionTier.HASH_ONLY,
    )

    selected_summary = getattr(summary, field_name)
    assert external_effect_payload_retention_tier(original) is PayloadRetentionTier.FULL
    assert (
        selected_summary["$agent_libos_payload_retention"]["sha256"]
        != forged_sha256
    )
    assert external_effect_payload_sha256(summary) == original_sha256
    assert external_effect_payload_sha256(hash_only) == original_sha256


@pytest.mark.parametrize(
    ("metadata_tier", "receipt_tier"),
    [("summary", "summary"), ("summary", "hash_only")],
)
def test_full_effect_does_not_trust_two_embedded_retention_envelopes(
    metadata_tier: str,
    receipt_tier: str,
) -> None:
    metadata_sha256 = "c" * 64
    receipt_sha256 = "d" * 64

    def forged_envelope(tier: str, sha256: str) -> dict[str, Any]:
        return {
            "$agent_libos_payload_retention": {
                "schema_version": 1,
                "tier": tier,
                "sha256": sha256,
            }
        }

    original = replace(
        _effect(f"effect-double-forged-{metadata_tier}-{receipt_tier}"),
        provider_metadata=forged_envelope(metadata_tier, metadata_sha256),
        provider_receipt=forged_envelope(receipt_tier, receipt_sha256),
    )
    original_sha256 = external_effect_payload_sha256(original)

    summary = retain_external_effect_payload(original, PayloadRetentionTier.SUMMARY)
    hash_only = retain_external_effect_payload(
        summary,
        PayloadRetentionTier.HASH_ONLY,
    )

    assert external_effect_payload_retention_tier(original) is PayloadRetentionTier.FULL
    assert summary.payload_retention_sha256 == original_sha256
    assert (
        summary.provider_metadata["$agent_libos_payload_retention"]["sha256"]
        != metadata_sha256
    )
    assert (
        summary.provider_receipt["$agent_libos_payload_retention"]["sha256"]
        != receipt_sha256
    )
    assert external_effect_payload_sha256(summary) == original_sha256
    assert external_effect_payload_sha256(hash_only) == original_sha256


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_external_effect_retention_schema_version_is_a_strict_integer(
    schema_version: Any,
) -> None:
    with pytest.raises(ValueError, match="schema version"):
        replace(
            _effect("effect-invalid-retention-schema"),
            payload_retention_schema_version=schema_version,
        )


def test_llm_payload_reduction_preserves_identity_usage_and_monotonic_hash() -> None:
    original = _llm_call(status="error")
    original_sha = llm_call_payload_sha256(original)

    summary = retain_llm_call_payload(original, PayloadRetentionTier.SUMMARY)
    hash_only = retain_llm_call_payload(summary, PayloadRetentionTier.HASH_ONLY)

    assert summary.call_id == original.call_id
    assert summary.pid == original.pid
    assert summary.request_options == original.request_options
    assert summary.usage == original.usage
    assert summary.created_at == original.created_at
    assert summary.completed_at == original.completed_at
    assert llm_call_payload_retention_tier(summary) is PayloadRetentionTier.SUMMARY
    assert llm_call_payload_retention_tier(hash_only) is PayloadRetentionTier.HASH_ONLY
    assert llm_call_payload_sha256(summary) == original_sha
    assert llm_call_payload_sha256(hash_only) == original_sha
    assert _SENTINEL not in json.dumps(summary.__dict__, sort_keys=True)
    assert _SENTINEL not in json.dumps(hash_only.__dict__, sort_keys=True)
    assert "preview" not in json.dumps(summary.__dict__, sort_keys=True)


def test_legacy_persist_full_io_opt_out_row_normalizes_without_copying_preview() -> None:
    legacy = _llm_call()
    legacy_observation = {
        "preview": _SENTINEL,
        "sha256": "b" * 64,
        "bytes": 999,
        "truncated": True,
    }
    complete_tool_calls_observation = {
        "preview": "[]",
        "sha256": "d" * 64,
        "bytes": 2,
        "truncated": False,
    }
    legacy = replace(
        legacy,
        messages=dict(legacy_observation),
        tools=dict(legacy_observation),
        tool_calls=dict(complete_tool_calls_observation),
        response_content=_SENTINEL,
        raw_response=dict(legacy_observation),
        reasoning=dict(legacy_observation),
        observability={
            "messages": dict(legacy_observation),
            "tools": dict(legacy_observation),
            "response_content": dict(legacy_observation),
            "tool_calls": dict(complete_tool_calls_observation),
            "reasoning": dict(legacy_observation),
            "raw_response": dict(legacy_observation),
        },
    )

    assert llm_call_payload_retention_tier(legacy) is PayloadRetentionTier.SUMMARY
    normalized = retain_llm_call_payload(legacy, PayloadRetentionTier.SUMMARY)

    assert llm_call_payload_retention_tier(normalized) is PayloadRetentionTier.SUMMARY
    assert _SENTINEL not in json.dumps(normalized.__dict__, sort_keys=True)
    assert normalized.messages["$agent_libos_payload_retention"]["sha256"] == "b" * 64


def test_real_persist_full_io_opt_out_view_is_recognized_and_can_be_normalized() -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
    )
    observable = observable_llm_call_fields(
        messages=[{"role": "user", "content": _SENTINEL}],
        tools=[],
        response_content=_SENTINEL,
        tool_calls=[],
        reasoning={"trace": _SENTINEL},
        raw_response={"raw": _SENTINEL},
        config=config,
    )
    legacy = replace(
        _llm_call(tool_calls=[]),
        **observable,
    )

    assert llm_call_payload_retention_tier(legacy) is PayloadRetentionTier.SUMMARY
    assert not llm_call_payload_is_runtime_dependency(legacy)
    summary_sha256 = llm_call_payload_sha256(legacy)
    assert summary_sha256 == legacy.observability[
        "$agent_libos_payload_retention"
    ]["payload_sha256"]

    normalized = retain_llm_call_payload(legacy, PayloadRetentionTier.SUMMARY)
    hash_only = retain_llm_call_payload(legacy, PayloadRetentionTier.HASH_ONLY)

    assert llm_call_payload_retention_tier(normalized) is PayloadRetentionTier.SUMMARY
    assert llm_call_payload_retention_tier(hash_only) is PayloadRetentionTier.HASH_ONLY
    assert llm_call_payload_sha256(hash_only) == summary_sha256
    assert _SENTINEL not in json.dumps(normalized.__dict__, sort_keys=True)
    assert _SENTINEL not in json.dumps(hash_only.__dict__, sort_keys=True)
    assert "preview" not in json.dumps(normalized.__dict__, sort_keys=True)


def test_external_effect_reduction_preserves_core_evidence_identity_and_hashes() -> None:
    original = _effect("effect-terminal")
    original_sha = external_effect_payload_sha256(original)

    summary = retain_external_effect_payload(original, PayloadRetentionTier.SUMMARY)
    hash_only = retain_external_effect_payload(summary, PayloadRetentionTier.HASH_ONLY)

    assert summary.effect_id == original.effect_id
    assert summary.record_id == original.record_id
    assert summary.event_id == original.event_id
    assert summary.canonical_args_hash == original.canonical_args_hash
    assert summary.idempotency_key == original.idempotency_key
    assert summary.effect_state == "finalized"
    assert summary.transaction_state == "committed"
    assert external_effect_payload_sha256(summary) == original_sha
    assert external_effect_payload_sha256(hash_only) == original_sha
    assert external_effect_payload_retention_tier(summary) is PayloadRetentionTier.SUMMARY
    assert external_effect_payload_retention_tier(hash_only) is PayloadRetentionTier.HASH_ONLY
    assert _SENTINEL not in json.dumps(summary.__dict__, sort_keys=True)


def test_task_run_external_effect_redaction_uses_two_canonical_cas_steps() -> None:
    original = _effect("effect-run-terminal")
    store = _TaskRunEffectStore([original])

    assert redact_task_run_external_effects(
        store,
        store.run_id,
        [original.pid],
    ) == 1

    persisted = store.effects[original.effect_id]
    assert external_effect_payload_retention_tier(
        persisted
    ) is PayloadRetentionTier.HASH_ONLY
    assert store.cas_calls == [
        (original.effect_id, "full", "summary"),
        (original.effect_id, "summary", "hash_only"),
    ]
    assert store.links == {original.effect_id}
    assert replace(
        persisted,
        provider_metadata=original.provider_metadata,
        provider_receipt=original.provider_receipt,
        payload_retention_schema_version=(
            original.payload_retention_schema_version
        ),
        payload_retention_tier=original.payload_retention_tier,
        payload_retention_sha256=original.payload_retention_sha256,
    ) == original
    assert _SENTINEL not in json.dumps(persisted.__dict__, sort_keys=True)


def test_task_run_external_effect_redaction_is_resumable_and_idempotent() -> None:
    summary = retain_external_effect_payload(
        _effect("effect-run-summary"),
        PayloadRetentionTier.SUMMARY,
    )
    store = _TaskRunEffectStore([summary])

    assert redact_task_run_external_effects(
        store,
        store.run_id,
        [summary.pid],
    ) == 1
    assert store.cas_calls == [
        (summary.effect_id, "summary", "hash_only"),
    ]
    assert redact_task_run_external_effects(
        store,
        store.run_id,
        [summary.pid],
    ) == 0
    assert len(store.cas_calls) == 1


@pytest.mark.parametrize("conflict_at", [1, 2])
def test_task_run_external_effect_redaction_conflict_fails_closed(
    conflict_at: int,
) -> None:
    effect = _effect(f"effect-run-conflict-{conflict_at}")
    store = _TaskRunEffectStore([effect], conflict_at=conflict_at)

    with pytest.raises(RuntimeError, match="redaction conflicted"):
        redact_task_run_external_effects(
            store,
            store.run_id,
            [effect.pid],
        )

    assert len(store.cas_calls) == conflict_at


def test_task_run_external_effect_redaction_rejects_unlinked_and_nonterminal_rows() -> None:
    effect = _effect("effect-run-membership")
    store = _TaskRunEffectStore([effect])
    store.listed_override = [replace(effect, pid="pid-other-run")]

    with pytest.raises(RuntimeError, match="unlinked row"):
        redact_task_run_external_effects(
            store,
            store.run_id,
            [effect.pid],
        )
    assert store.cas_calls == []

    store.listed_override = [
        replace(
            effect,
            effect_state="pending",
            transaction_state="unknown",
        )
    ]
    with pytest.raises(ValueError, match="nonterminal"):
        redact_task_run_external_effects(
            store,
            store.run_id,
            [effect.pid],
        )
    assert store.cas_calls == []


def test_task_run_external_effect_redaction_rechecks_link_membership_at_cas() -> None:
    effect = _effect("effect-run-link-race")
    store = _TaskRunEffectStore([effect])
    store.listed_override = [effect]
    store.links.clear()

    with pytest.raises(RuntimeError, match="redaction conflicted"):
        redact_task_run_external_effects(
            store,
            store.run_id,
            [effect.pid],
        )
    assert store.effects[effect.effect_id] == effect


def test_terminal_task_run_human_redaction_is_canonical_and_content_free() -> None:
    original = _human_request()
    original_sha256 = human_request_payload_sha256(original)

    redacted = redact_terminal_task_run_human_request(original)

    assert human_request_payload_sha256(redacted) == original_sha256
    assert redacted.request_id == original.request_id
    assert redacted.pid == original.pid
    assert redacted.human == original.human
    assert redacted.status is original.status
    assert redacted.blocking is original.blocking
    assert redacted.created_at == original.created_at
    assert redacted.updated_at == original.updated_at
    assert redacted.payload[
        "$agent_libos_task_run_human_redaction"
    ]["request_type"] == "question"
    assert _SENTINEL not in json.dumps(redacted.__dict__, sort_keys=True)
    assert redact_terminal_task_run_human_request(redacted) is redacted


def test_task_run_human_redaction_is_linked_and_idempotent() -> None:
    original = _human_request("human-run-linked")
    store = _TaskRunHumanStore([original])

    assert redact_task_run_human_requests(
        store,
        store.run_id,
        [original.pid],
    ) == 1

    persisted = store.requests[original.request_id]
    assert store.cas_calls == 1
    assert _SENTINEL not in json.dumps(persisted.__dict__, sort_keys=True)
    assert human_request_payload_sha256(persisted) == human_request_payload_sha256(
        original
    )
    assert redact_task_run_human_requests(
        store,
        store.run_id,
        [original.pid],
    ) == 0
    assert store.cas_calls == 1


def test_task_run_human_redaction_conflict_and_link_race_fail_closed() -> None:
    original = _human_request("human-run-conflict")
    conflict = _TaskRunHumanStore([original], conflict=True)
    with pytest.raises(RuntimeError, match="redaction conflicted"):
        redact_task_run_human_requests(
            conflict,
            conflict.run_id,
            [original.pid],
        )
    assert conflict.requests[original.request_id] == original

    unlinked = _TaskRunHumanStore([original])
    unlinked.listed_override = [original]
    unlinked.links.clear()
    with pytest.raises(RuntimeError, match="redaction conflicted"):
        redact_task_run_human_requests(
            unlinked,
            unlinked.run_id,
            [original.pid],
        )
    assert unlinked.requests[original.request_id] == original


def test_task_run_human_redaction_rejects_wrong_pid_and_preserves_null_decision() -> None:
    original = replace(
        _human_request("human-run-null"),
        status=HumanRequestStatus.CANCELLED,
        decision=None,
    )
    redacted = redact_terminal_task_run_human_request(original)
    assert redacted.decision is None
    assert human_request_payload_sha256(redacted) == human_request_payload_sha256(
        original
    )

    store = _TaskRunHumanStore([original])
    store.listed_override = [replace(original, pid="pid-other-run")]
    with pytest.raises(RuntimeError, match="unlinked row"):
        redact_task_run_human_requests(
            store,
            store.run_id,
            [original.pid],
        )
    assert store.cas_calls == 0


@pytest.mark.parametrize(
    ("effect_state", "transaction_state"),
    [
        ("pending", "prepared"),
        ("pending", "dispatched"),
        ("pending", "unknown"),
        ("finalized", "unknown"),
    ],
)
def test_nonterminal_external_effect_payload_cannot_be_reduced(
    effect_state: str,
    transaction_state: str,
) -> None:
    record = _effect(
        "effect-protected",
        effect_state=effect_state,
        transaction_state=transaction_state,
    )

    with pytest.raises(ValueError, match="nonterminal"):
        retain_external_effect_payload(record, PayloadRetentionTier.SUMMARY)


def test_direct_reducers_cannot_skip_summary_or_trim_nonterminal_llm_calls() -> None:
    with pytest.raises(ValueError, match="skip the summary"):
        retain_llm_call_payload(_llm_call(tool_calls=[]), PayloadRetentionTier.HASH_ONLY)
    with pytest.raises(ValueError, match="skip the summary"):
        retain_external_effect_payload(
            _effect("effect-no-skip"),
            PayloadRetentionTier.HASH_ONLY,
        )
    with pytest.raises(ValueError, match="nonterminal LLM"):
        retain_llm_call_payload(
            _llm_call("call-live", status="pending", completed_at=None),
            PayloadRetentionTier.SUMMARY,
        )


def test_disabled_policy_does_not_scan_or_mutate_but_audits_attempt() -> None:
    store = _Store(llm_calls=[_llm_call()])
    audit = _Audit()
    maintenance = PayloadRetentionMaintenance(store, audit)

    result = maintenance.run(
        PayloadRetentionRequest(kind=PayloadRetentionKind.LLM_CALL),
        now=_NOW,
    )

    assert result.status == "disabled"
    assert result.audit_record_id == "audit-1"
    assert store.scan_count == 0
    assert store.update_count == 0
    assert audit.records[0].decision["status"] == "disabled"


def test_dry_run_is_bounded_and_returns_opaque_cursor_audit_summary() -> None:
    store = _Store(llm_calls=[_llm_call("call-a"), _llm_call("call-b")])
    audit = _Audit()
    maintenance = PayloadRetentionMaintenance(store, audit, _policy())

    result = maintenance.run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.LLM_CALL,
            dry_run=True,
            limit=1,
        ),
        now=_NOW,
    )

    assert result.status == "dry_run"
    assert result.scanned == 1
    assert result.would_update == 1
    assert result.updated == 0
    assert result.next_cursor == PayloadRetentionCursor(_OLD, "call-a")
    assert store.update_count == 0
    decision_json = json.dumps(audit.records[0].decision, sort_keys=True)
    assert _SENTINEL not in decision_json
    assert "call-a" not in decision_json
    assert audit.records[0].decision["next_cursor_present"] is True
    assert audit.records[0].decision["next_cursor_sha256"]


def test_maintenance_stages_full_to_summary_then_hash_only() -> None:
    store = _Store(llm_calls=[_llm_call(tool_calls=[])])
    audit = _Audit()
    maintenance = PayloadRetentionMaintenance(store, audit, _policy())
    request = PayloadRetentionRequest(
        kind=PayloadRetentionKind.LLM_CALL,
        dry_run=False,
    )

    first = maintenance.run(request, now=_NOW)
    first_record = store.llm_calls["call-1"]
    second = maintenance.run(request, now=_NOW)
    second_record = store.llm_calls["call-1"]

    assert first.updated == 1
    assert first.conflicts == 0
    assert llm_call_payload_retention_tier(first_record) is PayloadRetentionTier.SUMMARY
    assert second.updated == 1
    assert llm_call_payload_retention_tier(second_record) is PayloadRetentionTier.HASH_ONLY
    assert llm_call_payload_sha256(first_record) == llm_call_payload_sha256(second_record)


def test_effect_maintenance_never_trims_pending_or_unknown_rows() -> None:
    terminal = _effect("effect-a-terminal")
    pending = _effect(
        "effect-b-pending",
        effect_state="pending",
        transaction_state="dispatched",
    )
    unknown = _effect(
        "effect-c-unknown",
        effect_state="pending",
        transaction_state="unknown",
    )
    finalized_unknown = _effect(
        "effect-d-finalized-unknown",
        effect_state="finalized",
        transaction_state="unknown",
    )
    store = _Store(effects=[terminal, pending, unknown, finalized_unknown])
    audit = _Audit()
    maintenance = PayloadRetentionMaintenance(store, audit, _policy())

    result = maintenance.run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.EXTERNAL_EFFECT,
            dry_run=False,
        ),
        now=_NOW,
    )

    assert result.updated == 1
    assert result.protected_nonterminal == 3
    assert external_effect_payload_retention_tier(
        store.effects[terminal.effect_id]
    ) is PayloadRetentionTier.SUMMARY
    for protected in (pending, unknown, finalized_unknown):
        assert store.effects[protected.effect_id] == protected
        assert _SENTINEL in json.dumps(protected.__dict__, sort_keys=True)


def test_cas_conflict_is_reported_without_overwriting_record() -> None:
    original = _effect("effect-conflict")
    store = _Store(effects=[original], conflict_ids={original.effect_id})
    audit = _Audit()

    result = PayloadRetentionMaintenance(store, audit, _policy()).run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.EXTERNAL_EFFECT,
            dry_run=False,
        ),
        now=_NOW,
    )

    assert result.would_update == 1
    assert result.updated == 0
    assert result.conflicts == 1
    assert store.effects[original.effect_id] == original


def test_audit_failure_rolls_back_payload_updates() -> None:
    original = _effect("effect-audit-failure")
    store = _Store(effects=[original])
    maintenance = PayloadRetentionMaintenance(store, _Audit(fail=True), _policy())

    with pytest.raises(RuntimeError, match="audit failure"):
        maintenance.run(
            PayloadRetentionRequest(
                kind=PayloadRetentionKind.EXTERNAL_EFFECT,
                dry_run=False,
            ),
            now=_NOW,
        )

    assert store.effects[original.effect_id] == original
    assert store.update_count == 0


def test_live_responses_chain_and_process_exit_fallback_are_runtime_dependencies() -> None:
    live_chain = _llm_call(
        "call-live-chain",
        tool_calls=[],
        api="responses",
        response_id="resp-live",
        request_options={"openai_provider_chain_eligible": True},
    )
    exit_fallback = replace(
        _llm_call(
            "call-exit-fallback",
            tool_calls=[
                {
                    "name": "process_exit",
                    "arguments": {"payload": {"summary": _SENTINEL}},
                }
            ],
        ),
        pid="pid-exit-fallback",
    )
    ordinary = replace(
        _llm_call("call-ordinary", tool_calls=[]),
        pid="pid-ordinary",
    )
    transparent_head = replace(
        _llm_call(
            "call-transparent-head",
            tool_calls=[],
            request_options={
                "image_only_transcript": {
                    "schema_version": 2,
                    "output_key": "call-transparent-head",
                }
            },
        ),
        pid="pid-transparent-head",
    )
    truncated_legacy = replace(
        _llm_call("call-truncated-legacy", tool_calls=[]),
        tool_calls={
            "preview": '[{"name":"process_',
            "sha256": "c" * 64,
            "bytes": 500,
            "truncated": True,
        },
    )

    assert llm_call_payload_is_runtime_dependency(live_chain)
    assert llm_call_payload_is_runtime_dependency(exit_fallback)
    assert llm_call_payload_is_runtime_dependency(truncated_legacy)
    assert llm_call_payload_is_runtime_dependency(transparent_head)
    assert not llm_call_payload_is_runtime_dependency(
        transparent_head,
        provider_chain_head=False,
    )
    assert not llm_call_payload_is_runtime_dependency(ordinary)
    assert not llm_call_payload_is_runtime_dependency(
        replace(live_chain, call_id="call-pidless-chain", pid=None)
    )
    assert llm_call_payload_is_runtime_dependency(
        replace(live_chain, call_id="call-empty-pid-chain", pid="")
    )

    with pytest.raises(ValueError, match="runtime-dependent"):
        retain_llm_call_payload(live_chain, PayloadRetentionTier.SUMMARY)

    store = _Store(
        llm_calls=[live_chain, exit_fallback, transparent_head, ordinary]
    )
    audit = _Audit()
    result = PayloadRetentionMaintenance(store, audit, _policy()).run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.LLM_CALL,
            dry_run=False,
        ),
        now=_NOW,
    )

    assert result.protected_runtime_dependency == 3
    assert result.updated == 1
    assert store.llm_calls[live_chain.call_id] == live_chain
    assert store.llm_calls[exit_fallback.call_id] == exit_fallback
    assert store.llm_calls[transparent_head.call_id] == transparent_head
    assert llm_call_payload_retention_tier(
        store.llm_calls[ordinary.call_id]
    ) is PayloadRetentionTier.SUMMARY


def test_pidless_responses_call_can_be_retained() -> None:
    pidless = replace(
        _llm_call(
            "call-pidless-responses",
            tool_calls=[],
            api="responses",
            response_id="resp-pidless",
            request_options={"openai_provider_chain_eligible": True},
        ),
        pid=None,
    )
    store = _Store(llm_calls=[pidless])

    result = PayloadRetentionMaintenance(store, _Audit(), _policy()).run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.LLM_CALL,
            dry_run=False,
        ),
        now=_NOW,
    )

    assert result.protected_runtime_dependency == 0
    assert result.updated == 1
    assert (
        llm_call_payload_retention_tier(store.llm_calls[pidless.call_id])
        is PayloadRetentionTier.SUMMARY
    )


def test_llm_retention_page_requires_typed_latest_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _llm_call("call-unclassified-page")
    store = _Store(llm_calls=[record])

    def unclassified_page(**_kwargs: Any) -> PayloadRetentionPage[LLMCallRecord]:
        return PayloadRetentionPage(records=(record,))

    monkeypatch.setattr(
        store,
        "scan_llm_call_payloads_for_retention",
        unclassified_page,
    )

    with pytest.raises(ValueError, match="did not classify latest LLM calls"):
        PayloadRetentionMaintenance(store, _Audit(), _policy()).run(
            PayloadRetentionRequest(kind=PayloadRetentionKind.LLM_CALL),
            now=_NOW,
        )

    with pytest.raises(ValueError, match="must belong to the page"):
        PayloadRetentionPage(
            records=(record,),
            latest_llm_call_ids=frozenset({"call-outside-page"}),
        )


def test_nonterminal_llm_call_is_never_trimmed() -> None:
    pending = _llm_call("call-pending", status="pending", completed_at=None)
    store = _Store(llm_calls=[pending])
    audit = _Audit()

    result = PayloadRetentionMaintenance(store, audit, _policy()).run(
        PayloadRetentionRequest(
            kind=PayloadRetentionKind.LLM_CALL,
            dry_run=False,
        ),
        now=_NOW,
    )

    assert result.protected_nonterminal == 1
    assert result.updated == 0
    assert store.llm_calls[pending.call_id] == pending


@pytest.mark.parametrize(
    "policy",
    [
        PayloadRetentionPolicy(),
        PayloadRetentionPolicy(enabled=False, summary_after_seconds=0),
    ],
)
def test_default_and_disabled_policies_are_inert(policy: PayloadRetentionPolicy) -> None:
    assert policy.enabled is False


def test_policy_is_derived_from_runtime_defaults_without_hidden_overrides() -> None:
    runtime_defaults = replace(
        DEFAULT_CONFIG.runtime,
        payload_retention_enabled=True,
        payload_retention_summary_after_seconds=60,
        payload_retention_hash_only_after_seconds=120,
        payload_retention_page_size=7,
        payload_retention_page_hard_limit=9,
    )

    assert PayloadRetentionPolicy.from_runtime_defaults(runtime_defaults) == (
        PayloadRetentionPolicy(
            enabled=True,
            summary_after_seconds=60,
            hash_only_after_seconds=120,
            batch_limit=7,
            hard_limit=9,
        )
    )


def test_policy_rejects_hash_tier_that_precedes_summary_tier() -> None:
    with pytest.raises(ValueError, match="at least summary"):
        PayloadRetentionPolicy(
            enabled=True,
            summary_after_seconds=20,
            hash_only_after_seconds=10,
        )


def test_runtime_exposed_maintenance_is_lifecycle_admission_gated() -> None:
    class Admission:
        entered = 0
        exited = 0
        accepting = True

        @contextmanager
        def admit(self, *, read_only: bool = False) -> Iterator[None]:
            assert read_only is False
            if not self.accepting:
                raise RuntimeError("runtime is not accepting operations")
            self.entered += 1
            try:
                yield
            finally:
                self.exited += 1

    admission = Admission()
    audit = _Audit()
    maintenance = PayloadRetentionMaintenance(
        _Store(),
        audit,
        admission=admission,
    )

    result = maintenance.run(
        PayloadRetentionRequest(kind=PayloadRetentionKind.LLM_CALL),
        now=_NOW,
    )

    assert result.status == "disabled"
    assert admission.entered == admission.exited == 1
    admission.accepting = False
    with pytest.raises(RuntimeError, match="not accepting"):
        maintenance.run(
            PayloadRetentionRequest(kind=PayloadRetentionKind.LLM_CALL),
            now=_NOW,
        )
    assert len(audit.records) == 1
