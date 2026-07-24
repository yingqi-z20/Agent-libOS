from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.config import AgentLibOSConfig, RuntimeDefaults
from agent_libos.api.cli import main as cli_main
from agent_libos.evidence.payload_retention import (
    PayloadRetentionCursor,
    PayloadRetentionKind,
    PayloadRetentionRequest,
    PayloadRetentionTier,
    external_effect_payload_retention_tier,
    external_effect_payload_sha256,
    llm_call_payload_retention_tier,
    llm_call_payload_sha256,
    retain_external_effect_payload,
    retain_llm_call_payload,
)
from agent_libos.models import (
    ExternalEffectRecord,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    LLMCallRecord,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.runtime import Runtime
from agent_libos.storage import PostgresStore, SQLiteStore


STORE_BACKENDS = [
    "sqlite",
    pytest.param("postgres", marks=pytest.mark.postgres),
]


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_llm_retention_store_rejects_forged_target_provenance(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        current = _llm_call("call-forged-target", "2026-01-01T00:00:00+00:00")
        store.insert_llm_call(current)
        expected_sha256 = llm_call_payload_sha256(current)
        retained = retain_llm_call_payload(
            current,
            PayloadRetentionTier.SUMMARY,
        )

        forged_payload = replace(retained, messages={"forged": "payload"})
        forged_observability = deepcopy(retained.observability)
        marker = forged_observability["$agent_libos_payload_retention"]
        marker["source_observability_sha256"] = "0" * 64
        extra_content = deepcopy(retained.messages)
        extra_content["$agent_libos_payload_retention"]["leak"] = "secret"
        wrong_field_sha = deepcopy(retained.messages)
        wrong_field_sha["$agent_libos_payload_retention"]["sha256"] = "0" * 64
        noncanonical_schema = deepcopy(retained.observability)
        noncanonical_schema["$agent_libos_payload_retention"][
            "schema_version"
        ] = True
        forged_targets = (
            forged_payload,
            replace(retained, observability=forged_observability),
            replace(retained, messages=extra_content),
            replace(retained, messages=wrong_field_sha),
            replace(retained, messages=json.dumps(retained.messages)),
            replace(retained, observability=noncanonical_schema),
        )
        for forged in forged_targets:
            assert not store.update_llm_call_payload_retention(
                forged,
                expected_payload_sha256=expected_sha256,
                expected_tier=PayloadRetentionTier.FULL,
            )
            assert store.get_llm_call(current.call_id) == current

        assert store.update_llm_call_payload_retention(
            retained,
            expected_payload_sha256=expected_sha256,
            expected_tier=PayloadRetentionTier.FULL,
        )
        persisted_summary = store.get_llm_call(current.call_id)
        assert persisted_summary is not None
        hash_only = retain_llm_call_payload(
            persisted_summary,
            PayloadRetentionTier.HASH_ONLY,
        )
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE llm_calls SET messages_json = ? WHERE call_id = ?",
                (json.dumps({"forged": "persisted"}), current.call_id),
            )
        assert not store.update_llm_call_payload_retention(
            hash_only,
            expected_payload_sha256=expected_sha256,
            expected_tier=PayloadRetentionTier.SUMMARY,
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_external_effect_retention_store_rejects_noncanonical_target(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        current = _external_effect(
            "effect-forged-target",
            created_at="2026-01-01T00:00:00+00:00",
        )
        store.insert_external_effect(current)
        expected_sha256 = external_effect_payload_sha256(current)
        retained = retain_external_effect_payload(
            current,
            PayloadRetentionTier.SUMMARY,
        )
        extra_content = deepcopy(retained.provider_metadata)
        extra_content["$agent_libos_payload_retention"]["leak"] = "secret"
        wrong_field_sha = deepcopy(retained.provider_receipt)
        wrong_field_sha["$agent_libos_payload_retention"]["sha256"] = "0" * 64
        forged_targets = (
            replace(retained, provider_metadata={"forged": "payload"}),
            replace(retained, provider_metadata=extra_content),
            replace(retained, provider_receipt=wrong_field_sha),
        )
        for forged in forged_targets:
            assert not store.update_external_effect_payload_retention(
                forged,
                expected_payload_sha256=expected_sha256,
                expected_tier=PayloadRetentionTier.FULL,
                expected_effect_state="finalized",
                expected_transaction_state="committed",
            )
            assert store.get_external_effect(current.effect_id) == current

        assert store.update_external_effect_payload_retention(
            retained,
            expected_payload_sha256=expected_sha256,
            expected_tier=PayloadRetentionTier.FULL,
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )


@pytest.mark.postgres
def test_postgres_llm_retention_cas_rejects_concurrent_payload_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _retention_store("postgres") as store:
        assert isinstance(store, PostgresStore)
        current = _llm_call("call-retention-race", "2026-01-01T00:00:00+00:00")
        store.insert_llm_call(current)
        expected_sha256 = llm_call_payload_sha256(current)
        retained = retain_llm_call_payload(
            current,
            PayloadRetentionTier.SUMMARY,
        )
        selected = Event()
        allow_update = Event()
        original_decode = store._row_to_llm_call

        def pause_after_select(row: object) -> LLMCallRecord:
            decoded = original_decode(row)
            selected.set()
            assert allow_update.wait(timeout=10)
            return decoded

        monkeypatch.setattr(store, "_row_to_llm_call", pause_after_select)
        results: list[bool] = []
        errors: list[BaseException] = []

        def retain() -> None:
            try:
                results.append(
                    store.update_llm_call_payload_retention(
                        retained,
                        expected_payload_sha256=expected_sha256,
                        expected_tier=PayloadRetentionTier.FULL,
                    )
                )
            except BaseException as exc:  # pragma: no cover - diagnostic path
                errors.append(exc)

        worker = Thread(target=retain, daemon=True)
        worker.start()
        assert selected.wait(timeout=10)
        import psycopg

        with psycopg.connect(store.dsn, autocommit=True) as connection:
            connection.execute(
                "UPDATE llm_calls SET messages_json = %s WHERE call_id = %s",
                (json.dumps({"concurrent": "winner"}), current.call_id),
            )
        allow_update.set()
        worker.join(timeout=10)
        assert not worker.is_alive()
        assert errors == []
        assert results == [False]
        persisted = store.get_llm_call(current.call_id)
        assert persisted is not None
        assert persisted.messages == {"concurrent": "winner"}


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_image_only_retention_protects_only_the_active_transcript_head(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        marker = {
            "image_only_transcript": {
                "schema_version": 1,
                "output_key": "transcript-output",
            }
        }
        older = replace(
            _llm_call("transparent-older", "2026-01-01T00:00:00+00:00"),
            purpose="action_selection",
            request_options=marker,
        )
        head = replace(
            _llm_call("transparent-head", "2026-01-01T00:00:01+00:00"),
            purpose="action_selection",
            request_options=marker,
        )
        store.insert_llm_call(older)
        store.insert_llm_call(head)

        page = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=10,
        )

        assert page.latest_llm_call_ids == frozenset({head.call_id})
        older_summary = retain_llm_call_payload(
            older,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert store.update_llm_call_payload_retention(
            older_summary,
            expected_payload_sha256=llm_call_payload_sha256(older),
            expected_tier=PayloadRetentionTier.FULL,
        )
        head_summary = retain_llm_call_payload(
            head,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert not store.update_llm_call_payload_retention(
            head_summary,
            expected_payload_sha256=llm_call_payload_sha256(head),
            expected_tier=PayloadRetentionTier.FULL,
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_payload_retention_store_is_paged_monotonic_and_cas_protected(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        first = _llm_call("call-a", "2026-01-01T00:00:00+00:00")
        second = _llm_call("call-b", "2026-01-01T00:00:01+00:00")
        store.insert_llm_call(first)
        store.insert_llm_call(second)

        first_page = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=1,
        )
        assert [item.call_id for item in first_page.records] == ["call-a"]
        assert first_page.next_cursor is not None
        second_page = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=first_page.next_cursor,
            limit=1,
        )
        assert [item.call_id for item in second_page.records] == ["call-b"]
        assert second_page.next_cursor is None

        original_sha = llm_call_payload_sha256(first)
        summarized = retain_llm_call_payload(first, PayloadRetentionTier.SUMMARY)
        assert store.update_llm_call_payload_retention(
            summarized,
            expected_payload_sha256=original_sha,
            expected_tier=PayloadRetentionTier.FULL,
        )
        persisted_summary = store.get_llm_call(first.call_id)
        assert persisted_summary is not None
        assert llm_call_payload_retention_tier(persisted_summary) is PayloadRetentionTier.SUMMARY
        assert persisted_summary.usage == first.usage
        assert persisted_summary.request_options == first.request_options
        assert persisted_summary.response_content != first.response_content

        assert not store.update_llm_call_payload_retention(
            summarized,
            expected_payload_sha256=original_sha,
            expected_tier=PayloadRetentionTier.FULL,
        )
        summary_sha = llm_call_payload_sha256(persisted_summary)
        hashed = retain_llm_call_payload(
            persisted_summary,
            PayloadRetentionTier.HASH_ONLY,
        )
        assert store.update_llm_call_payload_retention(
            hashed,
            expected_payload_sha256=summary_sha,
            expected_tier=PayloadRetentionTier.SUMMARY,
        )
        persisted_hash = store.get_llm_call(first.call_id)
        assert persisted_hash is not None
        assert llm_call_payload_retention_tier(persisted_hash) is PayloadRetentionTier.HASH_ONLY
        assert persisted_hash.usage == first.usage
        nonterminal = replace(
            _llm_call("call-nonterminal", "2026-01-01T00:00:00+00:00"),
            status="streaming",
            completed_at=None,
        )
        store.insert_llm_call(nonterminal)
        remaining_llm_page = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=10,
        )
        assert [item.call_id for item in remaining_llm_page.records] == [
            "call-b"
        ]

        finalized = _external_effect(
            "effect-finalized",
            created_at="2026-01-01T00:00:02+00:00",
        )
        pending = _external_effect(
            "effect-pending",
            created_at="2026-01-01T00:00:03+00:00",
            effect_state="pending",
            transaction_state="unknown",
        )
        store.insert_external_effect(finalized)
        store.insert_external_effect(pending)
        effect_page = store.scan_external_effect_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=10,
        )
        assert [item.effect_id for item in effect_page.records] == [
            "effect-finalized"
        ]

        effect_sha = external_effect_payload_sha256(finalized)
        retained_effect = retain_external_effect_payload(
            finalized,
            PayloadRetentionTier.SUMMARY,
        )
        assert store.update_external_effect_payload_retention(
            retained_effect,
            expected_payload_sha256=effect_sha,
            expected_tier=PayloadRetentionTier.FULL,
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        persisted_effect = store.get_external_effect(finalized.effect_id)
        assert persisted_effect is not None
        assert external_effect_payload_retention_tier(persisted_effect) is PayloadRetentionTier.SUMMARY
        assert persisted_effect.canonical_args_hash == finalized.canonical_args_hash
        assert persisted_effect.idempotency_key == finalized.idempotency_key
        retained_effect_hash = retain_external_effect_payload(
            persisted_effect,
            PayloadRetentionTier.HASH_ONLY,
        )
        assert store.update_external_effect_payload_retention(
            retained_effect_hash,
            expected_payload_sha256=effect_sha,
            expected_tier=PayloadRetentionTier.SUMMARY,
            expected_effect_state="finalized",
            expected_transaction_state="committed",
        )
        exhausted_effect_page = store.scan_external_effect_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=10,
        )
        assert exhausted_effect_page.records == ()
        assert exhausted_effect_page.next_cursor is None

        forged_terminal = retain_external_effect_payload(
            replace(
                pending,
                effect_state="finalized",
                transaction_state="committed",
            ),
            PayloadRetentionTier.SUMMARY,
        )
        assert not store.update_external_effect_payload_retention(
            forged_terminal,
            expected_payload_sha256=external_effect_payload_sha256(pending),
            expected_tier=PayloadRetentionTier.FULL,
            expected_effect_state="pending",
            expected_transaction_state="unknown",
        )
        assert store.get_external_effect(pending.effect_id) == pending


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_llm_retention_classifies_only_actual_latest_responses_head(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        tied_older = _responses_call(
            "responses-a",
            "2026-01-01T00:00:00+00:00",
            pid="pid-tied-chain",
        )
        tied_head = _responses_call(
            "responses-z",
            "2026-01-01T00:00:00+00:00",
            pid="pid-tied-chain",
        )
        superseded = _responses_call(
            "responses-superseded",
            "2026-01-01T00:00:00+00:00",
            pid="pid-pending-chain",
        )
        newer_pending = replace(
            _llm_call(
                "pending-newer",
                "2026-01-02T00:00:00+00:00",
            ),
            pid="pid-pending-chain",
            purpose="action_selection",
            status="pending",
            completed_at=None,
        )
        for record in (tied_older, tied_head, superseded, newer_pending):
            store.insert_llm_call(record)

        assert store.get_latest_llm_call(
            pid="pid-tied-chain",
            purpose="action_selection",
        ) == tied_head
        assert store.get_latest_llm_call(
            pid="pid-pending-chain",
            purpose="action_selection",
        ) == newer_pending

        first = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=1,
        )
        assert [record.call_id for record in first.records] == ["responses-a"]
        # The real head is outside this bounded page, but the correlated latest
        # lookup still prevents the older tied row from being misclassified.
        assert first.latest_llm_call_ids == frozenset()
        assert first.next_cursor is not None

        remaining = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=first.next_cursor,
            limit=10,
        )
        assert remaining.latest_llm_call_ids == frozenset({"responses-z"})
        assert "responses-superseded" not in remaining.latest_llm_call_ids

        older_sha = llm_call_payload_sha256(tied_older)
        older_summary = retain_llm_call_payload(
            tied_older,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert store.update_llm_call_payload_retention(
            older_summary,
            expected_payload_sha256=older_sha,
            expected_tier=PayloadRetentionTier.FULL,
        )

        head_summary = retain_llm_call_payload(
            tied_head,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert not store.update_llm_call_payload_retention(
            head_summary,
            expected_payload_sha256=llm_call_payload_sha256(tied_head),
            expected_tier=PayloadRetentionTier.FULL,
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_pidless_responses_call_is_not_a_runtime_chain_head(backend: str) -> None:
    with _retention_store(backend) as store:
        pidless = replace(
            _responses_call(
                "responses-without-pid",
                "2026-01-01T00:00:00+00:00",
                pid="discarded-pid",
            ),
            pid=None,
        )
        store.insert_llm_call(pidless)

        page = store.scan_llm_call_payloads_for_retention(
            older_than="2026-02-01T00:00:00+00:00",
            after=None,
            limit=10,
        )
        assert [record.call_id for record in page.records] == [pidless.call_id]
        assert page.latest_llm_call_ids == frozenset()

        summarized = retain_llm_call_payload(
            pidless,
            PayloadRetentionTier.SUMMARY,
            provider_chain_head=False,
        )
        assert store.update_llm_call_payload_retention(
            summarized,
            expected_payload_sha256=llm_call_payload_sha256(pidless),
            expected_tier=PayloadRetentionTier.FULL,
        )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_responses_head_retention_order_survives_restart(
    backend: str,
    tmp_path: Path,
) -> None:
    with _retention_runtime_target(backend, tmp_path) as (target, config):
        older = _responses_call(
            "restart-responses-a",
            "2026-01-01T00:00:00+00:00",
            pid="pid-restart-chain",
        )
        head = _responses_call(
            "restart-responses-z",
            "2026-01-01T00:00:00+00:00",
            pid="pid-restart-chain",
        )
        seeded = Runtime.open(target, config=config)
        try:
            seeded.store.insert_llm_call(older)
            seeded.store.insert_llm_call(head)
        finally:
            seeded.close()

        retained = Runtime.open(target, config=config)
        try:
            result = retained.payload_retention.run(
                PayloadRetentionRequest(
                    kind=PayloadRetentionKind.LLM_CALL,
                    dry_run=False,
                )
            )
            assert result.updated == 1
            assert result.protected_runtime_dependency == 1
            persisted_older = retained.store.get_llm_call(older.call_id)
            persisted_head = retained.store.get_llm_call(head.call_id)
            assert persisted_older is not None
            assert persisted_head is not None
            assert (
                llm_call_payload_retention_tier(persisted_older)
                is PayloadRetentionTier.SUMMARY
            )
            assert (
                llm_call_payload_retention_tier(persisted_head)
                is PayloadRetentionTier.FULL
            )
        finally:
            retained.close()

        verified = Runtime.open(target, config=config)
        try:
            page = verified.store.scan_llm_call_payloads_for_retention(
                older_than="2026-02-01T00:00:00+00:00",
                after=None,
                limit=10,
            )
            assert page.latest_llm_call_ids == frozenset({head.call_id})
            assert (
                llm_call_payload_retention_tier(
                    verified.store.get_llm_call(older.call_id)
                )
                is PayloadRetentionTier.SUMMARY
            )
        finally:
            verified.close()


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_llm_retention_tier_projection_is_fail_closed(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        record = _llm_call(
            "call-retention-tier-drift",
            "2026-01-01T00:00:00+00:00",
        )
        store.insert_llm_call(record)
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE llm_calls SET payload_retention_tier = ? WHERE call_id = ?",
                (PayloadRetentionTier.HASH_ONLY.value, record.call_id),
            )
        with pytest.raises(
            ValidationError,
            match="tier disagrees with its durable marker",
        ):
            store.get_llm_call(record.call_id)


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_retention_scans_use_terminal_non_hash_composite_indexes(
    backend: str,
) -> None:
    with _retention_store(backend) as store:
        store.insert_llm_call(
            _llm_call("call-retention-plan", "2026-01-01T00:00:00+00:00")
        )
        store.insert_external_effect(
            _external_effect(
                "effect-retention-plan",
                created_at="2026-01-01T00:00:00+00:00",
            )
        )
        if isinstance(store, PostgresStore):
            store.conn.execute("SET enable_seqscan = off")

        captured: list[tuple[str, tuple[object, ...]]] = []
        original_query = store._query

        def tracked_query(sql: str, params: object = ()) -> list[object]:
            selected_params = tuple(params)  # type: ignore[arg-type]
            if "payload_retention_tier IN" in sql and "LIMIT" in sql:
                captured.append((sql, selected_params))
            return original_query(sql, selected_params)

        store._query = tracked_query  # type: ignore[method-assign]
        try:
            store.scan_llm_call_payloads_for_retention(
                older_than="2026-02-01T00:00:00+00:00",
                after=PayloadRetentionCursor(
                    "2025-12-31T23:59:59+00:00",
                    "before-retention-plan",
                ),
                limit=10,
            )
            store.scan_external_effect_payloads_for_retention(
                older_than="2026-02-01T00:00:00+00:00",
                after=PayloadRetentionCursor(
                    "2025-12-31T23:59:59+00:00",
                    "before-retention-plan",
                ),
                limit=10,
            )
        finally:
            store._query = original_query  # type: ignore[method-assign]

        assert len(captured) == 2
        expected_ranges = (
            (
                "idx_llm_calls_retention_eligible",
                "(created_at, call_id) > (?, ?)",
            ),
            (
                "idx_external_effects_retention_eligible",
                "(created_at, effect_id) > (?, ?)",
            ),
        )
        for (sql, params), (expected_index, expected_range) in zip(
            captured,
            expected_ranges,
            strict=True,
        ):
            assert expected_range in sql
            if isinstance(store, PostgresStore):
                plan_rows = list(store.conn.execute(f"EXPLAIN {sql}", params))
                plan = "\n".join(str(row["QUERY PLAN"]) for row in plan_rows)
            else:
                plan_rows = list(
                    store.conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)
                )
                plan = "\n".join(str(row["detail"]) for row in plan_rows)
            assert expected_index in plan
            if expected_index == "idx_llm_calls_retention_eligible":
                assert "idx_llm_calls_provider_chain_head" in plan


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_runtime_config_wires_explicit_retention_without_running_it_at_startup(
    backend: str,
    tmp_path: Path,
) -> None:
    with _retention_runtime_target(backend, tmp_path) as (target, config):
        first = Runtime.open(target, config=config)
        call = _llm_call(
            "runtime-config-call",
            "2026-01-01T00:00:00+00:00",
        )
        try:
            first.store.insert_llm_call(call)
        finally:
            first.close()

        reopened = Runtime.open(target, config=config)
        maintenance = reopened.payload_retention
        try:
            before = reopened.store.get_llm_call(call.call_id)
            assert before is not None
            assert (
                llm_call_payload_retention_tier(before)
                is PayloadRetentionTier.FULL
            )

            preview = maintenance.run(
                PayloadRetentionRequest(
                    kind=PayloadRetentionKind.LLM_CALL,
                    dry_run=True,
                )
            )
            assert preview.status == "dry_run"
            assert preview.would_update == 1
            assert preview.updated == 0

            applied = maintenance.run(
                PayloadRetentionRequest(
                    kind=PayloadRetentionKind.LLM_CALL,
                    dry_run=False,
                )
            )
            assert applied.status == "applied"
            assert applied.updated == 1
            persisted = reopened.store.get_llm_call(call.call_id)
            assert persisted is not None
            assert (
                llm_call_payload_retention_tier(persisted)
                is PayloadRetentionTier.SUMMARY
            )
            audits = [
                record
                for record in reopened.audit.trace()
                if record.action == "evidence.payload_retention.maintenance"
            ]
            assert [record.decision["status"] for record in audits] == [
                "dry_run",
                "applied",
            ]
        finally:
            reopened.close()

        with pytest.raises(RuntimeError, match="not accepting operations"):
            maintenance.run(
                PayloadRetentionRequest(kind=PayloadRetentionKind.LLM_CALL)
            )


@pytest.mark.parametrize("backend", STORE_BACKENDS)
def test_external_effect_retention_provenance_survives_restart_and_cas(
    backend: str,
    tmp_path: Path,
) -> None:
    forged_sha256 = "f" * 64
    forged_envelope = {
        "$agent_libos_payload_retention": {
            "schema_version": 1,
            "tier": "summary",
            "sha256": forged_sha256,
        }
    }
    effect = replace(
        _external_effect(
            "effect-forged-retention-provenance",
            created_at="2026-01-01T00:00:00+00:00",
        ),
        provider_metadata=forged_envelope,
        provider_receipt=forged_envelope,
    )
    full_sha256 = external_effect_payload_sha256(effect)

    with _retention_runtime_target(backend, tmp_path) as (target, config):
        seeded = Runtime.open(target, config=config)
        try:
            forged_insert = replace(
                effect,
                payload_retention_tier="summary",
                payload_retention_sha256=full_sha256,
            )
            with pytest.raises(ValidationError, match="must contain full"):
                seeded.store.insert_external_effect(forged_insert)
            seeded.store.insert_external_effect(effect)
        finally:
            seeded.close()

        summarized_runtime = Runtime.open(target, config=config)
        try:
            persisted_full = summarized_runtime.store.get_external_effect(
                effect.effect_id
            )
            assert persisted_full is not None
            assert (
                external_effect_payload_retention_tier(persisted_full)
                is PayloadRetentionTier.FULL
            )
            forged_target = replace(
                persisted_full,
                payload_retention_tier="summary",
                payload_retention_sha256=full_sha256,
            )
            assert not summarized_runtime.store.update_external_effect_payload_retention(
                forged_target,
                expected_payload_sha256=full_sha256,
                expected_tier=PayloadRetentionTier.FULL,
                expected_effect_state="finalized",
                expected_transaction_state="committed",
            )
            assert summarized_runtime.store.get_external_effect(effect.effect_id) == (
                persisted_full
            )

            summary = retain_external_effect_payload(
                persisted_full,
                PayloadRetentionTier.SUMMARY,
            )
            assert summarized_runtime.store.update_external_effect_payload_retention(
                summary,
                expected_payload_sha256=full_sha256,
                expected_tier=PayloadRetentionTier.FULL,
                expected_effect_state="finalized",
                expected_transaction_state="committed",
            )
        finally:
            summarized_runtime.close()

        hashed_runtime = Runtime.open(target, config=config)
        try:
            persisted_summary = hashed_runtime.store.get_external_effect(
                effect.effect_id
            )
            assert persisted_summary is not None
            assert (
                external_effect_payload_retention_tier(persisted_summary)
                is PayloadRetentionTier.SUMMARY
            )
            assert persisted_summary.payload_retention_sha256 == full_sha256
            assert external_effect_payload_sha256(persisted_summary) == full_sha256
            hash_only = retain_external_effect_payload(
                persisted_summary,
                PayloadRetentionTier.HASH_ONLY,
            )
            assert hashed_runtime.store.update_external_effect_payload_retention(
                hash_only,
                expected_payload_sha256=full_sha256,
                expected_tier=PayloadRetentionTier.SUMMARY,
                expected_effect_state="finalized",
                expected_transaction_state="committed",
            )
        finally:
            hashed_runtime.close()

        verified = Runtime.open(target, config=config)
        try:
            persisted_hash = verified.store.get_external_effect(effect.effect_id)
            assert persisted_hash is not None
            assert (
                external_effect_payload_retention_tier(persisted_hash)
                is PayloadRetentionTier.HASH_ONLY
            )
            assert persisted_hash.payload_retention_sha256 == full_sha256
            assert external_effect_payload_sha256(persisted_hash) == full_sha256
        finally:
            verified.close()


def test_runtime_retention_rolls_back_sql_update_when_audit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            payload_retention_enabled=True,
            payload_retention_summary_after_seconds=0,
        )
    )
    runtime = Runtime.open("local", config=config)
    call = _llm_call("audit-rollback-call", "2026-01-01T00:00:00+00:00")
    try:
        runtime.store.insert_llm_call(call)

        def fail_audit(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("injected retention audit failure")

        with monkeypatch.context() as scoped:
            scoped.setattr(runtime.audit, "record", fail_audit)
            with pytest.raises(RuntimeError, match="audit failure"):
                runtime.payload_retention.run(
                    PayloadRetentionRequest(
                        kind=PayloadRetentionKind.LLM_CALL,
                        dry_run=False,
                    )
                )

        persisted = runtime.store.get_llm_call(call.call_id)
        assert persisted is not None
        assert llm_call_payload_retention_tier(persisted) is PayloadRetentionTier.FULL
    finally:
        runtime.close()


def test_cli_payload_retention_defaults_to_preview_and_requires_apply_to_mutate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "retention-cli.sqlite"
    config_path = tmp_path / "retention-config.yaml"
    config_path.write_text(
        "runtime:\n"
        "  payload_retention_enabled: true\n"
        "  payload_retention_summary_after_seconds: 0\n",
        encoding="utf-8",
    )
    config = AgentLibOSConfig(
        runtime=RuntimeDefaults(
            payload_retention_enabled=True,
            payload_retention_summary_after_seconds=0,
        )
    )
    seeded = Runtime.open(database, config=config)
    call = _llm_call("cli-retention-call", "2026-01-01T00:00:00+00:00")
    try:
        seeded.store.insert_llm_call(call)
    finally:
        seeded.close()

    cli_main(
        [
            "--config",
            str(config_path),
            "--db",
            str(database),
            "payload-retention",
            "llm_call",
        ]
    )
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] is True
    assert preview["would_update"] == 1
    inspected = Runtime.open(database, config=config)
    try:
        persisted = inspected.store.get_llm_call(call.call_id)
        assert persisted is not None
        assert llm_call_payload_retention_tier(persisted) is PayloadRetentionTier.FULL
    finally:
        inspected.close()

    cli_main(
        [
            "--config",
            str(config_path),
            "--db",
            str(database),
            "payload-retention",
            "llm_call",
            "--apply",
        ]
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert applied["updated"] == 1


def _llm_call(call_id: str, created_at: str) -> LLMCallRecord:
    return LLMCallRecord(
        call_id=call_id,
        pid="pid-retention",
        image_id="base-agent:v0",
        purpose="provider_observation",
        status="ok",
        api="chat_completions",
        model="test-model",
        messages=[{"role": "user", "content": "secret prompt"}],
        tools=[{"name": "lookup", "description": "secret schema"}],
        request_options={"temperature": 0},
        response_content="secret response",
        tool_calls=[{"name": "lookup", "arguments": {"query": "secret"}}],
        reasoning={"content": "secret reasoning"},
        usage={"input_tokens": 10, "output_tokens": 5},
        raw_response={"provider": "secret response payload"},
        observability={"trace": "stable evidence"},
        created_at=created_at,
        completed_at=created_at,
    )


def _responses_call(
    call_id: str,
    created_at: str,
    *,
    pid: str,
) -> LLMCallRecord:
    return replace(
        _llm_call(call_id, created_at),
        pid=pid,
        purpose="action_selection",
        api="responses",
        response_id=f"response-{call_id}",
        request_options={"openai_provider_chain_eligible": True},
    )


def _external_effect(
    effect_id: str,
    *,
    created_at: str,
    effect_state: str = "finalized",
    transaction_state: str = "committed",
) -> ExternalEffectRecord:
    return ExternalEffectRecord(
        effect_id=effect_id,
        record_id=f"audit-{effect_id}",
        event_id=f"event-{effect_id}",
        pid="pid-retention",
        provider="test-provider",
        operation="write",
        target="resource:test",
        rollback_class=ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED,
        rollback_status=ExternalEffectRollbackStatus.NOT_REQUIRED,
        state_mutation=True,
        information_flow=False,
        provider_metadata={"request": "secret request payload"},
        provider_receipt={"response": "secret receipt payload"},
        created_at=created_at,
        updated_at=created_at,
        effect_state=effect_state,
        transaction_state=transaction_state,
        canonical_args_hash="a" * 64,
        idempotency_key=f"idempotency-{effect_id}",
    )


@contextlib.contextmanager
def _retention_store(backend: str) -> Iterator[SQLiteStore | PostgresStore]:
    if backend == "sqlite":
        store = SQLiteStore(":memory:")
        try:
            yield store
        finally:
            store.close()
        return
    if backend != "postgres":
        raise AssertionError(f"unknown store backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        config = AgentLibOSConfig(
            runtime=RuntimeDefaults(store_backend="postgres", store_dsn=dsn)
        )
        store = PostgresStore(dsn, config=config)
        try:
            yield store
        finally:
            store.close()


@contextlib.contextmanager
def _retention_runtime_target(
    backend: str,
    tmp_path: Path,
) -> Iterator[tuple[str | Path, AgentLibOSConfig]]:
    runtime_options = {
        "payload_retention_enabled": True,
        "payload_retention_summary_after_seconds": 0,
        "payload_retention_hash_only_after_seconds": 0,
        "payload_retention_page_size": 10,
        "payload_retention_page_hard_limit": 20,
    }
    if backend == "sqlite":
        yield (
            tmp_path / "retention-runtime.sqlite",
            AgentLibOSConfig(runtime=RuntimeDefaults(**runtime_options)),
        )
        return
    if backend != "postgres":
        raise AssertionError(f"unknown store backend: {backend}")
    with _postgres_schema_dsn() as dsn:
        yield (
            dsn,
            AgentLibOSConfig(
                runtime=RuntimeDefaults(
                    store_backend="postgres",
                    store_dsn=dsn,
                    **runtime_options,
                )
            ),
        )


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_retention_{uuid4().hex}"
    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        parsed = urlsplit(dsn)
        query = [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key != "options"
        ]
        query.append(("options", f"-csearch_path={schema}"))
        yield urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )
