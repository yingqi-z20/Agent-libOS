from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent_libos.evidence.payload_retention import (
    PayloadRetentionTier,
    llm_call_payload_is_runtime_dependency,
    llm_call_payload_sha256,
    retain_llm_call_payload,
)
from agent_libos.models import Checkpoint, LLMCallRecord
from agent_libos.storage.sqlite import SQLiteStore


def _source(call_id: str = "hosted-call", *, second: int = 0) -> LLMCallRecord:
    timestamp = f"2026-01-01T00:00:{second:02d}+00:00"
    return LLMCallRecord(
        call_id=call_id, pid="hosted-pid", image_id="base-agent:v0",
        purpose="action_selection", status="ok", tool_calls=[],
        request_options={
            "provider_tools_enabled": True,
            "provider_tools_configured": {"provider": "openai", "code_interpreter": True},
        },
        response_content="PRIVATE_HOSTED_RESULT", created_at=timestamp, completed_at=timestamp,
    )


def _marker(*, state: str = "pending", second: int = 1) -> LLMCallRecord:
    return replace(
        _source(f"marker-{state}-{second}", second=second),
        purpose="provider_continuation", response_content="", messages=[
            {"role": "assistant", "content": "PRIVATE_HOSTED_RESULT"},
        ],
        request_options={"provider_continuation": {
            "schema_version": 1, "state": state, "call_id": "hosted-call",
        }},
    )


def _retention_ids(store: SQLiteStore) -> frozenset[str]:
    page = store.scan_llm_call_payloads_for_retention(
        older_than="2099-01-01T00:00:00+00:00", after=None, limit=100,
    )
    assert page.provider_continuation_call_ids is not None
    return page.provider_continuation_call_ids


def _attempt_retention(store: SQLiteStore, record: LLMCallRecord) -> bool:
    # A caller can construct an unprotected projection, but cannot bypass the
    # store's current reference check with a stale/forged classification.
    summary = retain_llm_call_payload(
        record, PayloadRetentionTier.SUMMARY,
        provider_chain_head=False, provider_continuation_pending=False,
    )
    return store.update_llm_call_payload_retention(
        summary, expected_payload_sha256=llm_call_payload_sha256(record),
        expected_tier=PayloadRetentionTier.FULL,
    )


def test_pending_continuation_source_and_marker_survive_retention_and_repairs() -> None:
    store = SQLiteStore(":memory:")
    try:
        source, marker = _source(), _marker()
        store.insert_llm_call(source)
        store.insert_llm_call(marker)
        # A later local-only successful but invalid decision and a failed
        # repair must not release the earlier provider result's dependency.
        for second, status in ((2, "ok"), (3, "error")):
            repair = replace(
                _source(f"repair-{second}", second=second), status=status,
                request_options={"provider_tools_enabled": False},
            )
            store.insert_llm_call(repair)
        assert _retention_ids(store) == {source.call_id, marker.call_id}
        for record in (source, marker):
            assert llm_call_payload_is_runtime_dependency(record)
            with pytest.raises(ValueError, match="runtime-dependent"):
                retain_llm_call_payload(record, PayloadRetentionTier.SUMMARY)
            assert _attempt_retention(store, record) is False
    finally:
        store.close()


def test_consumed_continuation_releases_both_retained_payloads() -> None:
    store = SQLiteStore(":memory:")
    try:
        source, marker = _source(), _marker()
        for record in (source, marker, _marker(state="consumed", second=2)):
            store.insert_llm_call(record)
        assert _retention_ids(store) == frozenset()
        assert _attempt_retention(store, source) is True
        assert _attempt_retention(store, marker) is True
    finally:
        store.close()


def test_checkpoint_references_keep_consumed_source_and_marker_payloads() -> None:
    store = SQLiteStore(":memory:")
    try:
        source, marker = _source(), _marker()
        for record in (source, marker, _marker(state="consumed", second=2)):
            store.insert_llm_call(record)
        store.insert_checkpoint(
            Checkpoint("hosted-checkpoint", source.pid, "retain continuation", marker.created_at),
            {"provider_continuation_refs": {source.pid: {
                "pid": source.pid, "marker_call_id": marker.call_id,
                "marker_sha256": "a" * 64, "source_call_id": source.call_id,
                "source_pid": source.pid, "source_sha256": "b" * 64,
                "profile_identity_sha256": "c" * 64, "context_generation": "initial",
                "payload_sha256": "d" * 64,
            }}},
        )
        assert _retention_ids(store) == {source.call_id, marker.call_id}
        assert _attempt_retention(store, source) is False
        assert _attempt_retention(store, marker) is False
    finally:
        store.close()


def test_checkpoint_body_mentions_do_not_become_continuation_references() -> None:
    store = SQLiteStore(":memory:")
    try:
        source, marker = _source(), _marker()
        for record in (source, marker, _marker(state="consumed", second=2)):
            store.insert_llm_call(record)
        store.insert_checkpoint(
            Checkpoint("unrelated-checkpoint", source.pid, "unrelated data", marker.created_at),
            {"object_payloads": {"arbitrary": {
                "marker_call_id": marker.call_id, "source_call_id": source.call_id,
            }}, "provider_continuation_refs": {}},
        )
        assert _retention_ids(store) == frozenset()
        assert _attempt_retention(store, source) is True
        assert _attempt_retention(store, marker) is True
    finally:
        store.close()


def test_fork_pending_marker_protects_source_in_original_pid_until_consumption() -> None:
    store = SQLiteStore(":memory:")
    try:
        source = _source()
        fork_marker = replace(
            _marker(second=3), pid="fork-pid", call_id="fork-marker",
            request_options={"provider_continuation": {
                "schema_version": 2, "state": "pending", "call_id": source.call_id,
                "source_pid": source.pid,
            }},
        )
        for record in (source, _marker(), _marker(state="consumed", second=2), fork_marker):
            store.insert_llm_call(record)
        assert _retention_ids(store) == {source.call_id, fork_marker.call_id}
        assert _attempt_retention(store, source) is False
        consumed = replace(
            _marker(state="consumed", second=4), pid="fork-pid", call_id="fork-consumed",
            request_options={"provider_continuation": {
                **fork_marker.request_options["provider_continuation"], "state": "consumed",
            }},
        )
        store.insert_llm_call(consumed)
        assert _retention_ids(store) == frozenset()
        assert _attempt_retention(store, source) is True
        assert _attempt_retention(store, fork_marker) is True
    finally:
        store.close()


def test_postgres_translates_only_fixed_continuation_reference_paths(monkeypatch) -> None:
    from agent_libos.storage.postgres import _PostgresDialect

    store = SQLiteStore(":memory:")
    captured = []
    try:
        source = _source()
        monkeypatch.setattr(store, "_query", lambda query, params: captured.append((query, params)) or [])
        store._external_provider_continuation_retention_ids((source,))
        query, params = captured[0]
        translated = _PostgresDialect().prepare(query, with_params=True)
        assert "json_extract" not in translated
        assert "json_each((checkpoint.snapshot_json::json -> 'provider_continuation_refs'))" in translated
        assert "continuation_ref.value ->> 'marker_call_id'" in translated
        assert "continuation_ref.value ->> 'source_call_id'" in translated
        assert "marker.request_options_json::json -> 'provider_continuation' ->> 'call_id'" in translated
        assert translated.count("%s") == len(params) == 1
        assert "COLLATE BINARY" not in translated
        assert _PostgresDialect().prepare("SELECT json_extract(arbitrary_json, '$.arbitrary')") == "SELECT json_extract(arbitrary_json, '$.arbitrary')"
    finally:
        store.close()


def test_retention_cannot_erase_hosted_success_before_continuation_is_committed() -> None:
    store = SQLiteStore(":memory:")
    try:
        source = _source()
        store.insert_llm_call(source)
        assert _retention_ids(store) == {source.call_id}
        assert _attempt_retention(store, source) is False
    finally:
        store.close()


def test_content_free_continuation_still_protects_its_integrity_bound_envelopes() -> None:
    from agent_libos.config import DEFAULT_CONFIG
    from agent_libos.llm.records import observable_llm_call_fields

    store = SQLiteStore(":memory:")
    try:
        source = _source()
        source = replace(
            source,
            request_options={**source.request_options, "provider_tools_function_call_count": 0},
            **observable_llm_call_fields(
                messages=[], tools=[], tool_calls=[], response_content=source.response_content,
                config=replace(DEFAULT_CONFIG, llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False)),
            ),
        )
        store.insert_llm_call(source)
        assert llm_call_payload_is_runtime_dependency(source)
        assert source.call_id in _retention_ids(store)
        hashed = retain_llm_call_payload(
            source, PayloadRetentionTier.HASH_ONLY,
            provider_chain_head=False, provider_continuation_pending=False,
        )
        assert not store.update_llm_call_payload_retention(
            hashed, expected_payload_sha256=llm_call_payload_sha256(source),
            expected_tier=PayloadRetentionTier.SUMMARY,
        )
    finally:
        store.close()


def test_retention_rechecks_new_pending_reference_after_page_selection() -> None:
    store = SQLiteStore(":memory:")
    try:
        source = _source()
        store.insert_llm_call(source)
        store.insert_llm_call(_marker(state="consumed", second=1))
        assert _retention_ids(store) == frozenset()
        store.insert_llm_call(_marker(second=2))
        assert _attempt_retention(store, source) is False
    finally:
        store.close()


def test_malformed_continuation_marker_preserves_remaining_evidence() -> None:
    store = SQLiteStore(":memory:")
    try:
        source = _source()
        marker = replace(_marker(), request_options={"provider_continuation": {"schema_version": 2}})
        store.insert_llm_call(source)
        store.insert_llm_call(marker)
        assert _retention_ids(store) == {source.call_id, marker.call_id}
        assert _attempt_retention(store, source) is False
    finally:
        store.close()


def test_task_run_continuation_source_survives_without_executor_marker(tmp_path: Path) -> None:
    from agent_libos import Runtime
    from agent_libos.llm.task_runs import validated_action_manifest
    from tests.runtime.test_task_run_provider_continuation import CONFIG, _call, _create, _record

    runtime = Runtime.open(tmp_path / "run-retention.sqlite", config=CONFIG)
    try:
        created = _create(runtime)
        source = _call(runtime, created.root_pid)
        _record(runtime, source)
        successor = replace(
            _call(runtime, created.root_pid, "local-successor"),
            request_options={**source.request_options, "provider_tools_enabled": False},
        )
        runtime.store.insert_llm_call(successor)
        assert source.call_id in _retention_ids(runtime.store)
        assert _attempt_retention(runtime.store, source) is False
        manifest = validated_action_manifest(
            [{"action": "process_exit"}], call_id=successor.call_id,
            parallel_tool_calls=False, host_auto_wait=False, tool_call_count=1, data_labels={},
        )
        runtime.task_runs.record_validated_transcript(
            pid=created.root_pid, call_id=successor.call_id, action_manifest=manifest,
            context_generation=runtime.store.get_llm_context_generation(created.root_pid),
        )
        assert source.call_id not in _retention_ids(runtime.store)
        assert _attempt_retention(runtime.store, source) is True
    finally:
        runtime.close()
