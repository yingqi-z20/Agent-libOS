from __future__ import annotations

from pathlib import Path
import json
import signal

import pytest

from benchmarks.durable_task_runs import (
    CRASH_EXIT_CODE,
    DurabilityBarrier,
    FsyncIdempotentJsonRpcProvider,
    FsyncProviderLedger,
    ProviderOutcome,
    RecoveryClass,
    run_crash_matrix,
    run_unpaired_committed_result_scenario,
)
from experiments.run_task_run_crash_matrix import main as crash_matrix_main


def test_six_durability_barriers_use_independent_fsync_provider_truth(
    tmp_path: Path,
) -> None:
    results = run_crash_matrix(tmp_path)

    assert {result.barrier for result in results} == set(DurabilityBarrier)
    by_barrier = {result.barrier: result for result in results}
    assert (
        by_barrier[DurabilityBarrier.PROVIDER_DISPATCHED].process_returncode
        == -signal.SIGKILL
    )
    assert all(
        result.process_returncode == CRASH_EXIT_CODE
        for result in results
        if result.barrier is not DurabilityBarrier.PROVIDER_DISPATCHED
    )
    assert all(result.runtime_reopened and result.root_present for result in results)
    assert all(result.settlement_reopen_stable for result in results)
    assert all(result.passed for result in results)
    assert {
        result.recovery_class for result in results
    } == set(RecoveryClass)
    unknown = [
        result
        for result in results
        if result.recovery_class is RecoveryClass.UNKNOWN_EFFECT
    ]
    assert unknown
    assert all(result.provider_outcome is ProviderOutcome.UNKNOWN for result in unknown)
    assert all(result.dispatch_count == 1 for result in unknown)
    assert all(result.recovered_status == "needs_attention" for result in unknown)
    assert all("unknown_effect" in result.blocker_kinds for result in unknown)
    action = by_barrier[DurabilityBarrier.ACTION_COMMITTED]
    assert action.validated_action_present
    assert action.pending_action_present
    assert action.local_llm_call_count == 1
    assert by_barrier[DurabilityBarrier.RESUME_POINT_COMMITTED].resume_point_present
    assert by_barrier[DurabilityBarrier.PROVIDER_RESULT_DURABLE].resume_point_present
    assert all(
        by_barrier[barrier].tool_call_present
        and by_barrier[barrier].effect_link_present
        for barrier in {
            DurabilityBarrier.PROVIDER_RESULT_DURABLE,
            DurabilityBarrier.RESUME_POINT_COMMITTED,
        }
    )
    assert not by_barrier[DurabilityBarrier.PROVIDER_RESULT_DURABLE].pending_action_present
    assert not by_barrier[DurabilityBarrier.RESUME_POINT_COMMITTED].pending_action_present
    assert by_barrier[DurabilityBarrier.PROVIDER_RESULT_DURABLE].completed_step_count == 1
    assert by_barrier[DurabilityBarrier.RESUME_POINT_COMMITTED].completed_step_count == 1
    assert (
        by_barrier[DurabilityBarrier.PROVIDER_RESULT_DURABLE].local_effect_transaction_state
        == "committed"
    )
    assert by_barrier[DurabilityBarrier.RUN_COMMITTED].recovered_status == "queued"
    assert all(
        by_barrier[barrier].recovered_status == "paused"
        for barrier in {
            DurabilityBarrier.ACTION_COMMITTED,
            DurabilityBarrier.EFFECT_PREPARED,
            DurabilityBarrier.PROVIDER_RESULT_DURABLE,
            DurabilityBarrier.RESUME_POINT_COMMITTED,
        }
    )
    assert all(
        by_barrier[barrier].idempotency_dedupe_verified
        for barrier in {
            DurabilityBarrier.PROVIDER_RESULT_DURABLE,
            DurabilityBarrier.RESUME_POINT_COMMITTED,
        }
    )


def test_provider_ledger_rejects_torn_or_noncanonical_records(tmp_path: Path) -> None:
    ledger = FsyncProviderLedger(tmp_path / "provider.jsonl")
    ledger.append(
        effect_id="effect-1",
        kind="dispatch",
        outcome=ProviderOutcome.UNKNOWN,
        idempotency_key="stable-key",
    )
    with (tmp_path / "provider.jsonl").open("ab") as stream:
        stream.write(b'{"partial":')

    with pytest.raises(ValueError, match="incomplete"):
        ledger.records()

    noncanonical_path = tmp_path / "noncanonical.jsonl"
    noncanonical_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sequence": 1,
                "effect_id": "effect-2",
                "kind": "dispatch",
                "outcome": "unknown",
                "idempotency_key": "stable-key-2",
                "receipt": None,
            },
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not canonical"):
        FsyncProviderLedger(noncanonical_path).records()

    duplicate_key_path = tmp_path / "duplicate-key.jsonl"
    duplicate_key_path.write_text(
        '{"effect_id":"effect-3","effect_id":"forged",'
        '"idempotency_key":"stable-key-3","kind":"dispatch",'
        '"outcome":"unknown","receipt":null,"schema_version":1,'
        '"sequence":1}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        FsyncProviderLedger(duplicate_key_path).records()


def test_committed_effect_after_safe_point_without_full_result_never_replays(
    tmp_path: Path,
) -> None:
    result = run_unpaired_committed_result_scenario(tmp_path)

    assert result.passed
    assert result.process_returncode == CRASH_EXIT_CODE
    assert result.provider_outcome is ProviderOutcome.SUCCEEDED
    assert result.dispatch_count == result.receipt_count == 1
    assert result.resume_point_present
    assert result.pending_action_present
    assert result.local_llm_call_count == 2
    assert result.completed_step_count == 1  # the older baseline only
    assert result.settlement_reopen_stable
    assert result.local_effect_transaction_state == "committed"
    assert result.recovered_status == "needs_attention"
    assert "unknown_effect" in result.blocker_kinds


def test_provider_idempotency_key_refuses_redispatch_while_outcome_is_unknown(
    tmp_path: Path,
) -> None:
    ledger = FsyncProviderLedger(tmp_path / "unknown-provider.jsonl")
    ledger.append(
        effect_id="effect-unknown",
        kind="dispatch",
        outcome=ProviderOutcome.UNKNOWN,
        idempotency_key="unknown-key",
    )
    provider = FsyncIdempotentJsonRpcProvider(ledger)
    request = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "retry",
            "method": "durability.commit",
            "params": {"idempotency_key": "unknown-key"},
        }
    ).encode("utf-8")

    with pytest.raises(RuntimeError, match="redispatch refused"):
        provider.call(None, None, request)

    assert ledger.count("effect-unknown", "dispatch") == 1
    assert ledger.count("effect-unknown", "receipt") == 0


def test_crash_matrix_cli_writes_only_a_complete_passing_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "matrix.json"

    assert crash_matrix_main(["--output", str(output)]) == 0

    printed = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == printed
    assert persisted["schema_version"] == 1
    assert persisted["barrier_count"] == len(DurabilityBarrier) == 6
    assert persisted["passed"] is True
    assert all(row["passed"] is True for row in persisted["results"])
    assert persisted["unpaired_committed_after_safe_point"]["passed"] is True
    assert (
        persisted["unpaired_committed_after_safe_point"]["recovery_class"]
        == RecoveryClass.UNKNOWN_EFFECT.value
    )
