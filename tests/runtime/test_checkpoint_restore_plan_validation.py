from __future__ import annotations

import pytest
from types import SimpleNamespace

from agent_libos import Runtime
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.checkpoint_reconciliation import CheckpointRestorePlan


@pytest.mark.parametrize("invalid", [True, 1.0, "1", None])
def test_restore_plan_operation_binding_version_requires_exact_integer(
    invalid: object,
) -> None:
    plan = CheckpointRestorePlan(
        checkpoint_id="checkpoint_test",
        pid="pid_test",
        actor="runtime",
        operation_id="operation_test",
        snapshot_version=1,
        snapshot_sha256="a" * 64,
        current_pids=("pid_test",),
        snapshot_pids=("pid_test",),
        scoped_pids=("pid_test",),
        stale_tool_ids=(),
        finalizer_work_items=(),
    ).to_mapping()
    plan["operation_binding_version"] = invalid

    with pytest.raises(
        ValidationError,
        match="operation binding version is invalid",
    ):
        CheckpointRestorePlan.from_mapping(plan)


def test_checkpoint_receipt_failures_are_correlated_and_text_free() -> None:
    runtime = Runtime.open("local")
    secret = "SECRET /private/checkpoint/driver-token"
    try:
        fork_failure = runtime.checkpoint._fork_failure(
            phase="fork_commit_acknowledgement",
            exc=RuntimeError(secret),
        )
        restore_failure = runtime.checkpoint._restore_post_commit_failure(
            actor="runtime",
            checkpoint=object(),  # unused when record_failure is false
            phase="image_reconciliation",
            exc=RuntimeError(secret),
            record_failure=False,
        )
        receipt = runtime.checkpoint._fork_unknown_outcome_receipt(
            checkpoint=SimpleNamespace(checkpoint_id="checkpoint_test", pid="pid_test"),
            root_pid="pid_fork",
            remapped={"pid_map": {}, "object_map": {}, "tool_map": {}},
            interruption=KeyboardInterrupt(secret),
            diagnostic_error=RuntimeError(secret),
        )
        diagnostic: dict[str, object] = {}
        runtime.checkpoint._append_diagnostic_error(
            diagnostic,
            "recovery_signal",
            RuntimeError(secret),
            code="checkpoint_fork_recovery_signal_failed",
        )

        serialized = repr(
            {
                "fork": fork_failure,
                "restore": restore_failure,
                "receipt": receipt,
                "diagnostic": diagnostic,
            }
        )
        assert secret not in serialized
        for failure in (fork_failure, restore_failure):
            assert failure["correlation_id"].startswith("corr_")
            assert failure["correlation_id"] in failure["message"]
            assert len(failure["internal_error"]["exception_text"]["sha256"]) == 64
        outcome = receipt["outcome_diagnostic"]
        assert outcome["interruption_correlation_id"] in outcome["interruption"]
        assert outcome["diagnostic_correlation_id"] in outcome["diagnostic_error"]
        assert diagnostic["recovery_signal_error_correlation_id"] in diagnostic[
            "recovery_signal_error"
        ]
    finally:
        runtime.close()
