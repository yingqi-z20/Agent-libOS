from __future__ import annotations

from agent_libos import Runtime
from agent_libos.storage import SemanticAssessmentRecord
from agent_libos.utils.serde import dumps


_DIGEST = "7" * 64


def _assessment(
    pid: str,
    *,
    suffix: str,
    created_at: str,
    completed_at: str,
) -> SemanticAssessmentRecord:
    return SemanticAssessmentRecord(
        assessment_id=f"assessment-{suffix}",
        job_id=f"job-{suffix}",
        kind="approval",
        status="abstained",
        domain="filesystem",
        action_id="filesystem.read",
        pid=pid,
        request_id=f"request-{suffix}",
        operation_id=f"operation-{suffix}",
        shadow_outcome="require_human",
        reason_codes=("abstained",),
        ood=False,
        abstain=True,
        confidence_bps=0,
        calibration_bucket="unknown",
        classifier_id="checkpoint-regression-scripted",
        classifier_version="1",
        artifact_sha256=_DIGEST,
        input_sha256=_DIGEST,
        feature_snapshot_sha256=_DIGEST,
        policy_sha256=_DIGEST,
        manifest_sha256=_DIGEST,
        action_sha256=_DIGEST,
        resource_sha256=_DIGEST,
        args_sha256=_DIGEST,
        state_sha256=_DIGEST,
        projection_sha256=_DIGEST,
        created_at=created_at,
        completed_at=completed_at,
        latency_ms=1,
    )


def _all_assessments(runtime: Runtime) -> tuple[SemanticAssessmentRecord, ...]:
    return runtime.uow.semantic.query_semantic_assessments(
        after=None,
        limit=100,
    ).records


def test_checkpoint_fork_and_restore_leave_semantic_assessment_ledger_global() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="semantic checkpoint ledger boundary")
        before = _assessment(
            pid,
            suffix="before-checkpoint",
            created_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
        )
        runtime.uow.semantic.append_semantic_assessment(before)

        checkpoint_id = runtime.checkpoint.create(
            pid,
            "semantic assessment ledger exclusion",
            actor=pid,
        )
        found = runtime.store.get_checkpoint_snapshot(checkpoint_id)
        assert found is not None
        _, snapshot = found
        assert "semantic_assessments" not in snapshot["rows"]
        assert before.assessment_id not in dumps(snapshot)

        after = _assessment(
            pid,
            suffix="after-checkpoint",
            created_at="2026-01-01T00:00:02Z",
            completed_at="2026-01-01T00:00:03Z",
        )
        runtime.uow.semantic.append_semantic_assessment(after)
        expected = (before, after)

        forked = runtime.checkpoint.fork_from_checkpoint(
            "host",
            checkpoint_id,
            require_capability=False,
        )
        fork_pid = str(forked["fork_root_pid"])
        assert _all_assessments(runtime) == expected
        assert runtime.uow.semantic.query_semantic_assessments(
            after=None,
            limit=100,
            pid=fork_pid,
        ).records == ()

        restored = runtime.checkpoint.restore(
            "host",
            checkpoint_id,
            require_capability=False,
        )
        assert restored["main_state_committed"] is True
        assert _all_assessments(runtime) == expected
        assert runtime.uow.semantic.get_semantic_assessment(
            before.assessment_id
        ) == before
        assert runtime.uow.semantic.get_semantic_assessment(
            after.assessment_id
        ) == after
    finally:
        runtime.close()
