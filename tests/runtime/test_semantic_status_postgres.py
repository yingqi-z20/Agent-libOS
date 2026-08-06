from __future__ import annotations

import contextlib
from dataclasses import replace
import hashlib
import json
import os
from collections.abc import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.models.semantic import SemanticAssessmentStatus, SemanticDomain
from agent_libos.storage import (
    PostgresStore,
    SemanticAssessmentCursor,
    SemanticAssessmentJobRecord,
    SemanticAssessmentJobStatus,
    SemanticAssessmentRecord,
    SemanticProjectionRetention,
    UnitOfWork,
)


_DIGEST = "1" * 64


def _projection_sha256(projection: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _job(
    job_id: str,
    *,
    created_at: str = "2026-01-01T00:00:00Z",
) -> SemanticAssessmentJobRecord:
    projection = {"action_id": "filesystem.read", "resource_count": 1}
    return SemanticAssessmentJobRecord(
        job_id=job_id,
        kind="approval",
        status=SemanticAssessmentJobStatus.QUEUED,
        domain="filesystem",
        pid="postgres-pid",
        request_id=f"request-{job_id}",
        operation_id=f"operation-{job_id}",
        bindings={
            "artifact_sha256": _DIGEST,
            "input_sha256": _DIGEST,
            "feature_snapshot_sha256": _DIGEST,
            "policy_sha256": _DIGEST,
            "manifest_sha256": _DIGEST,
            "action_sha256": _DIGEST,
            "resource_sha256": _DIGEST,
            "args_sha256": _DIGEST,
            "state_sha256": _DIGEST,
            "sink_identity_sha256": None,
            "tenant_bucket_sha256": _DIGEST,
        },
        projection=projection,
        projection_sha256=_projection_sha256(projection),
        projection_retention=SemanticProjectionRetention.REDACTED,
        projection_expires_at="2026-01-01T02:00:00Z",
        created_at=created_at,
        updated_at=created_at,
    )


def _assessment(
    job: SemanticAssessmentJobRecord,
    *,
    assessment_id: str | None = None,
) -> SemanticAssessmentRecord:
    return SemanticAssessmentRecord(
        assessment_id=assessment_id or f"assessment-{job.job_id}",
        job_id=job.job_id,
        kind=job.kind,
        status="success",
        domain=job.domain,
        action_id="filesystem.read",
        pid=job.pid,
        request_id=job.request_id,
        operation_id=job.operation_id,
        effect_id=job.effect_id,
        shadow_outcome="require_human",
        reason_codes=("abstained",),
        ood=False,
        abstain=False,
        confidence_bps=7500,
        calibration_bucket="high",
        classifier_id="postgres-scripted",
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
        projection_sha256=job.projection_sha256,
        created_at=job.created_at,
        completed_at="2026-01-01T00:00:02Z",
        latency_ms=2,
        tenant_bucket_sha256=_DIGEST,
        missing_predicates=("profile_pinned",),
    )


def _claim(
    unit: UnitOfWork,
    *,
    lease_id: str,
) -> SemanticAssessmentJobRecord:
    claimed = unit.semantic.claim_next_semantic_assessment_job(
        lease_owner_id="postgres-worker",
        lease_id=lease_id,
        lease_expires_at="2026-01-01T01:00:00Z",
        updated_at="2026-01-01T00:00:01Z",
    )
    assert claimed is not None
    return claimed


def _target(
    claimed: SemanticAssessmentJobRecord,
    assessment: SemanticAssessmentRecord,
) -> SemanticAssessmentJobRecord:
    return replace(
        claimed,
        assessment_id=assessment.assessment_id,
        status=SemanticAssessmentJobStatus.SUCCEEDED,
        revision=claimed.revision + 1,
        lease_owner_id=None,
        lease_id=None,
        lease_expires_at=None,
        projection={},
        projection_retention=SemanticProjectionRetention.HASH_ONLY,
        projection_expires_at=None,
        updated_at=assessment.completed_at,
        completed_at=assessment.completed_at,
    )


@contextlib.contextmanager
def _postgres_schema_dsn() -> Iterator[str]:
    import psycopg
    from psycopg import sql

    base_dsn = os.environ["AGENT_LIBOS_POSTGRES_DSN"]
    schema = f"agent_libos_semantic_aggregate_{uuid4().hex}"
    parsed = urlsplit(base_dsn)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "options"
    ]
    query.append(("options", f"-csearch_path={schema}"))
    selected_dsn = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )
    with psycopg.connect(base_dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
        )
    try:
        yield selected_dsn
    finally:
        with psycopg.connect(base_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


@pytest.mark.postgres
def test_postgres_semantic_status_aggregate_counts_one_snapshot() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            with store.transaction() as cursor:
                cursor.execute(
                    """
                    INSERT INTO semantic_assessment_jobs (
                        job_id, kind, status, domain, bindings_json,
                        projection_json, projection_sha256,
                        projection_retention, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "postgres-job",
                        "approval",
                        "queued",
                        "filesystem",
                        "{}",
                        "{}",
                        "1" * 64,
                        "redacted",
                        "2026-01-01T00:00:00.000000+00:00",
                        "2026-01-01T00:00:00.000000+00:00",
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO semantic_assessments (
                        assessment_id, job_id, kind, status, domain,
                        action_id, shadow_outcome, ood, record_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "postgres-assessment-1",
                        "postgres-assessment-job-1",
                        "approval",
                        "success",
                        "filesystem",
                        "filesystem.read",
                        "would_issue_exact_once",
                        0,
                        "{}",
                        "2026-01-01T00:00:00.000000+00:00",
                    ),
                )
                cursor.execute(
                    """
                    INSERT INTO semantic_assessments (
                        assessment_id, job_id, kind, status, domain,
                        action_id, shadow_outcome, ood, record_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "postgres-assessment-2",
                        "postgres-assessment-job-2",
                        "provider_ingress",
                        "ood",
                        "mcp",
                        "mcp.invoke",
                        "require_human",
                        1,
                        "{}",
                        "2026-01-01T00:00:01.000000+00:00",
                    ),
                )

            aggregate = UnitOfWork(store).semantic.semantic_status_aggregate()

            assert aggregate.job_total == 1
            assert aggregate.job_counts == {
                status.value: (
                    1 if status is SemanticAssessmentJobStatus.QUEUED else 0
                )
                for status in SemanticAssessmentJobStatus
            }
            assert aggregate.assessment_total == 2
            assert aggregate.assessment_status_counts == {
                status.value: (
                    1
                    if status
                    in {
                        SemanticAssessmentStatus.SUCCESS,
                        SemanticAssessmentStatus.OOD,
                    }
                    else 0
                )
                for status in SemanticAssessmentStatus
            }
            assert aggregate.assessment_domain_counts == {
                domain.value: (
                    1
                    if domain in {SemanticDomain.FILESYSTEM, SemanticDomain.MCP}
                    else 0
                )
                for domain in SemanticDomain
            }
            assert aggregate.assessment_ood_count == 1
            assert aggregate.shadow_outcome_counts == {
                "would_issue_exact_once": 1,
                "require_human": 1,
                "would_deny": 0,
            }
        finally:
            store.close()


@pytest.mark.postgres
def test_postgres_semantic_repository_lifecycle_and_conflict_rollback() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            unit = UnitOfWork(store)
            queued = _job("postgres-lifecycle")
            assert unit.semantic.enqueue_semantic_assessment_job(queued) == queued
            assert unit.semantic.enqueue_semantic_assessment_job(queued) == queued

            conflicting_projection = {
                "action_id": "filesystem.read",
                "resource_count": 2,
            }
            with pytest.raises(ValidationError, match="conflicts"):
                unit.semantic.enqueue_semantic_assessment_job(
                    replace(
                        queued,
                        projection=conflicting_projection,
                        projection_sha256=_projection_sha256(
                            conflicting_projection
                        ),
                    )
                )

            claimed = _claim(unit, lease_id="postgres-lease-lifecycle")
            assert claimed.revision == 1
            assert claimed.attempt_count == 1
            assert unit.semantic.claim_next_semantic_assessment_job(
                lease_owner_id="postgres-other-worker",
                lease_id="postgres-other-lease",
                lease_expires_at="2026-01-01T01:00:00Z",
                updated_at="2026-01-01T00:00:01Z",
            ) is None

            assessment = _assessment(claimed)
            target = _target(claimed, assessment)
            assert unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                target,
                assessment,
            ) is True
            assert unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                target,
                assessment,
            ) is False
            persisted_job = unit.semantic.get_semantic_assessment_job(
                claimed.job_id
            )
            assert persisted_job is not None
            assert persisted_job.projection == {}
            assert (
                persisted_job.projection_retention
                is SemanticProjectionRetention.HASH_ONLY
            )
            assert persisted_job.projection_expires_at is None
            assert unit.semantic.get_semantic_assessment(
                assessment.assessment_id
            ) == assessment

            rollback_job = _job("postgres-rollback")
            unit.semantic.enqueue_semantic_assessment_job(rollback_job)
            rollback_claimed = _claim(
                unit,
                lease_id="postgres-lease-rollback",
            )
            candidate = _assessment(rollback_claimed)
            conflicting_assessment = replace(
                candidate,
                human_outcome="pending",
            )
            assert unit.semantic.append_semantic_assessment(
                conflicting_assessment
            ) == conflicting_assessment
            assert unit.semantic.append_semantic_assessment(
                conflicting_assessment
            ) == conflicting_assessment
            with pytest.raises(ValidationError, match="conflicts"):
                unit.semantic.append_semantic_assessment(candidate)

            rollback_target = _target(rollback_claimed, candidate)
            with pytest.raises(ValidationError, match="conflicts"):
                unit.semantic.terminalize_semantic_assessment_job(
                    rollback_claimed,
                    rollback_target,
                    candidate,
                )
            assert unit.semantic.get_semantic_assessment_job(
                rollback_claimed.job_id
            ) == rollback_claimed
            assert unit.semantic.get_semantic_assessment(
                candidate.assessment_id
            ) == conflicting_assessment
        finally:
            store.close()


@pytest.mark.postgres
def test_postgres_semantic_lease_recovery_normalizes_utc_offsets() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        try:
            unit = UnitOfWork(store)
            unit.semantic.enqueue_semantic_assessment_job(
                _job(
                    "postgres-offset-lease",
                    created_at="2025-12-31T09:00:00Z",
                )
            )
            claimed = unit.semantic.claim_next_semantic_assessment_job(
                lease_owner_id="postgres-offset-worker",
                lease_id="postgres-offset-lease",
                lease_expires_at="2026-01-01T00:00:00+14:00",
                updated_at="2025-12-31T09:00:00Z",
            )
            assert claimed is not None
            assert (
                claimed.lease_expires_at
                == "2025-12-31T10:00:00.000000+00:00"
            )
            assert unit.semantic.query_expired_semantic_assessment_jobs(
                expired_before="2025-12-31T11:00:00-00:00",
                limit=1,
            ) == (claimed,)
            assert unit.semantic.claim_next_semantic_assessment_job(
                lease_owner_id="postgres-replay-worker",
                lease_id="postgres-replay-lease",
                lease_expires_at="2025-12-31T12:00:00Z",
                updated_at="2025-12-31T11:00:00Z",
            ) is None

            assessment = replace(
                _assessment(claimed),
                status="provider_outcome_unknown",
                reason_codes=("provider_outcome_unknown",),
                completed_at="2025-12-31T11:00:00Z",
            )
            recovered = replace(
                claimed,
                assessment_id=assessment.assessment_id,
                status=SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN,
                revision=claimed.revision + 1,
                lease_owner_id=None,
                lease_id=None,
                lease_expires_at=None,
                projection={},
                projection_retention=SemanticProjectionRetention.HASH_ONLY,
                projection_expires_at=None,
                error_code="provider_outcome_unknown",
                updated_at=assessment.completed_at,
                completed_at=assessment.completed_at,
            )
            assert unit.semantic.terminalize_semantic_assessment_job(
                claimed,
                recovered,
                assessment,
            ) is True
            persisted = unit.semantic.get_semantic_assessment_job(claimed.job_id)
            assert persisted is not None
            assert (
                persisted.status
                is SemanticAssessmentJobStatus.PROVIDER_OUTCOME_UNKNOWN
            )
            assert persisted.projection == {}
            assert (
                persisted.projection_retention
                is SemanticProjectionRetention.HASH_ONLY
            )
            assert unit.semantic.get_semantic_assessment(
                assessment.assessment_id
            ) == assessment
            assert unit.semantic.claim_next_semantic_assessment_job(
                lease_owner_id="postgres-post-recovery-worker",
                lease_id="postgres-post-recovery-lease",
                lease_expires_at="2025-12-31T13:00:00Z",
                updated_at="2025-12-31T12:00:00Z",
            ) is None
        finally:
            store.close()


@pytest.mark.postgres
def test_postgres_semantic_unicode_keyset_survives_reopen() -> None:
    with _postgres_schema_dsn() as dsn:
        store = PostgresStore(dsn)
        records = tuple(
            _assessment(
                _job(job_id),
                assessment_id=f"assessment-{job_id}",
            )
            for job_id in ("a", "é", "中")
        )
        try:
            unit = UnitOfWork(store)
            for record in records:
                unit.semantic.append_semantic_assessment(record)

            first = unit.semantic.query_semantic_assessments(
                after=None,
                limit=2,
            )
            assert len(first.records) == 2
            assert isinstance(first.next_cursor, SemanticAssessmentCursor)
            second = unit.semantic.query_semantic_assessments(
                after=first.next_cursor,
                limit=2,
            )
            assert len(second.records) == 1
            assert {
                record.assessment_id
                for record in (*first.records, *second.records)
            } == {record.assessment_id for record in records}
        finally:
            store.close()

        reopened = PostgresStore(dsn)
        try:
            page = UnitOfWork(reopened).semantic.query_semantic_assessments(
                after=None,
                limit=3,
            )
            assert {record.assessment_id for record in page.records} == {
                record.assessment_id for record in records
            }
        finally:
            reopened.close()
