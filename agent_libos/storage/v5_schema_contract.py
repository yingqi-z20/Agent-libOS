from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class V5ColumnContract:
    """Backend-neutral shape for a security-relevant schema-v5 column."""

    sql_type: str
    nullable: bool
    default: str | None = None
    primary_key_position: int = 0
    keyset_collation: bool = False


# ``human_requests`` predates schema v5; its full canonical v4 shape plus the
# new revision column are security relevant to CAS.  PostgreSQL can retain a
# dropped-column attnum when a v4 fixture is produced from v5, so human column
# names/order are exact but their physical ordinal numbers are not compared.
V5_STORAGE_COLUMN_CONTRACTS: dict[
    str,
    tuple[tuple[str, V5ColumnContract], ...],
] = {
    "human_requests": (
        (
            "request_id",
            V5ColumnContract(
                "text",
                nullable=False,
                primary_key_position=1,
                keyset_collation=True,
            ),
        ),
        ("pid", V5ColumnContract("text", nullable=False)),
        ("human", V5ColumnContract("text", nullable=False)),
        ("payload_json", V5ColumnContract("text", nullable=False)),
        ("status", V5ColumnContract("text", nullable=False)),
        ("decision_json", V5ColumnContract("text", nullable=True)),
        ("blocking", V5ColumnContract("integer", nullable=False)),
        (
            "created_at",
            V5ColumnContract("text", nullable=False, keyset_collation=True),
        ),
        ("updated_at", V5ColumnContract("text", nullable=False)),
        (
            "revision",
            V5ColumnContract("bigint", nullable=False, default="0"),
        ),
    ),
    "semantic_assessment_jobs": (
        (
            "job_id",
            V5ColumnContract(
                "text",
                nullable=False,
                primary_key_position=1,
                keyset_collation=True,
            ),
        ),
        (
            "assessment_id",
            V5ColumnContract("text", nullable=True, keyset_collation=True),
        ),
        ("kind", V5ColumnContract("text", nullable=False)),
        ("status", V5ColumnContract("text", nullable=False)),
        ("domain", V5ColumnContract("text", nullable=False)),
        ("pid", V5ColumnContract("text", nullable=True)),
        ("request_id", V5ColumnContract("text", nullable=True)),
        ("operation_id", V5ColumnContract("text", nullable=True)),
        ("effect_id", V5ColumnContract("text", nullable=True)),
        (
            "revision",
            V5ColumnContract("bigint", nullable=False, default="0"),
        ),
        (
            "attempt_count",
            V5ColumnContract("integer", nullable=False, default="0"),
        ),
        ("lease_owner_id", V5ColumnContract("text", nullable=True)),
        ("lease_id", V5ColumnContract("text", nullable=True)),
        ("lease_expires_at", V5ColumnContract("text", nullable=True)),
        ("bindings_json", V5ColumnContract("text", nullable=False)),
        ("projection_json", V5ColumnContract("text", nullable=False)),
        ("projection_sha256", V5ColumnContract("text", nullable=False)),
        ("projection_retention", V5ColumnContract("text", nullable=False)),
        ("projection_expires_at", V5ColumnContract("text", nullable=True)),
        ("error_code", V5ColumnContract("text", nullable=True)),
        (
            "created_at",
            V5ColumnContract("text", nullable=False, keyset_collation=True),
        ),
        ("updated_at", V5ColumnContract("text", nullable=False)),
        ("completed_at", V5ColumnContract("text", nullable=True)),
    ),
    "semantic_assessments": (
        (
            "assessment_id",
            V5ColumnContract(
                "text",
                nullable=False,
                primary_key_position=1,
                keyset_collation=True,
            ),
        ),
        ("job_id", V5ColumnContract("text", nullable=False)),
        ("kind", V5ColumnContract("text", nullable=False)),
        ("status", V5ColumnContract("text", nullable=False)),
        ("domain", V5ColumnContract("text", nullable=False)),
        ("action_id", V5ColumnContract("text", nullable=False)),
        ("tenant_bucket_sha256", V5ColumnContract("text", nullable=True)),
        ("pid", V5ColumnContract("text", nullable=True)),
        ("request_id", V5ColumnContract("text", nullable=True)),
        ("operation_id", V5ColumnContract("text", nullable=True)),
        ("effect_id", V5ColumnContract("text", nullable=True)),
        ("shadow_outcome", V5ColumnContract("text", nullable=True)),
        ("ood", V5ColumnContract("integer", nullable=False)),
        ("record_json", V5ColumnContract("text", nullable=False)),
        (
            "created_at",
            V5ColumnContract("text", nullable=False, keyset_collation=True),
        ),
        ("completed_at", V5ColumnContract("text", nullable=True)),
    ),
}


V4_HUMAN_REQUEST_COLUMN_CONTRACTS = {
    "human_requests": V5_STORAGE_COLUMN_CONTRACTS["human_requests"][:-1],
}


HUMAN_REQUEST_INDEX_CONTRACTS: dict[
    str,
    tuple[str, tuple[str, ...], bool, bool],
] = {
    "idx_human_requests_pid_created": (
        "human_requests",
        ("pid", "created_at", "request_id"),
        False,
        False,
    ),
    "idx_human_requests_human_status_created": (
        "human_requests",
        ("human", "status", "created_at", "request_id"),
        False,
        False,
    ),
    "idx_human_requests_status_created": (
        "human_requests",
        ("status", "created_at", "request_id"),
        False,
        False,
    ),
}


V4_HUMAN_REQUEST_CHECKS: dict[str, tuple[str, ...]] = {
    "human_requests": (),
}


V5_STORAGE_KEY_CONSTRAINTS: dict[
    str,
    tuple[tuple[str, tuple[str, ...]], ...],
] = {
    "human_requests": (("primary_key", ("request_id",)),),
    "semantic_assessment_jobs": (
        ("primary_key", ("job_id",)),
        ("unique", ("assessment_id",)),
    ),
    "semantic_assessments": (
        ("primary_key", ("assessment_id",)),
        ("unique", ("job_id",)),
    ),
}


V4_HUMAN_REQUEST_KEY_CONSTRAINTS = {
    "human_requests": V5_STORAGE_KEY_CONSTRAINTS["human_requests"],
}


# These expressions are compared after backend-specific lexical
# canonicalization.  The comparison is a multiset comparison: a duplicate,
# missing, or additional CHECK is rejected just like an altered expression.
V5_STORAGE_SQLITE_CHECKS: dict[str, tuple[str, ...]] = {
    "human_requests": ("revision >= 0",),
    "semantic_assessment_jobs": (
        "status IN ('queued', 'claimed', 'succeeded', 'failed', "
        "'egress_blocked', 'provider_outcome_unknown', 'cancelled', 'expired')",
        "revision >= 0",
        "attempt_count >= 0 AND attempt_count <= 1",
        "(lease_owner_id IS NULL AND lease_id IS NULL "
        "AND lease_expires_at IS NULL) OR "
        "(lease_owner_id IS NOT NULL AND lease_id IS NOT NULL "
        "AND lease_expires_at IS NOT NULL)",
        "(status = 'claimed') = (lease_id IS NOT NULL)",
        "(status IN ('queued', 'claimed') AND completed_at IS NULL) OR "
        "(status NOT IN ('queued', 'claimed') AND completed_at IS NOT NULL)",
        "status IN ('queued', 'claimed') OR "
        "(projection_retention = 'hash_only' "
        "AND projection_json = '{}' AND projection_expires_at IS NULL)",
        "projection_retention IN ('redacted', 'hash_only')",
    ),
    "semantic_assessments": ("ood IN (0, 1)",),
}


# ``pg_get_constraintdef(..., true)`` renders IN/NOT IN as ANY/ALL arrays.
# Comparing its normalized output avoids relying on generated constraint names
# while still rejecting every expression or multiplicity drift.
V5_STORAGE_POSTGRES_CHECKS: dict[str, tuple[str, ...]] = {
    "human_requests": ("CHECK (revision >= 0)",),
    "semantic_assessment_jobs": (
        "CHECK (status = ANY (ARRAY['queued'::text, 'claimed'::text, "
        "'succeeded'::text, 'failed'::text, 'egress_blocked'::text, "
        "'provider_outcome_unknown'::text, 'cancelled'::text, "
        "'expired'::text]))",
        "CHECK (revision >= 0)",
        "CHECK (attempt_count >= 0 AND attempt_count <= 1)",
        "CHECK (lease_owner_id IS NULL AND lease_id IS NULL "
        "AND lease_expires_at IS NULL OR lease_owner_id IS NOT NULL "
        "AND lease_id IS NOT NULL AND lease_expires_at IS NOT NULL)",
        "CHECK ((status = 'claimed'::text) = (lease_id IS NOT NULL))",
        "CHECK ((status = ANY (ARRAY['queued'::text, 'claimed'::text])) "
        "AND completed_at IS NULL OR "
        "(status <> ALL (ARRAY['queued'::text, 'claimed'::text])) "
        "AND completed_at IS NOT NULL)",
        "CHECK ((status = ANY (ARRAY['queued'::text, 'claimed'::text])) "
        "OR projection_retention = 'hash_only'::text "
        "AND projection_json = '{}'::text "
        "AND projection_expires_at IS NULL)",
        "CHECK (projection_retention = ANY "
        "(ARRAY['redacted'::text, 'hash_only'::text]))",
    ),
    "semantic_assessments": ("CHECK (ood = ANY (ARRAY[0, 1]))",),
}
