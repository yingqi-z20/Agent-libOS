from __future__ import annotations

import re

from agent_libos.storage.v5_schema_contract import V5ColumnContract


C = V5ColumnContract


V6_STORAGE_COLUMN_CONTRACTS: dict[str, tuple[tuple[str, V5ColumnContract], ...]] = {
    "semantic_flow_entities": (
        ("entity_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("kind", C("text", False)),
        ("pid", C("text", True)),
        ("tenant_bucket_sha256", C("text", False)),
        ("content_sha256", C("text", False)),
        ("version_sha256", C("text", False)),
        ("provenance_sha256", C("text", False)),
        ("baseline_labels_json", C("text", False)),
        ("identity_present", C("integer", False)),
        ("identity_mixed", C("integer", False)),
        ("coverage", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_flow_activities": (
        ("activity_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("kind", C("text", False)),
        ("pid", C("text", False)),
        ("action_id", C("text", True)),
        ("effect_id", C("text", True)),
        ("state_sha256", C("text", False)),
        ("provider_spec_sha256", C("text", True)),
        ("tool_schema_sha256", C("text", True)),
        ("model_artifact_sha256", C("text", True)),
        ("tenant_bucket_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_flow_edges": (
        ("edge_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("relation", C("text", False)),
        ("source_node_id", C("text", False, keyset_collation=True)),
        ("source_node_type", C("text", False)),
        ("target_node_id", C("text", False, keyset_collation=True)),
        ("target_node_type", C("text", False)),
        ("pid", C("text", False)),
        ("provenance_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_flow_label_assertions": (
        ("assertion_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("entity_id", C("text", False, keyset_collation=True)),
        ("source", C("text", False)),
        ("sensitivity_floor", C("text", False)),
        ("integrity_ceiling", C("text", False)),
        ("trust_ceiling", C("text", False)),
        ("evidence_sha256", C("text", False)),
        ("assessment_id", C("text", True)),
        ("locator_sha256", C("text", True)),
        ("locator_kind", C("text", True)),
        ("path_sha256s_json", C("text", False)),
        ("value_sha256", C("text", True)),
        ("ordinal", C("integer", True)),
        ("offset_start", C("integer", True)),
        ("offset_end", C("integer", True)),
        ("category", C("text", True)),
        ("coverage", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_legacy_coverage": (
        ("singleton", C("integer", False, primary_key_position=1)),
        ("source_schema_version", C("integer", False)),
        ("assessment_count", C("bigint", False)),
        ("coverage", C("text", False)),
        ("evidence_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_policy_epochs": (
        ("epoch_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("generation", C("bigint", False)),
        ("catalog_version", C("integer", False)),
        ("policy_sha256", C("text", False, keyset_collation=True)),
        ("expected_previous_sha256", C("text", True)),
        ("rollout_scope_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_control_state": (
        ("singleton", C("integer", False, primary_key_position=1)),
        ("revision", C("bigint", False, default="0")),
        ("generation", C("bigint", False, default="0")),
        ("mode", C("text", False)),
        ("active_epoch_id", C("text", True)),
        ("active_policy_sha256", C("text", True)),
        ("tripped", C("integer", False, default="0")),
        ("trip_code", C("text", True)),
        ("updated_at", C("text", False)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_control_transitions": (
        ("transition_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("revision", C("bigint", False)),
        ("generation", C("bigint", False)),
        ("mode", C("text", False)),
        ("active_epoch_id", C("text", True)),
        ("active_policy_sha256", C("text", True)),
        ("tripped", C("integer", False)),
        ("trip_code", C("text", True)),
        ("evidence_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_machine_settlements": (
        ("settlement_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("assessment_id", C("text", True)),
        ("job_id", C("text", True)),
        ("request_id", C("text", False)),
        ("request_revision", C("bigint", False)),
        ("pid", C("text", False)),
        ("operation_id", C("text", True)),
        ("effect_id", C("text", False)),
        ("epoch_id", C("text", False)),
        ("policy_sha256", C("text", False)),
        ("tenant_bucket_sha256", C("text", False)),
        ("action_id", C("text", False)),
        ("outcome", C("text", False)),
        ("capability_id", C("text", True)),
        ("binding_sha256", C("text", False)),
        ("decision_sha256", C("text", False)),
        ("matched_rule_id", C("text", True)),
        ("reason_codes_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_review_labels": (
        ("review_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("settlement_id", C("text", False)),
        ("outcome", C("text", False)),
        ("reviewer_sha256", C("text", False)),
        ("evidence_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_health_events": (
        ("event_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("event_kind", C("text", False)),
        ("severity", C("text", False)),
        ("epoch_id", C("text", True)),
        ("tenant_bucket_sha256", C("text", True)),
        ("evidence_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_human_outcome_links": (
        ("link_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("request_id", C("text", False, keyset_collation=True)),
        ("request_revision", C("bigint", False)),
        ("pid", C("text", False)),
        ("assessment_id", C("text", True)),
        ("job_id", C("text", True)),
        ("settlement_id", C("text", True)),
        ("outcome", C("text", False)),
        ("source", C("text", False)),
        ("decision_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_machine_outcomes": (
        ("outcome_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("settlement_id", C("text", False)),
        ("effect_id", C("text", False)),
        ("outcome", C("text", False)),
        ("evidence_sha256", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("schema_version", C("integer", False, default="1")),
    ),
    "semantic_rate_budgets": (
        ("bucket_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("epoch_id", C("text", False)),
        ("tenant_bucket_sha256", C("text", False)),
        ("rule_id", C("text", False)),
        ("minute_window_started_at", C("text", False)),
        ("day_window_started_at", C("text", False)),
        ("minute_count", C("integer", False, default="0")),
        ("day_count", C("integer", False, default="0")),
        ("inflight_count", C("integer", False, default="0")),
        ("revision", C("bigint", False, default="0")),
        ("updated_at", C("text", False)),
    ),
}


V6_STORAGE_KEY_CONSTRAINTS = {
    table: (("primary_key", (columns[0][0],)),)
    for table, columns in V6_STORAGE_COLUMN_CONTRACTS.items()
}
V6_STORAGE_KEY_CONSTRAINTS["semantic_policy_epochs"] += (
    ("unique", ("generation",)),
    ("unique", ("policy_sha256",)),
)
V6_STORAGE_KEY_CONSTRAINTS["semantic_human_outcome_links"] += (
    ("unique", ("request_id",)),
)


V6_STORAGE_SQLITE_CHECKS = {
    "semantic_flow_entities": (
        "kind IN ('root_goal', 'object_version', 'file_binding_version', 'provider_result', 'tool_result', 'materialization', 'model_output')",
        "identity_present IN (0, 1)",
        "identity_mixed IN (0, 1)",
        "identity_mixed = 0 OR identity_present = 1",
        "coverage IN ('complete', 'partial', 'unknown', 'conflict', 'stale')",
        "schema_version = 1",
    ),
    "semantic_flow_activities": (
        "kind IN ('process_spawn', 'provider_call', 'tool_call', 'llm_call', 'object_create', 'object_update', 'object_append', 'object_materialize', 'object_read', 'file_read', 'file_write', 'transformation', 'aggregation', 'conditional', 'tool_selection', 'memory_retrieval')",
        "schema_version = 1",
    ),
    "semantic_flow_edges": (
        "relation IN ('direct', 'indirect', 'control')",
        "source_node_type IN ('entity', 'activity')",
        "target_node_type IN ('entity', 'activity')",
        "source_node_id <> target_node_id OR source_node_type <> target_node_type",
        "schema_version = 1",
    ),
    "semantic_flow_label_assertions": (
        "source IN ('host', 'model', 'deterministic')",
        "coverage IN ('complete', 'partial', 'unknown', 'conflict', 'stale')",
        "(locator_sha256 IS NULL AND locator_kind IS NULL AND path_sha256s_json = '[]' AND value_sha256 IS NULL AND ordinal IS NULL AND offset_start IS NULL AND offset_end IS NULL) OR (locator_sha256 IS NOT NULL AND locator_kind = 'json_field' AND path_sha256s_json <> '[]' AND value_sha256 IS NOT NULL AND ordinal IS NULL AND offset_start IS NULL AND offset_end IS NULL) OR (locator_sha256 IS NOT NULL AND locator_kind = 'text_chunk' AND path_sha256s_json = '[]' AND value_sha256 IS NOT NULL AND ordinal >= 0 AND offset_start >= 0 AND offset_end > offset_start)",
        "schema_version = 1",
    ),
    "semantic_legacy_coverage": (
        "singleton = 1",
        "source_schema_version = 5",
        "assessment_count >= 0",
        "coverage = 'unknown'",
        "schema_version = 1",
    ),
    "semantic_policy_epochs": (
        "generation > 0",
        "catalog_version = 1",
        "schema_version = 1",
    ),
    "semantic_control_state": (
        "singleton = 1",
        "revision >= 0",
        "generation >= 0",
        "mode IN ('off', 'shadow', 'enforce_deny', 'canary_auto')",
        "(active_epoch_id IS NULL) = (active_policy_sha256 IS NULL)",
        "tripped IN (0, 1)",
        "(tripped = 1) = (trip_code IS NOT NULL)",
        "schema_version = 1",
    ),
    "semantic_control_transitions": (
        "revision >= 0",
        "generation >= 0",
        "mode IN ('off', 'shadow', 'enforce_deny', 'canary_auto')",
        "(active_epoch_id IS NULL) = (active_policy_sha256 IS NULL)",
        "tripped IN (0, 1)",
        "(tripped = 1) = (trip_code IS NOT NULL)",
        "schema_version = 1",
    ),
    "semantic_machine_settlements": (
        "request_revision >= 0",
        "outcome IN ('issued', 'denied', 'require_human', 'race_lost', 'stale', 'budget_exhausted', 'revoked', 'expired', 'failed')",
        "(outcome = 'issued') = (capability_id IS NOT NULL)",
        "schema_version = 1",
    ),
    "semantic_review_labels": (
        "outcome IN ('safe', 'unsafe', 'inconclusive')",
        "schema_version = 1",
    ),
    "semantic_health_events": (
        "severity IN ('info', 'warning', 'critical')",
        "schema_version = 1",
    ),
    "semantic_human_outcome_links": (
        "request_revision >= 0",
        "outcome IN ('approved', 'rejected', 'cancelled')",
        "source IN ('human', 'machine_policy', 'cancel')",
        "(source = 'cancel') = (outcome = 'cancelled')",
        "schema_version = 1",
    ),
    "semantic_machine_outcomes": (
        "outcome IN ('issued', 'consumed', 'succeeded', 'failed', 'outcome_unknown', 'expired', 'revoked', 'race_lost')",
        "schema_version = 1",
    ),
    "semantic_rate_budgets": (
        "minute_count >= 0",
        "day_count >= 0",
        "inflight_count >= 0",
        "revision >= 0",
    ),
}


V6_INDEX_CONTRACTS: dict[str, tuple[str, tuple[str, ...], bool, bool]] = {
    "idx_semantic_flow_entities_created": ("semantic_flow_entities", ("created_at", "entity_id"), False, False),
    "idx_semantic_flow_entities_pid_created": ("semantic_flow_entities", ("pid", "created_at", "entity_id"), False, False),
    "idx_semantic_flow_entities_tenant_created": ("semantic_flow_entities", ("tenant_bucket_sha256", "created_at", "entity_id"), False, False),
    "idx_semantic_flow_activities_created": ("semantic_flow_activities", ("created_at", "activity_id"), False, False),
    "idx_semantic_flow_activities_pid_created": ("semantic_flow_activities", ("pid", "created_at", "activity_id"), False, False),
    "idx_semantic_flow_edges_created": ("semantic_flow_edges", ("created_at", "edge_id"), False, False),
    "idx_semantic_flow_edges_source_created": ("semantic_flow_edges", ("source_node_id", "created_at", "edge_id"), False, False),
    "idx_semantic_flow_edges_target_created": ("semantic_flow_edges", ("target_node_id", "created_at", "edge_id"), False, False),
    "idx_semantic_flow_assertions_entity_created": ("semantic_flow_label_assertions", ("entity_id", "created_at", "assertion_id"), False, False),
    "idx_semantic_policy_epochs_created": ("semantic_policy_epochs", ("created_at", "epoch_id"), False, False),
    "idx_semantic_control_transitions_created": ("semantic_control_transitions", ("created_at", "transition_id"), False, False),
    "idx_semantic_control_transitions_epoch_created": ("semantic_control_transitions", ("active_epoch_id", "created_at", "transition_id"), False, False),
    "idx_semantic_settlements_created": ("semantic_machine_settlements", ("created_at", "settlement_id"), False, False),
    "idx_semantic_settlements_request_created": ("semantic_machine_settlements", ("request_id", "created_at", "settlement_id"), False, False),
    "idx_semantic_settlements_epoch_tenant_created": ("semantic_machine_settlements", ("epoch_id", "tenant_bucket_sha256", "created_at", "settlement_id"), False, False),
    "idx_semantic_reviews_settlement_created": ("semantic_review_labels", ("settlement_id", "created_at", "review_id"), False, False),
    "idx_semantic_health_created": ("semantic_health_events", ("created_at", "event_id"), False, False),
    "idx_semantic_health_epoch_created": ("semantic_health_events", ("epoch_id", "created_at", "event_id"), False, False),
    "idx_semantic_human_outcomes_created": ("semantic_human_outcome_links", ("created_at", "link_id"), False, False),
    "idx_semantic_human_outcomes_pid_created": ("semantic_human_outcome_links", ("pid", "created_at", "link_id"), False, False),
    "idx_semantic_human_outcomes_assessment_created": ("semantic_human_outcome_links", ("assessment_id", "created_at", "link_id"), False, False),
    "idx_semantic_human_outcomes_settlement_created": ("semantic_human_outcome_links", ("settlement_id", "created_at", "link_id"), False, False),
    "idx_semantic_outcomes_created": ("semantic_machine_outcomes", ("created_at", "outcome_id"), False, False),
    "idx_semantic_outcomes_settlement_created": ("semantic_machine_outcomes", ("settlement_id", "created_at", "outcome_id"), False, False),
    "idx_semantic_budgets_scope": ("semantic_rate_budgets", ("tenant_bucket_sha256", "rule_id"), True, False),
}


def _postgres_check(expression: str) -> str:
    match = re.fullmatch(r"([a-z_]+) IN \((.+)\)", expression)
    if match is not None:
        column, values = match.groups()
        selected: list[str] = []
        for raw in (item.strip() for item in values.split(",")):
            if raw.startswith("'") and raw.endswith("'"):
                selected.append(f"{raw}::text")
            else:
                selected.append(raw)
        return f"CHECK ({column} = ANY (ARRAY[{', '.join(selected)}]))"
    selected_expression = re.sub(
        r"= ('[^']*')",
        r"= \1::text",
        expression,
    )
    return f"CHECK ({selected_expression})"


V6_STORAGE_POSTGRES_CHECKS = {
    table: tuple(_postgres_check(expression) for expression in expressions)
    for table, expressions in V6_STORAGE_SQLITE_CHECKS.items()
}
V6_STORAGE_POSTGRES_CHECKS["semantic_flow_label_assertions"] = (
    "CHECK (source = ANY (ARRAY['host'::text, 'model'::text, "
    "'deterministic'::text]))",
    "CHECK (coverage = ANY (ARRAY['complete'::text, 'partial'::text, "
    "'unknown'::text, 'conflict'::text, 'stale'::text]))",
    "CHECK (locator_sha256 IS NULL AND locator_kind IS NULL AND "
    "path_sha256s_json = '[]'::text AND value_sha256 IS NULL AND "
    "ordinal IS NULL AND offset_start IS NULL AND offset_end IS NULL OR "
    "locator_sha256 IS NOT NULL AND locator_kind = 'json_field'::text AND "
    "path_sha256s_json <> '[]'::text AND value_sha256 IS NOT NULL AND "
    "ordinal IS NULL AND offset_start IS NULL AND offset_end IS NULL OR "
    "locator_sha256 IS NOT NULL AND locator_kind = 'text_chunk'::text AND "
    "path_sha256s_json = '[]'::text AND value_sha256 IS NOT NULL AND "
    "ordinal >= 0 AND offset_start >= 0 AND offset_end > offset_start)",
    "CHECK (schema_version = 1)",
)


V6_TABLES = frozenset(V6_STORAGE_COLUMN_CONTRACTS)
