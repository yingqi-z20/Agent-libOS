"""Canonical RuntimeStore schema-v7 contracts for durable MCP client state."""

from __future__ import annotations

import re

from agent_libos.storage.v5_schema_contract import V5ColumnContract


C = V5ColumnContract


V7_STORAGE_COLUMN_CONTRACTS: dict[str, tuple[tuple[str, V5ColumnContract], ...]] = {
    "mcp_continuations": (
        ("continuation_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("server_id", C("text", False, keyset_collation=True)),
        ("server_spec_sha256", C("text", False)),
        ("server_generation", C("bigint", False)),
        ("owner_id", C("text", False, keyset_collation=True)),
        ("auth_principal_sha256", C("text", False)),
        ("auth_scope_sha256", C("text", False)),
        ("request_sha256", C("text", False)),
        ("effect_id", C("text", False)),
        ("capability_sha256", C("text", False)),
        ("data_flow_sha256", C("text", False)),
        ("human_request_id", C("text", False)),
        ("broker_ref", C("text", True)),
        ("broker_value_sha256", C("text", True)),
        ("status", C("text", False)),
        ("revision", C("bigint", False, default="0")),
        ("expires_at", C("text", False, keyset_collation=True)),
        ("metadata_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("updated_at", C("text", False, keyset_collation=True)),
    ),
    "mcp_remote_tasks": (
        ("task_ref", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("server_id", C("text", False, keyset_collation=True)),
        ("server_spec_sha256", C("text", False)),
        ("server_generation", C("bigint", False)),
        ("owner_id", C("text", False, keyset_collation=True)),
        ("auth_principal_sha256", C("text", False)),
        ("auth_scope_sha256", C("text", False)),
        ("origin_request_sha256", C("text", False)),
        ("origin_effect_id", C("text", False)),
        ("human_request_id", C("text", True)),
        ("broker_ref", C("text", True)),
        ("remote_id_sha256", C("text", False)),
        ("status", C("text", False)),
        ("revision", C("bigint", False, default="0")),
        ("expires_at", C("text", True, keyset_collation=True)),
        ("poll_interval_ms", C("bigint", True)),
        ("status_message_sha256", C("text", True)),
        ("result_ref", C("text", True)),
        ("result_sha256", C("text", True)),
        ("metadata_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("updated_at", C("text", False, keyset_collation=True)),
    ),
    "mcp_subscriptions": (
        ("subscription_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("server_id", C("text", False, keyset_collation=True)),
        ("server_spec_sha256", C("text", False)),
        ("server_generation", C("bigint", False)),
        ("owner_id", C("text", False, keyset_collation=True)),
        ("auth_principal_sha256", C("text", False)),
        ("auth_scope_sha256", C("text", False)),
        ("requested_filter_sha256", C("text", False)),
        ("acknowledged_filter_sha256", C("text", True)),
        ("status", C("text", False)),
        ("queue_limit", C("bigint", False)),
        ("event_max_bytes", C("bigint", False)),
        ("received_count", C("bigint", False, default="0")),
        ("dropped_count", C("bigint", False, default="0")),
        ("revision", C("bigint", False, default="0")),
        ("last_event_at", C("text", True, keyset_collation=True)),
        ("metadata_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("updated_at", C("text", False, keyset_collation=True)),
    ),
    "mcp_auth_metadata": (
        ("profile_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("server_id", C("text", False, keyset_collation=True)),
        ("server_spec_sha256", C("text", False)),
        ("server_generation", C("bigint", False)),
        ("status", C("text", False)),
        ("issuer_sha256", C("text", True)),
        ("resource_sha256", C("text", True)),
        ("audience_sha256", C("text", True)),
        ("scopes_sha256", C("text", False)),
        ("principal_sha256", C("text", True)),
        ("expires_at", C("text", True, keyset_collation=True)),
        ("credential_generation", C("bigint", False, default="0")),
        ("revision", C("bigint", False, default="0")),
        ("metadata_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("updated_at", C("text", False, keyset_collation=True)),
    ),
    "mcp_side_effect_preparations": (
        ("preparation_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("operation_kind", C("text", False)),
        ("operation_id", C("text", False, keyset_collation=True)),
        ("operation_revision", C("bigint", True)),
        ("server_id", C("text", False, keyset_collation=True)),
        ("server_spec_sha256", C("text", False)),
        ("server_generation", C("bigint", False)),
        ("owner_id", C("text", False, keyset_collation=True)),
        ("auth_principal_sha256", C("text", False)),
        ("auth_scope_sha256", C("text", False)),
        ("human_request_id", C("text", True)),
        ("human_preview_sha256", C("text", True)),
        ("broker_ref", C("text", True)),
        ("broker_value_sha256", C("text", True)),
        ("result_ref", C("text", True)),
        ("result_sha256", C("text", True)),
        ("status", C("text", False)),
        ("revision", C("bigint", False, default="0")),
        ("expires_at", C("text", False, keyset_collation=True)),
        ("metadata_json", C("text", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("updated_at", C("text", False, keyset_collation=True)),
    ),
}


V7_STORAGE_KEY_CONSTRAINTS = {
    table: (("primary_key", (columns[0][0],)),)
    for table, columns in V7_STORAGE_COLUMN_CONTRACTS.items()
}
V7_STORAGE_KEY_CONSTRAINTS["mcp_continuations"] += (
    ("unique", ("human_request_id",)),
    ("unique", ("broker_ref",)),
)
V7_STORAGE_KEY_CONSTRAINTS["mcp_remote_tasks"] += (
    ("unique", ("human_request_id",)),
    ("unique", ("broker_ref",)),
)
V7_STORAGE_KEY_CONSTRAINTS["mcp_side_effect_preparations"] += (
    ("unique", ("operation_kind", "operation_id")),
    ("unique", ("human_request_id",)),
    ("unique", ("broker_ref",)),
    ("unique", ("result_ref",)),
)


# These references deliberately use the default NO ACTION policy.  Human
# requests are durable audit records; MCP lifecycle cleanup may redact their
# payload but must neither delete them nor cascade-delete continuation/task
# evidence.
V7_STORAGE_FOREIGN_KEYS: dict[
    str,
    tuple[tuple[str, str, str, str, str], ...],
] = {
    "mcp_continuations": (
        ("human_request_id", "human_requests", "request_id", "NO ACTION", "NO ACTION"),
    ),
    "mcp_remote_tasks": (
        ("human_request_id", "human_requests", "request_id", "NO ACTION", "NO ACTION"),
    ),
}


_HASH_CHECKS = {
    "mcp_continuations": (
        "server_spec_sha256", "auth_principal_sha256", "auth_scope_sha256",
        "request_sha256", "capability_sha256", "data_flow_sha256",
        "broker_value_sha256",
    ),
    "mcp_remote_tasks": (
        "server_spec_sha256", "auth_principal_sha256", "auth_scope_sha256",
        "origin_request_sha256", "remote_id_sha256", "status_message_sha256",
        "result_sha256",
    ),
    "mcp_subscriptions": (
        "server_spec_sha256", "auth_principal_sha256", "auth_scope_sha256",
        "requested_filter_sha256", "acknowledged_filter_sha256",
    ),
    "mcp_auth_metadata": (
        "server_spec_sha256", "issuer_sha256", "resource_sha256",
        "audience_sha256", "scopes_sha256", "principal_sha256",
    ),
    "mcp_side_effect_preparations": (
        "server_spec_sha256", "auth_principal_sha256", "auth_scope_sha256",
        "human_preview_sha256", "broker_value_sha256", "result_sha256",
    ),
}


def _digest_checks(table: str) -> tuple[str, ...]:
    nullable = {
        name
        for name, contract in V7_STORAGE_COLUMN_CONTRACTS[table]
        if contract.nullable
    }
    return tuple(
        (
            f"length({column}) = 64"
            if column not in nullable
            else f"{column} IS NULL OR length({column}) = 64"
        )
        for column in _HASH_CHECKS[table]
    )


V7_STORAGE_SQLITE_CHECKS: dict[str, tuple[str, ...]] = {
    "mcp_continuations": (
        "server_generation >= 0",
        "(broker_ref IS NULL) = (broker_value_sha256 IS NULL)",
        "status IN ('input_required', 'dispatching', 'complete', 'cancelled', 'expired', 'needs_attention')",
        "revision >= 0",
        *_digest_checks("mcp_continuations"),
    ),
    "mcp_remote_tasks": (
        "server_generation >= 0",
        "status IN ('working', 'input_required', 'completed', 'failed', 'cancelled', 'cancel_requested', 'update_dispatching', 'cancel_dispatching', 'needs_attention')",
        "status <> 'input_required' OR human_request_id IS NOT NULL",
        "revision >= 0",
        "poll_interval_ms IS NULL OR poll_interval_ms >= 0",
        "(result_ref IS NULL) = (result_sha256 IS NULL)",
        *_digest_checks("mcp_remote_tasks"),
    ),
    "mcp_subscriptions": (
        "server_generation >= 0",
        "status IN ('starting', 'active', 'stopping', 'stopped', 'lost', 'needs_attention')",
        "queue_limit > 0",
        "event_max_bytes > 0",
        "received_count >= 0",
        "dropped_count >= 0",
        "dropped_count <= received_count",
        "revision >= 0",
        *_digest_checks("mcp_subscriptions"),
    ),
    "mcp_auth_metadata": (
        "server_generation >= 0",
        "status IN ('unconfigured', 'authorization_required', 'authorized', 'expired', 'revoked', 'needs_attention')",
        "credential_generation >= 0",
        "revision >= 0",
        *_digest_checks("mcp_auth_metadata"),
    ),
    "mcp_side_effect_preparations": (
        "operation_kind IN ('continuation', 'remote_task')",
        "operation_revision IS NULL OR operation_revision >= 0",
        "server_generation >= 0",
        "(human_request_id IS NULL) = (human_preview_sha256 IS NULL)",
        "(broker_ref IS NULL) = (broker_value_sha256 IS NULL)",
        "(result_ref IS NULL) = (result_sha256 IS NULL)",
        "broker_ref IS NULL OR result_ref IS NULL OR broker_ref <> result_ref",
        "status IN ('prepared', 'cleaning')",
        "revision >= 0",
        *_digest_checks("mcp_side_effect_preparations"),
    ),
}


V7_INDEX_CONTRACTS: dict[str, tuple[str, tuple[str, ...], bool, bool]] = {
    "idx_mcp_continuations_created": (
        "mcp_continuations", ("created_at", "continuation_id"), False, False
    ),
    "idx_mcp_continuations_owner_status": (
        "mcp_continuations", ("owner_id", "status", "created_at", "continuation_id"), False, False
    ),
    "idx_mcp_continuations_server_generation": (
        "mcp_continuations", ("server_id", "server_generation", "status", "created_at", "continuation_id"), False, False
    ),
    "idx_mcp_continuations_expiry": (
        "mcp_continuations", ("expires_at", "continuation_id"), False, False
    ),
    "idx_mcp_remote_tasks_created": (
        "mcp_remote_tasks", ("created_at", "task_ref"), False, False
    ),
    "idx_mcp_remote_tasks_owner_status": (
        "mcp_remote_tasks", ("owner_id", "status", "created_at", "task_ref"), False, False
    ),
    "idx_mcp_remote_tasks_server_generation": (
        "mcp_remote_tasks", ("server_id", "server_generation", "status", "created_at", "task_ref"), False, False
    ),
    "idx_mcp_remote_tasks_expiry": (
        "mcp_remote_tasks", ("expires_at", "task_ref"), False, False
    ),
    "idx_mcp_remote_tasks_remote_id": (
        "mcp_remote_tasks", ("server_id", "remote_id_sha256"), True, False
    ),
    "idx_mcp_subscriptions_created": (
        "mcp_subscriptions", ("created_at", "subscription_id"), False, False
    ),
    "idx_mcp_subscriptions_owner_status": (
        "mcp_subscriptions", ("owner_id", "status", "created_at", "subscription_id"), False, False
    ),
    "idx_mcp_subscriptions_server_generation": (
        "mcp_subscriptions", ("server_id", "server_generation", "status", "created_at", "subscription_id"), False, False
    ),
    "idx_mcp_auth_server_status": (
        "mcp_auth_metadata", ("server_id", "server_generation", "status", "profile_id"), False, False
    ),
    "idx_mcp_auth_expiry": (
        "mcp_auth_metadata", ("expires_at", "profile_id"), False, False
    ),
    "idx_mcp_side_effect_preparations_owner_status": (
        "mcp_side_effect_preparations",
        ("owner_id", "status", "created_at", "preparation_id"),
        False,
        False,
    ),
    "idx_mcp_side_effect_preparations_expiry": (
        "mcp_side_effect_preparations", ("expires_at", "preparation_id"), False, False
    ),
}


def _postgres_check(expression: str) -> str:
    match = re.fullmatch(r"([a-z_]+) IN \((.+)\)", expression)
    if match is not None:
        column, values = match.groups()
        selected: list[str] = []
        for raw in (item.strip() for item in values.split(",")):
            selected.append(f"{raw}::text" if raw.startswith("'") and raw.endswith("'") else raw)
        return f"CHECK ({column} = ANY (ARRAY[{', '.join(selected)}]))"
    selected_expression = re.sub(
        r"(=|<>) ('[^']*')",
        r"\1 \2::text",
        expression,
    )
    return f"CHECK ({selected_expression})"


V7_STORAGE_POSTGRES_CHECKS = {
    table: tuple(_postgres_check(expression) for expression in expressions)
    for table, expressions in V7_STORAGE_SQLITE_CHECKS.items()
}

V7_TABLES = frozenset(V7_STORAGE_COLUMN_CONTRACTS)


__all__ = [
    "V7_INDEX_CONTRACTS",
    "V7_STORAGE_COLUMN_CONTRACTS",
    "V7_STORAGE_FOREIGN_KEYS",
    "V7_STORAGE_KEY_CONSTRAINTS",
    "V7_STORAGE_POSTGRES_CHECKS",
    "V7_STORAGE_SQLITE_CHECKS",
    "V7_TABLES",
]
