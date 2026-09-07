"""Canonical schema-v8 Host-private Responses replay storage contract."""

from agent_libos.storage.v5_schema_contract import V5ColumnContract as C

V8_STORAGE_COLUMN_CONTRACTS = {
    "llm_replay_turns": (
        ("turn_id", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("pid", C("text", False, keyset_collation=True)),
        ("run_id", C("text", True, keyset_collation=True)),
        ("provider_fingerprint", C("text", False)),
        ("model", C("text", False)),
        ("context_generation", C("text", False)),
        ("payload_json", C("text", True)),
        ("source_labels_json", C("text", False)),
        ("payload_sha256", C("text", False)),
        ("payload_bytes", C("bigint", False)),
        ("created_at", C("text", False, keyset_collation=True)),
        ("purged_at", C("text", True)),
    ),
    "llm_replay_heads": (
        ("pid", C("text", False, primary_key_position=1, keyset_collation=True)),
        ("turn_id", C("text", False, keyset_collation=True)),
        ("revision", C("bigint", False)),
        ("updated_at", C("text", False)),
    ),
}
V8_STORAGE_KEY_CONSTRAINTS = {
    table: (("primary_key", (columns[0][0],)),)
    for table, columns in V8_STORAGE_COLUMN_CONTRACTS.items()
}
V8_STORAGE_SQLITE_CHECKS = {
    "llm_replay_turns": (
        "length(payload_sha256) = 64",
        "payload_bytes >= 0",
        "(payload_json IS NULL) = (purged_at IS NOT NULL)",
    ),
    "llm_replay_heads": ("revision > 0",),
}
V8_STORAGE_POSTGRES_CHECKS = {
    table: tuple(f"CHECK ({expression})" for expression in expressions)
    for table, expressions in V8_STORAGE_SQLITE_CHECKS.items()
}
V8_INDEX_CONTRACTS = {
    "idx_llm_pending_replay_recovery": ("llm_pending_actions", ("status", "pid"), False, False),
    "idx_llm_replay_turns_pid": ("llm_replay_turns", ("pid", "created_at", "turn_id"), False, False),
    "idx_llm_replay_turns_run": ("llm_replay_turns", ("run_id", "turn_id"), False, False),
}
V8_TABLES = frozenset(V8_STORAGE_COLUMN_CONTRACTS)

V8_STORAGE_FOREIGN_KEYS = {table: () for table in V8_TABLES}
