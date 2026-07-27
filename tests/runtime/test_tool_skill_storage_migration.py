from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any

import pytest

from agent_libos import Runtime
import agent_libos.storage.tool_skill_migration as migration_module
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.storage import SQLiteStore
from agent_libos.storage.tool_skill_migration import (
    LEGACY_TOOL_GROUP_TOOLS,
    SKILL_LIFECYCLE_TOOLS,
    SKILL_PROJECTION_BOOTSTRAP,
    ToolSkillMigrationError,
    migrate_tool_groups_to_skills,
)
from agent_libos.tools.registry import stable_static_tool_id
from agent_libos.utils.serde import dumps, loads


def _tool_id(name: str) -> str:
    return stable_static_tool_id(
        name,
        digest_chars=DEFAULT_CONFIG.tools.static_tool_id_digest_chars,
    )


def _legacy_tool_ids() -> dict[str, str]:
    return {name: _tool_id(name) for name in LEGACY_TOOL_GROUP_TOOLS}


def _open_seeded_store(path: Path) -> tuple[SQLiteStore, str]:
    runtime = Runtime.open(path)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="migrate old tools")
    finally:
        runtime.close()
    store = SQLiteStore(path)
    _downgrade_builtin_image_and_process(store, pid)
    return store, pid


def _downgrade_builtin_image_and_process(store: SQLiteStore, pid: str) -> None:
    legacy_ids = _legacy_tool_ids()
    with store.transaction() as cursor:
        for name, tool_id in legacy_ids.items():
            cursor.execute(
                "INSERT INTO tools "
                "(tool_id, name, spec_json, scope, registered_by, created_at, ephemeral) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tool_id,
                    name,
                    dumps({"name": name, "description": "legacy Tool Group lifecycle"}),
                    "static",
                    "runtime.core",
                    "2026-01-01T00:00:00+00:00",
                    0,
                ),
            )
        process = _one(cursor, "SELECT * FROM processes WHERE pid = ?", (pid,))
        full = _legacy_projection(loads(process["tool_table_json"]), legacy_ids)
        model = _legacy_projection(loads(process["model_tool_table_json"]), legacy_ids)
        cursor.execute(
            "UPDATE processes SET tool_table_json = ?, model_tool_table_json = ? "
            "WHERE pid = ?",
            (dumps(full), dumps(model), pid),
        )
        _write_bindings(cursor, pid=pid, full=full, model=model)

        image = _one(
            cursor,
            "SELECT * FROM images WHERE image_id = ?",
            ("base-agent:v0",),
        )
        manifest = loads(image["manifest_json"])
        manifest["metadata"].pop("tool_projection", None)
        manifest["metadata"]["lazy_tool_groups"] = True
        manifest["metadata"]["initial_tool_groups"] = ["memory", "authority"]
        manifest["default_tools"] = _legacy_names(manifest["default_tools"])
        cursor.execute(
            "UPDATE images SET manifest_json = ? WHERE image_id = ?",
            (dumps(manifest), "base-agent:v0"),
        )


def _legacy_projection(
    mapping: dict[str, str],
    legacy_ids: dict[str, str],
) -> dict[str, str]:
    selected = {
        name: tool_id
        for name, tool_id in mapping.items()
        if name not in SKILL_LIFECYCLE_TOOLS
    }
    selected.update(legacy_ids)
    return selected


def _legacy_names(names: list[str]) -> list[str]:
    selected = [name for name in names if name not in SKILL_LIFECYCLE_TOOLS]
    selected.extend(LEGACY_TOOL_GROUP_TOOLS)
    return selected


def _as_legacy_core_skill_row(row: dict[str, Any]) -> dict[str, Any]:
    selected = deepcopy(row)
    selected["spec_json"] = dumps(
        {
            "name": selected["name"],
            "description": "pre-tool-skills lifecycle schema",
        }
    )
    selected["scope"] = "static"
    selected["registered_by"] = "runtime.core"
    selected["ephemeral"] = 0
    return selected


def _write_bindings(
    cursor: Any,
    *,
    pid: str,
    full: dict[str, str],
    model: dict[str, str],
) -> None:
    cursor.execute("DELETE FROM process_tool_bindings WHERE pid = ?", (pid,))
    cursor.executemany(
        "INSERT INTO process_tool_bindings "
        "(pid, binding_kind, tool_name, tool_id, jit_rehydration_eligible) "
        "VALUES (?, ?, ?, ?, 0)",
        [
            (pid, kind, name, tool_id)
            for kind, mapping in (("callable", full), ("model", model))
            for name, tool_id in mapping.items()
        ],
    )


def _downgrade_checkpoint_snapshot(store: SQLiteStore, checkpoint_id: str) -> None:
    legacy_ids = _legacy_tool_ids()
    with store.transaction() as cursor:
        row = _one(
            cursor,
            "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        snapshot = loads(row["snapshot_json"])
        for process in snapshot["rows"]["processes"]:
            process["tool_table_json"] = dumps(
                _legacy_projection(loads(process["tool_table_json"]), legacy_ids)
            )
            process["model_tool_table_json"] = dumps(
                _legacy_projection(loads(process["model_tool_table_json"]), legacy_ids)
            )
        legacy_discover = _as_legacy_core_skill_row(
            next(
                tool
                for tool in snapshot["rows"]["tools"]
                if tool["name"] == "discover_skills"
            )
        )
        snapshot["rows"]["tools"] = [
            tool
            for tool in snapshot["rows"]["tools"]
            if tool["name"] not in SKILL_PROJECTION_BOOTSTRAP
        ]
        snapshot["rows"]["tools"].append(legacy_discover)
        snapshot["rows"]["tools"].extend(
            {
                "tool_id": tool_id,
                "name": name,
                "spec_json": dumps({"name": name, "description": "legacy"}),
                "scope": "static",
                "registered_by": "runtime.core",
                "created_at": "2026-01-01T00:00:00+00:00",
                "ephemeral": 0,
            }
            for name, tool_id in legacy_ids.items()
        )
        for manifest in snapshot["images"].values():
            if manifest["image_id"] == "base-agent:v0":
                manifest["metadata"].pop("tool_projection", None)
                manifest["metadata"]["lazy_tool_groups"] = True
                manifest["default_tools"] = _legacy_names(manifest["default_tools"])
        cursor.execute(
            "UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?",
            (dumps(snapshot), checkpoint_id),
        )


def _checkpoint_pending_action(pid: str) -> dict[str, Any]:
    return {
        "pid": pid,
        "wait_type": "tool",
        "tool_name": None,
        "tool_call_id": None,
        "tool_operation_id": None,
        "filters_json": "{}",
        "action_json": dumps({"tool_name": "discover_tool_groups"}),
        "data_flow_context_json": "{}",
    }


def _insert_terminal_checkpoint_restore_reference(
    cursor: Any,
    *,
    publication_id: str,
    pid: str,
    checkpoint_id: str,
    snapshot_sha256: str,
    delivery_state: str | None,
    attempt_state: str | None,
) -> None:
    owner_instance_id = "offline-test"
    attempt_id = "completed-delivery-attempt" if delivery_state else None
    started_at = "2026-01-01T00:00:00+00:00" if attempt_id else None
    receipt: dict[str, Any] = {"phases": [], "artifacts": []}
    if delivery_state is not None:
        receipt["payload_delivery"] = {"state": delivery_state}
        receipt["payload_delivery_attempt"] = {
            "attempt_id": attempt_id,
            "started_at": started_at,
        }
    if attempt_state is not None:
        assert attempt_id is not None
        cursor.execute(
            "INSERT INTO checkpoint_payload_delivery_attempts "
            "(attempt_id, owner_instance_id, state, started_at, acked_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                attempt_id,
                owner_instance_id,
                attempt_state,
                started_at,
                "2026-01-01T00:00:01+00:00" if attempt_state == "acked" else None,
                "2026-01-01T00:00:01+00:00",
            ),
        )
    cursor.execute(
        "INSERT INTO runtime_publications "
        "(publication_id, kind, pid, owner_instance_id, state, phase, "
        "plan_json, receipt_json, operation_reconciled, payload_delivery_state, "
        "payload_delivery_attempt_id, payload_delivery_started_at, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            publication_id,
            "checkpoint_restore",
            pid,
            owner_instance_id,
            "committed",
            "reconciled",
            dumps(
                {
                    "checkpoint_id": checkpoint_id,
                    "snapshot_sha256": snapshot_sha256,
                }
            ),
            dumps(receipt),
            1,
            delivery_state,
            attempt_id,
            started_at,
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T00:00:00+00:00",
        ),
    )


def _downgrade_checkpoint_artifact(
    store: SQLiteStore,
    *,
    image_id: str,
    artifact_id: str,
    checkpoint_id: str,
) -> str:
    legacy_ids = _legacy_tool_ids()
    with store.transaction() as cursor:
        artifact_row = _one(
            cursor,
            "SELECT * FROM image_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        )
        artifact = loads(artifact_row["artifact_json"])
        process = artifact["source_process"]
        process["tool_table_json"] = dumps(
            _legacy_projection(loads(process["tool_table_json"]), legacy_ids)
        )
        process["model_tool_table_json"] = dumps(
            _legacy_projection(loads(process["model_tool_table_json"]), legacy_ids)
        )
        artifact["tool_table"] = _legacy_projection(artifact["tool_table"], legacy_ids)
        artifact["static_default_tools"] = _legacy_names(
            artifact["static_default_tools"]
        )
        legacy_discover = _as_legacy_core_skill_row(
            next(
                row
                for row in artifact["rows"]["tools"]
                if row["name"] == "discover_skills"
            )
        )
        artifact["rows"]["tools"] = [
            row
            for row in artifact["rows"]["tools"]
            if row["name"] not in SKILL_PROJECTION_BOOTSTRAP
        ]
        artifact["rows"]["tools"].append(legacy_discover)
        artifact["rows"]["tools"].extend(
            {
                "tool_id": tool_id,
                "name": name,
                "spec_json": dumps({"name": name, "description": "legacy"}),
                "scope": "static",
                "registered_by": "runtime.core",
                "created_at": "2026-01-01T00:00:00+00:00",
                "ephemeral": 0,
            }
            for name, tool_id in legacy_ids.items()
        )
        artifact["counts"]["tools"] = len(artifact["tool_table"])
        old_artifact_json = dumps(artifact)
        old_sha = hashlib.sha256(old_artifact_json.encode("utf-8")).hexdigest()
        cursor.execute(
            "UPDATE image_artifacts SET artifact_json = ?, sha256 = ? "
            "WHERE artifact_id = ?",
            (old_artifact_json, old_sha, artifact_id),
        )

        image_row = _one(
            cursor,
            "SELECT * FROM images WHERE image_id = ?",
            (image_id,),
        )
        manifest = loads(image_row["manifest_json"])
        manifest["metadata"].pop("tool_projection", None)
        manifest["metadata"]["lazy_tool_groups"] = True
        manifest["metadata"]["artifact_sha256"] = old_sha
        manifest["boot"]["artifact_sha256"] = old_sha
        manifest["default_tools"] = _legacy_names(manifest["default_tools"])
        cursor.execute(
            "UPDATE images SET manifest_json = ? WHERE image_id = ?",
            (dumps(manifest), image_id),
        )

        checkpoint_row = _one(
            cursor,
            "SELECT * FROM checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        snapshot = loads(checkpoint_row["snapshot_json"])
        for snapshot_process in snapshot["rows"]["processes"]:
            snapshot_process["tool_table_json"] = dumps(
                _legacy_projection(
                    loads(snapshot_process["tool_table_json"]),
                    legacy_ids,
                )
            )
            snapshot_process["model_tool_table_json"] = dumps(
                _legacy_projection(
                    loads(snapshot_process["model_tool_table_json"]),
                    legacy_ids,
                )
            )
        snapshot["rows"]["tools"] = [
            row
            for row in snapshot["rows"]["tools"]
            if row["name"] not in SKILL_PROJECTION_BOOTSTRAP
        ]
        snapshot["rows"]["tools"].extend(deepcopy(artifact["rows"]["tools"][-2:]))
        snapshot_manifest = snapshot["images"][image_id]
        snapshot_manifest["metadata"].pop("tool_projection", None)
        snapshot_manifest["metadata"]["lazy_tool_groups"] = True
        snapshot_manifest["metadata"]["artifact_sha256"] = old_sha
        snapshot_manifest["boot"]["artifact_sha256"] = old_sha
        snapshot_manifest["default_tools"] = _legacy_names(
            snapshot_manifest["default_tools"]
        )
        embedded = snapshot["image_artifacts"][artifact_id]
        embedded["artifact"] = deepcopy(artifact)
        embedded["sha256"] = old_sha
        cursor.execute(
            "UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?",
            (dumps(snapshot), checkpoint_id),
        )
    return old_sha


def _one(cursor: Any, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
    rows = [dict(row) for row in cursor.execute(sql, params)]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize(
    "raw",
    [
        '{"key": 1, "key": 2}',
        '{"value": NaN}',
        "[" * 257 + "0" + "]" * 257,
        "[" + ",".join("0" for _ in range(100_001)) + "]",
    ],
    ids=("duplicate-key", "non-finite", "depth", "nodes"),
)
def test_migration_json_boundary_rejects_ambiguous_or_excessive_input(
    raw: str,
) -> None:
    with pytest.raises(
        ToolSkillMigrationError,
        match="legacy.test contains malformed JSON",
    ):
        migration_module._json_value(raw, "legacy.test")


def test_migration_json_boundary_enforces_encoded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = '{"key": "\u00e9"}'
    encoded_bytes = len(raw.encode("utf-8"))
    monkeypatch.setattr(
        migration_module,
        "_LEGACY_JSON_HARD_LIMIT_BYTES",
        encoded_bytes,
    )

    assert migration_module._json_value(raw, "legacy.test") == {"key": "\u00e9"}

    monkeypatch.setattr(
        migration_module,
        "_LEGACY_JSON_HARD_LIMIT_BYTES",
        encoded_bytes - 1,
    )
    with pytest.raises(
        ToolSkillMigrationError,
        match="legacy.test contains malformed JSON",
    ):
        migration_module._json_value(raw, "legacy.test")


def test_migration_rejects_duplicate_persisted_json_without_writes(
    tmp_path: Path,
) -> None:
    store, pid = _open_seeded_store(tmp_path / "ambiguous-json.sqlite")
    ambiguous = (
        '{"discover_tool_groups": "first", '
        '"discover_tool_groups": "second"}'
    )
    try:
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE processes SET tool_table_json = ? WHERE pid = ?",
                (ambiguous, pid),
            )
        before = store.select_table_rows("processes", "pid = ?", (pid,))[0]

        with pytest.raises(ToolSkillMigrationError, match="malformed JSON"):
            migrate_tool_groups_to_skills(store, apply=True)

        assert store.select_table_rows("processes", "pid = ?", (pid,))[0] == before
    finally:
        store.close()


def test_migration_dry_run_apply_and_idempotence(tmp_path: Path) -> None:
    database = tmp_path / "tool-skill-migration.sqlite"
    store, pid = _open_seeded_store(database)
    try:
        audit_before = store.select_table_rows("audit_records", order_by="record_id")
        revision_before = int(
            store.select_table_rows("processes", "pid = ?", (pid,))[0]["revision"]
        )
        loaded_skills_before = store.select_table_rows(
            "processes", "pid = ?", (pid,)
        )[0]["loaded_skills_json"]
        dry_run = migrate_tool_groups_to_skills(store)

        assert dry_run.applied is False
        assert dry_run.changed is True
        unchanged = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        assert set(LEGACY_TOOL_GROUP_TOOLS) <= set(loads(unchanged["tool_table_json"]))

        applied = migrate_tool_groups_to_skills(store, apply=True)
        assert applied.applied is True
        assert applied.builtin_images_rewritten == 1
        assert applied.processes_migrated == 1
        assert applied.old_static_tools_deleted == 2

        process = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        full = loads(process["tool_table_json"])
        model = loads(process["model_tool_table_json"])
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(full)
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(model)
        assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(full)
        assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(model)
        assert int(process["revision"]) == revision_before + 1
        assert process["loaded_skills_json"] == loaded_skills_before
        counter = store.select_table_rows(
            "runtime_counters",
            "counter_name = ?",
            (f"process_revision:{pid}",),
        )
        assert len(counter) == 1
        assert int(counter[0]["value"]) == int(process["revision"])
        bindings = store.select_table_rows(
            "process_tool_bindings",
            "pid = ?",
            (pid,),
        )
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & {
            str(row["tool_name"]) for row in bindings
        }
        for old_id in _legacy_tool_ids().values():
            assert store.get_tool_spec(old_id) is None
        assert store.select_table_rows("audit_records", order_by="record_id") == audit_before

        second = migrate_tool_groups_to_skills(store, apply=True)
        assert second.changed is False
        assert second.to_dict()["artifact_id_remaps"] == {}
    finally:
        store.close()

    reopened = Runtime.open(database)
    try:
        with pytest.raises(Exception, match="tool not found|unknown tool"):
            reopened.tools.resolve("discover_tool_groups", pid=pid)
    finally:
        reopened.close()


def test_cli_dry_run_does_not_create_a_missing_sqlite_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "missing" / "runtime.sqlite"
    lease = database.with_suffix(database.suffix + ".runtime.lock")

    with pytest.raises(SystemExit) as caught:
        migration_module.cli([str(database)])

    assert caught.value.code == 2
    assert "requires an existing initialized Agent libOS store" in capsys.readouterr().err
    assert not database.parent.exists()
    assert not database.exists()
    assert not lease.exists()


def test_cli_dry_run_skips_schema_initialization_before_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "canonical.sqlite"
    store = SQLiteStore(database)
    store.close()

    def unexpected_initialize(_store: SQLiteStore) -> None:
        raise AssertionError("dry-run initialized the store before migration")

    monkeypatch.setattr(SQLiteStore, "initialize", unexpected_initialize)

    assert migration_module.cli([str(database)]) == 0
    output = loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["changed"] is False


def test_apply_rejects_an_active_outer_transaction(tmp_path: Path) -> None:
    database = tmp_path / "tool-skill-migration-nested-apply.sqlite"
    store, pid = _open_seeded_store(database)
    try:
        before = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        with store.transaction():
            with pytest.raises(
                ToolSkillMigrationError,
                match="inside an active store transaction",
            ):
                migrate_tool_groups_to_skills(store, apply=True)

        after = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        assert after["tool_table_json"] == before["tool_table_json"]
        assert after["model_tool_table_json"] == before["model_tool_table_json"]
        assert set(LEGACY_TOOL_GROUP_TOOLS) <= set(loads(after["tool_table_json"]))
    finally:
        store.close()


@pytest.mark.parametrize(
    "metadata",
    [
        {"lazy_tool_groups": "yes"},
        {"initial_tool_groups": ["memory"]},
        {"lazy_tool_groups": False, "initial_tool_groups": ["memory"]},
        {"lazy_tool_groups": True, "initial_tool_groups": ["memory", "memory"]},
        {"lazy_tool_groups": True, "initial_tool_groups": ["unknown"]},
        {"lazy_tool_groups": True, "tool_projection": "skills"},
    ],
)
def test_malformed_or_unknown_legacy_metadata_rolls_back_whole_store(
    tmp_path: Path,
    metadata: dict[str, Any],
) -> None:
    store, pid = _open_seeded_store(tmp_path / "malformed.sqlite")
    try:
        before = deepcopy(store.select_table_rows("processes", "pid = ?", (pid,))[0])
        with store.transaction() as cursor:
            row = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("base-agent:v0",),
            )
            manifest = loads(row["manifest_json"])
            manifest["metadata"] = metadata
            cursor.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (dumps(manifest), "base-agent:v0"),
            )

        with pytest.raises(ToolSkillMigrationError):
            migrate_tool_groups_to_skills(store, apply=True)

        assert store.select_table_rows("processes", "pid = ?", (pid,))[0] == before
        assert all(store.get_tool_spec(tool_id) is not None for tool_id in _legacy_tool_ids().values())
    finally:
        store.close()


def test_legacy_builtin_review_filesystem_read_group_migrates(
    tmp_path: Path,
) -> None:
    store, _pid = _open_seeded_store(tmp_path / "legacy-review.sqlite")
    try:
        with store.transaction() as cursor:
            row = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("review-agent:v0",),
            )
            manifest = loads(row["manifest_json"])
            manifest["metadata"].pop("tool_projection", None)
            manifest["metadata"]["lazy_tool_groups"] = True
            manifest["metadata"]["initial_tool_groups"] = ["filesystem_read"]
            manifest["default_tools"] = [
                "read_text_file",
                "read_directory",
                "create_object_from_file",
                *LEGACY_TOOL_GROUP_TOOLS,
            ]
            cursor.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (dumps(manifest), "review-agent:v0"),
            )

        report = migrate_tool_groups_to_skills(store, apply=True)

        assert report.builtin_images_rewritten == 2
        migrated = loads(
            store.select_table_rows(
                "images", "image_id = ?", ("review-agent:v0",)
            )[0]["manifest_json"]
        )
        assert migrated["metadata"]["tool_projection"] == "skills"
        assert "lazy_tool_groups" not in migrated["metadata"]
        assert "initial_tool_groups" not in migrated["metadata"]
    finally:
        store.close()


@pytest.mark.parametrize(
    "remaining",
    [[], ["discover_tool_groups"], ["activate_tool_group"]],
)
def test_lazy_legacy_image_requires_complete_lifecycle_pair(
    tmp_path: Path,
    remaining: list[str],
) -> None:
    store, _pid = _open_seeded_store(tmp_path / "partial-lifecycle.sqlite")
    try:
        with store.transaction() as cursor:
            row = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("base-agent:v0",),
            )
            manifest = loads(row["manifest_json"])
            manifest["default_tools"] = [
                name
                for name in manifest["default_tools"]
                if name not in LEGACY_TOOL_GROUP_TOOLS
            ] + remaining
            cursor.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (dumps(manifest), "base-agent:v0"),
            )

        with pytest.raises(ToolSkillMigrationError, match="both removed lifecycle"):
            migrate_tool_groups_to_skills(store, apply=True)
    finally:
        store.close()


def test_custom_lazy_image_falls_back_to_full_projection(tmp_path: Path) -> None:
    store, _pid = _open_seeded_store(tmp_path / "custom.sqlite")
    try:
        with store.transaction() as cursor:
            source = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("coding-agent:v0",),
            )
            manifest = loads(source["manifest_json"])
            manifest["image_id"] = "custom-legacy:v1"
            manifest["name"] = "custom-legacy"
            manifest["metadata"] = {
                "owner": "user",
                "lazy_tool_groups": True,
                "initial_tool_groups": ["filesystem"],
            }
            manifest["default_tools"] = _legacy_names(manifest["default_tools"])
            cursor.execute(
                "INSERT INTO images "
                "(image_id, manifest_json, registered_by, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    manifest["image_id"],
                    dumps(manifest),
                    "user",
                    "test",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        report = migrate_tool_groups_to_skills(store, apply=True)
        assert report.custom_images_migrated == 1
        assert any("full tool projection" in warning for warning in report.warnings)
        migrated = loads(
            store.select_table_rows(
                "images",
                "image_id = ?",
                ("custom-legacy:v1",),
            )[0]["manifest_json"]
        )
        assert "lazy_tool_groups" not in migrated["metadata"]
        assert "initial_tool_groups" not in migrated["metadata"]
        assert "tool_projection" not in migrated["metadata"]
        assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(migrated["default_tools"])
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(migrated["default_tools"])
    finally:
        store.close()


@pytest.mark.parametrize(
    ("lazy", "legacy_tools", "expected_added"),
    [
        (False, [], set()),
        (True, list(LEGACY_TOOL_GROUP_TOOLS), set(SKILL_LIFECYCLE_TOOLS)),
    ],
)
def test_custom_migration_does_not_add_process_exit_or_unrelated_authority(
    tmp_path: Path,
    lazy: bool,
    legacy_tools: list[str],
    expected_added: set[str],
) -> None:
    store, _pid = _open_seeded_store(tmp_path / f"authority-{lazy}.sqlite")
    try:
        with store.transaction() as cursor:
            source = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("coding-agent:v0",),
            )
            manifest = loads(source["manifest_json"])
            manifest["image_id"] = f"custom-authority-{lazy}:v1"
            manifest["name"] = "custom-authority"
            manifest["metadata"] = {"lazy_tool_groups": lazy}
            manifest["default_tools"] = ["compact_process_context", *legacy_tools]
            cursor.execute(
                "INSERT INTO images "
                "(image_id, manifest_json, registered_by, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    manifest["image_id"],
                    dumps(manifest),
                    "user",
                    "test",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        migrate_tool_groups_to_skills(store, apply=True)
        migrated = loads(
            store.select_table_rows(
                "images",
                "image_id = ?",
                (manifest["image_id"],),
            )[0]["manifest_json"]
        )
        assert set(migrated["default_tools"]) == {
            "compact_process_context",
            *expected_added,
        }
        assert "process_exit" not in migrated["default_tools"]
    finally:
        store.close()


def test_checkpoint_full_and_model_tables_are_migrated_without_revising_safe_point(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="checkpoint migration")
        checkpoint_id = runtime.checkpoint.create(pid, "legacy checkpoint", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        before = loads(
            store.select_table_rows(
                "checkpoints",
                "checkpoint_id = ?",
                (checkpoint_id,),
            )[0]["snapshot_json"]
        )["rows"]["processes"][0]
        safe_point = {
            "revision": before["revision"],
            "updated_at": before["updated_at"],
            "state_generation": before["state_generation"],
            "execution_generation": before["execution_generation"],
        }

        report = migrate_tool_groups_to_skills(store, apply=True)

        assert report.checkpoints_migrated == 1
        snapshot = loads(
            (checkpoint_row := store.select_table_rows(
                "checkpoints",
                "checkpoint_id = ?",
                (checkpoint_id,),
            )[0])["snapshot_json"]
        )
        assert loads(checkpoint_row["metadata_json"])["snapshot_bytes"] == len(
            dumps(snapshot).encode("utf-8")
        )
        process = snapshot["rows"]["processes"][0]
        assert {
            "revision": process["revision"],
            "updated_at": process["updated_at"],
            "state_generation": process["state_generation"],
            "execution_generation": process["execution_generation"],
        } == safe_point
        for field in ("tool_table_json", "model_tool_table_json"):
            mapping = loads(process[field])
            assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(mapping)
            assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(mapping)
        tool_rows = {row["name"]: row for row in snapshot["rows"]["tools"]}
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(tool_rows)
        assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(tool_rows)
        assert tool_rows["discover_skills"]["scope"] == "module:agent-libos-core:v0"
        assert loads(tool_rows["discover_skills"]["spec_json"])["description"] != (
            "pre-tool-skills lifecycle schema"
        )
    finally:
        store.close()


def test_checkpoint_pending_action_with_removed_tool_aborts_atomically(
    tmp_path: Path,
) -> None:
    database = tmp_path / "checkpoint-pending.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="pending checkpoint")
        checkpoint_id = runtime.checkpoint.create(pid, "pending", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        with store.transaction() as cursor:
            row = _one(
                cursor,
                "SELECT snapshot_json FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,),
            )
            snapshot = loads(row["snapshot_json"])
            snapshot["rows"]["llm_pending_actions"].append(
                _checkpoint_pending_action(pid)
            )
            cursor.execute(
                "UPDATE checkpoints SET snapshot_json = ? WHERE checkpoint_id = ?",
                (dumps(snapshot), checkpoint_id),
            )
        before = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]

        with pytest.raises(
            ToolSkillMigrationError,
            match=r"rows\.llm_pending_actions\[0\] references a removed tool",
        ):
            migrate_tool_groups_to_skills(store, apply=True)

        assert store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0] == before
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


def test_checkpoint_pending_action_postcondition_rejects_reintroduced_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "checkpoint-postcondition.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="postcondition")
        checkpoint_id = runtime.checkpoint.create(pid, "postcondition", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        original = migration_module._migrate_checkpoint_snapshot

        def reintroduce_legacy_action(*args: Any, **kwargs: Any) -> dict[str, Any]:
            migrated = original(*args, **kwargs)
            migrated["rows"]["llm_pending_actions"].append(
                _checkpoint_pending_action(pid)
            )
            return migrated

        monkeypatch.setattr(
            migration_module,
            "_migrate_checkpoint_snapshot",
            reintroduce_legacy_action,
        )

        with pytest.raises(
            ToolSkillMigrationError,
            match=r"postcondition failed: .*rows\.llm_pending_actions\[0\]",
        ):
            migrate_tool_groups_to_skills(store, apply=True)
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


@pytest.mark.parametrize("pending_kind", ["llm", "publication"])
def test_pending_activity_with_removed_tool_reference_aborts_atomically(
    tmp_path: Path,
    pending_kind: str,
) -> None:
    store, pid = _open_seeded_store(tmp_path / f"pending-{pending_kind}.sqlite")
    try:
        before = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        with store.transaction() as cursor:
            if pending_kind == "llm":
                cursor.execute(
                    "INSERT INTO llm_pending_actions "
                    "(pid, wait_type, tool_name, filters_json, action_json, "
                    "data_flow_context_json, content_preview, tool_call_count, status, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        pid,
                        "tool",
                        None,
                        "{}",
                        dumps({"tool_name": "activate_tool_group"}),
                        "{}",
                        "legacy",
                        1,
                        "pending",
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )
            else:
                cursor.execute(
                    "INSERT INTO runtime_publications "
                    "(publication_id, kind, pid, owner_instance_id, state, phase, "
                    "plan_json, receipt_json, operation_reconciled, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "publication-legacy",
                        "process_launch",
                        pid,
                        "offline-test",
                        "planning",
                        "planning",
                        dumps({"before": {"tool": "discover_tool_groups"}}),
                        "{}",
                        0,
                        "2026-01-01T00:00:00+00:00",
                        "2026-01-01T00:00:00+00:00",
                    ),
                )

        with pytest.raises(ToolSkillMigrationError, match="pending LLM|unresolved runtime"):
            migrate_tool_groups_to_skills(store, apply=True)
        assert store.select_table_rows("processes", "pid = ?", (pid,))[0] == before
        assert all(store.get_tool_spec(tool_id) is not None for tool_id in _legacy_tool_ids().values())
    finally:
        store.close()


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        pytest.param(False, False, id="bool-false"),
        pytest.param(True, True, id="bool-true"),
        pytest.param(0, False, id="integer-zero"),
        pytest.param(1, True, id="integer-one"),
    ],
)
def test_persisted_boolean_accepts_only_normal_backend_boolean_values(
    stored: object,
    expected: bool,
) -> None:
    assert (
        migration_module._persisted_boolean(stored, path="test.operation_reconciled")
        is expected
    )


@pytest.mark.parametrize(
    "stored",
    [
        pytest.param("false", id="string-false"),
        pytest.param("0", id="string-zero"),
        pytest.param("1", id="string-one"),
        pytest.param(0.0, id="float-zero"),
        pytest.param(1.0, id="float-one"),
        pytest.param(2, id="integer-two"),
        pytest.param(None, id="none"),
    ],
)
def test_persisted_boolean_rejects_truthy_and_falsey_type_confusion(
    stored: object,
) -> None:
    with pytest.raises(ToolSkillMigrationError, match="must be stored as 0 or 1"):
        migration_module._persisted_boolean(
            stored,
            path="test.operation_reconciled",
        )


@pytest.mark.parametrize("apply", [False, True], ids=["dry-run", "apply"])
def test_malformed_publication_boolean_aborts_before_any_migration_write(
    tmp_path: Path,
    apply: bool,
) -> None:
    database = tmp_path / f"malformed-publication-boolean-{apply}.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="strict migration bool")
        checkpoint_id = runtime.checkpoint.create(pid, "strict migration bool", actor=pid)
    finally:
        runtime.close()

    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        checkpoint_before = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]
        snapshot_sha256 = hashlib.sha256(
            checkpoint_before["snapshot_json"].encode("utf-8")
        ).hexdigest()
        publication_id = f"malformed-publication-boolean-{apply}"
        with store.transaction() as cursor:
            cursor.execute("PRAGMA ignore_check_constraints = ON")
            cursor.execute(
                "INSERT INTO runtime_publications "
                "(publication_id, kind, pid, owner_instance_id, state, phase, "
                "plan_json, receipt_json, operation_reconciled, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publication_id,
                    "checkpoint_restore",
                    pid,
                    "offline-test",
                    "committed",
                    "reconciled",
                    dumps(
                        {
                            "checkpoint_id": checkpoint_id,
                            "snapshot_sha256": snapshot_sha256,
                        }
                    ),
                    dumps({"phases": [], "artifacts": []}),
                    "false",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )
            cursor.execute("PRAGMA ignore_check_constraints = OFF")

        process_before = store.select_table_rows(
            "processes", "pid = ?", (pid,)
        )[0]
        publication_before = store.select_table_rows(
            "runtime_publications", "publication_id = ?", (publication_id,)
        )[0]
        tools_before = store.select_table_rows("tools", order_by="tool_id")

        with pytest.raises(
            ToolSkillMigrationError,
            match=r"operation_reconciled must be stored as 0 or 1",
        ):
            migrate_tool_groups_to_skills(store, apply=apply)

        assert store.select_table_rows(
            "processes", "pid = ?", (pid,)
        )[0] == process_before
        assert store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0] == checkpoint_before
        assert store.select_table_rows(
            "runtime_publications", "publication_id = ?", (publication_id,)
        )[0] == publication_before
        assert store.select_table_rows("tools", order_by="tool_id") == tools_before
    finally:
        store.close()


def test_incomplete_checkpoint_payload_receipt_with_removed_tool_aborts(
    tmp_path: Path,
) -> None:
    store, pid = _open_seeded_store(tmp_path / "pending-delivery.sqlite")
    try:
        with store.transaction() as cursor:
            cursor.execute(
                "INSERT INTO runtime_publications "
                "(publication_id, kind, pid, owner_instance_id, state, phase, "
                "plan_json, receipt_json, operation_reconciled, payload_delivery_state, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "checkpoint-delivery-legacy",
                    "checkpoint_restore",
                    pid,
                    "offline-test",
                    "committed",
                    "reconciled",
                    "{}",
                    dumps({"before": {"tool": "discover_tool_groups"}}),
                    1,
                    "pending",
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        with pytest.raises(ToolSkillMigrationError, match="incomplete checkpoint"):
            migrate_tool_groups_to_skills(store, apply=True)
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("state", "operation_reconciled", "payload_delivery_state"),
    [
        pytest.param("reconciliation_pending", False, None, id="reconciliation-pending"),
        pytest.param("failed", True, None, id="failed"),
        pytest.param("manual", True, None, id="manual"),
        pytest.param("committed", False, "completed", id="operation-unreconciled"),
    ],
)
def test_unresolved_checkpoint_restore_plan_blocks_snapshot_migration(
    tmp_path: Path,
    state: str,
    operation_reconciled: bool,
    payload_delivery_state: str | None,
) -> None:
    database = tmp_path / f"restore-plan-{state}-{payload_delivery_state}.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="restore plan")
        checkpoint_id = runtime.checkpoint.create(pid, "restore plan", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        checkpoint_before = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]
        snapshot_sha256 = hashlib.sha256(
            checkpoint_before["snapshot_json"].encode("utf-8")
        ).hexdigest()
        publication_id = f"restore-publication-{state}-{payload_delivery_state}"
        delivery_attempt_id = (
            "completed-delivery-attempt"
            if payload_delivery_state == "completed"
            else None
        )
        delivery_started_at = (
            "2026-01-01T00:00:00+00:00"
            if delivery_attempt_id is not None
            else None
        )
        with store.transaction() as cursor:
            cursor.execute(
                "INSERT INTO runtime_publications "
                "(publication_id, kind, pid, owner_instance_id, state, phase, "
                "plan_json, receipt_json, operation_reconciled, "
                "payload_delivery_state, payload_delivery_attempt_id, "
                "payload_delivery_started_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    publication_id,
                    "checkpoint_restore",
                    pid,
                    "offline-test",
                    state,
                    "reconciled" if state == "committed" else "recovery",
                    dumps(
                        {
                            "checkpoint_id": checkpoint_id,
                            "snapshot_sha256": snapshot_sha256,
                        }
                    ),
                    dumps({"phases": [], "artifacts": []}),
                    int(operation_reconciled),
                    payload_delivery_state,
                    delivery_attempt_id,
                    delivery_started_at,
                    "2026-01-01T00:00:00+00:00",
                    "2026-01-01T00:00:00+00:00",
                ),
            )

        with pytest.raises(
            ToolSkillMigrationError,
            match="unresolved checkpoint restore references a checkpoint snapshot",
        ):
            migrate_tool_groups_to_skills(store, apply=True)

        assert store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0] == checkpoint_before
        publication = store.select_table_rows(
            "runtime_publications", "publication_id = ?", (publication_id,)
        )[0]
        assert loads(publication["plan_json"])["snapshot_sha256"] == snapshot_sha256
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


@pytest.mark.parametrize("attempt_state", ["preparing", "aborted", None])
def test_completed_checkpoint_delivery_requires_exact_acked_attempt(
    tmp_path: Path,
    attempt_state: str | None,
) -> None:
    database = tmp_path / f"restore-delivery-attempt-{attempt_state}.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="delivery attempt")
        checkpoint_id = runtime.checkpoint.create(pid, "delivery attempt", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        checkpoint_before = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]
        snapshot_sha256 = hashlib.sha256(
            checkpoint_before["snapshot_json"].encode("utf-8")
        ).hexdigest()
        with store.transaction() as cursor:
            _insert_terminal_checkpoint_restore_reference(
                cursor,
                publication_id=f"restore-delivery-{attempt_state}",
                pid=pid,
                checkpoint_id=checkpoint_id,
                snapshot_sha256=snapshot_sha256,
                delivery_state="completed",
                attempt_state=attempt_state,
            )

        with pytest.raises(
            ToolSkillMigrationError,
            match="incomplete checkpoint payload delivery",
        ):
            migrate_tool_groups_to_skills(store, apply=True)

        assert store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0] == checkpoint_before
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("delivery_state", "attempt_state"),
    [
        pytest.param(None, None, id="no-delivery"),
        pytest.param("completed", "acked", id="acked-delivery"),
    ],
)
def test_resolved_checkpoint_restore_allows_snapshot_migration(
    tmp_path: Path,
    delivery_state: str | None,
    attempt_state: str | None,
) -> None:
    database = tmp_path / f"resolved-restore-{delivery_state}.sqlite"
    runtime = Runtime.open(database)
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="resolved restore")
        checkpoint_id = runtime.checkpoint.create(pid, "resolved restore", actor=pid)
    finally:
        runtime.close()
    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        _downgrade_checkpoint_snapshot(store, checkpoint_id)
        checkpoint_before = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]
        snapshot_sha256 = hashlib.sha256(
            checkpoint_before["snapshot_json"].encode("utf-8")
        ).hexdigest()
        with store.transaction() as cursor:
            _insert_terminal_checkpoint_restore_reference(
                cursor,
                publication_id=f"resolved-restore-{delivery_state}",
                pid=pid,
                checkpoint_id=checkpoint_id,
                snapshot_sha256=snapshot_sha256,
                delivery_state=delivery_state,
                attempt_state=attempt_state,
            )

        report = migrate_tool_groups_to_skills(store, apply=True)

        assert report.checkpoints_migrated == 1
        checkpoint_after = store.select_table_rows(
            "checkpoints", "checkpoint_id = ?", (checkpoint_id,)
        )[0]
        assert checkpoint_after["snapshot_json"] != checkpoint_before["snapshot_json"]
        publication = store.select_table_rows(
            "runtime_publications",
            "publication_id = ?",
            (f"resolved-restore-{delivery_state}",),
        )[0]
        assert loads(publication["plan_json"])["snapshot_sha256"] == snapshot_sha256
    finally:
        store.close()


@pytest.mark.parametrize(
    "forgery",
    [
        {"scope": "external"},
        {"registered_by": "attacker"},
    ],
)
def test_forged_existing_skill_lifecycle_row_rolls_back(
    tmp_path: Path,
    forgery: dict[str, str],
) -> None:
    store, pid = _open_seeded_store(tmp_path / "forged-bootstrap.sqlite")
    try:
        before = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        assignments = ", ".join(f"{column} = ?" for column in forgery)
        with store.transaction() as cursor:
            cursor.execute(
                f"UPDATE tools SET {assignments} WHERE name = ?",
                (*forgery.values(), "discover_skills"),
            )

        with pytest.raises(ToolSkillMigrationError, match="provenance"):
            migrate_tool_groups_to_skills(store, apply=True)
        assert store.select_table_rows("processes", "pid = ?", (pid,))[0] == before
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


@pytest.mark.parametrize(
    "forgery",
    [
        {"scope": "external"},
        {"registered_by": "attacker"},
    ],
)
def test_forged_legacy_lifecycle_row_rolls_back(
    tmp_path: Path,
    forgery: dict[str, str],
) -> None:
    store, pid = _open_seeded_store(tmp_path / "forged-legacy.sqlite")
    try:
        before = store.select_table_rows("processes", "pid = ?", (pid,))[0]
        assignments = ", ".join(f"{column} = ?" for column in forgery)
        with store.transaction() as cursor:
            cursor.execute(
                f"UPDATE tools SET {assignments} WHERE name = ?",
                (*forgery.values(), "discover_tool_groups"),
            )

        with pytest.raises(ToolSkillMigrationError, match="malformed"):
            migrate_tool_groups_to_skills(store, apply=True)
        assert store.select_table_rows("processes", "pid = ?", (pid,))[0] == before
        assert all(
            store.get_tool_spec(tool_id) is not None
            for tool_id in _legacy_tool_ids().values()
        )
    finally:
        store.close()


def test_old_core_skill_lifecycle_spec_is_canonicalized(tmp_path: Path) -> None:
    store, _pid = _open_seeded_store(tmp_path / "old-bootstrap-spec.sqlite")
    try:
        with store.transaction() as cursor:
            cursor.execute(
                "UPDATE tools SET spec_json = ?, scope = ?, registered_by = ? "
                "WHERE name = ?",
                (
                    dumps(
                        {
                            "name": "discover_skills",
                            "description": "pre-tool-skills lifecycle schema",
                        }
                    ),
                    "static",
                    "runtime.core",
                    "discover_skills",
                ),
            )

        report = migrate_tool_groups_to_skills(store, apply=True)

        assert report.static_tools_canonicalized == 1
        row = store.select_table_rows(
            "tools", "name = ?", ("discover_skills",)
        )[0]
        assert row["scope"] == "module:agent-libos-core:v0"
        assert row["registered_by"] == "module:agent-libos-core:v0"
        assert loads(row["spec_json"])["description"] != (
            "pre-tool-skills lifecycle schema"
        )
    finally:
        store.close()


def test_raw_image_package_artifact_is_unchanged_and_reported(
    tmp_path: Path,
) -> None:
    database = tmp_path / "image-package.sqlite"
    runtime = Runtime.open(database)
    try:
        registered = runtime.image_registry.register_from_package_files(
            {
                "IMAGE.yaml": """
image_id: legacy-package:v0
name: legacy-package
version: v0
prompt: prompt.md
default_tools:
  - human_output
metadata:
  owner: user
workspace:
  source: workspace
  working_directory: .
  grants: []
""".lstrip(),
                "prompt.md": "Package image prompt.\n",
                "workspace/seed.txt": "seed\n",
            },
            actor="test",
        )
        artifact_id = str(registered.image.boot["artifact_id"])
        pid = runtime.process.spawn(image="base-agent:v0", goal="migrate package")
    finally:
        runtime.close()

    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, pid)
        with store.transaction() as cursor:
            row = _one(
                cursor,
                "SELECT * FROM images WHERE image_id = ?",
                ("legacy-package:v0",),
            )
            manifest = loads(row["manifest_json"])
            manifest["metadata"]["lazy_tool_groups"] = True
            manifest["default_tools"] = [
                "human_output",
                *LEGACY_TOOL_GROUP_TOOLS,
            ]
            cursor.execute(
                "UPDATE images SET manifest_json = ? WHERE image_id = ?",
                (dumps(manifest), "legacy-package:v0"),
            )
        artifact_before = store.select_table_rows(
            "image_artifacts", "artifact_id = ?", (artifact_id,)
        )[0]

        report = migrate_tool_groups_to_skills(store, apply=True)

        assert any(
            "immutable image_package raw files" in warning
            for warning in report.warnings
        )
        assert store.select_table_rows(
            "image_artifacts", "artifact_id = ?", (artifact_id,)
        )[0] == artifact_before
        migrated = loads(
            store.select_table_rows(
                "images", "image_id = ?", ("legacy-package:v0",)
            )[0]["manifest_json"]
        )
        assert migrated["boot"]["artifact_id"] == artifact_id
        assert "lazy_tool_groups" not in migrated["metadata"]
        assert "tool_projection" not in migrated["metadata"]
        assert set(SKILL_LIFECYCLE_TOOLS) <= set(migrated["default_tools"])
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(migrated["default_tools"])
    finally:
        store.close()


def test_checkpoint_commit_artifact_is_republished_and_references_are_cascaded(
    tmp_path: Path,
) -> None:
    database = tmp_path / "artifact.sqlite"
    runtime = Runtime.open(database)
    try:
        source = runtime.process.spawn(image="base-agent:v0", goal="artifact source")
        source_checkpoint = runtime.checkpoint.create(source, "source", actor=source)
        runtime.image_registry.grant_register(source, "legacy-artifact:v0", issued_by="test")
        committed = runtime.image_registry.commit_from_checkpoint(
            actor=source,
            checkpoint_id=source_checkpoint,
            image_id="legacy-artifact:v0",
            name="legacy-artifact",
        )
        old_artifact_id = str(committed.image.boot["artifact_id"])
        booted = runtime.process.spawn(
            image="legacy-artifact:v0",
            goal="checkpoint artifact consumer",
        )
        consumer_checkpoint = runtime.checkpoint.create(
            booted,
            "consumer",
            actor=booted,
        )
    finally:
        runtime.close()

    store = SQLiteStore(database)
    try:
        _downgrade_builtin_image_and_process(store, source)
        old_sha = _downgrade_checkpoint_artifact(
            store,
            image_id="legacy-artifact:v0",
            artifact_id=old_artifact_id,
            checkpoint_id=consumer_checkpoint,
        )

        dry_run = migrate_tool_groups_to_skills(store)
        dry_run_artifact_id = dry_run.artifact_id_remaps[old_artifact_id]
        report = migrate_tool_groups_to_skills(store, apply=True)

        assert report.checkpoint_artifacts_created == 1
        new_artifact_id = report.artifact_id_remaps[old_artifact_id]
        assert new_artifact_id == dry_run_artifact_id
        assert new_artifact_id != old_artifact_id
        old_row = store.get_image_artifact(old_artifact_id)
        new_row = store.get_image_artifact(new_artifact_id)
        assert old_row is not None and new_row is not None
        assert old_row[1]["sha256"] == old_sha
        assert new_row[1]["sha256"] != old_sha
        artifact = new_row[0]
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & set(artifact["tool_table"])
        assert set(SKILL_PROJECTION_BOOTSTRAP) <= set(artifact["tool_table"])
        assert artifact["counts"]["tools"] == len(artifact["tool_table"])
        assert not set(LEGACY_TOOL_GROUP_TOOLS) & {
            row["name"] for row in artifact["rows"]["tools"]
        }
        artifact_tool_rows = {
            row["name"]: row for row in artifact["rows"]["tools"]
        }
        assert artifact_tool_rows["discover_skills"]["scope"] == (
            "module:agent-libos-core:v0"
        )
        assert loads(artifact_tool_rows["discover_skills"]["spec_json"])[
            "description"
        ] != "pre-tool-skills lifecycle schema"

        image = loads(
            store.select_table_rows(
                "images", "image_id = ?", ("legacy-artifact:v0",)
            )[0]["manifest_json"]
        )
        assert image["boot"]["artifact_id"] == new_artifact_id
        assert image["boot"]["artifact_sha256"] == new_row[1]["sha256"]
        snapshot = loads(
            store.select_table_rows(
                "checkpoints",
                "checkpoint_id = ?",
                (consumer_checkpoint,),
            )[0]["snapshot_json"]
        )
        assert snapshot["images"]["legacy-artifact:v0"]["boot"]["artifact_id"] == new_artifact_id
        assert old_artifact_id not in snapshot["image_artifacts"]
        assert new_artifact_id in snapshot["image_artifacts"]

        assert migrate_tool_groups_to_skills(store, apply=True).changed is False
    finally:
        store.close()

    reopened = Runtime.open(database)
    try:
        pid = reopened.process.spawn(
            image="legacy-artifact:v0",
            goal="boot migrated artifact",
        )
        assert reopened.process.get(pid).image_id == "legacy-artifact:v0"
    finally:
        reopened.close()
