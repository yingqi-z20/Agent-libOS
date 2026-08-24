from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import pytest

from agent_libos.models import (
    Checkpoint,
    ChildProcessWait,
    ExitedProcessOutcome,
    process_outcome_from_json,
    process_outcome_to_mapping,
    process_wait_state_from_json,
    process_wait_state_to_mapping,
)
from agent_libos.models.exceptions import ValidationError
from agent_libos.runtime.snapshots import (
    SNAPSHOT_SCHEMA_VERSION,
    SnapshotCodec,
    SnapshotCoordinator,
    SnapshotIdentityMap,
    SnapshotRemapper,
    SnapshotRows,
    SnapshotVersionError,
)
from agent_libos.utils.serde import dumps


def _row(table: str, **values: object) -> dict[str, object]:
    return {**{column: None for column in SnapshotRows.ROW_COLUMNS[table]}, **values}


def _capability_row(**values: object) -> dict[str, object]:
    row = _row(
        "capabilities",
        cap_id="cap_1",
        subject="pid_1",
        resource="test:snapshot",
        rights_json=dumps(["read"]),
        constraints_json=dumps({}),
        issued_by="test",
        issued_at="2040-01-01T00:00:00+00:00",
        expires_at=None,
        delegable=False,
        revocable=True,
        effect="allow",
        issuer_cap_id=None,
        parent_cap_id=None,
        delegation_depth=0,
        max_delegation_depth=None,
        uses_remaining=None,
        status="active",
        metadata_json=dumps({}),
    )
    row.update(values)
    return row


def _snapshot() -> dict:
    return {
        "version": SNAPSHOT_SCHEMA_VERSION,
        "checkpoint_id": "ckpt_1",
        "pid": "pid_1",
        "reason": "test",
        "created_at": "2040-01-01T00:00:00Z",
        "created_by": "test",
        "subtree_pids": ["pid_1"],
        "object_oids": ["obj_1"],
        "owned_object_oids": ["obj_1"],
        "referenced_object_oids": ["obj_1"],
        "referenced_object_types": {"obj_1": "TEXT"},
        "namespaces": ["proc:pid_1"],
        "owned_namespaces": ["proc:pid_1"],
        "rows": {
            "processes": [
                _row(
                    "processes",
                    pid="pid_1",
                    status="runnable",
                    goal_oid="obj_1",
                    wait_state_json=dumps(None),
                    outcome_json=dumps(None),
                    state_generation=0,
                )
            ],
            "object_namespaces": [_row("object_namespaces", namespace="proc:pid_1")],
            "objects": [_row("objects", oid="obj_1", namespace="proc:pid_1")],
            "object_links": [],
            "capabilities": [_capability_row()],
            "process_resource_reservations": [],
            "process_messages": [],
            "llm_pending_actions": [],
            "skills": [],
            "tools": [_row("tools", tool_id="tool_1")],
            "tool_candidates": [],
        },
        "object_payloads": {"obj_1": {"text": "hello"}},
        "images": {},
        "image_artifacts": {},
        "jit_sources": {"tool_1": "export default {}"},
        "modules": [{"module_id": "core", "source_sha256": "a" * 64}],
    }


def test_snapshot_codec_round_trip_is_strict_and_lossless() -> None:
    snapshot = SnapshotCodec.decode_mapping(_snapshot())
    encoded = SnapshotCodec.dumps(snapshot)
    assert SnapshotCodec.loads(encoded) == snapshot
    assert SnapshotCodec.encode_mapping(snapshot) == _snapshot()


@pytest.mark.parametrize(
    "replacement",
    [
        "NaN",
        "[" * 257 + "0" + "]" * 257,
        "[" + ",".join("0" for _ in range(100_001)) + "]",
    ],
    ids=("non-finite", "excessive-depth", "excessive-nodes"),
)
def test_snapshot_codec_loads_rejects_nonstandard_or_excessive_json(
    replacement: str,
) -> None:
    encoded = SnapshotCodec.dumps(SnapshotCodec.decode_mapping(_snapshot()))
    invalid = encoded.replace(
        '{"text": "hello"}',
        replacement,
        1,
    )

    with pytest.raises(ValidationError, match="invalid snapshot document JSON"):
        SnapshotCodec.loads(invalid)


def test_snapshot_codec_loads_rejects_duplicate_object_keys() -> None:
    encoded = SnapshotCodec.dumps(SnapshotCodec.decode_mapping(_snapshot()))
    invalid = encoded.replace(
        '"reason": "test"',
        '"reason": "test", "reason": "ambiguous"',
        1,
    )

    with pytest.raises(ValidationError, match="invalid snapshot document JSON"):
        SnapshotCodec.loads(invalid)


def test_snapshot_codec_loads_enforces_encoded_byte_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = SnapshotCodec.dumps(SnapshotCodec.decode_mapping(_snapshot()))
    monkeypatch.setattr(
        SnapshotCodec,
        "max_document_bytes",
        len(encoded.encode("utf-8")) - 1,
    )

    with pytest.raises(ValidationError, match="max_bytes"):
        SnapshotCodec.loads(encoded)


def test_checkpoint_model_uses_the_codec_protocol_version() -> None:
    checkpoint = Checkpoint(
        checkpoint_id="ckpt_1",
        pid="pid_1",
        reason="test",
        created_at="2040-01-01T00:00:00Z",
    )

    assert checkpoint.snapshot_version == SnapshotCodec.schema_version


def test_snapshot_codec_rejects_old_versions_and_unknown_tables() -> None:
    old = _snapshot()
    old["version"] = 1
    with pytest.raises(SnapshotVersionError):
        SnapshotCodec.decode_mapping(old)

    unknown = _snapshot()
    unknown["rows"]["arbitrary_table"] = []
    with pytest.raises(ValidationError, match="unsupported row tables"):
        SnapshotCodec.decode_mapping(unknown)

    incomplete = _snapshot()
    incomplete["rows"]["processes"][0].pop("status")
    with pytest.raises(ValidationError, match="not canonical"):
        SnapshotCodec.decode_mapping(incomplete)


def test_snapshot_codec_rejects_missing_tables_and_process_rows() -> None:
    missing_table = _snapshot()
    missing_table["rows"].pop("processes")
    with pytest.raises(ValidationError, match="missing tables"):
        SnapshotCodec.decode_mapping(missing_table)

    missing_process = _snapshot()
    missing_process["rows"]["processes"] = []
    with pytest.raises(ValidationError, match="exactly match subtree_pids"):
        SnapshotCodec.decode_mapping(missing_process)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda module: module.pop("module_id"), "module_id"),
        (lambda module: module.pop("source_sha256"), "source_sha256"),
        (
            lambda module: module.__setitem__("source_sha256", "not-a-sha256"),
            "64-character SHA-256",
        ),
    ],
)
def test_snapshot_codec_rejects_incomplete_or_malformed_module_requirements(
    mutation,
    message: str,
) -> None:
    snapshot = _snapshot()
    mutation(snapshot["modules"][0])

    with pytest.raises(ValidationError, match=message):
        SnapshotCodec.decode_mapping(snapshot)


def test_snapshot_codec_rejects_duplicate_module_requirements() -> None:
    snapshot = _snapshot()
    snapshot["modules"].append(dict(snapshot["modules"][0]))

    with pytest.raises(ValidationError, match="duplicate module_id"):
        SnapshotCodec.decode_mapping(snapshot)


@pytest.mark.parametrize(
    ("status", "wait_state_json", "outcome_json", "state_generation"),
    [
        ("waiting_event", dumps(None), dumps(None), 1),
        ("exited", dumps(None), dumps(None), 1),
        ("runnable", dumps(None), dumps(None), -1),
    ],
)
def test_snapshot_codec_rejects_invalid_typed_process_product_before_restore(
    status: str,
    wait_state_json: str,
    outcome_json: str,
    state_generation: int,
) -> None:
    invalid = _snapshot()
    invalid["rows"]["processes"][0].update(
        {
            "status": status,
            "wait_state_json": wait_state_json,
            "outcome_json": outcome_json,
            "state_generation": state_generation,
        }
    )
    with pytest.raises(ValidationError, match="invalid snapshot rows.processes"):
        SnapshotCodec.decode_mapping(invalid)


def test_snapshot_codec_rejects_sql_null_typed_process_state_fields() -> None:
    for field_name in ("wait_state_json", "outcome_json"):
        invalid = _snapshot()
        invalid["rows"]["processes"][0][field_name] = None

        with pytest.raises(
            ValidationError,
            match=f"{field_name} must be canonical JSON text",
        ):
            SnapshotCodec.decode_mapping(invalid)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("delegable", 0),
        ("revocable", "true"),
        ("revocable", 1),
        ("delegation_depth", False),
        ("delegation_depth", "0"),
        ("delegation_depth", 0.0),
        ("delegation_depth", -1),
        ("max_delegation_depth", True),
        ("max_delegation_depth", "1"),
        ("max_delegation_depth", 1.0),
        ("max_delegation_depth", -1),
        ("uses_remaining", False),
        ("uses_remaining", "1"),
        ("uses_remaining", 1.0),
        ("uses_remaining", -1),
        ("rights_json", dumps({"read": True})),
        ("rights_json", dumps([1])),
        ("constraints_json", dumps([])),
        ("metadata_json", dumps([])),
        ("effect", 1),
        ("effect", "permit"),
        ("status", False),
        ("status", "enabled"),
        ("cap_id", 1),
        ("subject", False),
        ("resource", None),
        ("issued_by", 1.0),
        ("issued_at", None),
        ("expires_at", 1),
        ("issuer_cap_id", False),
        ("parent_cap_id", 1),
    ],
)
def test_snapshot_rows_reject_malformed_capability_types_directly(
    field_name: str,
    value: object,
) -> None:
    row = _capability_row(**{field_name: value})

    with pytest.raises(
        ValidationError,
        match=rf"rows\.capabilities\[0\].{field_name}",
    ):
        SnapshotRows(capabilities=(row,))


def test_snapshot_rows_reject_string_boolean_directly() -> None:
    row = _capability_row(delegable="false")

    with pytest.raises(
        ValidationError,
        match=r"rows\.capabilities\[0\]\.delegable must be a boolean",
    ):
        SnapshotRows(capabilities=(row,))


def test_snapshot_rows_reject_active_exhausted_capability_directly() -> None:
    row = _capability_row(
        effect="deny",
        uses_remaining=0,
        status="active",
    )

    with pytest.raises(
        ValidationError,
        match="active capability uses_remaining must be positive",
    ):
        SnapshotRows(capabilities=(row,))


@pytest.mark.parametrize("status", ["revoked", "disabled", "exec_revoked"])
def test_snapshot_rows_preserve_exhausted_nonactive_capability_history(
    status: str,
) -> None:
    row = _capability_row(
        effect="deny",
        uses_remaining=0,
        status=status,
    )

    rows = SnapshotRows(capabilities=(row,))

    assert rows.capabilities[0]["uses_remaining"] == 0
    assert rows.capabilities[0]["status"] == status


def test_snapshot_rows_distinguish_serialized_booleans_from_trusted_sqlite_rows() -> None:
    rows = _snapshot()["rows"]
    rows["capabilities"][0]["delegable"] = 0
    rows["capabilities"][0]["revocable"] = 1

    with pytest.raises(ValidationError, match="delegable must be a boolean"):
        SnapshotRows.from_mapping(rows)

    durable = SnapshotRows.from_trusted_durable_mapping(rows)

    assert durable.capabilities[0]["delegable"] is False
    assert durable.capabilities[0]["revocable"] is True


def test_snapshot_remapper_updates_typed_identity_fields() -> None:
    snapshot = SnapshotCodec.decode_mapping(_snapshot())
    remapped = SnapshotRemapper.remap(
        snapshot,
        SnapshotIdentityMap(
            pids={"pid_1": "pid_2"},
            objects={"obj_1": "obj_2"},
            namespaces={"proc:pid_1": "proc:pid_2"},
            capabilities={"cap_1": "cap_2"},
            tools={"tool_1": "tool_2"},
        ),
    )
    encoded = SnapshotCodec.encode_mapping(remapped)
    assert encoded["pid"] == "pid_2"
    assert encoded["rows"]["processes"][0]["goal_oid"] == "obj_2"
    assert encoded["rows"]["capabilities"][0]["cap_id"] == "cap_2"
    assert encoded["rows"]["capabilities"][0]["subject"] == "pid_2"
    assert encoded["object_payloads"] == {"obj_2": {"text": "hello"}}
    assert encoded["jit_sources"] == {"tool_2": "export default {}"}


def test_snapshot_remapper_updates_nested_identity_carriers_without_global_replacement() -> None:
    raw = _snapshot()
    raw["rows"]["processes"][0].update(
        {
            "capabilities_json": dumps(["cap_1"]),
            "tool_table_json": dumps({"echo": "tool_1"}),
            "model_tool_table_json": dumps({"echo": "tool_1"}),
            "loaded_skills_json": dumps(
                {
                    "skill_1": {
                        "tool_ids": {"echo": "tool_1"},
                        "jit_tool_ids": {"jit": "tool_1"},
                        "base_tool_ids": {"echo": "tool_1"},
                        "base_model_tool_ids": {"echo": "tool_1"},
                    }
                }
            ),
            "memory_view_json": dumps(
                {
                    "view_id": "view_1",
                    "owner_pid": "pid_1",
                    "roots": [
                        {
                            "oid": "obj_1",
                            "rights": ["read"],
                            "capability_id": "cap_1",
                        }
                    ],
                    "created_from": None,
                }
            ),
        }
    )
    raw["rows"]["objects"][0].update(
        {
            "owner_kind": "process",
            "owner_id": "pid_1",
            "provenance_json": dumps(
                {
                    "source_refs": [],
                    "created_from_action": "test",
                    "parent_oids": ["obj_1"],
                    "source_operation_ids": [],
                }
            ),
        }
    )
    raw["rows"]["capabilities"][0]["resource"] = "object:obj_1"
    raw["rows"]["process_messages"] = [
        _row(
            "process_messages",
            message_id="message_1",
            sender="pid_1",
            recipient_pid="pid_1",
            metadata_json=dumps({"label_carrier_oid": "obj_1"}),
        )
    ]
    raw["rows"]["tool_candidates"] = [
        _row(
            "tool_candidates",
            candidate_id="candidate_1",
            pid="pid_1",
            registered_tool_id="tool_1",
            requested_capabilities_json=dumps(
                [{"resource": "object:obj_1", "rights": ["read"]}]
            ),
        )
    ]
    raw["object_payloads"]["obj_1"]["literal"] = (
        "obj_1 pid_1 cap_1 tool_1 must remain ordinary text"
    )
    before = deepcopy(raw)
    snapshot = SnapshotCodec.decode_mapping(raw)

    remapped = SnapshotRemapper.remap(
        snapshot,
        SnapshotIdentityMap(
            pids={"pid_1": "pid_2"},
            objects={"obj_1": "obj_2"},
            namespaces={"proc:pid_1": "proc:pid_2"},
            capabilities={"cap_1": "cap_2"},
            tools={"tool_1": "tool_2"},
            candidates={"candidate_1": "candidate_2"},
        ),
    )
    encoded = SnapshotCodec.encode_mapping(remapped)
    process = encoded["rows"]["processes"][0]
    obj = encoded["rows"]["objects"][0]
    capability = encoded["rows"]["capabilities"][0]
    memory_view = SnapshotRemapper._json_container(
        process["memory_view_json"],
        field_name="test.memory_view_json",
        expected_type=dict,
    )
    loaded_skills = SnapshotRemapper._json_container(
        process["loaded_skills_json"],
        field_name="test.loaded_skills_json",
        expected_type=dict,
    )

    assert SnapshotRemapper._json_container(
        process["capabilities_json"],
        field_name="test.capabilities_json",
        expected_type=list,
    ) == ["cap_2"]
    assert SnapshotRemapper._json_container(
        process["tool_table_json"],
        field_name="test.tool_table_json",
        expected_type=dict,
    ) == {"echo": "tool_2"}
    assert loaded_skills["skill_1"]["jit_tool_ids"] == {"jit": "tool_2"}
    assert memory_view["owner_pid"] == "pid_2"
    assert memory_view["roots"][0]["oid"] == "obj_2"
    assert memory_view["roots"][0]["capability_id"] == "cap_2"
    assert obj["owner_id"] == "pid_2"
    assert SnapshotRemapper._json_container(
        obj["provenance_json"],
        field_name="test.provenance_json",
        expected_type=dict,
    )["parent_oids"] == ["obj_2"]
    assert capability["resource"] == "object:obj_2"
    assert SnapshotRemapper._json_container(
        encoded["rows"]["process_messages"][0]["metadata_json"],
        field_name="test.metadata_json",
        expected_type=dict,
    )["label_carrier_oid"] == "obj_2"
    assert SnapshotRemapper._json_container(
        encoded["rows"]["tool_candidates"][0][
            "requested_capabilities_json"
        ],
        field_name="test.requested_capabilities_json",
        expected_type=list,
    )[0]["resource"] == "object:obj_2"
    assert encoded["object_payloads"]["obj_2"]["literal"] == (
        "obj_1 pid_1 cap_1 tool_1 must remain ordinary text"
    )
    assert raw == before


@pytest.mark.parametrize(
    ("resource", "expected"),
    [
        ("object:obj_1", "object:obj_2"),
        ("object:obj_1/*", "object:obj_2/*"),
        ("object:obj_1:*", "object:obj_2:*"),
    ],
    ids=("exact", "subtree", "prefix"),
)
def test_snapshot_remapper_preserves_object_resource_scope(
    resource: str,
    expected: str,
) -> None:
    raw = _snapshot()
    raw["rows"]["capabilities"][0]["resource"] = resource
    raw["rows"]["tool_candidates"] = [
        _row(
            "tool_candidates",
            candidate_id="candidate_1",
            pid="pid_1",
            requested_capabilities_json=dumps(
                [{"resource": resource, "rights": ["read"]}]
            ),
        )
    ]

    remapped = SnapshotRemapper.remap(
        SnapshotCodec.decode_mapping(raw),
        SnapshotIdentityMap(objects={"obj_1": "obj_2"}),
    )
    encoded = SnapshotCodec.encode_mapping(remapped)
    requested = SnapshotRemapper._json_container(
        encoded["rows"]["tool_candidates"][0][
            "requested_capabilities_json"
        ],
        field_name="test.requested_capabilities_json",
        expected_type=list,
    )

    assert encoded["rows"]["capabilities"][0]["resource"] == expected
    assert requested[0]["resource"] == expected


@pytest.mark.parametrize(
    "carrier",
    ["capability", "tool_candidate"],
)
def test_snapshot_remapper_rejects_malformed_capability_resources(
    carrier: str,
) -> None:
    raw = _snapshot()
    malformed = "object:obj_1/*/escape"
    if carrier == "capability":
        raw["rows"]["capabilities"][0]["resource"] = malformed
    else:
        raw["rows"]["tool_candidates"] = [
            _row(
                "tool_candidates",
                candidate_id="candidate_1",
                pid="pid_1",
                requested_capabilities_json=dumps(
                    [{"resource": malformed, "rights": ["read"]}]
                ),
            )
        ]

    with pytest.raises(ValidationError, match="valid capability resource"):
        SnapshotRemapper.remap(
            SnapshotCodec.decode_mapping(raw),
            SnapshotIdentityMap(objects={"obj_1": "obj_2"}),
        )


@pytest.mark.parametrize("domain", ["pids", "objects", "capabilities", "tools"])
def test_snapshot_remapper_rejects_partial_map_targeting_unchanged_identity(
    domain: str,
) -> None:
    raw = _snapshot()
    if domain == "pids":
        raw["subtree_pids"].append("pid_2")
        second = deepcopy(raw["rows"]["processes"][0])
        second["pid"] = "pid_2"
        raw["rows"]["processes"].append(second)
        identities = SnapshotIdentityMap(pids={"pid_1": "pid_2"})
    elif domain == "objects":
        raw["object_oids"].append("obj_2")
        raw["owned_object_oids"].append("obj_2")
        raw["rows"]["objects"].append(
            _row("objects", oid="obj_2", namespace="proc:pid_1")
        )
        raw["object_payloads"]["obj_2"] = {"text": "second"}
        identities = SnapshotIdentityMap(objects={"obj_1": "obj_2"})
    elif domain == "capabilities":
        raw["rows"]["capabilities"].append(
            _capability_row(cap_id="cap_2")
        )
        identities = SnapshotIdentityMap(
            capabilities={"cap_1": "cap_2"}
        )
    else:
        raw["rows"]["tools"].append(_row("tools", tool_id="tool_2"))
        identities = SnapshotIdentityMap(tools={"tool_1": "tool_2"})

    snapshot = SnapshotCodec.decode_mapping(raw)

    with pytest.raises(ValidationError, match="collides with an unchanged"):
        SnapshotRemapper.remap(snapshot, identities)


def test_snapshot_remapper_supports_injective_identity_swap_without_payload_loss() -> None:
    raw = _snapshot()
    raw["object_oids"].append("obj_2")
    raw["owned_object_oids"].append("obj_2")
    raw["rows"]["objects"].append(
        _row("objects", oid="obj_2", namespace="proc:pid_1")
    )
    raw["object_payloads"]["obj_2"] = {"text": "second"}
    snapshot = SnapshotCodec.decode_mapping(raw)

    remapped = SnapshotRemapper.remap(
        snapshot,
        SnapshotIdentityMap(objects={"obj_1": "obj_2", "obj_2": "obj_1"}),
    )

    assert remapped.object_oids == ("obj_2", "obj_1")
    assert remapped.object_payloads == {
        "obj_2": {"text": "hello"},
        "obj_1": {"text": "second"},
    }


def test_snapshot_remapper_postflight_rejects_a_stale_nested_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _snapshot()
    raw["rows"]["processes"][0]["capabilities_json"] = dumps(["cap_1"])
    snapshot = SnapshotCodec.decode_mapping(raw)
    monkeypatch.setattr(
        SnapshotRemapper,
        "_remap_process_carriers",
        classmethod(lambda cls, row, identities: None),
    )

    with pytest.raises(
        ValidationError,
        match=r"rows\.processes\[0\]\.capabilities_json",
    ):
        SnapshotRemapper.remap(
            snapshot,
            SnapshotIdentityMap(capabilities={"cap_1": "cap_2"}),
        )


def test_snapshot_row_remapper_updates_nested_process_state_identities() -> None:
    identities = SnapshotIdentityMap(
        pids={"pid_1": "pid_2"},
        objects={"obj_1": "obj_2"},
    )
    child_wait = SnapshotRemapper.remap_row(
        _row(
            "processes",
            pid="pid_1",
            status="waiting_event",
            status_message="waiting for pid_1",
            wait_state_json=dumps(
                process_wait_state_to_mapping(ChildProcessWait(child_pid="pid_1"))
            ),
            outcome_json=dumps(None),
            state_generation=7,
        ),
        identities,
    )
    terminal = SnapshotRemapper.remap_row(
        _row(
            "processes",
            pid="pid_1",
            status="exited",
            status_message="result_oid:obj_1",
            wait_state_json=dumps(None),
            outcome_json=dumps(
                process_outcome_to_mapping(ExitedProcessOutcome(result_oid="obj_1"))
            ),
            state_generation=8,
        ),
        identities,
    )

    assert process_wait_state_from_json(child_wait["wait_state_json"]) == (
        ChildProcessWait(child_pid="pid_2")
    )
    assert child_wait["status_message"] == "waiting for pid_2"
    assert process_outcome_from_json(terminal["outcome_json"]) == (
        ExitedProcessOutcome(result_oid="obj_2")
    )
    assert terminal["status_message"] == "result_oid:obj_2"


def test_snapshot_identity_maps_must_be_one_to_one() -> None:
    with pytest.raises(ValidationError, match="one-to-one"):
        SnapshotIdentityMap(pids={"pid_1": "pid_3", "pid_2": "pid_3"})


class _CoordinatorStore:
    def __init__(self) -> None:
        self.calls: list[str] = []

    @contextmanager
    def transaction(self, *, include_object_payloads: bool = False):
        assert include_object_payloads is True
        self.calls.append("transaction.enter")
        try:
            yield self
        finally:
            self.calls.append("transaction.exit")

    @contextmanager
    def locked(self):
        self.calls.append("store.enter")
        try:
            yield
        finally:
            self.calls.append("store.exit")


def test_snapshot_coordinator_rolls_back_its_transaction_after_publish_failure() -> None:
    store = _CoordinatorStore()
    coordinator = SnapshotCoordinator(store)

    def fail_publish(snapshot, prepared):
        assert snapshot.header.checkpoint_id == "ckpt_1"
        assert prepared == "prepared"
        store.calls.append("publish")
        raise RuntimeError("injected publish failure")

    with pytest.raises(RuntimeError, match="publish failure"):
        coordinator.atomic_publish(
            _snapshot(),
            prepare=lambda _snapshot: store.calls.append("prepare") or "prepared",
            publish=fail_publish,
        )

    assert store.calls == [
        "transaction.enter",
        "prepare",
        "publish",
        "transaction.exit",
    ]


def test_snapshot_coordinator_exposes_canonical_restore_lock_order() -> None:
    store = _CoordinatorStore()
    coordinator = SnapshotCoordinator(store)

    def scope(name: str):
        @contextmanager
        def selected():
            store.calls.append(f"{name}.enter")
            try:
                yield
            finally:
                store.calls.append(f"{name}.exit")

        return selected

    with coordinator.restore_runtime_scope(scope("runtime")):
        with coordinator.restore_registry_scope(scope("registry")):
            with coordinator.restore_atomic_scope(scope("ownership")):
                store.calls.append("publish")
            store.calls.append("registry.finalize")
        store.calls.append("host.finalize")

    assert store.calls == [
        "runtime.enter",
        "registry.enter",
        "ownership.enter",
        "store.enter",
        "publish",
        "store.exit",
        "ownership.exit",
        "registry.finalize",
        "registry.exit",
        "host.finalize",
        "runtime.exit",
    ]
