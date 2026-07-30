from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos import AgentImage, Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.evidence.initial_goal_recovery import (
    initial_goal_object_identity,
    initial_goal_object_identity_sha256,
)
from agent_libos.llm.client import LLMCompletion
from agent_libos.models import (
    CapabilityRight,
    ObjectLifecycleState,
    ObjectType,
    PROMPT_MODE_IMAGE_ONLY,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    ProcessError,
    RuntimePublicationPending,
    ValidationError,
)


class _ExitClient:
    def __init__(self) -> None:
        self.calls = 0
        self.message_batches: list[list[dict[str, Any]]] = []

    def complete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMCompletion:
        del tools
        self.calls += 1
        self.message_batches.append(json.loads(json.dumps(messages)))
        return LLMCompletion(
            content="",
            tool_calls=[
                {
                    "id": "initial_goal_exit",
                    "name": "process_exit",
                    "arguments": '{"payload":{"done":true}}',
                }
            ],
            raw=SimpleNamespace(id="initial_goal_raw"),
            api="chat",
            model="fake",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def _raw_launch_receipt(runtime: Runtime, pid: str) -> tuple[str, dict[str, Any]]:
    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "pid = ? AND kind = ?",
        (pid, "process_launch"),
    )
    assert len(rows) == 1
    return str(rows[0]["publication_id"]), json.loads(rows[0]["receipt_json"])


def _goal_envelope(receipt: dict[str, Any]) -> dict[str, Any]:
    phases = [
        phase
        for phase in receipt["phases"]
        if phase.get("phase") == "goal_created"
    ]
    assert len(phases) == 1
    return phases[0]["initial_goal_recovery"]


@pytest.mark.parametrize(
    ("goal", "expected"),
    [
        ("reopen exact string goal", {"text": "reopen exact string goal"}),
        (
            {"task": "reopen exact structure", "nested": [1, {"ok": True}]},
            {"task": "reopen exact structure", "nested": [1, {"ok": True}]},
        ),
    ],
)
def test_root_spawn_reopen_recovers_only_exact_initial_goal(
    tmp_path: Path,
    goal: Any,
    expected: dict[str, Any],
) -> None:
    database = tmp_path / "root-goal.sqlite"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal=goal)
    goal_oid = runtime.process.get(pid).goal_oid
    assert goal_oid is not None
    ordinary = runtime.memory.create_object(
        pid,
        ObjectType.OBSERVATION,
        {"ordinary": "must remain volatile"},
    )
    publication_id, _receipt = _raw_launch_receipt(runtime, pid)
    assert "reopen exact" not in repr(
        runtime.store.get_runtime_publication(publication_id)
    )
    runtime.close()

    reopened = Runtime.open(database)
    try:
        restored = reopened.store.get_object(goal_oid)
        assert restored is not None
        assert restored.payload == expected
        assert reopened.recovered_root_spawn_initial_goal_payloads == (goal_oid,)
        assert reopened.store.get_object(ordinary.oid) is None
        ordinary_state = reopened.store.get_persisted_object_state(ordinary.oid)
        assert ordinary_state is not None
        assert ordinary_state.lifecycle_state is ObjectLifecycleState.RELEASED
    finally:
        reopened.close()


def test_image_only_first_quantum_receives_goal_after_spawn_reopen(
    tmp_path: Path,
) -> None:
    database = tmp_path / "image-only-root-goal.sqlite"
    runtime = Runtime.open(database)
    runtime.register_image(
        AgentImage(
            image_id="initial-goal-image-only:v0",
            name="initial-goal-image-only",
            system_prompt="Exact first-quantum prompt.",
            prompt_mode=PROMPT_MODE_IMAGE_ONLY,
            default_tools=["process_exit"],
        ),
        actor="test",
    )
    pid = runtime.process.spawn(
        image="initial-goal-image-only:v0",
        goal="goal survives before first quantum",
    )
    runtime.close()

    reopened = Runtime.open(database)
    try:
        client = _ExitClient()
        reopened.llm.client = client
        result = reopened.run_process_once(pid)
        assert result["ok"], result
        assert client.message_batches[0] == [
            {"role": "system", "content": "Exact first-quantum prompt."},
            {"role": "user", "content": "goal survives before first quantum"},
        ]
    finally:
        reopened.close()


def test_root_exec_image_change_still_recovers_original_spawn_goal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "root-exec-goal.sqlite"
    runtime = Runtime.open(database)
    runtime.register_image(
        AgentImage(
            image_id="initial-goal-after-exec:v0",
            name="initial-goal-after-exec",
            system_prompt="Changed image.",
        ),
        actor="test",
    )
    pid = runtime.process.spawn(goal="original root goal")
    goal_oid = runtime.process.get(pid).goal_oid
    runtime.capability.grant(
        pid,
        "image:initial-goal-after-exec:v0",
        [CapabilityRight.READ],
        issued_by="test",
    )
    runtime.exec_process(pid, "initial-goal-after-exec:v0")
    assert runtime.process.get(pid).goal_oid == goal_oid
    runtime.close()

    reopened = Runtime.open(database)
    try:
        restored = reopened.store.get_object(goal_oid)
        assert restored is not None
        assert restored.payload == {"text": "original root goal"}
    finally:
        reopened.close()


def test_persist_full_io_false_never_writes_or_recovers_goal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "redacted-root-goal.sqlite"
    secret = "never-persist-this-goal-7f3c"
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
    )
    runtime = Runtime.open(database, config=config)
    pid = runtime.process.spawn(goal=secret)
    goal_oid = runtime.process.get(pid).goal_oid
    _, receipt = _raw_launch_receipt(runtime, pid)
    assert secret not in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "hash_only"
    runtime.close()
    assert secret.encode() not in database.read_bytes()

    reopened = Runtime.open(database, config=config)
    try:
        assert reopened.store.get_object(goal_oid) is None
        client = _ExitClient()
        reopened.llm.client = client
        with pytest.raises(
            ValidationError,
            match="goal payload is unavailable before the first LLM quantum",
        ):
            reopened.run_process_once(pid)
        assert client.calls == 0
    finally:
        reopened.close()


def test_reopen_with_full_io_disabled_redacts_existing_recovery_payload(
    tmp_path: Path,
) -> None:
    database = tmp_path / "policy-change-root-goal.sqlite"
    secret = "redact-on-policy-change-193a"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal=secret)
    goal_oid = runtime.process.get(pid).goal_oid
    runtime.close()

    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, persist_full_io=False),
    )
    reopened = Runtime.open(database, config=config)
    try:
        assert reopened.store.get_object(goal_oid) is None
        _, receipt = _raw_launch_receipt(reopened, pid)
        assert secret not in json.dumps(receipt)
        assert _goal_envelope(receipt)["retention"] == "hash_only"
    finally:
        reopened.close()


def test_tampered_goal_recovery_fails_closed_without_rewriting_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "tampered-root-goal.sqlite"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal={"task": "original"})
    goal_oid = runtime.process.get(pid).goal_oid
    publication_id, receipt = _raw_launch_receipt(runtime, pid)
    runtime.close()

    envelope = _goal_envelope(receipt)
    envelope["payload"] = {"task": "tampered"}
    tampered_receipt = json.dumps(receipt, sort_keys=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE runtime_publications SET receipt_json = ? WHERE publication_id = ?",
            (tampered_receipt, publication_id),
        )
        payload_marker = connection.execute(
            "SELECT payload_json FROM objects WHERE oid = ?",
            (goal_oid,),
        ).fetchone()[0]

    with pytest.raises(ValidationError, match="payload digest changed"):
        Runtime.open(database)

    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT receipt_json FROM runtime_publications WHERE publication_id = ?",
            (publication_id,),
        ).fetchone()[0] == tampered_receipt
        assert connection.execute(
            "SELECT payload_json FROM objects WHERE oid = ?",
            (goal_oid,),
        ).fetchone()[0] == payload_marker


def test_rehydrate_rejects_crafted_mutable_goal_even_with_matching_identity(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mutable-root-goal-row.sqlite"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal="crafted mutable root goal")
    goal_oid = runtime.process.get(pid).goal_oid
    assert goal_oid is not None
    publication_id, receipt = _raw_launch_receipt(runtime, pid)
    runtime.close()

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT oid, namespace, name, type, schema_version, metadata_json, "
            "provenance_json, version, created_by, owner_kind, owner_id, created_at "
            "FROM objects WHERE oid = ?",
            (goal_oid,),
        ).fetchone()
        assert row is not None
        mutable_identity = initial_goal_object_identity(
            oid=str(row["oid"]),
            namespace=str(row["namespace"]),
            name=str(row["name"]),
            object_type=str(row["type"]),
            schema_version=str(row["schema_version"]),
            metadata=json.loads(row["metadata_json"]),
            provenance=json.loads(row["provenance_json"]),
            version=int(row["version"]),
            immutable=False,
            created_by=str(row["created_by"]),
            owner_kind=str(row["owner_kind"]),
            owner_id=str(row["owner_id"]),
            created_at=str(row["created_at"]),
        )
        _goal_envelope(receipt)["object_identity_sha256"] = (
            initial_goal_object_identity_sha256(mutable_identity)
        )
        connection.execute(
            "UPDATE objects SET immutable = 0 WHERE oid = ?",
            (goal_oid,),
        )
        connection.execute(
            "UPDATE runtime_publications SET receipt_json = ? "
            "WHERE publication_id = ?",
            (json.dumps(receipt, sort_keys=True), publication_id),
        )

    with pytest.raises(ValidationError, match="goal Object state changed"):
        Runtime.open(database)


def test_terminal_root_process_redacts_and_cannot_rehydrate_goal(
    tmp_path: Path,
) -> None:
    database = tmp_path / "terminal-root-goal.sqlite"
    secret = "terminal-goal-redaction-c84d"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(goal=secret)
    goal_oid = runtime.process.get(pid).goal_oid
    _, before = _raw_launch_receipt(runtime, pid)
    assert secret in json.dumps(before)

    runtime.process.exit(pid)

    _, after = _raw_launch_receipt(runtime, pid)
    envelope = _goal_envelope(after)
    assert secret not in json.dumps(after)
    assert envelope["retention"] == "hash_only"
    assert "payload" not in envelope
    assert runtime.store.get_persisted_object_state(
        goal_oid
    ).lifecycle_state is ObjectLifecycleState.RELEASED
    runtime.close()

    reopened = Runtime.open(database)
    try:
        assert reopened.store.get_object(goal_oid) is None
        assert goal_oid not in reopened.recovered_root_spawn_initial_goal_payloads
    finally:
        reopened.close()


def test_late_root_spawn_hook_failure_redacts_recovery_envelope(
    tmp_path: Path,
) -> None:
    database = tmp_path / "late-spawn-failure.sqlite"
    secret = "late-hook-failure-goal-41e2"
    runtime = Runtime.open(database)

    def fail_after_goal(_pid: str, _image_id: str, _publication_id: str) -> None:
        raise RuntimeError("late spawn hook failed")

    runtime.process.add_after_spawn_hook(fail_after_goal)
    with pytest.raises(RuntimeError, match="late spawn hook failed"):
        runtime.process.spawn(goal=secret)

    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "kind = ?",
        ("process_launch",),
    )
    assert len(rows) == 1
    assert rows[0]["state"] == "rolled_back"
    receipt = json.loads(rows[0]["receipt_json"])
    assert secret not in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "hash_only"
    assert runtime.store.get_process(str(rows[0]["pid"])) is None
    runtime.close()


def test_failed_launch_redaction_failure_rolls_back_noncommittable_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "atomic-online-spawn-redaction.sqlite"
    secret = "atomic-online-redaction-goal-b540"
    runtime = Runtime.open(database)

    def fail_after_goal(_pid: str, _image_id: str, _publication_id: str) -> None:
        raise RuntimeError("late spawn hook failed")

    original_redact = runtime.process._redact_root_spawn_initial_goal

    def fail_redaction(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected launch redaction failure")

    runtime.process.add_after_spawn_hook(fail_after_goal)
    monkeypatch.setattr(
        runtime.process,
        "_redact_root_spawn_initial_goal",
        fail_redaction,
    )
    with pytest.raises(RuntimePublicationPending):
        runtime.process.spawn(goal=secret)

    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "kind = ?",
        ("process_launch",),
    )
    assert len(rows) == 1
    assert rows[0]["state"] == "applying"
    receipt = json.loads(rows[0]["receipt_json"])
    assert secret in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "full"

    monkeypatch.setattr(
        runtime.process,
        "_redact_root_spawn_initial_goal",
        original_redact,
    )
    released = runtime.release_recovery_diagnostics()
    assert released["ok"] is True

    reopened = Runtime.open(database)
    try:
        row = reopened.store.select_table_rows(
            "runtime_publications",
            "kind = ?",
            ("process_launch",),
        )[0]
        assert row["state"] == "rolled_back"
        receipt = json.loads(row["receipt_json"])
        assert secret not in json.dumps(receipt)
        assert _goal_envelope(receipt)["retention"] == "hash_only"
    finally:
        reopened.close()


def test_crashed_late_spawn_is_redacted_by_startup_compensation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "startup-spawn-compensation.sqlite"
    secret = "startup-compensation-goal-a916"
    runtime = Runtime.open(database)

    def fail_after_goal(_pid: str, _image_id: str, _publication_id: str) -> None:
        raise RuntimeError("simulate crash after goal")

    def interrupt_rollback(*_args: Any, **_kwargs: Any) -> bool:
        raise KeyboardInterrupt("simulated crash before rollback")

    runtime.process.add_after_spawn_hook(fail_after_goal)
    monkeypatch.setattr(
        runtime.process,
        "_rollback_launch_publication",
        interrupt_rollback,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        runtime.process.spawn(goal=secret)
    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "kind = ?",
        ("process_launch",),
    )
    assert len(rows) == 1
    assert secret in str(rows[0]["receipt_json"])
    publication_id = str(rows[0]["publication_id"])
    runtime.close()

    reopened = Runtime.open(database)
    try:
        row = reopened.store.select_table_rows(
            "runtime_publications",
            "publication_id = ?",
            (publication_id,),
        )[0]
        assert row["state"] == "rolled_back"
        receipt = json.loads(row["receipt_json"])
        assert secret not in json.dumps(receipt)
        assert _goal_envelope(receipt)["retention"] == "hash_only"
    finally:
        reopened.close()


def test_startup_claim_redaction_failure_rolls_back_claim_and_plaintext(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "atomic-startup-spawn-redaction.sqlite"
    secret = "atomic-startup-redaction-goal-390f"
    runtime = Runtime.open(database)

    def fail_after_goal(_pid: str, _image_id: str, _publication_id: str) -> None:
        raise RuntimeError("simulate crash after goal")

    def interrupt_rollback(*_args: Any, **_kwargs: Any) -> bool:
        raise KeyboardInterrupt("simulated crash before rollback")

    runtime.process.add_after_spawn_hook(fail_after_goal)
    monkeypatch.setattr(
        runtime.process,
        "_rollback_launch_publication",
        interrupt_rollback,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        runtime.process.spawn(goal=secret)
    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "kind = ?",
        ("process_launch",),
    )
    assert len(rows) == 1
    publication_id = str(rows[0]["publication_id"])
    assert rows[0]["state"] == "applying"
    runtime.close()

    manager_type = type(runtime.process)
    original_redact = manager_type._redact_root_spawn_initial_goal

    def fail_recovery_redaction(
        self: Any,
        publication: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if publication["publication_id"] == publication_id:
            raise RuntimeError("injected startup redaction failure")
        original_redact(self, publication, **kwargs)

    monkeypatch.setattr(
        manager_type,
        "_redact_root_spawn_initial_goal",
        fail_recovery_redaction,
    )
    with pytest.raises(RuntimeError, match="injected startup redaction failure"):
        Runtime.open(database)

    with sqlite3.connect(database) as connection:
        state, receipt_json = connection.execute(
            "SELECT state, receipt_json FROM runtime_publications "
            "WHERE publication_id = ?",
            (publication_id,),
        ).fetchone()
    assert state == "applying"
    receipt = json.loads(receipt_json)
    assert secret in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "full"

    monkeypatch.setattr(
        manager_type,
        "_redact_root_spawn_initial_goal",
        original_redact,
    )
    reopened = Runtime.open(database)
    try:
        row = reopened.store.select_table_rows(
            "runtime_publications",
            "publication_id = ?",
            (publication_id,),
        )[0]
        assert row["state"] == "rolled_back"
        receipt = json.loads(row["receipt_json"])
        assert secret not in json.dumps(receipt)
        assert _goal_envelope(receipt)["retention"] == "hash_only"
    finally:
        reopened.close()


def test_exhausted_startup_claim_cannot_commit_manual_before_redaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "atomic-manual-spawn-redaction.sqlite"
    secret = "atomic-manual-redaction-goal-87ca"
    config = replace(
        DEFAULT_CONFIG,
        runtime=replace(
            DEFAULT_CONFIG.runtime,
            publication_recovery_max_attempts=1,
        ),
    )
    runtime = Runtime.open(database, config=config)

    def fail_after_goal(_pid: str, _image_id: str, _publication_id: str) -> None:
        raise RuntimeError("simulate crash after goal")

    def interrupt_rollback(*_args: Any, **_kwargs: Any) -> bool:
        raise KeyboardInterrupt("simulated crash before rollback")

    runtime.process.add_after_spawn_hook(fail_after_goal)
    monkeypatch.setattr(
        runtime.process,
        "_rollback_launch_publication",
        interrupt_rollback,
    )
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        runtime.process.spawn(goal=secret)
    row = runtime.store.list_runtime_publications(states=("applying",))[0]
    publication_id = str(row["publication_id"])
    first_claim = runtime.store.claim_runtime_publication_recovery(
        publication_id,
        claimant_instance_id="prior-recovery-runtime",
        expected_owner_instance_id=str(row["owner_instance_id"]),
        expected_state="applying",
        classification="compensate_process_launch",
        max_attempts=1,
        allow_orphaned_claim_takeover=True,
    )
    assert first_claim is not None
    recovery_lease_id = str(first_claim["receipt"]["recovery"]["lease_id"])
    assert runtime.store.advance_runtime_publication(
        publication_id,
        state="failed",
        phase="startup_compensation_failed",
        expected_states={"rollback_pending"},
        recovery_lease_id=recovery_lease_id,
    )
    runtime.close()

    manager_type = type(runtime.process)
    original_redact = manager_type._redact_root_spawn_initial_goal

    def fail_manual_redaction(
        self: Any,
        publication: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if publication["publication_id"] == publication_id:
            raise RuntimeError("injected exhausted-claim redaction failure")
        original_redact(self, publication, **kwargs)

    monkeypatch.setattr(
        manager_type,
        "_redact_root_spawn_initial_goal",
        fail_manual_redaction,
    )
    with pytest.raises(
        RuntimeError,
        match="injected exhausted-claim redaction failure",
    ):
        Runtime.open(database, config=config)

    with sqlite3.connect(database) as connection:
        state, receipt_json = connection.execute(
            "SELECT state, receipt_json FROM runtime_publications "
            "WHERE publication_id = ?",
            (publication_id,),
        ).fetchone()
    assert state == "failed"
    receipt = json.loads(receipt_json)
    assert receipt["recovery"]["attempt"] == 1
    assert secret in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "full"

    monkeypatch.setattr(
        manager_type,
        "_redact_root_spawn_initial_goal",
        original_redact,
    )
    with pytest.raises(ProcessError, match="requires manual recovery"):
        Runtime.open(database, config=config)

    with sqlite3.connect(database) as connection:
        state, receipt_json = connection.execute(
            "SELECT state, receipt_json FROM runtime_publications "
            "WHERE publication_id = ?",
            (publication_id,),
        ).fetchone()
    assert state == "manual"
    receipt = json.loads(receipt_json)
    assert receipt["recovery"]["attempt"] == 2
    assert secret not in json.dumps(receipt)
    assert _goal_envelope(receipt)["retention"] == "hash_only"


def test_child_goal_never_gets_root_recovery_envelope(
    tmp_path: Path,
) -> None:
    database = tmp_path / "child-no-root-recovery.sqlite"
    runtime = Runtime.open(database)
    parent = runtime.process.spawn(goal="parent")
    child = runtime.process.spawn_child(parent, "child remains volatile")
    child_goal_oid = runtime.process.get(child).goal_oid
    _, receipt = _raw_launch_receipt(runtime, child)
    assert all(
        "initial_goal_recovery" not in phase for phase in receipt["phases"]
    )
    runtime.close()

    reopened = Runtime.open(database)
    try:
        assert reopened.store.get_object(child_goal_oid) is None
        state = reopened.store.get_persisted_object_state(child_goal_oid)
        assert state is not None
        assert state.lifecycle_state is ObjectLifecycleState.RELEASED
    finally:
        reopened.close()


def test_mutable_goal_handle_does_not_get_recovery_envelope(
    tmp_path: Path,
) -> None:
    database = tmp_path / "mutable-goal-handle.sqlite"
    runtime = Runtime.open(database)
    owner = runtime.process.spawn(goal="mutable goal owner")
    mutable_goal = runtime.memory.create_object(
        owner,
        ObjectType.GOAL,
        {"task": "mutable"},
        immutable=False,
    )
    with pytest.raises(CapabilityDenied, match="capability subject mismatch"):
        runtime.process.spawn(goal=mutable_goal)
    rows = runtime.store.select_table_rows(
        "runtime_publications",
        "kind = ? AND state = ?",
        ("process_launch", "rolled_back"),
    )
    assert len(rows) == 1
    receipt = json.loads(rows[0]["receipt_json"])
    assert all(
        "initial_goal_recovery" not in phase for phase in receipt["phases"]
    )
    runtime.close()

    reopened = Runtime.open(database)
    try:
        assert reopened.store.get_object(mutable_goal.oid) is None
    finally:
        reopened.close()
