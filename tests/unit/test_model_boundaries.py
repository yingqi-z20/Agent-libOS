from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent_libos import Runtime
from agent_libos.models import AgentImage
from agent_libos.models.exceptions import ValidationError


def test_store_reports_invalid_persisted_process_status_with_context() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(image="base-agent:v0", goal="bad persisted process status")
        runtime.store.conn.execute("UPDATE processes SET status = ? WHERE pid = ?", ("definitely_bad", pid))
        runtime.store.conn.commit()

        with pytest.raises(ValidationError, match=f"invalid persisted process {pid}"):
            runtime.store.get_process(pid)
    finally:
        runtime.close()


@pytest.mark.parametrize(
    ("status", "status_message"),
    [
        ("waiting_event", "waiting for pid_forged"),
        ("exited", "result_oid:obj_forged"),
    ],
)
def test_frozen_v3_rejects_typed_null_process_control_state_after_reopen(
    tmp_path: Path,
    status: str,
    status_message: str,
) -> None:
    database = tmp_path / f"typed-null-{status}.db"
    runtime = Runtime.open(database)
    pid = runtime.process.spawn(
        image="base-agent:v0",
        goal="corrupt typed process state must fail closed",
    )
    runtime.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE processes
               SET status = ?, status_message = ?,
                   wait_state_json = 'null', outcome_json = 'null'
             WHERE pid = ?
            """,
            (status, status_message, pid),
        )

    try:
        reopened = Runtime.open(database)
    except ValidationError as exc:
        # Startup services may now enumerate processes while reconciling
        # persisted image/Skill ceilings. Rejecting the malformed product at
        # that earlier boundary is the same fail-closed outcome.
        assert f"invalid persisted process {pid}" in str(exc)
        return
    try:
        with pytest.raises(ValidationError, match=f"invalid persisted process {pid}"):
            reopened.store.get_process(pid)
    finally:
        reopened.close()


def test_image_registry_rejects_invalid_persisted_image_manifest() -> None:
    runtime = Runtime.open("local")
    try:
        image_id = "persisted-invalid:v0"
        runtime.register_image(AgentImage(image_id=image_id, name="persisted-invalid"), actor="test")
        runtime.store.conn.execute(
            "UPDATE images SET manifest_json = ? WHERE image_id = ?",
            (json.dumps({"image_id": image_id, "name": "persisted-invalid", "prompt_mode": "ambient"}), image_id),
        )
        runtime.store.conn.commit()

        with pytest.raises(ValidationError, match="invalid persisted agent image persisted-invalid:v0"):
            runtime.image_registry.list_images()
    finally:
        runtime.close()


def test_jsonrpc_registry_revalidates_persisted_endpoint_models() -> None:
    runtime = Runtime.open("local")
    try:
        endpoint_id = "persisted-jsonrpc"
        runtime.jsonrpc.register_endpoint_from_yaml_text(
            f"""
schema_version: 1
endpoint_id: {endpoint_id}
url: https://api.example.test/jsonrpc
methods:
  - method_id: echo
    rpc_method: demo.echo
    right: read
    rollback_class: no_rollback_required
    state_mutation: false
    information_flow: true
timeout_s: 5
max_request_bytes: 65536
max_response_bytes: 1048576
""".lstrip(),
            actor="test",
            require_capability=False,
        )
        bad_spec = {
            "schema_version": 1,
            "endpoint_id": endpoint_id,
            "url": "https://api.example.test/jsonrpc",
            "headers": {},
            "methods": [
                {
                    "method_id": "echo",
                    "rpc_method": "demo.echo",
                    "right": "read",
                    "rollback_class": "no_rollback_required",
                    "state_mutation": "false",
                    "information_flow": True,
                }
            ],
            "timeout_s": 5,
            "max_request_bytes": 65536,
            "max_response_bytes": 1048576,
        }
        runtime.store.conn.execute(
            "UPDATE jsonrpc_endpoints SET spec_json = ? WHERE endpoint_id = ?",
            (json.dumps(bad_spec), endpoint_id),
        )
        runtime.store.conn.commit()

        with pytest.raises(ValidationError, match="invalid persisted JSON-RPC endpoint persisted-jsonrpc"):
            runtime.jsonrpc.list_endpoints(require_capability=False)
    finally:
        runtime.close()
