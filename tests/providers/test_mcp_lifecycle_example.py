from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.mcp

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPOSITORY_ROOT / "examples" / "mcp" / "run_lifecycle_e2e.py"
PRIVATE_MARKER = "MUST-NOT-PERSIST"


def test_modern_lifecycle_example_is_deterministic_and_opaque_state_safe() -> None:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert PRIVATE_MARKER not in completed.stdout
    result = json.loads(completed.stdout)

    assert result["schema_version"] == 1
    assert result["mrtr"]["initial_result"] == "input_required"
    assert result["mrtr"]["reopened_result"] == "input_required"
    assert result["mrtr"]["explicit_response_result"] == "complete"
    assert result["mrtr"]["initial_tool_dispatches"] == 1
    assert result["mrtr"]["continuation_dispatches"] == 1
    assert result["mrtr"]["automatic_initial_replay"] is False

    assert result["remote_tasks"]["input_flow"] == [
        "working",
        "input_required",
        "working",
        "completed",
    ]
    assert result["remote_tasks"]["cancel_flow"] == [
        "working",
        "cancel_requested",
        "cancelled",
    ]
    assert result["remote_tasks"]["provider_calls_after_restart"] == 0
    assert result["remote_tasks"]["automatic_poll_or_replay"] is False

    assert result["subscriptions"]["event_sequences"] == [1]
    assert result["subscriptions"]["event_provenance"] == [
        "untrusted_mcp_notification"
    ]
    assert result["subscriptions"]["explicit_stop_status"] == "closed"
    assert result["subscriptions"]["reopened_status"] == "lost"
    assert result["subscriptions"]["lost_reason"] == "runtime_restart"
    assert result["subscriptions"]["reopened_events"] == "unavailable"
    assert result["subscriptions"]["queued_event_before_restart"] is True
    assert result["subscriptions"]["automatic_relisten"] is False

    assert result["recovery"]["missing_broker_failed_closed"] is True
    assert result["recovery"]["raw_remote_state_in_sqlite"] is False
    effects = set(result["recovery"]["protected_effect_operations"])
    assert {
        "call_tool",
        "continuation.respond",
        "tasks.get",
        "tasks.update",
        "tasks.cancel",
        "subscriptions.start",
        "subscriptions.events",
        "subscriptions.stop",
        "subscriptions.status",
    } <= effects
    actions = set(result["recovery"]["audit_actions"])
    assert {
        "primitive.mcp.call",
        "primitive.mcp.continuation.respond",
        "primitive.mcp.tasks.get",
        "primitive.mcp.tasks.update",
        "primitive.mcp.tasks.cancel",
        "primitive.mcp.subscriptions.start",
        "primitive.mcp.subscriptions.events",
        "primitive.mcp.subscriptions.stop",
        "primitive.mcp.subscriptions.status",
    } <= actions
