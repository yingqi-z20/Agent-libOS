from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import (
    RuntimePublicationPending,
    RuntimeRecoveryRequired,
)
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.checkpoint import (
    CreateCheckpointArgs,
    CreateCheckpointTool,
    DiffCheckpointTool,
    ForkCheckpointTool,
    InspectCheckpointTool,
    ListCheckpointsTool,
    RestoreCheckpointTool,
)


_RECEIPT_SENTINEL = "provider-receipt-must-stay-out-of-model-output"
_AUDIT_SENTINEL = "audit-event-id-must-stay-out-of-model-output"
_LARGE_TEXT = "x" * 32_000
_V2_CONFIG = replace(
    DEFAULT_CONFIG,
    llm=replace(DEFAULT_CONFIG.llm, prompt_layout="cache_optimized_v2"),
)


@pytest.mark.parametrize("pid", (None, "", "None", " null "))
def test_create_checkpoint_normalizes_omitted_pid_sentinels(pid: str | None) -> None:
    args = CreateCheckpointArgs(reason="verified milestone", pid=pid)

    assert args.pid is None


def _context(checkpoint: Any, *, config: Any = DEFAULT_CONFIG) -> ToolContext:
    return ToolContext(
        trace_id="trace_checkpoint_projection",
        call_id="call_checkpoint_projection",
        pid="pid_caller",
        runtime=SimpleNamespace(
            checkpoint=checkpoint,
            config=config,
        ),
    )


def _effect(index: int) -> dict[str, Any]:
    return {
        "effect_id": f"eff_{index:06d}",
        "record_id": f"{_AUDIT_SENTINEL}_{index}",
        "event_id": f"event_{index}",
        "pid": "pid_source",
        "provider": "filesystem",
        "operation": f"write_{index:06d}",
        "target": f"workspace:{index}:{_LARGE_TEXT}",
        "rollback_class": "irreversible",
        "rollback_status": "not_supported",
        "state_mutation": True,
        "information_flow": False,
        "provider_metadata": {
            "provider_receipt": _RECEIPT_SENTINEL,
            "raw": _LARGE_TEXT,
        },
        "provider_receipt": {"raw": _RECEIPT_SENTINEL + _LARGE_TEXT},
        "canonical_args_hash": "f" * 64,
        "idempotency_key": f"idempotency_{index}",
        "created_at": "2026-07-25T01:02:03.123456+00:00",
        "updated_at": "2026-07-25T01:02:04.123456+00:00",
        "effect_state": "finalized",
        "transaction_state": "committed",
    }


def _effect_summary(count: int) -> dict[str, Any]:
    return {
        "total": count,
        "by_rollback_class": {"irreversible": count},
        "by_provider_operation": {
            f"filesystem.write_{index:06d}{_LARGE_TEXT}": 1
            for index in range(count)
        },
        "state_mutations": count,
        "information_flows": 0,
        "by_state": {"finalized": count},
        "pending": 0,
        "provider_metadata": _RECEIPT_SENTINEL,
    }


def _page_assertions(page: dict[str, Any], *, count: int, returned: int) -> None:
    assert page["count"] == count
    assert page["returned_count"] == returned
    assert page["truncated"] is (returned < count)


def test_diff_tool_pages_sanitized_external_effects_and_keeps_policy() -> None:
    effect_count = 250
    effects = [_effect(index) for index in range(effect_count)]
    oversized_ids = [f"oid_{index}_{_LARGE_TEXT}" for index in range(100)]

    class Checkpoint:
        def diff(self, checkpoint_id: str, *, actor: str) -> dict[str, Any]:
            assert checkpoint_id == "ckpt_large"
            assert actor == "pid_caller"
            return {
                "checkpoint_id": checkpoint_id,
                "pid": "pid_source",
                "tables": {
                    "objects": {
                        "added": oversized_ids,
                        "removed": [],
                        "changed": [],
                        "added_count": len(oversized_ids),
                        "removed_count": 0,
                        "changed_count": 0,
                    }
                },
                "external_effects_since_checkpoint": effects,
                "external_effect_summary": _effect_summary(effect_count),
                "restore_external_policy": "report_only",
            }

    first = DiffCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_large"},
        _context(Checkpoint()),
    )
    assert first.ok
    assert first.data["checkpoint_id"] == "ckpt_large"
    assert first.data["pid"] == "pid_source"
    assert first.data["restore_external_policy"] == "report_only"
    assert first.data["external_effect_summary"]["total"] == effect_count
    _page_assertions(
        first.data["external_effects_page"],
        count=effect_count,
        returned=DEFAULT_CONFIG.checkpoint.diff_preview_items,
    )
    assert first.data["external_effects_page"]["next_cursor"] == 25
    assert len(first.data["external_effects_since_checkpoint"]) == 25
    assert len(first.data["tables"]["objects"]["added"]) == 25
    assert all(
        len(item) <= DEFAULT_CONFIG.tools.tool_observability_preview_chars
        for item in first.data["tables"]["objects"]["added"]
    )
    assert _RECEIPT_SENTINEL not in first.content
    assert _AUDIT_SENTINEL not in first.content
    for item in first.data["external_effects_since_checkpoint"]:
        assert set(item) == {
            "provider",
            "operation",
            "target",
            "rollback_class",
            "rollback_status",
            "state_mutation",
            "information_flow",
            "effect_state",
            "transaction_state",
        }
        assert len(item["target"]) <= DEFAULT_CONFIG.tools.tool_observability_preview_chars
    assert len(first.content.encode("utf-8")) < DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes

    second = DiffCheckpointTool().invoke(
        {
            "checkpoint_id": "ckpt_large",
            "external_effect_cursor": 25,
            "external_effect_limit": 5,
        },
        _context(Checkpoint()),
    )
    assert second.ok
    assert second.data["external_effects_page"] == {
        "count": effect_count,
        "returned_count": 5,
        "truncated": True,
        "next_cursor": 30,
    }
    assert second.data["external_effects_since_checkpoint"][0]["operation"] == "write_000025"


def test_inspect_tool_pages_large_subtree_and_bounds_nested_process_state() -> None:
    process_count = 2_000
    processes = [
        {
            "pid": f"pid_{index:06d}",
            "parent_pid": None if index == 0 else "pid_000000",
            "image_id": "base-agent:v0",
            "status": "waiting_event",
            "working_directory": ".",
            "goal_oid": f"oid_goal_{index:06d}",
            "wait_state": {
                "kind": "message",
                "filter": {
                    "body": _LARGE_TEXT,
                    "provider_metadata": _RECEIPT_SENTINEL,
                },
            },
            "outcome": None,
            "state_generation": index,
            "private_snapshot_payload": _LARGE_TEXT,
        }
        for index in range(process_count)
    ]

    class Checkpoint:
        def list(
            self,
            pid: str,
            *,
            actor: str,
            limit: int | None,
        ) -> list[dict[str, Any]]:
            assert pid == "pid_caller"
            assert actor == "pid_caller"
            assert limit == 2
            return [{"checkpoint_id": "ckpt_large"}]

        def inspect(self, checkpoint_id: str, *, actor: str) -> dict[str, Any]:
            assert checkpoint_id == "ckpt_large"
            assert actor == "pid_caller"
            return {
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "pid": "pid_000000",
                    "reason": "large subtree",
                    "created_at": "2026-07-25T01:02:03+00:00",
                    "created_by": "pid_caller",
                    "snapshot_version": 4,
                    "effect_ledger_seq": 999,
                    "metadata": {"raw": _LARGE_TEXT},
                },
                "snapshot_version": 4,
                "subtree_pids": [item["pid"] for item in processes],
                "modules": [
                    {
                        "module_id": f"module_{index:06d}",
                        "name": "module",
                        "version": "1",
                        "source_kind": "package",
                        "entrypoint": "module:register",
                        "source_sha256": _AUDIT_SENTINEL,
                        "source_files": [{"content": _LARGE_TEXT}],
                    }
                    for index in range(100)
                ],
                "counts": {"processes": process_count, "objects": 4_000},
                "processes": processes,
            }

    result = InspectCheckpointTool().invoke(
        {
            "checkpoint_id": "ckpt_large",
            "process_cursor": 25,
            "module_cursor": 25,
        },
        _context(Checkpoint()),
    )
    assert result.ok
    assert result.data["checkpoint"]["checkpoint_id"] == "ckpt_large"
    assert "metadata" not in result.data["checkpoint"]
    assert "effect_ledger_seq" not in result.data["checkpoint"]
    assert result.data["subtree_pids"][0] == "pid_000025"
    assert result.data["processes"][0]["pid"] == "pid_000025"
    assert result.data["modules"][0]["module_id"] == "module_000025"
    assert result.data["processes_page"]["next_cursor"] == 50
    assert result.data["modules_page"]["next_cursor"] == 50
    assert _RECEIPT_SENTINEL not in result.content
    assert _AUDIT_SENTINEL not in result.content
    assert _LARGE_TEXT not in result.content
    assert len(result.content.encode("utf-8")) < DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes

    semantic = InspectCheckpointTool().invoke(
        {"checkpoint_id": "only"},
        _context(Checkpoint()),
    )
    assert semantic.ok
    assert semantic.data["checkpoint"]["checkpoint_id"] == "ckpt_large"


def test_restore_tool_preserves_commit_identity_while_bounding_large_reports() -> None:
    item_count = 5_000
    effect_count = 250

    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str) -> dict[str, Any]:
            assert actor == "pid_caller"
            assert checkpoint_id == "ckpt_large"
            return {
                "checkpoint_id": checkpoint_id,
                "publication_id": "pub_restore_large",
                "pid": "pid_source",
                "status": "restored",
                "main_state_committed": True,
                "reconciliation_pending": False,
                "post_commit_failures": [],
                "restored_pids": [f"pid_restored_{index:06d}" for index in range(item_count)],
                "previous_pids": [f"pid_previous_{index:06d}" for index in range(item_count)],
                "cancelled_human_requests": [
                    f"hreq_{index:06d}" for index in range(item_count)
                ],
                "superseded_messages": [f"msg_{index:06d}" for index in range(item_count)],
                "superseded_object_tasks": [
                    f"task_{index:06d}" for index in range(item_count)
                ],
                "external_effects_since_checkpoint": [
                    _effect(index) for index in range(effect_count)
                ],
                "external_effect_summary": _effect_summary(effect_count),
                "restore_external_policy": "report_only",
            }

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_large"},
        _context(Checkpoint()),
    )
    assert result.ok
    assert result.data["checkpoint_id"] == "ckpt_large"
    assert result.data["publication_id"] == "pub_restore_large"
    assert result.data["pid"] == "pid_source"
    assert result.data["status"] == "restored"
    assert result.data["main_state_committed"] is True
    assert result.data["restore_external_policy"] == "report_only"
    assert result.data["external_effect_summary"]["total"] == effect_count
    for field in (
        "restored_pids_page",
        "previous_pids_page",
        "cancelled_human_requests_page",
        "superseded_messages_page",
        "superseded_object_tasks_page",
    ):
        _page_assertions(result.data[field], count=item_count, returned=25)
        assert result.data[field]["next_cursor"] is None
    _page_assertions(
        result.data["external_effects_page"],
        count=effect_count,
        returned=25,
    )
    assert _RECEIPT_SENTINEL not in result.content
    assert _AUDIT_SENTINEL not in result.content
    assert len(result.content.encode("utf-8")) < DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes


def test_fork_tool_preserves_root_identity_while_bounding_large_maps() -> None:
    item_count = 20_000

    class Checkpoint:
        def fork_from_checkpoint(
            self,
            actor: str,
            checkpoint_id: str,
            *,
            parent_pid: str | None,
        ) -> dict[str, Any]:
            assert actor == "pid_caller"
            assert checkpoint_id == "ckpt_large"
            assert parent_pid is None
            return {
                "checkpoint_id": checkpoint_id,
                "source_pid": "pid_source",
                "fork_root_pid": "pid_fork_root",
                "pid_map": {
                    f"pid_source_{index:06d}": f"pid_fork_{index:06d}"
                    for index in range(item_count)
                },
                "object_map": {
                    f"oid_source_{index:06d}": f"oid_fork_{index:06d}"
                    for index in range(item_count)
                },
                "tool_map": {
                    f"tool_source_{index:06d}": f"tool_fork_{index:06d}"
                    for index in range(item_count)
                },
                "status": "forked",
                "main_state_committed": True,
                "post_commit_failures": [],
            }

    result = ForkCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_large"},
        _context(Checkpoint()),
    )
    assert result.ok
    assert result.data["checkpoint_id"] == "ckpt_large"
    assert result.data["source_pid"] == "pid_source"
    assert result.data["fork_root_pid"] == "pid_fork_root"
    assert result.data["status"] == "forked"
    assert result.data["main_state_committed"] is True
    for field in ("pid_map_page", "object_map_page", "tool_map_page"):
        _page_assertions(result.data[field], count=item_count, returned=25)
        assert result.data[field]["next_cursor"] is None
    assert len(result.data["pid_map"]) == 25
    assert len(result.data["object_map"]) == 25
    assert len(result.data["tool_map"]) == 25
    assert len(result.content.encode("utf-8")) < DEFAULT_CONFIG.tools.memory_payload_hard_limit_bytes


def test_inspect_and_diff_honor_runtime_preview_limit_in_schema_and_execution() -> None:
    config = replace(
        DEFAULT_CONFIG,
        checkpoint=replace(
            DEFAULT_CONFIG.checkpoint,
            diff_preview_items=3,
        ),
    )
    processes = [
        {
            "pid": f"pid_{index}",
            "parent_pid": None,
            "image_id": "base-agent:v0",
            "status": "runnable",
            "working_directory": ".",
            "goal_oid": None,
            "wait_state": {f"key_{item}": item for item in range(10)},
            "outcome": None,
            "state_generation": index,
        }
        for index in range(10)
    ]

    class Checkpoint:
        def inspect(self, checkpoint_id: str, *, actor: str) -> dict[str, Any]:
            return {
                "checkpoint": {
                    "checkpoint_id": checkpoint_id,
                    "pid": "pid_0",
                    "reason": "runtime preview",
                    "created_at": "2026-07-25T01:02:03+00:00",
                    "created_by": actor,
                    "snapshot_version": 4,
                },
                "snapshot_version": 4,
                "subtree_pids": [item["pid"] for item in processes],
                "modules": [
                    {
                        "module_id": f"module_{index}",
                        "name": "module",
                        "version": "1",
                        "source_kind": "package",
                        "entrypoint": "module:register",
                    }
                    for index in range(10)
                ],
                "counts": {"processes": len(processes)},
                "processes": processes,
            }

        def diff(self, checkpoint_id: str, *, actor: str) -> dict[str, Any]:
            return {
                "checkpoint_id": checkpoint_id,
                "pid": actor,
                "tables": {
                    "objects": {
                        "added": [f"oid_{index}" for index in range(10)],
                        "removed": [],
                        "changed": [],
                        "added_count": 10,
                        "removed_count": 0,
                        "changed_count": 0,
                    }
                },
                "external_effects_since_checkpoint": [
                    _effect(index) for index in range(10)
                ],
                "external_effect_summary": _effect_summary(10),
                "restore_external_policy": "report_only",
            }

    inspect_schema = InspectCheckpointTool().spec(config=config).input_schema[
        "properties"
    ]["detail_limit"]
    diff_schema = DiffCheckpointTool().spec(config=config).input_schema[
        "properties"
    ]["external_effect_limit"]
    assert inspect_schema["default"] == 3
    assert inspect_schema["maximum"] == 3
    assert diff_schema["default"] == 3
    assert diff_schema["maximum"] == 3

    context = _context(Checkpoint(), config=config)
    inspect = InspectCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_runtime"},
        context,
    )
    assert inspect.ok
    assert len(inspect.data["processes"]) == 3
    assert len(inspect.data["modules"]) == 3
    assert inspect.data["processes_page"]["next_cursor"] == 3
    nested_wait = inspect.data["processes"][0]["wait_state"]
    assert len([key for key in nested_wait if key != "_projection"]) == 3

    diff = DiffCheckpointTool().invoke(
        {
            "checkpoint_id": "ckpt_runtime",
            "external_effect_limit": 9,
        },
        context,
    )
    assert diff.ok
    assert len(diff.data["external_effects_since_checkpoint"]) == 3
    assert len(diff.data["tables"]["objects"]["added"]) == 3
    assert diff.data["external_effects_page"]["next_cursor"] == 3


def test_create_checkpoint_bounds_reason_and_hides_identity_until_selection() -> None:
    calls: list[tuple[str, str, str]] = []

    class Checkpoint:
        def create(self, pid: str, reason: str, *, actor: str) -> str:
            calls.append((pid, reason, actor))
            return "ckpt_identity_survives"

    tool = CreateCheckpointTool()
    schema = tool.spec().input_schema["properties"]["reason"]
    assert schema["maxLength"] == 512
    assert "1024 bytes" in schema["description"]

    reason = "界" * 300
    result = tool.invoke(
        {"reason": reason},
        _context(Checkpoint(), config=_V2_CONFIG),
    )
    assert result.ok
    assert list(result.data) == ["checkpoint_id", "pid", "reason"]
    assert result.data["checkpoint_id"] == "ckpt_identity_survives"
    assert result.model_projection(limit_bytes=2_048) == {
        "created": True,
        "reason": result.data["reason"],
    }
    assert calls == [("pid_caller", reason, "pid_caller")]
    assert len(result.data["reason"]) <= DEFAULT_CONFIG.tools.tool_observability_preview_chars
    assert len(result.content.encode("utf-8")) < 2_048

    byte_overflow = tool.invoke(
        {"reason": "界" * 342},
        _context(Checkpoint()),
    )
    assert not byte_overflow.ok
    char_overflow = tool.invoke(
        {"reason": "a" * 513},
        _context(Checkpoint()),
    )
    assert not char_overflow.ok
    assert len(calls) == 1


def test_legacy_checkpoint_projection_preserves_exact_identity() -> None:
    class Checkpoint:
        def create(self, pid: str, reason: str, *, actor: str) -> str:
            return "ckpt_legacy"

        def list(
            self,
            pid: str,
            *,
            actor: str,
            limit: int | None,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "checkpoint_id": "ckpt_legacy",
                    "pid": pid,
                    "reason": "legacy rollback",
                    "created_at": "2026-08-11T00:00:00+00:00",
                    "created_by": actor,
                    "snapshot_version": 4,
                }
            ]

    context = _context(Checkpoint())
    created = CreateCheckpointTool().invoke(
        {"reason": "legacy rollback"},
        context,
    )
    listed = ListCheckpointsTool().invoke({}, context)

    assert created.ok and listed.ok
    assert created.model_projection(limit_bytes=8_192) == created.data
    assert listed.model_projection(limit_bytes=8_192) == listed.data


def test_list_checkpoints_projects_bounded_allowlist_and_window_metadata() -> None:
    config = replace(
        DEFAULT_CONFIG,
        llm=replace(DEFAULT_CONFIG.llm, prompt_layout="cache_optimized_v2"),
        checkpoint=replace(DEFAULT_CONFIG.checkpoint, list_limit=5),
    )
    observed_limits: list[int | None] = []
    rows = [
        {
            "checkpoint_id": f"ckpt_{index}",
            "pid": "pid_source",
            "reason": f"reason_{index}_{_LARGE_TEXT}",
            "created_at": "2026-07-25T01:02:03+00:00",
            "created_by": "pid_caller",
            "snapshot_version": 4,
            "effect_ledger_seq": index,
            "metadata": {"secret": _RECEIPT_SENTINEL + _LARGE_TEXT},
        }
        for index in range(5)
    ]

    class Checkpoint:
        def list(
            self,
            pid: str,
            *,
            actor: str,
            limit: int | None,
        ) -> list[dict[str, Any]]:
            assert pid == "pid_caller"
            assert actor == "pid_caller"
            observed_limits.append(limit)
            return rows[:limit]

    tool = ListCheckpointsTool()
    limit_schema = tool.spec(config=config).input_schema["properties"]["limit"]
    assert limit_schema["default"] == 5
    assert limit_schema["maximum"] == 5

    context = _context(Checkpoint(), config=config)
    result = tool.invoke({"limit": 2}, context)
    assert result.ok
    assert result.data["count"] == 5
    assert result.data["has_more"] is True
    assert len(result.data["checkpoints"]) == 2
    assert observed_limits == [5]
    assert set(result.data["checkpoints"][0]) == {
        "checkpoint_id",
        "pid",
        "reason",
        "created_at",
        "created_by",
        "snapshot_version",
    }
    assert len(result.data["checkpoints"][0]["reason"]) <= DEFAULT_CONFIG.tools.tool_observability_preview_chars
    assert _RECEIPT_SENTINEL not in result.content
    assert _LARGE_TEXT not in result.content
    model_projection = result.model_projection(limit_bytes=8_192)
    assert model_projection["checkpoints"] == [
        {
            "checkpoint_id": "ckpt_0",
            "reason": result.data["checkpoints"][0]["reason"],
        },
        {
            "checkpoint_id": "ckpt_1",
            "reason": result.data["checkpoints"][1]["reason"],
        },
    ]
    assert "pid_source" not in str(model_projection)
    assert "created_at" not in str(model_projection)

    full_window = tool.invoke({}, context)
    assert full_window.ok
    assert len(full_window.data["checkpoints"]) == 5
    assert full_window.data["count"] == 5
    assert full_window.data["has_more"] is False
    assert observed_limits == [5, 5]

    class SingleCheckpoint(Checkpoint):
        def list(
            self,
            pid: str,
            *,
            actor: str,
            limit: int | None,
        ) -> list[dict[str, Any]]:
            return rows[:1]

    singular = tool.invoke({}, _context(SingleCheckpoint(), config=config))
    assert singular.ok
    assert singular.model_projection(limit_bytes=8_192)["checkpoints"] == [
        {
            "checkpoint_ref": "only",
            "reason": singular.data["checkpoints"][0]["reason"],
        }
    ]
    assert "ckpt_0" not in str(singular.model_projection(limit_bytes=8_192))

    cross_process = tool.invoke(
        {"pid": "pid_source"},
        _context(SingleCheckpoint(), config=config),
    )
    assert cross_process.ok
    assert cross_process.model_projection(limit_bytes=8_192)["checkpoints"] == [
        {
            "checkpoint_id": "ckpt_0",
            "reason": cross_process.data["checkpoints"][0]["reason"],
        }
    ]


def test_fork_failure_preserves_bounded_retry_safety_receipt() -> None:
    class Checkpoint:
        def fork_from_checkpoint(self, actor: str, checkpoint_id: str, parent_pid: str | None):
            error = RuntimeError("commit outcome unknown")
            error.checkpoint_fork_receipt = {
                "checkpoint_id": checkpoint_id,
                "source_pid": actor,
                "fork_root_pid": "pid_fork",
                "pid_map": {actor: "pid_fork"},
                "object_map": {},
                "tool_map": {},
                "status": "fork_outcome_unknown",
                "main_state_committed": None,
                "reconciliation_pending": True,
                "post_commit_failures": [],
                "outcome_diagnostic": {
                    "phase": "fork_commit_confirmation",
                    "diagnostic_error": _RECEIPT_SENTINEL,
                },
            }
            raise error

    result = ForkCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_unknown"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.retryable is False
    receipt = result.error.details["checkpoint_fork_receipt"]
    assert receipt["main_state_committed"] is None
    assert receipt["reconciliation_pending"] is True
    projection = result.model_projection(limit_bytes=4096)
    projected_receipt = projection["error"]["details"]["checkpoint_fork_receipt"]
    assert projected_receipt == {
        "checkpoint_id": "ckpt_unknown",
        "fork_root_pid": "pid_fork",
        "status": "fork_outcome_unknown",
        "main_state_committed": None,
        "reconciliation_pending": True,
        "failure_phases": ["fork_commit_confirmation"],
    }
    assert _RECEIPT_SENTINEL not in str(projection)


def test_restore_pending_failure_preserves_bounded_reconciliation_receipt() -> None:
    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            raise RuntimePublicationPending(
                publication_id="pub_restore_pending",
                operation_id="op_restore_pending",
                state="applying",
                phase="restore_commit",
            )

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_pending"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    assert result.error.retryable is False
    expected = {
        "checkpoint_id": "ckpt_pending",
        "publication_id": "pub_restore_pending",
        "operation_id": "op_restore_pending",
        "state": "applying",
        "phase": "restore_commit",
        "status": "restore_publication_pending",
        "main_state_committed": None,
        "reconciliation_pending": True,
    }
    assert result.error.details["checkpoint_restore_receipt"] == expected
    projection = result.model_projection(limit_bytes=4096)
    assert projection["error"]["details"]["checkpoint_restore_receipt"] == expected


def test_restore_recovery_failure_preserves_publication_state() -> None:
    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            raise RuntimeRecoveryRequired(
                publication_id="pub_restore_recovery",
                operation_id="op_restore_recovery",
                pid="pid_recovery",
                state="rollback_pending",
                phase="restore_compensation",
            )

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_recovery"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    receipt = result.error.details["checkpoint_restore_receipt"]
    assert receipt == {
        "checkpoint_id": "ckpt_recovery",
        "publication_id": "pub_restore_recovery",
        "operation_id": "op_restore_recovery",
        "state": "rollback_pending",
        "phase": "restore_compensation",
        "status": "restore_recovery_required",
        "main_state_committed": None,
        "reconciliation_pending": True,
    }
    assert result.model_projection(limit_bytes=4096)["error"]["details"][
        "checkpoint_restore_receipt"
    ] == receipt


def test_grouped_restore_failure_preserves_unique_pending_signal() -> None:
    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            pending = RuntimePublicationPending(
                publication_id="pub_grouped",
                operation_id="op_grouped",
                state="reconciliation_pending",
                phase="restore_confirmation",
            )
            raise ExceptionGroup(
                "restore confirmation failed",
                [RuntimeError("primary"), ExceptionGroup("nested", [pending])],
            )

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_grouped"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    receipt = result.error.details["checkpoint_restore_receipt"]
    assert receipt["publication_id"] == "pub_grouped"
    assert receipt["operation_id"] == "op_grouped"
    assert receipt["state"] == "reconciliation_pending"
    assert receipt["phase"] == "restore_confirmation"
    assert receipt["reconciliation_pending"] is True


def test_grouped_restore_failure_rejects_conflicting_pending_signals() -> None:
    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            raise ExceptionGroup(
                "conflicting publications",
                [
                    RuntimePublicationPending(
                        publication_id="pub_one",
                        operation_id="op_one",
                        state="applying",
                        phase="first",
                    ),
                    RuntimePublicationPending(
                        publication_id="pub_two",
                        operation_id="op_two",
                        state="rollback_pending",
                        phase="second",
                    ),
                ],
            )

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_conflicting"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    assert "checkpoint_restore_receipt" not in result.error.details


@pytest.mark.parametrize("shape", ["deep", "wide"])
def test_grouped_restore_failure_bounds_recovery_signal_traversal(shape: str) -> None:
    pending = RuntimePublicationPending(
        publication_id="pub_bounded_tree",
        operation_id="op_bounded_tree",
        state="reconciliation_pending",
        phase="restore_confirmation",
    )
    if shape == "deep":
        grouped: Exception = pending
        for index in range(1_500):
            grouped = ExceptionGroup(f"nested {index}", [grouped])
    else:
        grouped = ExceptionGroup(
            "wide restore failure",
            [*(RuntimeError(f"leaf {index}") for index in range(1_100)), pending],
        )

    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            raise grouped

    result = RestoreCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_bounded_tree"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    assert "checkpoint_restore_receipt" not in result.error.details


def test_restore_does_not_convert_control_flow_base_exception_group() -> None:
    class Checkpoint:
        def restore(self, actor: str, checkpoint_id: str):
            raise BaseExceptionGroup(
                "control flow",
                [
                    KeyboardInterrupt(),
                    RuntimePublicationPending(
                        publication_id="pub_control",
                        operation_id="op_control",
                        state="applying",
                        phase="control",
                    ),
                ],
            )

    with pytest.raises(BaseExceptionGroup):
        RestoreCheckpointTool().invoke(
            {"checkpoint_id": "ckpt_control"},
            _context(Checkpoint()),
        )


def test_fork_tool_rejects_forged_receipt_fields() -> None:
    class Checkpoint:
        def fork_from_checkpoint(self, actor: str, checkpoint_id: str, parent_pid: str | None):
            error = RuntimeError("forged receipt")
            error.checkpoint_fork_receipt = {
                "checkpoint_id": checkpoint_id,
                "source_pid": actor,
                "fork_root_pid": "pid_fork",
                "pid_map": {actor: "pid_fork"},
                "object_map": {},
                "tool_map": {},
                "status": "forked",
                "main_state_committed": True,
                "reconciliation_pending": False,
                "post_commit_failures": [],
                "secret": _RECEIPT_SENTINEL,
            }
            raise error

    result = ForkCheckpointTool().invoke(
        {"checkpoint_id": "ckpt_forged"},
        _context(Checkpoint()),
    )

    assert not result.ok
    assert result.error is not None
    assert "checkpoint_fork_receipt" not in result.error.details
    assert _RECEIPT_SENTINEL not in str(result.model_projection(limit_bytes=4096))
