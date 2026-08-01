from __future__ import annotations

import re
from pathlib import Path

from agent_libos.models import EventPriority, EventType, ProcessStatus


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _words(value: str) -> str:
    return " ".join(value.split())


def _section(document: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(?P<body>.*?)(?=^## |\Z)",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_event_catalog_tracks_the_complete_runtime_enum_in_order() -> None:
    documentation = _read("docs/events.md")
    catalog = _section(documentation, "Event type catalog")
    documented = re.findall(r"^\| `([a-z][a-z0-9_]*)` \|", catalog, re.MULTILINE)

    assert documented == [event_type.value for event_type in EventType]
    for priority in EventPriority:
        assert f"`{priority.value}`" in documentation


def test_event_docs_keep_idempotency_ordering_and_atomicity_limits() -> None:
    documentation = _words(_read("docs/events.md"))

    for required in (
        "order rows by `(created_at, event_id)`",
        "it is not causal order",
        "returns the original row, including its original",
        "failed evidence link leaves no orphan event",
        "guarantees atomicity only for its own insert",
        "has no `DATA_FLOW_DECISION`",
        "not a substitute for a provider idempotency key",
    ):
        assert required in documentation


def test_runtime_lifecycle_docs_cover_typed_waits_and_non_scheduler_running() -> None:
    documentation = _words(_read("docs/runtime_model.md"))

    for status in ProcessStatus:
        assert f"`{status.value}`" in documentation
    for required in (
        "a claimed direct workflow",
        "an ObjectTask runner is managed by the",
        "`ChildProcessWait` or",
        "`MessageProcessWait`",
        "`ToolProcessWait`",
        "enumerates and claims only `runnable` rows",
        "does not convert typed waits to `runnable`",
        "there is no standalone generic wake event",
    ):
        assert required in documentation


def test_runtime_and_storage_docs_keep_typed_stale_recovery_boundary() -> None:
    runtime = _words(_read("docs/runtime_model.md"))
    storage = _words(_read("docs/storage.md"))

    for required in (
        "typed `StaleExecutionProcessWait` receipt is resumed",
        "`stale_execution_recovery` `status_message` is compatibility display text",
        "is not a control input",
    ):
        assert required in runtime
    assert "`stale_execution_recovery` pause is resumed" not in runtime

    for required in (
        "owner/lease-identity-hash-only",
        "canonical recovering-Runtime owner-id/prior-owner/prior-lease SHA-256 values",
        "identity hash rather than a cryptographic signature",
        "historical recovering-owner hash",
        "TaskRun admission/current epochs",
        "safe-point integrity, and current bindings remain authoritative",
        "including stale execution",
        "never parsed as the control protocol",
    ):
        assert required in storage


def test_runtime_and_storage_docs_keep_command_receipts_evidence_fenced() -> None:
    runtime = _words(_read("docs/runtime_model.md"))
    storage = _words(_read("docs/storage.md"))

    for required in (
        "split local-control pending and completed receipt retains its exact `admission_ledger_seq`",
        "`admission_ledger_item_id`, and `admission_evidence_sha256`",
        "even on terminal or superseded early-return paths",
        "Interrupt completion retains its `interrupt_provenance_sha256` while removing the raw admission epoch/fence list",
        "append-only effect transition plus `host_verified_receipt` audit",
        "global Runtime-epoch counter-row fence",
        "linked-recovery gap repair",
    ):
        assert required in runtime

    for required in (
        "rejects an oversized raw UTF-8 `result_json` before parsing",
        "complete canonical public TaskRun summary bound to `run_id` and `result_revision`",
        "exact key set for the command/request-selected variant",
        "pending-only fields on a completed receipt",
        "outside signed BIGINT bounds",
        "`admission_ledger_seq`, `admission_ledger_item_id`, and `admission_evidence_sha256`",
        "append-only `STATUS_TRANSITION` item",
        "completion drops the raw epoch/fences but retains that digest",
        "exact `settlement_transition_seq` and `settlement_audit_record_id`",
        "replay does not depend on purgeable provider metadata or receipt bodies",
        "conditional no-op update of the global `task_run_runtime_epoch` counter row",
        "linked-recovery missing-parent receipt path",
    ):
        assert required in storage


def test_runtime_docs_bound_cwd_human_and_object_task_resume_contracts() -> None:
    documentation = _words(_read("docs/runtime_model.md"))

    for required in (
        "activated Skill bodies",
        "Unactivated Skill metadata is not injected",
        "An explicit cwd supplied to child spawn, direct fork, or PTY creation",
        "omit cwd, they reuse the stored parent/current cwd identity",
        "<workspace-relative-dir>",
        "After the Host grants that process directory-read authority",
        "process in `PausedProcessWait` until an explicitly permitted resume",
        "Host or a direct parent",
        "An ambiguous provider outcome installs the stronger `HostResumeProcessWait`",
        "restricted `discover_skills` queries for message/mailbox help",
        "A read with `ack=false` leaves the interrupt unread",
        "message/child condition replay auto-resume is limited",
        "Human-blocked ObjectTasks resume through their durable request id",
    ):
        assert required in documentation


def test_runtime_docs_bound_image_only_anchor_and_root_goal_recovery() -> None:
    documentation = _words(_read("docs/runtime_model.md"))

    for required in (
        "durable LLM-context generation",
        "Image-only provider errors use a separate purpose",
        "legacy same-purpose error rows are likewise ignored",
        "not a usable head until normalization and dispatch validation",
        "preserves any first-request retry anchor",
        "checkpoint restore advances it",
        "attempts to append a content-free same-purpose tombstone",
        "A tombstone failure does not discard the successful transcript",
        "leaves the old request anchor conservatively protected",
        "committed root `ProcessManager.spawn` publication",
        "Ordinary publication reads expose only a hash-only projection",
        "replacement goal supplied by exec does not",
        "forks, child spawns, ObjectTask owner payloads",
        "Terminal root cleanup redacts any still-full envelope",
        "same outer transaction that reduces any full envelope to hashes",
        "never commits a non-committable launch state with reversible goal content",
    ):
        assert required in documentation

    checkpoint_docs = _words(_read("docs/checkpoints.md"))
    for required in (
        "restored full-I/O `image_only` conditional release",
        "rebinds only the frozen transcript/request anchor",
        "request payload did not change",
        "profile, trust, release, and single-claim checks still apply",
    ):
        assert required in checkpoint_docs


def test_data_flow_docs_do_not_promise_decisions_for_pre_flow_denials() -> None:
    documentation = _words(_read("docs/data_flow.md"))

    assert "A request that reaches the data-flow gate and is denied" in documentation
    assert "has not yet made a data-flow decision" in documentation
    assert "must not be expected to create that row or event" in documentation


def test_storage_docs_distinguish_product_and_schema_and_bound_backup_support() -> None:
    documentation = _words(_read("docs/storage.md"))

    assert "Agent libOS 1.1.0 stores durable runtime state" in documentation
    assert "## Strict store schema v4" in documentation
    assert "Product version and store schema version are independent" in documentation
    assert "There are no migrations, backfills, read-only compatibility modes, or dual schema paths from schema v3 to v4" in documentation
    assert "must be opened with Agent libOS 1.0.1" in documentation
    for required in (
        "## Backup and restore runbook",
        "SQLite's own backup command",
        "Runtime has released its active-store lease",
        "This must happen before shutdown",
        "`current_database()` plus the exact `current_schema()`",
        "`sqlite_master.type = 'table'`",
        "`pg_class.relkind = 'r'`",
        "`--schema=agent_libos_runtime`",
        "`--no-privileges`",
        "do not pre-create it",
        "Restore into the new target in one",
        "no application-level online-backup barrier",
        "supported full runbook therefore requires quiescence",
        "does not include other volatile Object payloads",
        "the live connection holds SQLite's exclusive database lock",
        "committed root `ProcessManager.spawn` publication carries",
        "Generic publication get/list paths redact the reversible value",
        "still-full internal root-spawn initial-goal recovery envelope",
        "cannot leave `rollback_pending`, `rolled_back`, `failed`, or `manual`",
    ):
        assert required in documentation


def test_retention_docs_cover_image_only_streams_and_goal_envelope_boundary() -> None:
    documentation = _words(_read("docs/evidence_payload_retention.md"))

    for required in (
        "`image_only_request:<anchor-fingerprint>`",
        "Image-only provider errors instead use `image_only_error`",
        "legacy newer same-purpose error",
        "separate successful action-validation marker",
        "a content-free tombstone is attempted afterward",
        "A tombstone write failure is audited",
        "the old request remains protected",
        "is not scanned or aged by `PayloadRetentionMaintenance`",
        "redaction failure rolls the state transition back",
        "does not promise a maximum deletion deadline",
        "image-only retry-request anchor",
    ):
        assert required in documentation


def test_architecture_docs_keep_root_goal_recovery_narrow() -> None:
    documentation = _words(_read("docs/architecture.md"))

    for required in (
        "internal, integrity-bound recovery envelope",
        "committed, live root spawn",
        "generic publication reads redact the payload",
        "does not apply to child/fork/ObjectTask or ordinary Object payloads",
        "an exec that preserves the original root goal remains eligible",
        "`persist_full_io=false` stores no reversible goal content",
    ):
        assert required in documentation


def test_object_docs_and_skills_do_not_generalize_root_goal_recovery() -> None:
    object_docs = _words(_read("docs/object_memory.md"))
    runtime_skill = _words(
        _read("agent_libos/skills/builtin/agent-libos-runtime-session/SKILL.md")
    )
    memory_skill = _words(
        _read("agent_libos/skills/builtin/agent-libos-object-memory/SKILL.md")
    )
    transfer_skill = _words(
        _read("agent_libos/skills/builtin/agent-libos-object-file-transfer/SKILL.md")
    )

    assert "sole current automatic exception is not stored in the Object row" in object_docs
    assert "an exec replacement goal, child/fork goal, ObjectTask owner" in object_docs
    assert "Host may already have rehydrated the exact initial goal" in runtime_skill
    assert "does not cover child/fork or exec replacement goals" in runtime_skill
    assert "does not make these ordinary Objects durable" in memory_skill
    assert "committed-root initial-GOAL recovery path does not apply" in transfer_skill
