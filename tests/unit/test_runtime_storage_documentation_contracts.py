from __future__ import annotations

import ast
import re
from pathlib import Path

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.human.descriptors import (
    PROTECTED_OPERATION_DESCRIPTORS as HUMAN_OPERATION_DESCRIPTORS,
)
from agent_libos.llm.prompt_cache_gate import PromptCacheGateThresholds
from agent_libos.models import EventPriority, EventType, ProcessStatus
from agent_libos.sdk import PostProviderFailureMode
from agent_libos.sdk.protected_operations import _HOST_RESULT_CONTRACT_PREFIXES


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


def _assert_in_order(document: str, *phrases: str) -> None:
    positions = [document.index(phrase) for phrase in phrases]
    assert positions == sorted(positions)


def test_runtime_safety_task_count_is_consistent_across_current_docs() -> None:
    task_count = len(
        tuple((ROOT / "benchmarks/runtime_safety/tasks").glob("*.yaml"))
    )
    assert task_count == 33

    expected_phrases = {
        "README.md": "33 checked-in schema-v1",
        "docs/benchmark.md": "contains 33 schema-v1 YAML tasks",
        "docs/paper_thesis.md": "with 33 schema-v1 adversarial tasks",
        "docs/release_status.md": "gate all 33 checked-in",
        "docs/support_matrix.md": "33 deterministic schema-v1 tasks",
    }
    for path, expected_phrase in expected_phrases.items():
        assert expected_phrase in _words(_read(path))


def test_security_docs_pin_provider_preintent_and_result_identity_contract() -> None:
    protected = _words(_read("docs/protected_operation_sdk.md"))
    semantic = _words(_read("docs/semantic_shadow.md"))
    threat = _words(_read("docs/threat_model.md"))

    for document in (protected, semantic, threat):
        assert "`GitPrimitive._semantic_read_flow_snapshot`" in document
        assert "pre-intent" in document
        assert "external-effect intent of its own" in document
        assert "Filesystem, Shell, Git, JSON-RPC, MCP, and LLM" in document
        assert "`bytearray`" in document
        assert "`StrEnum`" in document
        assert "`LLMCompletion`" in document
        assert "raw provider" in document.lower()

    for required in (
        "The default is `PostProviderFailureMode.PROPAGATE`",
        "Both `primitive.human.read` and `primitive.human.write` use `PostProviderFailureMode.PRESERVE_RESULT`",
        "pending/unknown intent must not be treated as safe to retry",
        "does not suppress a failure of the conservative fallback settlement itself",
    ):
        assert required in protected

    assert _HOST_RESULT_CONTRACT_PREFIXES == (
        "primitive.filesystem.",
        "primitive.shell.",
        "primitive.git.",
        "primitive.jsonrpc.",
        "primitive.mcp.",
        "primitive.llm.",
    )
    assert {
        descriptor.name: descriptor.post_provider_failure_mode
        for descriptor in HUMAN_OPERATION_DESCRIPTORS
    } == {
        "primitive.human.read": PostProviderFailureMode.PRESERVE_RESULT,
        "primitive.human.write": PostProviderFailureMode.PRESERVE_RESULT,
    }


def test_semantic_phase2_to_4_docs_keep_authority_privacy_and_control_contract() -> None:
    semantic = _words(_read("docs/semantic_shadow.md"))
    cli = _words(_read("docs/cli.md"))
    gui = _words(_read("docs/gui.md"))
    storage = _words(_read("docs/storage.md"))

    for required in (
        "The default remains `semantic.mode: off`",
        "The classifier remains evidence, not authority",
        "`mode: off` is the global kill switch and the default",
        "There is no semantic `POST`, `PUT`, `PATCH`, or `DELETE` route",
        "`action_id`, and `tenant_bucket_sha256` filters",
        "No aggregate accuracy value is treated as a safety conclusion",
        "The status response is schema v3",
        "complete, exact-key `by_status` counters",
        "A rate whose denominator is zero is `null`",
        "Model findings may prevent a positive Shadow outcome",
        "cannot supply a missing allow predicate",
        "A real hard denial is limited to",
        "Only `complete` coverage for one non-mixed tenant identity",
        "The machine path never installs `always_allow`",
        "There is no wildcard tenant, implicit epoch, auto-activation",
        "Approval and provider-ingress capture are metadata-only",
        "Root-goal capture may temporarily include a deterministic `redacted_intent`",
        "the goal sensitivity is `public` or `normal`",
        "4,096-node and 256 KiB budget",
        "500,000-node and 64 MiB ceiling",
        "payload-free descriptor of at most 4 KiB",
        "every other dynamic or hostile type is reported as the fixed string `opaque`",
        "neither the descriptor nor semantic persistence contains its original text or bytes",
        "Local Host DLP evidence is independent of the external model",
        "At most four local detections are frozen in a job",
        "These Host findings merge into every terminal assessment path",
        "remain advisory and never write labels back",
        "Assembly freezes both the selected profile snapshot identity and its explicit model",
        "an invocation-specific observer is additive and cannot replace or suppress it",
        "The assembled Runtime defaults to `null`",
        "`semantic_tenant_bucketer=`",
        "deployment-keyed HMAC",
        "completed external classifier dispatch, the producer may populate `input_tokens`, `output_tokens`, and `cost_microunits`",
        "`prompt_tokens` aliases `input_tokens`",
        "at most 64 exact string keys",
        "safe-integer maximum (`2^53 - 1`)",
        "canonical key and alias are both present, they must be valid and equal",
        "Unknown keys and the raw usage object are never copied",
        "Deterministic/scripted assessments and missing, malformed, conflicting, or untrusted counters remain `null`",
        "valid counters can accompany an `invalid_schema` terminal assessment",
        "provider-reported operational hints",
        "Queue and assessment counts come from exact store aggregation",
        "`capture_failures` is a runtime-local health counter and resets on reopen",
    ):
        assert required in semantic

    for required in (
        "All runtime and policy surfaces are read-only",
        "`semantic review import` is the sole write",
        "semantic flow lineage <node_id>",
        "semantic settlements --limit 50",
        "semantic control history --limit 50",
        "--action-id filesystem.read",
        "--tenant-bucket-sha256 <lowercase_sha256>",
        "raw Human or classifier response",
        "`prompt_tokens → input_tokens`",
        "Unknown usage keys and the raw usage object are never emitted",
    ):
        assert required in cli

    for required in (
        "The read-only Semantic tab",
        "`action_id`, `tenant_bucket_sha256`, `after`, and `limit`",
        "There are no semantic write routes",
        "reserved nullable input/output token and cost fields",
        "status schema v3 with complete exact-key `by_status` and `by_domain` maps",
        "never token aliases, unknown provider usage keys, or the raw usage object",
    ):
        assert required in gui

    for required in (
        "`2^53 - 1` are rejected before persistence",
        "uses that same ceiling for latency and nullable token/cost counters",
        "valid durable value remains exactly representable by the HTTP/CLI/GUI read",
    ):
        assert required in storage


def test_event_catalog_tracks_the_complete_runtime_enum_in_order() -> None:
    documentation = _read("docs/events.md")
    catalog = _section(documentation, "Event type catalog")
    documented = re.findall(r"^\| `([a-z][a-z0-9_]*)` \|", catalog, re.MULTILINE)

    assert documented == [event_type.value for event_type in EventType]
    assert "One of the values in the catalog below" in documentation
    assert re.search(r"One of the \d+ values in the catalog below", documentation) is None
    for priority in EventPriority:
        assert f"`{priority.value}`" in documentation


def test_event_docs_pin_task_run_and_semantic_machine_response_shapes() -> None:
    catalog = _section(_read("docs/events.md"), "Event type catalog")

    task_run = re.search(
        r"^\| `task_run_created` \|.*$",
        catalog,
        flags=re.MULTILINE,
    )
    assert task_run is not None
    task_run_row = task_run.group(0)
    assert "`runtime` → Run id" in task_run_row
    assert "`runtime.task_runs`" not in task_run_row
    for field in (
        "schema_version",
        "run_id",
        "revision",
        "status",
        "root_pid",
        "display_title",
        "retention",
    ):
        assert f"`{field}`" in task_run_row

    semantic = re.search(
        r"^\| `semantic_policy_response` \|.*$",
        catalog,
        flags=re.MULTILINE,
    )
    assert semantic is not None
    semantic_row = semantic.group(0)
    for required in (
        "`policy:semantic:auto` or `policy:semantic:hard-deny` → PID",
        "top level: `request_id`, `status`, `source` (`machine_policy`), sanitized `policy`",
        "nested `settlement_receipt`",
        "`schema_version`, `settlement_id`, `outcome`, `policy_sha256`, and `binding_sha256`",
        "`capability_id` only for an issued approval",
    ):
        assert required in semantic_row
    for stale in (
        "`policy:semantic:<epoch>`",
        "expected `revision`",
        "`epoch_id`",
    ):
        assert stale not in semantic_row


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


def test_data_flow_docs_pin_llm_sink_identity_and_host_result_domains() -> None:
    documentation = _words(_read("docs/data_flow.md"))

    for required in (
        "prompt-cache retention/mode/TTL",
        "`fallback_json_actions` policy identity hash",
        "changing any bound effective policy invalidates a trust rule tied to the old identity hash",
        "`prompt_cache_mode`, `prompt_cache_ttl`, or `fallback_json_actions` changes the profile identity hash",
        "trusted Sink rule bound to the old `identity_sha256` no longer matches",
        "six built-in provider domains—Filesystem, Shell, Git, JSON-RPC, MCP, and LLM—",
    ):
        assert required in documentation
    assert "five built-in provider domains" not in documentation


def test_storage_docs_distinguish_product_and_schema_and_bound_backup_support() -> None:
    documentation = _words(_read("docs/storage.md"))
    readme = _words(_read("README.md"))

    assert "Agent libOS 1.5.1 stores durable runtime state" in documentation
    assert "## Strict store schema v7" in documentation
    assert "Product version and store schema version are independent" in documentation
    assert "The only supported migrations are the explicit, offline, operator-invoked canonical v4-to-v5, v5-to-v6, and v6-to-v7 procedures" in documentation
    assert "There are no automatic migrations, backfills, read-only compatibility modes, or dual runtime schema paths" in documentation
    assert "must be opened with Agent libOS 1.0.1" in documentation
    assert "creates and opens only RuntimeStore schema v7" in readme
    assert "offline, digest-bound v6-to-v7 migration" in readme
    for required in (
        "## Offline v6 to v7 migration",
        "creates the five MCP v7 tables",
        "compare-and-swaps the singleton marker `6 -> 7`",
        "complete canonical v7 catalog",
        "## Offline v5 to v6 migration",
        "`semantic_flow_entities`, `semantic_flow_activities`, `semantic_flow_edges`, and `semantic_flow_label_assertions`",
        "`semantic_policy_epochs` is immutable",
        "`semantic_machine_settlements`, `semantic_machine_outcomes`, `semantic_review_labels`, and `semantic_health_events`",
        "singleton marker CAS `5 -> 6`",
        "complete canonical v6 storage catalog",
        "A v4 store is not accepted by `--to 6`",
        "## Offline v4 to v5 migration",
        "performs zero source/lease/sidecar writes",
        "`--expected-plan-sha256`",
        "`--sqlite-backup`",
        "`--postgres-snapshot-confirmed`",
        "compare-and-swaps the singleton marker from 4 to 5",
        "complete canonical v5 storage catalog validator",
        "both source and independent backup",
        "exact mode `0600`",
        "Dry-run does not chmod either file",
        "canonical UTC with six fractional digits",
        "## Backup and restore runbook",
        "SQLite's own backup command",
        "Runtime has released its active-store lease",
        "This must happen before shutdown",
        "`current_database()` plus the exact `current_schema()`",
        "`sqlite_master.type = 'table'`",
        "`pg_class.relkind = 'r'`",
        "requires the optional `postgres` dependency extra",
        "server whose major version is `17`",
        "PostgreSQL 17.10 is the tested, manifest-generation baseline",
        "require the complete backend canonical storage catalog",
        "The open-time validator does not compare raw DDL bytes",
        "complete canonical storage catalog captured for that backend",
        "PostgreSQL compares the pinned server major",
        "standalone functions, enum/base types, or domains",
        "complete canonical catalog comparison above still rejects drift",
        "`--schema=agent_libos_runtime`",
        "`--no-privileges`",
        "do not pre-create it",
        "Restore into the new target in one",
        "The expected output includes `ok` and `7`",
        "store schema version to equal `7` before opening the Runtime",
        "restored `server_version_num` to have major version 17",
        "server must be major version 17",
        "`SHOW server_version_num`",
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

    assert "does not compare column types" not in documentation
    assert "The expected output includes `ok` and `5`" not in documentation
    assert "schema version to equal `5` before opening the Runtime" not in documentation
    assert "store schema version to equal `5` before opening the Runtime" not in documentation
    assert "Planning a v5 or older store" not in documentation
    assert _read("docs/storage.md").count("SHOW server_version_num") >= 3


def test_current_evaluation_artifacts_do_not_name_stale_schema_v6_databases() -> None:
    for path in (
        "experiments/run_durable_task_run_evaluation.py",
        "experiments/run_knowledge_workflow_evaluation.py",
        "experiments/run_browser_customer_flow_evaluation.py",
    ):
        string_constants = {
            node.value
            for node in ast.walk(ast.parse(_read(path)))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert any("Runtime databases" in value for value in string_constants)
        assert not any("v6 Runtime databases" in value for value in string_constants)
        assert not any("v4 Runtime databases" in value for value in string_constants)

    benchmark_docs = _words(_read("benchmarks/durable_task_runs/README.md"))
    assert "retention v7 databases" in benchmark_docs
    assert "retention v6 databases" not in benchmark_docs
    assert "retention v4 databases" not in benchmark_docs


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


def test_architecture_docs_track_recovery_and_host_control_boundaries() -> None:
    documentation = _words(_read("docs/architecture.md"))
    recovery = documentation[
        documentation.index("Before the lifecycle becomes") : documentation.index(
            "Process/image/checkpoint publications"
        )
    ]

    for required in (
        "validates recoverable TaskRun plaintext and integrity bindings without dispatch",
        "Pending-effect reconciliation precedes stale capability-reservation abandonment",
        "provider receipt may prove an effect never started and restore its bound reservation",
        "reserves the maximum call/token envelope before Provider dispatch",
        "Public registry writes require configured `data_flow_sink_registry:*` admin authority",
        "Before `OPEN`, the trusted Host bootstrap may reconcile only the Host-configured rules without a process capability",
        "The Host-owned Python component graph does expose `semantic_control`",
        "trusted composition/control surface, not a remote or model-facing one",
        "bounded provider-response projection",
        "it is not the provider's raw response",
        "semantic/ Host semantic assessment, flow, control, settlement, and recovery",
        "closed event catalog",
    ):
        assert required in documentation

    _assert_in_order(
        recovery,
        "prepared protected operations",
        "pending external effects",
        "semantic authority",
        "stale capability-use reservations",
        "resource-usage reservations",
        "process-exec publications",
        "process-launch publications",
        "checkpoint-restore publications",
        "root-spawn initial-goal payloads",
        "missing volatile Object payloads",
        "registered JIT rehydration",
        "stale Explainable Operations",
        "stale process execution leases",
        "Object Tasks",
        "incomplete process-terminal cleanup intents",
        "TaskRun recovery",
    )
    assert "current 46-value event catalog" not in documentation


def test_core_runtime_docs_match_builder_startup_recovery_order() -> None:
    builder = _read("agent_libos/runtime/builder.py")
    builder_recovery = builder[
        builder.index("def _recover_runtime_state") : builder.index(
            "def _record_stale_execution_recovery"
        )
    ]
    builder_markers = (
        "validate_recoverable_payloads",
        "recovered_mcp_continuations",
        "recovered_mcp_remote_tasks",
        "recovered_mcp_subscriptions",
        "recovered_prepared_operations",
        "reconciled_external_effects",
        "recovered_semantic_authority",
        "recovered_capability_use_reservations",
        "recovered_resource_usage_reservations",
        "recovered_exec_publications",
        "recovered_runtime_publications",
        "recovered_checkpoint_restore_publications",
        "recovered_root_spawn_initial_goal_payloads",
        "recovered_missing_object_payloads",
        "rehydrate_registered_jit_tools",
        "recovered_stale_operations",
        "recovered_stale_executions",
        "recovered_object_tasks",
        "recovered_terminal_cleanups",
        "recovered_task_runs",
    )
    documentation_markers = (
        "recoverable TaskRun plaintext and integrity bindings",
        "MCP continuations",
        "MCP remote Tasks",
        "MCP subscriptions",
        "prepared protected operations",
        "pending external effects",
        "semantic authority",
        "stale capability-use reservations",
        "resource-usage reservations",
        "process-exec",
        "process-launch",
        "checkpoint-restore",
        "root-spawn initial-goal payloads",
        "missing volatile Object payloads",
        "registered JIT",
        "stale Explainable Operations",
        "stale process execution leases",
        "Object Tasks",
        "incomplete process-terminal cleanup intents",
        "TaskRun startup recovery",
    )

    _assert_in_order(builder_recovery, *builder_markers)
    for path, start_marker in (
        ("docs/runtime_model.md", "The builder first validates"),
        ("docs/storage.md", "While holding the lifecycle recovery lease"),
        ("docs/capabilities.md", "On reopen, the Runtime holds"),
    ):
        documentation = _words(_read(path)).split(start_marker, 1)[1]
        _assert_in_order(documentation, *documentation_markers)
        assert "provider receipt" in documentation
        assert "provider replay" in documentation
        assert "restore its bound reservation" in documentation


def test_semantic_docs_separate_evidence_control_settlement_and_remote_layers() -> None:
    python_api = _words(_read("docs/python_api.md"))
    threat_model = _words(_read("docs/threat_model.md"))

    for documentation in (python_api, threat_model):
        _assert_in_order(
            documentation,
            "Evidence and review",
            "Trusted Host kill switch and control",
            "Private settlement",
            "Remote and model boundary",
        )
        for required in (
            '`runtime.semantic.set_mode("off")`',
            "supported one-way live kill switch",
            "restart and startup admission",
            "semantic_control",
            "not Runtime facade methods",
            "no HTTP, GUI, model Tool, Skill, JIT",
            "`semantic review import`",
        ):
            assert required in documentation

    assert "only public evidence/review write" in python_api
    assert (
        "There is no Runtime/model/CLI/HTTP/GUI policy or settlement entrypoint"
        not in threat_model
    )


def test_runtime_docs_keep_prompt_layout_cache_and_projection_contract() -> None:
    documentation = _words(_read("docs/runtime_model.md"))
    thresholds = PromptCacheGateThresholds()

    for required in (
        "`prompt_mode` and the Host-selected `llm.prompt_layout` are separate prompt composition axes",
        "TaskRun supervision is the explicit transparency exception",
        "ordinary non-TaskRun `image_only` process",
        "every call remains subject to the applicable Capability, Task Authority, data-flow, budget, and policy checks",
        "Provider policy and durable external-effect settlement apply when the primitive crosses those respective boundaries",
        "bounded, redacted Provider-response projection",
        "It is not the complete Provider SDK response object",
        f"`llm.prompt_layout={DEFAULT_CONFIG.llm.prompt_layout}`",
        "`prompt_cache_mode=provider_default` sends no v2 `prompt_cache_options`",
        "Both opt-in modes require a nonempty Host-configured `prompt_cache_key`",
        "without a Run or process id",
        "removes the cache key, options, and all cache breakpoints as one group",
        "within the same logical protected LLM call",
        "separately record configured policy",
        "secret-free facts of the request that actually succeeded",
        "content-free `prompt_projection`",
        "strict gate rejects missing cache counters rather than accepting a numerical aggregate zero as complete evidence",
        "does not execute the prompt-cache release gate during dispatch",
        "paired `legacy_v1` and candidate arms",
        f"reduce uncached input by at least {thresholds.minimum_uncached_input_reduction:.0%}",
        f"total input by at least {thresholds.minimum_total_input_reduction:.0%}",
        "`legacy_v1` remains the default and rollback layout",
    ):
        assert required in documentation


def test_repository_guidelines_keep_semantic_and_prompt_cache_boundaries() -> None:
    documentation = _words(_read("AGENTS.md"))

    for required in (
        "`ports/`, `sdk/`, `semantic/`, `storage/`",
        "boundaries applicable to their effect class",
        "only trusted Host bootstrap/reconciliation before Runtime OPEN may bypass a process Capability",
        "Semantic policy control is a Host-composition surface reachable from the local Runtime Python object",
        "Treat prompt-cache v2 as an explicit Host opt-in",
        "same-logical-call compatibility downgrade",
        "configured-versus-effective evidence",
        "paired multi-provider release gate",
        "reconcile prepared/pending effects before abandoning stale Capability reservations",
        "Provider reconciliation may restore an effect-bound Capability reservation",
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


def test_object_docs_cover_payload_bounds_and_paged_model_reads() -> None:
    documentation = _words(_read("docs/object_memory.md"))
    tools = DEFAULT_CONFIG.tools

    for required in (
        "Object payloads have a separate bounded-JSON contract",
        "Cycles, non-finite numbers such as `NaN`/infinity",
        "Create and payload replacement apply `tools.memory_payload_hard_limit_bytes`",
        "Append applies `tools.memory_append_entry_max_bytes`",
        (
            f"current defaults are {tools.memory_payload_hard_limit_bytes:,} bytes "
            f"per complete payload and {tools.memory_append_entry_max_bytes:,} bytes "
            "per append entry"
        ),
        "A failed create, replacement, or append does not publish a partial Object change",
        "`json_pointer` is an RFC 6901 pointer",
        "`representation=json_value`",
        f"default is {tools.memory_payload_chars:,} characters",
        "`representation=canonical_json_page`",
        "`page_offset_bytes` and `next_cursor` are UTF-8 byte offsets",
        "positive cursor must include the preceding page's `sha256` as `expected_sha256`",
        "cursor is relative to the JSON selected by that exact `json_pointer`",
    ):
        assert required in documentation
