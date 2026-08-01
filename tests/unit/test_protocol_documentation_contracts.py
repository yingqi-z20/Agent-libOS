from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _words(value: str) -> str:
    return " ".join(value.split())


def test_jsonrpc_docs_keep_retry_and_dns_phase_contracts() -> None:
    documentation = _words(_read("docs/jsonrpc.md"))
    skill = _words(
        _read("agent_libos/skills/builtin/agent-libos-jsonrpc/SKILL.md")
    )

    assert (
        "only when the current address fails before request dispatch starts"
        in documentation
    )
    assert "at most one attempt enters request dispatch" in documentation
    assert (
        "after any exception during connect, TLS, request write"
        not in documentation
    )
    assert "DNS resolution is not local preflight" in skill
    assert "the first protected information-flow provider phase" in skill
    assert "never tries another address after a write" in skill


def test_mcp_docs_keep_transport_and_provider_bounds_distinct() -> None:
    documentation = _words(_read("docs/mcp.md"))
    skill = _words(_read("agent_libos/skills/builtin/agent-libos-mcp/SKILL.md"))

    for required in (
        "maximum nesting depth is 128",
        "maximum node count is `min(100,000, max_response_bytes)`",
        "aggregate UTF-8 bytes across all string values and mapping keys",
        "`config.mcp.list_limit` tools",
        "cannot exceed `max_response_bytes` or under-report",
        "`McpSubprocessLimitsProvider`",
        "`supports_runtime_environment_snapshots = True`",
        "`supports_executable_snapshots = True`",
    ):
        assert required in documentation

    assert (
        "Raw transport-limit failures are instead `transport_error`"
        in documentation
    )
    assert (
        "Raw stdio frame/stdout or HTTP body/SSE-frame overflow is `transport_error`"
        in skill
    )
    assert "depth 128" in skill
    assert "`SubprocessLimits`" in skill


def test_workspace_skills_keep_canonical_path_and_readback_contracts() -> None:
    navigation = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-workspace-navigation/SKILL.md"
        )
    )
    editing = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-workspace-editing/SKILL.md"
        )
    )

    assert "absolute paths are rejected" in navigation
    assert "on POSIX a backslash is an ordinary filename character" in navigation
    assert "canonical identity is stored exactly" in navigation
    assert "absolute paths and unknown input fields are rejected" in editing
    assert "write argument limit can be larger than the read" in editing
    assert "Establish a viable verification route before mutation" in editing


def test_shell_skill_keeps_character_and_cancellation_contracts() -> None:
    skill = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-command-execution/SKILL.md"
        )
    )

    assert "hard limits on decoded characters" in skill
    assert "Caller cancellation does not kill or abandon" in skill
    assert "shields and joins that worker" in skill


def test_echo_and_pytest_skills_keep_strict_bounded_protocol_contracts() -> None:
    echo = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-tool-protocol-diagnostics/SKILL.md"
        )
    )
    pytest_log = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-test-log-analysis/SKILL.md"
        )
    )

    assert "top-level strict JSON object" in echo
    assert "byte-limited before parsing" in echo
    assert "duplicate keys, nonstandard `NaN`/infinity values" in echo
    assert "Unknown fields are rejected" in pytest_log
    assert "itself has no cursor, offset, or paging contract" in pytest_log


def test_jit_skill_keeps_strict_expected_and_interruption_contracts() -> None:
    skill = _words(
        _read(
            "agent_libos/skills/builtin/agent-libos-jit-tool-authoring/SKILL.md"
        )
    )

    assert "Every test object must include `expected`" in skill
    assert "the `tests` list itself may be empty" in skill
    assert "finite integers and floats are one JSON number type" in skill
    assert "booleans are distinct from numbers" in skill
    assert "A missing `expected` rejects before source execution" in skill
    assert "recorded as a rejected candidate" in skill
    assert "do not revalidate the interrupted candidate" in skill


def test_skill_compatibility_is_opaque_metadata_not_a_runtime_gate() -> None:
    documentation = _words(_read("docs/skills.md"))

    assert "validated only as a non-empty string" in documentation
    assert "does not parse Semantic Versioning" in documentation
    assert "reject registration or activation" in documentation
    assert "Host or package-release workflow is responsible" in documentation
    assert "not a Runtime-enforced version gate" in documentation


def test_swe_skill_docs_keep_native_child_boundary_and_dependency_contracts() -> None:
    documentation = _words(_read("docs/skills.md"))
    skill = _words(_read("skills/swe-agent/SKILL.md"))

    for text in (documentation, skill):
        assert "does not re-enter" in text
        assert "not an operating-system sandbox" in text
        assert "container, WASM, VM, service" in text
        assert "safe PATH" in text

    assert "Skill activation validates the JIT package" in documentation
    assert "activating this Skill does not prove" in skill


def test_remote_skills_keep_deadline_pagination_and_phase_local_contracts() -> None:
    jsonrpc = _words(
        _read("agent_libos/skills/builtin/agent-libos-jsonrpc/SKILL.md")
    )
    mcp = _words(_read("agent_libos/skills/builtin/agent-libos-mcp/SKILL.md"))

    assert "one absolute deadline shared by DNS resolution" in jsonrpc
    assert "The certificate is phase-local" in jsonrpc
    assert "no phase gets a fresh timeout" in jsonrpc
    assert "v1 live `tools/list` is deliberately unpaginated" in mcp
    assert "continuation cursors are neither exposed nor followed" in mcp
    assert "a non-empty MCP `nextCursor` is rejected as an incomplete catalog" in _words(
        _read("docs/mcp.md")
    )
    assert "one absolute deadline across the live exchange" in mcp
    assert "certificate is phase-local" in mcp


def test_remote_manifest_docs_keep_explicit_null_rollback_default() -> None:
    for path in ("docs/jsonrpc.md", "docs/mcp.md"):
        documentation = _words(_read(path))
        assert (
            "An omitted `rollback_status` and an explicit YAML/JSON `null` "
            "have the same meaning"
        ) in documentation
        assert "A supplied non-null `rollback_status` is preserved" in documentation


def test_durable_task_run_docs_keep_create_and_auto_run_identities_separate() -> None:
    documentation = _words(_read("docs/durable_task_runs.md"))

    for required in (
        "For an existing-Run mutation, `expected_revision` is part of the canonical request identity",
        "canonical create identity contains only that request id and the complete spec",
        "it has no `expected_revision`",
        "does not add `auto_run` to the create identity",
        "separate deterministic `<client_request_id>:run` command",
        "reconstructed from the immutable create receipt",
        "raises a conflict rather than binding the old intent to the current revision",
        "matching pending run receipt follows the local-only replay rules",
    ):
        assert required in documentation


def test_durable_task_run_docs_keep_linked_recovery_gap_local_and_bound() -> None:
    documentation = _words(_read("docs/durable_task_runs.md"))

    for required in (
        "nested receipt's hidden parent command/request-hash binding",
        "target create receipt and Run row",
        "unique `rerun_of` link",
        "does not invoke a fresh rerun or create a second Run",
        "Startup likewise does not synthesize a missing outer linked-recovery receipt",
        "Only an exact retry of that original recover request",
        "current source state is never used to rebind the old command",
    ):
        assert required in documentation


def test_durable_task_run_docs_keep_typed_stale_interrupt_fences_exact() -> None:
    documentation = _words(_read("docs/durable_task_runs.md"))

    for required in (
        "Store-only typed `StaleExecutionProcessWait` receipt",
        "recovering Runtime owner-id SHA-256",
        "not a cryptographic signature",
        "generic process API omits the currently active process execution-owner or lease fields",
        "historical recovering-owner hash",
        "`(pid, admission_state_generation, admission_execution_generation)` fence",
        "contains each PID exactly once",
        "Repeating a PID with the same or different generation values makes the receipt malformed",
        "may not collapse such entries through a map overwrite",
        "current Runtime epoch must be later than admission",
        "`process.task_run_epoch == run.runtime_epoch == current Runtime epoch`",
        "`0 < point.task_run_epoch <= process.task_run_epoch`",
        "state-generation plus two and execution-generation plus one",
        "terminal member must be at admitted state-generation plus one",
        "supported event/Human/tool wait with a typed wait and its pending action",
        "Checkpoint restore and checkpoint fork both deliberately degrade",
        "clear the `stale_execution_recovery` compatibility `status_message`",
    ):
        assert required in documentation
    assert "Store signer" not in documentation


def test_durable_task_run_docs_keep_strict_command_receipt_provenance() -> None:
    documentation = _words(_read("docs/durable_task_runs.md"))

    for required in (
        "canonical strict version-1 object",
        "complete public `TaskRunSummary` mapping must bind the command row's Run and `result_revision`",
        "each command/settlement variant accepts its exact key set only",
        "pending-only fields on a completed variant",
        "outside the signed BIGINT range",
        "raw UTF-8 `result_json` before JSON decoding",
        "canonical request's server-issued `option_id` selects",
        "stored result keys cannot reclassify that request",
        "`admission_ledger_seq`, `admission_ledger_item_id`, and `admission_evidence_sha256`",
        "append-only `STATUS_TRANSITION` ledger item written in the same admission transaction",
        "Run, command id/kind, canonical request hash, from/to status",
        "Every terminal or superseded early-return path validates",
        "`interrupt_provenance_sha256` binds its positive `admission_runtime_epoch`",
        "completed receipt removes those raw pending-only fields",
        "global `task_run_runtime_epoch` counter-row lock",
        "missing outer receipt in the linked-recovery gap",
        "`settlement_transition_seq`, and `settlement_audit_record_id`",
        "`external_effect.recovery_settled` decision from `host_verified_receipt`",
        "remains usable after provider metadata/receipt bodies are purged",
    ):
        assert required in documentation

    providers = _words(_read("docs/providers.md"))
    for required in (
        "exact finalized external-effect transition sequence",
        "matching append-only `external_effect.recovery_settled` audit record",
        "source is `host_verified_receipt`",
        "remains available after terminal retention removes readable provider metadata and receipt bodies",
        "replay never depends on purgeable content",
    ):
        assert required in providers

    threat = _words(_read("docs/threat_model.md"))
    for required in (
        "strict version-1 result has an exact variant schema",
        "raw pre-decode byte cap, and signed BIGINT bounds",
        "same-transaction append-only status-transition ledger item",
        "Terminal or superseded early returns validate that evidence",
        "`host_verified_receipt` audit, which remain after provider-body purge",
        "global Runtime-epoch counter-row lock",
    ):
        assert required in threat


def test_stale_execution_receipt_docs_keep_projection_and_transfer_limits() -> None:
    checkpoints = _words(_read("docs/checkpoints.md"))
    threat = _words(_read("docs/threat_model.md"))
    gui = _words(_read("docs/gui.md"))
    cli = _words(_read("docs/cli.md"))
    python_api = _words(_read("docs/python_api.md"))

    for required in (
        "hash-and-generation receipt is diagnostic state, not transferable resume authority",
        "Before either restore or fork publishes a new process concurrency identity",
        "`PausedProcessWait(reason_oid=None)`",
        "detached from any captured TaskRun binding",
        "non-transferable stale-execution receipt instead keeps the conservative ordinary paused posture",
    ):
        assert required in checkpoints

    for required in (
        "Store-reserved typed `StaleExecutionProcessWait`",
        "never the `stale_execution_recovery` `status_message`",
        "hash is not a cryptographic signature",
        "interrupt admission Runtime epoch and exact per-PID state/execution-generation fence",
        "identity- and integrity-bound complete safe point",
        "Checkpoint restore and fork replace the non-transferable receipt",
    ):
        assert required in threat

    for required in (
        "`stale_execution` branch is a diagnostic projection",
        "not prior raw owner/lease tokens or TaskRun epoch, safe-point, or live-binding evidence",
        "must not infer that a Task Run is resumable",
        "Run controls continue to follow the server's `allowed_actions`",
        "presentation-only compatibility text",
    ):
        assert required in gui

    for required in (
        "`processes` and `resources` responses",
        "resulting process state from `exit`",
        "canonical `wait_state`, `outcome`, and `state_generation` fields",
        "diagnostic evidence, not client-held permission to resume",
        "must not parse `status_message`",
    ):
        assert required in cli

    assert "`StaleExecutionProcessWait`" in python_api
    for required in (
        "Store-only recovery receipt",
        "must not construct or submit one as an ordinary transition",
        "TaskRun epoch, safe-point integrity, and current binding evidence",
    ):
        assert required in python_api


def test_image_and_checkpoint_skills_keep_recovery_boundaries() -> None:
    images = _words(
        _read("agent_libos/skills/builtin/agent-libos-agent-images/SKILL.md")
    )
    checkpoints = _words(
        _read("agent_libos/skills/builtin/agent-libos-checkpoints/SKILL.md")
    )

    assert "mutation-only publication tools cannot inspect the registry" in images
    assert "one-shot target-image read remains consumed" in images
    assert "data-flow labels, including tenant boundaries" in images
    assert "not a stable snapshot token" in checkpoints
    assert "mutates the global image registry" in checkpoints
    assert "Release finalizers may themselves create external effects" in checkpoints
    assert "`checkpoint_restore_receipt`" in checkpoints


def test_checkpoint_image_docs_keep_direct_and_skill_projection_distinct() -> None:
    documentation = _words(_read("docs/checkpoints.md"))

    assert (
        "Without `metadata.tool_projection: skills`, a custom or committed Image "
        "that includes the wrapper in `default_tools` projects it directly"
    ) in documentation
    assert "Skill activation is not required" in documentation
    assert (
        "With the `skills` projection, the Image must bind the complete immutable "
        "`agent-libos-agent-images` tool set"
    ) in documentation
    assert "every call still enforces the same checkpoint/image authority" in documentation
