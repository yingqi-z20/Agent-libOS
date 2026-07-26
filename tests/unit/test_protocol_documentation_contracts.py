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
    assert "one absolute deadline across the live exchange" in mcp
    assert "certificate is phase-local" in mcp


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
