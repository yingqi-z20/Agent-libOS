from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_release_status_has_one_mcp_server_exclusion_clause() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")

    assert text.count("and an MCP server surface remain out of scope.") == 1


def test_release_status_live_task_run_command_has_one_repetition_option() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    command = text.split(
        "uv run --env-file .env python experiments/run_durable_task_run_evaluation.py",
        1,
    )[1].split("\n\n", 1)[0]

    assert command.count("--repetitions 3") == 1


def test_release_status_keeps_mrtr_out_of_v1_v2_tools_compatibility() -> None:
    text = (ROOT / "docs" / "release_status.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert (
        "Manifests v1/v2 remain governed Tools-only compatibility surfaces"
        in normalized
    )
    assert (
        "A typed MRTR continuation is available only to Manifest v3" in normalized
    )
    assert "and an MRTR input request is non-retryable" not in normalized


def test_mcp_docs_lock_manifest_client_info_identities() -> None:
    text = (ROOT / "docs" / "mcp.md").read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "| v1 legacy wire | `mcp` | `0.1.0` |" in normalized
    assert (
        "| v2 governed Tools compatibility | `agent-libos` | `1.4.2` |"
        in normalized
    )
    assert (
        "| v3 exact `2026-07-28` | `agent-libos` | `1.5.0` |"
        in normalized
    )
    assert "v1 and v2 values are frozen compatibility identities" in normalized
