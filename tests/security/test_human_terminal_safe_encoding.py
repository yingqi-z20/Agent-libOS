from __future__ import annotations

import sys
import unicodedata
from typing import Any

from agent_libos.human.manager import HumanObjectManager
from agent_libos.models import HumanRequest, HumanRequestStatus


_ATTACK = (
    "plain-ascii\nFORGED APPROVAL: yes\rRETURN-OVERWRITE"
    "\x00\x07\x08\x09\x0b\x0c\x1b]0;owned\x07"
    "\x7f\x80\x85\x9b31mCSI\x9dtitle\x9c"
    "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
    "\u2066\u2067\u2068\u2069\u200bZERO-WIDTH\u2060WORD-JOINER"
    "\ufeffBOM\u2028LINE\u2029PARAGRAPH\ud800LONE-SURROGATE"
)
_LEGACY_CONTEXT_ONLY = "LEGACY-CONTEXT-MUST-NOT-RENDER"
_CONTENT_BODY_ONLY = "CONTENT-BODY-MUST-NOT-RENDER"


def _request(payload: dict[str, Any]) -> HumanRequest:
    return HumanRequest(
        request_id="hreq-terminal-safe",
        pid="pid-terminal-safe",
        human="owner",
        payload=payload,
        status=HumanRequestStatus.PENDING,
        decision=None,
        blocking=True,
        created_at="2026-08-10T00:00:00Z",
        updated_at="2026-08-10T00:00:00Z",
    )


def _manager() -> HumanObjectManager:
    # Formatting is deliberately pure for these request variants.  Avoid a
    # Runtime/store fixture so the security regression test stays focused on
    # the terminal presentation boundary.
    return object.__new__(HumanObjectManager)


def _assert_no_untrusted_terminal_controls(rendered: str) -> None:
    assert "\nFORGED APPROVAL" not in rendered
    assert "\r" not in rendered
    for character in rendered:
        if character == "\n":  # Host-owned presentation layout is retained.
            continue
        assert unicodedata.category(character) not in {
            "Cc",
            "Cf",
            "Cs",
            "Zl",
            "Zp",
        }

    # The attack remains inspectable instead of disappearing or becoming an
    # executable terminal sequence.
    assert r"\nFORGED APPROVAL" in rendered
    assert r"\rRETURN-OVERWRITE" in rendered
    assert r"\x1b]0;owned\a" in rendered
    assert r"\x9b31mCSI" in rendered
    assert r"\x9dtitle\x9c" in rendered
    assert r"\u202e" in rendered
    assert r"\u200bZERO-WIDTH\u2060WORD-JOINER" in rendered
    assert r"\ufeffBOM" in rendered
    assert r"\u2028LINE\u2029PARAGRAPH" in rendered
    assert r"\ud800LONE-SURROGATE" in rendered


def test_question_terminal_fields_cannot_forge_lines_or_control_display() -> None:
    rendered = _manager().format_terminal_request(
        _request(
            {
                "type": "question",
                "question": _ATTACK,
                "context": {_ATTACK: _ATTACK},
            }
        )
    )

    assert rendered.count("\n") == 2
    assert "\nContext:\n" in rendered
    _assert_no_untrusted_terminal_controls(rendered)


def test_permission_terminal_fields_use_the_same_safe_encoding() -> None:
    rendered = _manager().format_terminal_request(
        _request(
            {
                "type": "permission_request",
                "question": _ATTACK,
                "context": {
                    "reason": _ATTACK,
                    "resource": _ATTACK,
                    "canonical_resource": _ATTACK,
                    "lease": {
                        "type": _ATTACK,
                        "choices": [_ATTACK],
                    },
                    "constraints": {
                        "authority_rules": [
                            {
                                "rule_id": _ATTACK,
                                "effect": _ATTACK,
                                "risk": _ATTACK,
                                "conditions": {"dynamic": _ATTACK},
                            }
                        ]
                    },
                },
                "requested_permission": {
                    "resource": _ATTACK,
                    "rights": [_ATTACK],
                },
            }
        )
    )

    assert rendered.startswith("Permission request details:\n")
    assert "\n- constraints:\n" in rendered
    assert "\n- requested policy target:\n" in rendered
    _assert_no_untrusted_terminal_controls(rendered)


def test_data_release_terminal_fields_use_the_same_safe_encoding() -> None:
    rendered = _manager().format_terminal_request(
        _request(
            {
                "type": "data_release_approval",
                "question": _ATTACK,
                "context": {
                    "sink": _ATTACK,
                    "tenant": _ATTACK,
                    "principal": _ATTACK,
                    "operation": _ATTACK,
                },
            }
        )
    )

    assert rendered.startswith("Data release details:\n")
    assert rendered.count("\n") == 5
    _assert_no_untrusted_terminal_controls(rendered)


class _UnsafePreview:
    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": 0,
            "pid": _ATTACK,
            "action_id": "filesystem.read",
            "resource_display": _ATTACK,
            "resource_sha256": "d" * 64,
            "rights": ["read", _ATTACK],
            "risk": "low",
            "effect_id": _ATTACK,
            "canonical_args_sha256": "a" * 64,
            "argument_projection": {
                "kind": "shell",
                "operation": _ATTACK,
                "display_argv": ["safe-tool", _ATTACK],
                "argv_count": 2,
                "argv_truncated": False,
                "argv_sha256": "b" * 64,
                "safe_cwd": "<workspace>",
                "cwd_sha256": "c" * 64,
                # A fake producer cannot make fields outside the typed variant
                # visible even though this formatter still encodes all values.
                "policy_reason": _LEGACY_CONTEXT_ONLY,
                "content_preview": _CONTENT_BODY_ONLY,
            },
            "target_state_sha256": _ATTACK,
            "expires_at": _ATTACK,
            "source_labels": {
                "sensitivity": _ATTACK,
                "integrity": _ATTACK,
                "trust_level": _ATTACK,
                "identity_present": False,
                "identity_mixed": False,
            },
        }

    def canonical_sha256(self) -> str:
        return _ATTACK


def test_external_preview_and_host_evidence_use_the_same_safe_encoding() -> None:
    manager = _manager()
    manager.canonical_approval_preview = lambda _request: _UnsafePreview()  # type: ignore[method-assign]
    rendered = manager.format_terminal_request(
        _request(
            {
                "type": "external_operation_approval",
                "question": _ATTACK,
                "context": {
                    "path": _LEGACY_CONTEXT_ONLY,
                    "argv": [_LEGACY_CONTEXT_ONLY],
                    "command": _LEGACY_CONTEXT_ONLY,
                    "policy_reason": _LEGACY_CONTEXT_ONLY,
                    "matched_rule": _LEGACY_CONTEXT_ONLY,
                    "risk": _LEGACY_CONTEXT_ONLY,
                    "sandbox_profile": {"operation": _LEGACY_CONTEXT_ONLY},
                    "target": {"kind": _LEGACY_CONTEXT_ONLY},
                    "content_preview": _CONTENT_BODY_ONLY,
                },
                "requested_once_capability": {
                    "resource": _LEGACY_CONTEXT_ONLY,
                    "rights": [_LEGACY_CONTEXT_ONLY],
                },
            }
        )
    )

    assert rendered.startswith("Canonical operation approval preview:\n")
    projection_label = "\nHost-bound canonical argument projection:\n"
    assert projection_label in rendered
    assert "\n- kind: shell\n" in rendered
    assert "\n- display argv: ['safe-tool', 'plain-ascii\\nFORGED" in rendered
    assert _LEGACY_CONTEXT_ONLY not in rendered
    assert _CONTENT_BODY_ONLY not in rendered
    assert "policy reason" not in rendered.lower()
    assert "matched rule" not in rendered.lower()
    assert "content preview" not in rendered.lower()
    assert "Requester-provided rationale" not in rendered
    _assert_no_untrusted_terminal_controls(rendered)


def test_git_reference_terminal_rows_are_role_bound_and_do_not_echo_secret_refs() -> None:
    secret = "refs/heads/ghp_SECRET_TERMINAL_SENTINEL_abcdefghijkl"

    class _GitPreview:
        def to_dict(self) -> dict[str, Any]:
            return {
                "revision": 0,
                "pid": "pid-safe",
                "action_id": "git.write",
                "resource_display": "git_remote:origin",
                "resource_sha256": "a" * 64,
                "rights": ["write"],
                "risk": "high",
                "effect_id": "effect-safe",
                "canonical_args_sha256": "b" * 64,
                "argument_projection": {
                    "kind": "git",
                    "operation": "push",
                    "worktree_id": "main",
                    "worktree_id_sha256": "c" * 64,
                    "source_args_sha256": "d" * 64,
                    "git_references": [
                        {
                            "role": "local_ref",
                            "display": "refs/heads/main",
                            "sha256": "e" * 64,
                        },
                        {
                            "role": "remote_ref",
                            "display": "<redacted>",
                            "sha256": "f" * 64,
                        },
                    ],
                    "git_fact_tokens": ["delete=false"],
                },
                "target_state_sha256": None,
                "expires_at": None,
                "source_labels": {
                    "sensitivity": "normal",
                    "integrity": "verified",
                    "trust_level": "trusted",
                    "identity_present": True,
                    "identity_mixed": False,
                },
            }

        def canonical_sha256(self) -> str:
            return "1" * 64

    manager = _manager()
    manager.canonical_approval_preview = lambda _request: _GitPreview()  # type: ignore[method-assign]
    rendered = manager.format_terminal_request(
        _request(
            {
                "type": "external_operation_approval",
                "question": secret,
                "context": {"remote_ref": secret},
            }
        )
    )

    assert "Git reference local_ref: refs/heads/main" in rendered
    assert "Git reference remote_ref: <redacted>" in rendered
    assert secret not in rendered


def test_every_unicode_format_control_is_encoded() -> None:
    format_controls = "".join(
        chr(codepoint)
        for codepoint in range(sys.maxunicode + 1)
        if unicodedata.category(chr(codepoint)) == "Cf"
    )
    assert format_controls

    rendered = _manager().format_terminal_request(
        _request(
            {
                "type": "question",
                "question": f"before{format_controls}after",
            }
        )
    )

    assert rendered.startswith("before")
    assert rendered.endswith("after")
    assert all(character not in rendered for character in format_controls)
    assert r"\u200b" in rendered
    assert r"\u2060" in rendered
    assert r"\ufeff" in rendered
    if unicodedata.category(chr(0xE0001)) == "Cf":
        assert r"\U000e0001" in rendered


def test_normal_ascii_terminal_fields_remain_readable() -> None:
    rendered = _manager().format_terminal_request(
        _request(
            {
                "type": "question",
                "question": "Review the quarterly report?",
                "context": {"owner": "alice", "priority": "normal"},
            }
        )
    )

    assert rendered == (
        "Review the quarterly report?\n"
        "Context:\n"
        "- owner: 'alice'\n"
        "- priority: 'normal'"
    )
