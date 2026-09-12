from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_libos.llm.client import LLMCompletion
from agent_libos.sdk.protected_operations import (
    _post_commit_result_identity,
    visit_bounded_host_result_text,
)
from agent_libos.tools.observability import sanitize_for_observability
from agent_libos.utils.serde import dumps, to_jsonable


def test_generic_serializer_excludes_nested_explicit_private_dataclass_fields() -> None:
    @dataclass
    class PrivateReplay:
        visible: str = "public"
        hidden: str = field(default="CIPHERTEXT_SECRET", metadata={"serialize": False})
        quiet: str = field(default="still serializable", repr=False)

    @dataclass
    class Envelope:
        replay: Any

    value = Envelope(replay={"nested": [PrivateReplay()]})

    assert to_jsonable(value) == {
        "replay": {"nested": [{"visible": "public", "quiet": "still serializable"}]},
    }
    assert "CIPHERTEXT_SECRET" not in dumps(value)


@pytest.mark.parametrize(
    "private_key",
    [
        "encrypted_content", "encryptedContent", "ciphertext", "cipher_text",
        "response_items", "responses_items", "responseItems", "responses-items",
    ],
)
def test_observability_never_previews_ciphertext_keys(private_key: str) -> None:
    result = sanitize_for_observability(
        {"nested": [{private_key: "CIPHERTEXT_SECRET"}]},
        preview_chars=10_000,
    )

    assert result["redacted"] is True
    assert "CIPHERTEXT_SECRET" not in json.dumps(result)


def test_completion_private_replay_is_excluded_from_generic_serialization() -> None:
    completion = LLMCompletion(
        content="public result",
        tool_calls=[],
        response_items=[{"type": "reasoning", "encrypted_content": "REPLAY_CIPHERTEXT"}],
        raw={"output": [{"type": "reasoning", "encrypted_content": "RAW_CIPHERTEXT"}]},
    )

    serialized = dumps(completion)

    assert "public result" in serialized
    assert "REPLAY_CIPHERTEXT" not in serialized
    assert "RAW_CIPHERTEXT" not in serialized
    assert "response_items" not in serialized


def test_private_replay_does_not_enter_protected_result_identity_or_text() -> None:
    first = LLMCompletion(
        content="public result",
        tool_calls=[],
        response_items=[{"type": "reasoning", "encrypted_content": "REPLAY_CIPHERTEXT"}],
    )
    second = LLMCompletion(content="public result", tool_calls=[])
    visited: list[str | bytes] = []

    first_identity = _post_commit_result_identity(first, contract_name="primitive.llm.complete")
    second_identity = _post_commit_result_identity(second, contract_name="primitive.llm.complete")
    visit_bounded_host_result_text(
        first,
        contract_name="primitive.llm.complete",
        visitor=visited.append,
    )

    assert first_identity == second_identity
    assert first_identity[0] is not None
    assert "REPLAY_CIPHERTEXT" not in repr(visited)
