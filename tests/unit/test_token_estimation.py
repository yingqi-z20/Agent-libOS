from __future__ import annotations

from agent_libos.llm.context_management import estimate_multilingual_tokens
from agent_libos.utils.ids import estimate_tokens


def test_shared_estimator_is_conservative_for_ascii_cjk_and_mixed_text() -> None:
    ascii_text = "abcdefghijkl"
    cjk_text = "你好世界"
    mixed_text = "cache 前缀稳定"

    assert estimate_tokens(ascii_text) == 4
    assert estimate_tokens(cjk_text) == 4
    assert estimate_tokens(mixed_text) == 2 + 4

    # Object/context materialization calls estimate_tokens directly. Request
    # pressure must never apply a larger estimate to the same prompt text.
    assert estimate_tokens(mixed_text) >= estimate_multilingual_tokens(mixed_text)


def test_estimator_canonicalizes_structured_values_without_repr_addresses() -> None:
    first = {"cjk": "你好", "ascii": ["abc", 1], "empty": None}
    second = {"empty": None, "ascii": ["abc", 1], "cjk": "你好"}

    assert estimate_tokens(first) == estimate_tokens(second)


def test_estimator_handles_empty_cyclic_and_non_json_values_stably() -> None:
    class NonJsonValue:
        def __repr__(self) -> str:
            raise AssertionError("token estimation must not call repr")

        def model_dump(self, **_: object) -> object:
            raise LookupError("not serializable")

    cyclic: list[object] = []
    cyclic.append(cyclic)

    assert estimate_tokens("") == 1
    assert estimate_tokens(None) > 0
    assert estimate_tokens(cyclic) > 0
    assert estimate_tokens(NonJsonValue()) == estimate_tokens(NonJsonValue())
