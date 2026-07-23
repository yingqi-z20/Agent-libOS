from __future__ import annotations

import json

import pytest

from agent_libos.utils.serde import bounded_json_loads


@pytest.mark.parametrize(
    "payload",
    [
        ("[" * 257) + "0" + ("]" * 257),
        ('{"value":' * 257) + "0" + ("}" * 257),
    ],
)
def test_bounded_json_loads_rejects_excessive_container_depth(payload: str) -> None:
    with pytest.raises(ValueError, match="JSON nesting exceeds maximum depth=256"):
        bounded_json_loads(payload)


def test_bounded_json_loads_accepts_container_depth_at_limit() -> None:
    decoded = bounded_json_loads(("[" * 256) + "0" + ("]" * 256))

    for _ in range(256):
        assert isinstance(decoded, list)
        assert len(decoded) == 1
        decoded = decoded[0]
    assert decoded == 0


def test_bounded_json_loads_does_not_count_brackets_inside_strings() -> None:
    value = '[[[{{{"}}}]]]'

    assert bounded_json_loads(json.dumps({"value": value})) == {"value": value}


def test_bounded_json_loads_rejects_oversized_integers() -> None:
    with pytest.raises(ValueError, match="JSON integer exceeds maximum digits=4300"):
        bounded_json_loads("9" * 4_301)
