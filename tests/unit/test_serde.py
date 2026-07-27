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


@pytest.mark.parametrize(
    "payload",
    [
        b'"\xff"',
        bytearray(b'"\xc3("'),
        "\ud800",
    ],
)
def test_bounded_json_loads_rejects_non_utf8_input_with_stable_value_error(
    payload: str | bytes | bytearray,
) -> None:
    with pytest.raises(ValueError, match="JSON input must be valid UTF-8"):
        bounded_json_loads(payload)


@pytest.mark.parametrize("payload", ["NaN", "Infinity", "-Infinity", "1e1000000"])
def test_bounded_json_loads_rejects_non_finite_numbers(payload: str) -> None:
    with pytest.raises(ValueError, match="JSON numbers must be finite"):
        bounded_json_loads(payload)


def test_bounded_json_loads_rejects_extreme_depth_without_recursion_error() -> None:
    payload = ("[" * 10_000) + "0" + ("]" * 10_000)

    with pytest.raises(ValueError, match="JSON nesting exceeds maximum depth=256"):
        bounded_json_loads(payload)


def test_bounded_json_loads_rejects_duplicate_object_keys_by_default() -> None:
    with pytest.raises(ValueError, match="duplicate keys"):
        bounded_json_loads('{"outer":{"value":1,"value":2}}')


def test_bounded_json_loads_requires_explicit_legacy_duplicate_key_mode() -> None:
    assert bounded_json_loads(
        '{"value":1,"value":2}',
        reject_duplicate_keys=False,
    ) == {"value": 2}

    with pytest.raises(ValueError, match="must be a boolean"):
        bounded_json_loads("{}", reject_duplicate_keys=1)  # type: ignore[arg-type]


def test_bounded_json_loads_enforces_utf8_byte_limit() -> None:
    payload = '{"value":"猫"}'
    encoded_size = len(payload.encode("utf-8"))

    assert bounded_json_loads(payload, max_bytes=encoded_size) == {"value": "猫"}
    with pytest.raises(ValueError, match="max_bytes"):
        bounded_json_loads(payload, max_bytes=encoded_size - 1)
