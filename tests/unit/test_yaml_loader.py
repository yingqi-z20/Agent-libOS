from __future__ import annotations

import asyncio
import sys

import pytest

from agent_libos.models.exceptions import ValidationError
from agent_libos.utils import yaml_loader
from agent_libos.utils.yaml_loader import (
    YAML_MAX_ALIASES,
    YAML_MAX_EXPANDED_SCALAR_BYTES,
    YAML_MAX_INTEGER_DIGITS,
    YAML_MAX_NESTING_DEPTH,
    YAML_MAX_NODES,
    YAML_MAX_PARSE_EVENTS,
    YAML_MAX_UTF8_BYTES,
    load_yaml_mapping,
)


class _YamlLoadCancelled(BaseException):
    pass


class TestYamlLoader:
    def test_empty_values_use_yaml_null_semantics(self) -> None:
        data = load_yaml_mapping(
            """
empty:
items:
  -
  - name: first
    optional:
inline: {left:, right: [1, true, null]}
""".lstrip()
        )

        assert data["empty"] is None
        assert data["items"] == [None, {"name": "first", "optional": None}]
        assert data["inline"] == {"left": None, "right": [1, True, None]}

    def test_duplicate_mapping_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate YAML key"):
            load_yaml_mapping(
                """
name: first
name: second
""".lstrip()
            )

    def test_duplicate_nested_mapping_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate YAML key"):
            load_yaml_mapping(
                """
items:
  - name: first
    name: second
""".lstrip()
            )

    def test_duplicate_inline_mapping_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate YAML key"):
            load_yaml_mapping("metadata: {role: review, role: audit}\n")

    def test_utf8_byte_limit_is_enforced_before_parsing(self) -> None:
        prefix = "value: "
        text = prefix + "é" * ((YAML_MAX_UTF8_BYTES // 2) + 1)

        assert len(text) < YAML_MAX_UTF8_BYTES
        with pytest.raises(ValidationError, match="YAML_MAX_UTF8_BYTES"):
            load_yaml_mapping(text)

    def test_nesting_limit_rejects_deep_input_before_construction(self) -> None:
        text = (
            "value: "
            + "[" * YAML_MAX_NESTING_DEPTH
            + "0"
            + "]" * YAML_MAX_NESTING_DEPTH
        )

        with pytest.raises(ValidationError, match="YAML_MAX_NESTING_DEPTH"):
            load_yaml_mapping(text)

    def test_parse_event_limit_is_enforced_independently_of_node_limit(self) -> None:
        empty_sequences = YAML_MAX_PARSE_EVENTS // 2
        text = "items: [" + ",".join("[]" for _ in range(empty_sequences)) + "]"

        with pytest.raises(ValidationError, match="YAML_MAX_PARSE_EVENTS"):
            load_yaml_mapping(text)

    def test_node_limit_rejects_large_flat_document(self) -> None:
        text = "items: [" + ",".join("0" for _ in range(YAML_MAX_NODES)) + "]"

        with pytest.raises(ValidationError, match="YAML_MAX_NODES"):
            load_yaml_mapping(text)

    def test_alias_count_is_bounded_before_construction(self) -> None:
        aliases = ",".join("*base" for _ in range(YAML_MAX_ALIASES + 1))
        text = f"base: &base {{value: 1}}\nitems: [{aliases}]\n"

        with pytest.raises(ValidationError, match="YAML_MAX_ALIASES"):
            load_yaml_mapping(text)

    def test_integer_digits_are_bounded_when_interpreter_guard_is_disabled(
        self,
    ) -> None:
        original_limit = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(0)
            for value in (
                "9" * (YAML_MAX_INTEGER_DIGITS + 1),
                "0x" + "f" * (YAML_MAX_INTEGER_DIGITS + 1),
            ):
                with pytest.raises(
                    ValidationError,
                    match="YAML_MAX_INTEGER_DIGITS",
                ):
                    load_yaml_mapping("value: " + value + "\n")
        finally:
            sys.set_int_max_str_digits(original_limit)

    def test_alias_graph_expansion_is_bounded_before_construction(self) -> None:
        lines = ["root: &node0 [leaf]"]
        for index in range(1, 16):
            lines.append(
                f"node{index}: &node{index} "
                f"[*node{index - 1}, *node{index - 1}]"
            )

        with pytest.raises(ValidationError, match="YAML_MAX_NODES"):
            load_yaml_mapping("\n".join(lines))

    def test_expanded_scalar_bytes_are_bounded_before_construction(self) -> None:
        lines = ["seed: &node0 \"" + "x" * (32 * 1024) + "\""]
        for index in range(1, 12):
            lines.append(
                f"node{index}: &node{index} "
                f"[*node{index - 1}, *node{index - 1}]"
            )
        lines.append("payload: *node11")
        text = "\n".join(lines)

        assert len(text.encode("utf-8")) < YAML_MAX_EXPANDED_SCALAR_BYTES
        with pytest.raises(
            ValidationError,
            match="YAML_MAX_EXPANDED_SCALAR_BYTES",
        ):
            load_yaml_mapping(text)

    def test_bounded_repeated_scalar_alias_remains_supported(self) -> None:
        scalar = "x" * (128 * 1024)
        data = load_yaml_mapping(
            f'seed: &seed "{scalar}"\nitems: &items [*seed, *seed]\n'
            "payload: *items\n"
        )

        assert data["payload"] == [scalar, scalar]

    def test_recursive_alias_is_rejected_before_construction(self) -> None:
        with pytest.raises(ValidationError, match="recursive alias"):
            load_yaml_mapping("root: &root [*root]\n")

    def test_non_string_mapping_keys_are_rejected_at_every_level(self) -> None:
        with pytest.raises(ValidationError, match="mapping keys must be strings"):
            load_yaml_mapping("outer:\n  1: value\n")

    def test_merge_alias_expansion_is_bounded_before_construction(self) -> None:
        fields = "\n".join(f"  field_{index}: {index}" for index in range(1024))
        merged = "\n".join("  - <<: *base" for _ in range(32))
        text = f"base: &base\n{fields}\nitems:\n{merged}\n"

        with pytest.raises(ValidationError, match="YAML_MAX_NODES"):
            load_yaml_mapping(text)

    def test_normal_merge_and_alias_semantics_remain_supported(self) -> None:
        data = load_yaml_mapping(
            """
primary: &primary
  retries: 2
  color: primary
fallback: &fallback
  timeout_s: 10
  color: fallback
server:
  <<: [*primary, *fallback]
  retries: 3
  endpoint: local
""".lstrip()
        )

        assert data["server"] == {
            "retries": 3,
            "color": "primary",
            "timeout_s": 10,
            "endpoint": "local",
        }

    def test_duplicate_keys_inside_inline_merge_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate YAML key"):
            load_yaml_mapping(
                "server:\n  <<: {role: review, role: audit}\n"
            )

    @pytest.mark.parametrize(
        "text",
        (
            pytest.param("value: !!int ____\n", id="int"),
            pytest.param("value: !!float ____\n", id="float"),
            pytest.param("value: !!timestamp nope\n", id="timestamp-shape"),
            pytest.param(
                "value: !!timestamp 2024-99-99\n",
                id="timestamp-value",
            ),
            pytest.param("value: !!bool maybe\n", id="bool"),
        ),
    )
    def test_malformed_explicit_scalar_tags_are_domain_validation_errors(
        self,
        text: str,
    ) -> None:
        with pytest.raises(
            ValidationError,
            match="scalar construction failed",
        ):
            load_yaml_mapping(text)

    def test_recursion_error_is_normalized_but_memory_error_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def recurse(_node: object) -> None:
            raise RecursionError("synthetic parser recursion")

        monkeypatch.setattr(yaml_loader, "_validate_composed_node_graph", recurse)
        with pytest.raises(ValidationError, match="nesting is too deep"):
            load_yaml_mapping("value: 1\n")

        def exhaust_memory(_node: object) -> None:
            raise MemoryError("synthetic parser exhaustion")

        monkeypatch.setattr(
            yaml_loader,
            "_validate_composed_node_graph",
            exhaust_memory,
        )
        with pytest.raises(MemoryError, match="synthetic parser exhaustion"):
            load_yaml_mapping("value: 1\n")

    @pytest.mark.parametrize(
        "error_type",
        (
            pytest.param(SystemExit, id="system-exit"),
            pytest.param(KeyboardInterrupt, id="keyboard-interrupt"),
            pytest.param(asyncio.CancelledError, id="asyncio-cancelled"),
            pytest.param(_YamlLoadCancelled, id="base-exception"),
        ),
    )
    def test_system_and_cancellation_exceptions_propagate(
        self,
        monkeypatch: pytest.MonkeyPatch,
        error_type: type[BaseException],
    ) -> None:
        def interrupt(_node: object) -> None:
            raise error_type("synthetic YAML interruption")

        monkeypatch.setattr(
            yaml_loader,
            "_validate_composed_node_graph",
            interrupt,
        )
        with pytest.raises(error_type, match="synthetic YAML interruption"):
            load_yaml_mapping("value: 1\n")
