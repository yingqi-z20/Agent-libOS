from __future__ import annotations

from typing import Any

import yaml

from agent_libos.models.exceptions import ValidationError


# This loader is used while bootstrap configuration and invariant metadata are
# being read, so the parser limits must not depend on Runtime configuration.
YAML_MAX_UTF8_BYTES = 1_048_576
YAML_MAX_PARSE_EVENTS = 60_000
YAML_MAX_NODES = 32_768
YAML_MAX_NESTING_DEPTH = 64
YAML_MAX_ALIASES = 64
YAML_MAX_INTEGER_DIGITS = 4_300
YAML_MAX_EXPANDED_SCALAR_BYTES = 1_048_576
_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


def load_yaml_mapping(text: str) -> dict[str, Any]:
    """Load one bounded YAML mapping with unique keys at every level."""

    _validate_yaml_text_size(text)
    loader: _UniqueKeyLoader | None = None
    try:
        loader = _UniqueKeyLoader(text)
        node = loader.get_single_node()
        if node is None:
            return {}
        _validate_composed_node_graph(node)
        data = loader.construct_document(node)
    except ValidationError:
        raise
    except RecursionError as exc:
        raise ValidationError("invalid YAML document: nesting is too deep") from exc
    except yaml.YAMLError as exc:
        raise ValidationError(f"invalid YAML document: {exc}") from exc
    except (
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValidationError(
            "invalid YAML document: scalar construction failed"
        ) from exc
    finally:
        if loader is not None:
            loader.dispose()
    if not isinstance(data, dict):
        raise ValidationError("YAML document must be a mapping")
    return data


class _UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream: str) -> None:
        self._bounded_event_count = 0
        self._bounded_node_count = 0
        self._bounded_alias_count = 0
        self._bounded_depth = 0
        super().__init__(stream)

    def get_event(self) -> yaml.events.Event | None:
        event = super().get_event()
        if event is None:
            return None
        self._bounded_event_count += 1
        _require_yaml_limit(
            self._bounded_event_count,
            YAML_MAX_PARSE_EVENTS,
            "YAML_MAX_PARSE_EVENTS",
        )
        if isinstance(
            event,
            (yaml.events.MappingStartEvent, yaml.events.SequenceStartEvent),
        ):
            self._bounded_node_count += 1
            self._bounded_depth += 1
            _require_yaml_limit(
                self._bounded_depth,
                YAML_MAX_NESTING_DEPTH,
                "YAML_MAX_NESTING_DEPTH",
            )
        elif isinstance(event, (yaml.events.ScalarEvent, yaml.events.AliasEvent)):
            self._bounded_node_count += 1
        elif isinstance(
            event,
            (yaml.events.MappingEndEvent, yaml.events.SequenceEndEvent),
        ):
            self._bounded_depth -= 1
        _require_yaml_limit(
            self._bounded_node_count,
            YAML_MAX_NODES,
            "YAML_MAX_NODES",
        )
        if isinstance(event, yaml.events.AliasEvent):
            self._bounded_alias_count += 1
            _require_yaml_limit(
                self._bounded_alias_count,
                YAML_MAX_ALIASES,
                "YAML_MAX_ALIASES",
            )
        return event


def _validate_yaml_text_size(text: str) -> None:
    if not isinstance(text, str):
        raise ValidationError("YAML document must be text")
    try:
        size = len(text.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ValidationError("YAML document must be valid UTF-8 text") from exc
    _require_yaml_limit(size, YAML_MAX_UTF8_BYTES, "YAML_MAX_UTF8_BYTES")


def _require_yaml_limit(value: int, limit: int, name: str) -> None:
    if value > limit:
        raise ValidationError(f"YAML document exceeds {name}={limit}")


def _validate_composed_node_graph(root: yaml.nodes.Node) -> None:
    """Bound alias expansion and reject recursive node graphs before construction."""

    expanded_nodes = 0
    expanded_scalar_bytes = 0
    scalar_sizes: dict[int, int] = {}
    active: set[int] = set()
    stack: list[tuple[yaml.nodes.Node, bool]] = [(root, False)]
    while stack:
        node, leaving = stack.pop()
        identity = id(node)
        if leaving:
            active.remove(identity)
            continue
        expanded_nodes += 1
        _require_yaml_limit(expanded_nodes, YAML_MAX_NODES, "YAML_MAX_NODES")
        if isinstance(node, yaml.nodes.ScalarNode):
            scalar_size = scalar_sizes.get(identity)
            if scalar_size is None:
                try:
                    scalar_size = len(node.value.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise ValidationError(
                        "YAML scalar must be valid UTF-8 text"
                    ) from exc
                scalar_sizes[identity] = scalar_size
            expanded_scalar_bytes += scalar_size
            _require_yaml_limit(
                expanded_scalar_bytes,
                YAML_MAX_EXPANDED_SCALAR_BYTES,
                "YAML_MAX_EXPANDED_SCALAR_BYTES",
            )
        children = _yaml_node_children(node)
        if not children:
            continue
        if identity in active:
            raise ValidationError("YAML document contains a recursive alias")
        active.add(identity)
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(children))


def _yaml_node_children(node: yaml.nodes.Node) -> list[yaml.nodes.Node]:
    if isinstance(node, yaml.nodes.MappingNode):
        return [child for pair in node.value for child in pair]
    if isinstance(node, yaml.nodes.SequenceNode):
        return list(node.value)
    if isinstance(node, yaml.nodes.ScalarNode):
        return []
    raise ValidationError(f"unsupported YAML node type: {type(node).__name__}")


def _construct_unique_mapping(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    _validate_unique_source_mapping_keys(loader, node, deep=deep)
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValidationError("YAML mapping keys must be strings")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


def _validate_unique_source_mapping_keys(
    loader: yaml.SafeLoader,
    node: yaml.nodes.MappingNode,
    *,
    deep: bool,
) -> None:
    keys: set[str] = set()
    merge_seen = False
    for key_node, value_node in node.value:
        if key_node.tag == _YAML_MERGE_TAG:
            if merge_seen:
                raise ValidationError("duplicate YAML key: '<<'")
            merge_seen = True
            _validate_merge_source_mapping_keys(loader, value_node, deep=deep)
            continue
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValidationError("YAML mapping keys must be strings")
        if key in keys:
            raise ValidationError(f"duplicate YAML key: {key!r}")
        keys.add(key)


def _validate_merge_source_mapping_keys(
    loader: yaml.SafeLoader,
    node: yaml.nodes.Node,
    *,
    deep: bool,
) -> None:
    if isinstance(node, yaml.nodes.MappingNode):
        _validate_unique_source_mapping_keys(loader, node, deep=deep)
        return
    if isinstance(node, yaml.nodes.SequenceNode):
        for child in node.value:
            if isinstance(child, yaml.nodes.MappingNode):
                _validate_unique_source_mapping_keys(loader, child, deep=deep)


def _construct_bounded_integer(
    loader: yaml.SafeLoader,
    node: yaml.nodes.ScalarNode,
) -> int:
    value = loader.construct_scalar(node)
    normalized = value.replace("_", "").lstrip("+-")
    if normalized.startswith(("0b", "0B", "0o", "0O", "0x", "0X")):
        normalized = normalized[2:]
    digits = len(normalized.replace(":", ""))
    _require_yaml_limit(
        digits,
        YAML_MAX_INTEGER_DIGITS,
        "YAML_MAX_INTEGER_DIGITS",
    )
    try:
        return yaml.constructor.SafeConstructor.construct_yaml_int(loader, node)
    except ValueError as exc:
        raise ValidationError("invalid YAML integer") from exc


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
_UniqueKeyLoader.add_constructor(
    "tag:yaml.org,2002:int",
    _construct_bounded_integer,
)
