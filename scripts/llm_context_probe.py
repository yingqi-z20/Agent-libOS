from __future__ import annotations

import ast
import json
import re
from typing import Any


def static_prefix(messages: list[dict[str, Any]]) -> dict[str, Any]:
    text = _message_text(messages)
    match = re.search(r"Static prefix:\n(?P<payload>.*?)\n\nAppend-only entries:", text, flags=re.DOTALL)
    if match is not None:
        try:
            payload = json.loads(match.group("payload"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            return payload
    return _source_process_facts(text)


def entries(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text = _message_text(messages)
    if "Append-only entries:" not in text:
        return []
    tail = text.split("Append-only entries:", 1)[1]
    result: list[dict[str, Any]] = []
    for block in re.split(r"(?m)^---\s*$", tail):
        block = block.strip()
        if not block:
            continue
        try:
            entry = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(entry, dict):
            result.append(entry)
    return result


def recent_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries(messages):
        if entry.get("kind") != "events_delta":
            continue
        events = entry.get("events")
        if isinstance(events, list):
            result.extend(event for event in events if isinstance(event, dict))
    return result or _source_context_events(messages)


def tool_result_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = _delta_tool_result_payloads(messages)
    return result or _source_context_tool_results(messages)


def last_tool_result(messages: list[dict[str, Any]], tool_name: str) -> dict[str, Any] | None:
    delta_results = _delta_tool_result_payloads(messages)
    for payload in reversed(delta_results):
        if payload.get("tool_name") == tool_name and isinstance(payload.get("result"), dict):
            return payload["result"]

    source_objects = _source_context_tool_result_objects(messages)
    by_oid = {oid: payload for oid, payload in source_objects}
    for event in reversed(_source_context_events(messages)):
        if event.get("type") != "tool_completed":
            continue
        event_payload = event.get("payload")
        if not isinstance(event_payload, dict):
            continue
        payload = by_oid.get(str(event_payload.get("result_oid") or ""))
        if (
            isinstance(payload, dict)
            and payload.get("tool_name") == tool_name
            and isinstance(payload.get("result"), dict)
        ):
            return payload["result"]

    # Minimal image-only prompts intentionally omit event metadata. Their
    # recency-first source context renders the newest Object first.
    for _oid, payload in source_objects:
        if payload.get("tool_name") == tool_name and isinstance(payload.get("result"), dict):
            return payload["result"]
    return None


def _delta_tool_result_payloads(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for entry in entries(messages):
        if entry.get("kind") != "memory_delta":
            continue
        objects = entry.get("objects")
        if not isinstance(objects, list):
            continue
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            payload = obj.get("payload")
            if isinstance(payload, dict) and isinstance(payload.get("tool_name"), str):
                result.append(payload)
    return result


def _message_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _source_process_facts(text: str) -> dict[str, Any]:
    """Read the current process envelope used by source-only prompt modes."""

    match = re.search(
        r"(?ms)^(?:Process|Process facts):\n(?P<body>(?:- [^\n]*\n?)+)",
        text,
    )
    if match is None:
        return {}
    result: dict[str, Any] = {}
    for line in match.group("body").splitlines():
        field, separator, raw_value = line.removeprefix("- ").partition(": ")
        if not separator or not field:
            continue
        value = _literal_value(raw_value)
        result[field] = raw_value if value is None and raw_value != "None" else value
    return result


def _source_context_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read events from the source-only prompt format used without enrichment."""

    result: list[dict[str, Any]] = []
    for match in re.finditer(
        r"(?ms)^Recent events:\n(?P<payload>\[.*?\])\n\nMaterialized context:\n",
        _message_text(messages),
    ):
        parsed = _literal_value(match.group("payload"))
        if isinstance(parsed, list):
            result.extend(event for event in parsed if isinstance(event, dict))
    return result


def _source_context_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read tool-result Objects rendered directly in a materialized context."""

    return [payload for _oid, payload in _source_context_tool_result_objects(messages)]


def _source_context_tool_result_objects(
    messages: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any]]]:
    result: list[tuple[str, dict[str, Any]]] = []
    blocks = re.split(
        r"(?m)(?=^\[[^\]\n]+\] namespace=)",
        _message_text(messages),
    )
    for block in blocks:
        header, separator, remainder = block.partition("\n")
        if not separator or not re.search(r"\btype=tool_result\b", header):
            continue
        payload_line = next(
            (line for line in remainder.splitlines() if line.startswith("payload: ")),
            None,
        )
        if payload_line is None:
            continue
        parsed = _literal_value(payload_line.removeprefix("payload: "))
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_name"), str):
            oid_match = re.match(r"^\[(?P<oid>[^\]\n]+)\]", header)
            if oid_match is not None:
                result.append((oid_match.group("oid"), parsed))
    return result


def _literal_value(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (MemoryError, RecursionError, SyntaxError, ValueError):
        return None
