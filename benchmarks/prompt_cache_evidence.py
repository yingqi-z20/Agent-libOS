from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

from agent_libos.llm.usage import aggregate_cache_usage, canonicalize_llm_usage


_HOST_ID_SUFFIX = r"[0-9a-f]{16}"
_HOST_CONTRACT_ID = rf"(?:run|trreq|trp)_{_HOST_ID_SUFFIX}"
_MATERIALIZATION_ID = rf"(?:ctxmat|view)_{_HOST_ID_SUFFIX}"
_COMPLETION_SOURCE_ID = rf"(?:obj|pmsg|trreq)_{_HOST_ID_SUFFIX}"

# Match libOS-owned field/value pairs rather than field names alone. User and
# external payloads may legitimately contain keys such as ``run_id`` or
# ``schema_version``; those values must survive the Host-to-Model projection.
FORBIDDEN_MODEL_TEXT_PATTERNS = {
    "host_contract_fields": re.compile(
        rf'"(?:run_id|task_run_id|requirement_id|payload_id)"\s*:\s*'
        rf'"{_HOST_CONTRACT_ID}"'
    ),
    "materialization_fields": re.compile(
        rf'"(?:materialization_id|view_id|generation_id|revision_id)"\s*:\s*'
        rf'"{_MATERIALIZATION_ID}"'
    ),
    "completion_binding_fields": re.compile(
        rf'"(?:goal_oid|reviewed_message_ids|source_refs)"\s*:\s*'
        rf'(?:"{_COMPLETION_SOURCE_ID}"|\[[^\]]*"{_COMPLETION_SOURCE_ID}")'
    ),
    "current_process_ids": re.compile(
        rf'"(?:current_pid|caller_pid|parent_pid)"\s*:\s*"pid_{_HOST_ID_SUFFIX}"'
    ),
}
FORBIDDEN_MODEL_TEXT_CATEGORIES = (
    *FORBIDDEN_MODEL_TEXT_PATTERNS,
    "terminal_host_identifiers",
)
TERMINAL_HOST_IDENTIFIER_PATTERN = re.compile(
    rf"\b(?:(?:pid|obj|cap|ckpt|pmsg|evt|run|trreq|trp|ctxmat|view)_"
    rf"{_HOST_ID_SUFFIX}|tool_static_[0-9a-f]{{12,64}})\b"
)


def collect_prompt_cache_call_evidence(calls: Iterable[Any]) -> dict[str, Any]:
    """Aggregate safe cache/leak evidence without retaining extra prompt text."""

    selected = list(calls)
    details = forbidden_model_text_leak_details(selected)
    categories = aggregate_model_text_leak_details(details)
    return {
        **aggregate_cache_usage(selected),
        **_aggregate_total_token_usage(selected),
        "forbidden_internal_id_leak_evidence_complete": True,
        "forbidden_internal_id_leaks": sum(categories.values()),
        "forbidden_internal_id_leaks_by_category": categories,
        "forbidden_internal_id_leak_calls": details,
        "forbidden_internal_id_leak_call_count": len(details),
    }


def aggregate_prompt_cache_run_evidence(
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = list(runs)
    validated_leak_evidence: list[Mapping[str, Any]] = []
    leak_evidence_complete = bool(selected)
    for run in selected:
        try:
            validated_leak_evidence.append(
                validate_prompt_cache_leak_evidence(run)
            )
        except ValueError:
            leak_evidence_complete = False
    total_calls = sum(
        _nonnegative_int(run.get("cache_total_calls")) for run in selected
    )
    reported_calls = sum(
        _nonnegative_int(run.get("cache_reported_calls")) for run in selected
    )
    read_reported_calls = sum(
        _nonnegative_int(run.get("cache_read_reported_calls")) for run in selected
    )
    write_reported_calls = sum(
        _nonnegative_int(run.get("cache_write_reported_calls")) for run in selected
    )
    metric_reported_calls = sum(
        _nonnegative_int(run.get("cache_metric_reported_calls")) for run in selected
    )
    cache_metric_input_tokens = sum(
        _nonnegative_int(run.get("cache_metric_input_tokens")) for run in selected
    )
    uncached_input_tokens = sum(
        _nonnegative_int(run.get("uncached_input_tokens")) for run in selected
    )
    categories: dict[str, int] | None = None
    leak_total: int | None = None
    leak_call_count: int | None = None
    if leak_evidence_complete:
        categories = {
            category: sum(
                evidence["forbidden_internal_id_leaks_by_category"][category]
                for evidence in validated_leak_evidence
            )
            for category in FORBIDDEN_MODEL_TEXT_CATEGORIES
        }
        leak_total = sum(categories.values())
        leak_call_count = sum(
            evidence["forbidden_internal_id_leak_call_count"]
            for evidence in validated_leak_evidence
        )
    known_write_tokens = sum(
        _nonnegative_int(run.get("cache_write_tokens")) for run in selected
    )
    return {
        "cache_read_tokens": sum(
            _nonnegative_int(run.get("cache_read_tokens")) for run in selected
        ),
        "cache_write_tokens": (
            known_write_tokens
            if total_calls > 0 and write_reported_calls == total_calls
            else None
        ),
        "cache_total_calls": total_calls,
        "cache_reported_calls": reported_calls,
        "cache_read_reported_calls": read_reported_calls,
        "cache_write_reported_calls": write_reported_calls,
        "cache_metric_reported_calls": metric_reported_calls,
        "cache_metric_input_tokens": cache_metric_input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "cache_hit_rate": (
            (cache_metric_input_tokens - uncached_input_tokens)
            / cache_metric_input_tokens
            if cache_metric_input_tokens > 0
            else None
        ),
        "total_input_tokens": sum(
            _nonnegative_int(run.get("total_input_tokens")) for run in selected
        ),
        "total_output_tokens": sum(
            _nonnegative_int(run.get("total_output_tokens")) for run in selected
        ),
        "completion_evidence_successful_runs": sum(
            _completion_evidence_passed(run) for run in selected
        ),
        "forbidden_internal_id_leak_evidence_complete": leak_evidence_complete,
        "forbidden_internal_id_leaks": leak_total,
        "forbidden_internal_id_leaks_by_category": categories,
        "forbidden_internal_id_leak_call_count": leak_call_count,
    }


def validate_prompt_cache_leak_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the closed, redacted leak-measurement contract for one run."""

    if (
        "forbidden_internal_id_leak_evidence_complete" in evidence
        and evidence.get("forbidden_internal_id_leak_evidence_complete") is not True
    ):
        raise ValueError(
            "forbidden_internal_id_leak_evidence_complete must be true"
        )
    total = _required_nonnegative_int(
        evidence.get("forbidden_internal_id_leaks"),
        "forbidden_internal_id_leaks",
    )
    raw_categories = evidence.get("forbidden_internal_id_leaks_by_category")
    if not isinstance(raw_categories, Mapping):
        raise ValueError(
            "forbidden_internal_id_leaks_by_category must be an object"
        )
    expected_categories = set(FORBIDDEN_MODEL_TEXT_CATEGORIES)
    if set(raw_categories) != expected_categories:
        raise ValueError(
            "forbidden_internal_id_leaks_by_category must contain exactly the "
            "closed category set"
        )
    categories = {
        category: _required_nonnegative_int(
            raw_categories[category],
            f"forbidden_internal_id_leaks_by_category.{category}",
        )
        for category in FORBIDDEN_MODEL_TEXT_CATEGORIES
    }
    if sum(categories.values()) != total:
        raise ValueError(
            "forbidden_internal_id_leaks must equal the category total"
        )

    raw_details = evidence.get("forbidden_internal_id_leak_calls")
    details_present = "forbidden_internal_id_leak_calls" in evidence
    details: list[Mapping[str, Any]] | None = None
    if details_present:
        if not isinstance(raw_details, list) or not all(
            isinstance(item, Mapping) for item in raw_details
        ):
            raise ValueError(
                "forbidden_internal_id_leak_calls must be a list of objects"
            )
        details = list(raw_details)

    if "forbidden_internal_id_leak_call_count" in evidence:
        call_count = _required_nonnegative_int(
            evidence.get("forbidden_internal_id_leak_call_count"),
            "forbidden_internal_id_leak_call_count",
        )
    elif details is not None:
        # Compatibility for v1 raw-run reports, which carried the redacted
        # detail list before an explicit count was added.
        call_count = len(details)
    else:
        raise ValueError(
            "forbidden_internal_id_leak_call_count must be reported"
        )

    if details is not None:
        if len(details) != call_count:
            raise ValueError(
                "forbidden_internal_id_leak_call_count must equal the detail count"
            )
        if aggregate_model_text_leak_details(details) != categories:
            raise ValueError(
                "forbidden_internal_id_leak_calls must reconcile with categories"
            )
    if (total == 0) != (call_count == 0) or call_count > total:
        raise ValueError(
            "forbidden_internal_id_leak_call_count must reconcile with the leak total"
        )
    return {
        "forbidden_internal_id_leaks": total,
        "forbidden_internal_id_leaks_by_category": categories,
        "forbidden_internal_id_leak_call_count": call_count,
    }


def forbidden_model_text_leak_details(
    calls: Iterable[Any],
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for ordinal, call in enumerate(calls, start=1):
        categories = {category: 0 for category in FORBIDDEN_MODEL_TEXT_PATTERNS}
        categories["terminal_host_identifiers"] = 0
        surfaces = {"messages": 0, "response_content": 0, "tool_calls": 0}
        for surface, text in _model_visible_text_fragments(call):
            for category, pattern in FORBIDDEN_MODEL_TEXT_PATTERNS.items():
                hits = len(pattern.findall(text))
                categories[category] += hits
                surfaces[surface] += hits
        terminal_hits = sum(
            len(TERMINAL_HOST_IDENTIFIER_PATTERN.findall(arguments))
            for name, arguments in _model_tool_call_arguments(
                _record_value(call, "tool_calls")
            )
            if name in {"human_output", "process_exit"}
        )
        categories["terminal_host_identifiers"] += terminal_hits
        surfaces["tool_calls"] += terminal_hits
        if sum(categories.values()) == 0:
            continue
        details.append(
            {
                "call_ordinal": ordinal,
                "categories": {
                    key: value for key, value in categories.items() if value
                },
                "surfaces": {
                    key: value for key, value in surfaces.items() if value
                },
                "response_tools": sorted(
                    {
                        name
                        for name, _arguments in _model_tool_call_arguments(
                            _record_value(call, "tool_calls")
                        )
                    }
                ),
            }
        )
    return details


def aggregate_model_text_leak_details(
    details: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    counts = {category: 0 for category in FORBIDDEN_MODEL_TEXT_PATTERNS}
    counts["terminal_host_identifiers"] = 0
    for detail in details:
        categories = detail.get("categories")
        if not isinstance(categories, Mapping):
            continue
        for category in counts:
            counts[category] += _nonnegative_int(categories.get(category))
    return counts


def _aggregate_total_token_usage(calls: Iterable[Any]) -> dict[str, int]:
    input_tokens = 0
    output_tokens = 0
    for call in calls:
        api = _record_value(call, "api")
        usage, _invalid = canonicalize_llm_usage(
            _record_value(call, "usage"),
            api=api if isinstance(api, str) else None,
        )
        input_tokens += _first_counter(
            usage,
            ("input_tokens", "prompt_tokens")
            if api == "responses"
            else ("prompt_tokens", "input_tokens"),
        )
        output_tokens += _first_counter(
            usage,
            ("output_tokens", "completion_tokens")
            if api == "responses"
            else ("completion_tokens", "output_tokens"),
        )
    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
    }


def _first_counter(value: Mapping[str, int], keys: tuple[str, ...]) -> int:
    for key in keys:
        selected = value.get(key)
        if selected is not None:
            return selected
    return 0


def _model_visible_text_fragments(call: Any) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    messages = _record_value(call, "messages")
    rows = messages if isinstance(messages, list) else [messages]
    for row in rows:
        if not isinstance(row, Mapping):
            fragments.append(("messages", _model_visible_text(row)))
            continue
        content = row.get("content")
        if content not in (None, ""):
            fragments.append(("messages", _model_visible_text(content)))
        fragments.extend(
            ("messages", arguments)
            for _name, arguments in _model_tool_call_arguments(row.get("tool_calls"))
        )
        envelope = {
            key: value
            for key, value in row.items()
            if key not in {"content", "tool_calls"}
        }
        if envelope:
            fragments.append(("messages", _model_visible_text(envelope)))
    response_content = _record_value(call, "response_content")
    if response_content not in (None, ""):
        fragments.append(("response_content", _model_visible_text(response_content)))
    fragments.extend(
        ("tool_calls", arguments)
        for _name, arguments in _model_tool_call_arguments(
            _record_value(call, "tool_calls")
        )
    )
    return fragments


def _model_visible_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _model_tool_call_arguments(value: Any) -> list[tuple[str, str]]:
    rows = value if isinstance(value, list) else [value]
    selected: list[tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        function = row.get("function")
        function_mapping = function if isinstance(function, Mapping) else {}
        name = row.get("name") or function_mapping.get("name")
        arguments = row.get("arguments", function_mapping.get("arguments", ""))
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(arguments, str):
            arguments = _model_visible_text(arguments)
        selected.append((name, arguments))
    return selected


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _completion_evidence_passed(run: Mapping[str, Any]) -> bool:
    explicit = run.get("completion_review_passed")
    if isinstance(explicit, bool):
        return explicit
    required = run.get("task_run_requirement_count")
    satisfied = run.get("task_run_satisfied_requirement_count")
    return (
        isinstance(required, int)
        and not isinstance(required, bool)
        and required > 0
        and satisfied == required
        and run.get("final_status") == "succeeded"
    )


def _nonnegative_int(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _required_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    raise ValueError(f"{field} must be a non-negative integer")


__all__ = [
    "FORBIDDEN_MODEL_TEXT_CATEGORIES",
    "FORBIDDEN_MODEL_TEXT_PATTERNS",
    "TERMINAL_HOST_IDENTIFIER_PATTERN",
    "aggregate_model_text_leak_details",
    "aggregate_prompt_cache_run_evidence",
    "collect_prompt_cache_call_evidence",
    "forbidden_model_text_leak_details",
    "validate_prompt_cache_leak_evidence",
]
