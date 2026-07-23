from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


_EVIDENCE_LEVELS = ("native-live", "modeled")
_REPORT_SCHEMA_PATH = Path(__file__).with_name("report.schema.json")


def validate_practical_report_schema(report: Mapping[str, Any]) -> list[str]:
    """Return stable diagnostics for violations of the checked-in v1 schema."""

    schema = json.loads(_REPORT_SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    diagnostics: list[str] = []
    for error in sorted(
        validator.iter_errors(report),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    ):
        location = ".".join(str(part) for part in error.absolute_path) or "$"
        diagnostics.append(f"{location}: {error.message}")
    return diagnostics


def validate_practical_report(report: Mapping[str, Any]) -> list[str]:
    """Validate report invariants that JSON Schema cannot express.

    The JSON Schema owns the shape and per-row constraints.  This validator
    owns cross-row identities, aggregates, and the exact gate semantics.  It
    intentionally accepts an empty evidence partition: the corresponding gate
    is the vacuous ``all(...)`` value documented by the programmatic API.
    """

    errors: list[str] = []
    results = report.get("results")
    if not isinstance(results, list):
        return ["results must be an array before semantic validation"]

    rows_by_level: dict[str, list[Mapping[str, Any]]] = {
        level: [] for level in _EVIDENCE_LEVELS
    }
    scenario_ids: set[str] = set()
    effect_ids: set[str] = set()
    operation_ids: set[str] = set()

    for index, raw_row in enumerate(results):
        if not isinstance(raw_row, Mapping):
            errors.append(f"results[{index}] must be an object")
            continue
        row = raw_row
        scenario_id = row.get("scenario_id")
        if isinstance(scenario_id, str):
            if scenario_id in scenario_ids:
                errors.append(f"duplicate scenario_id: {scenario_id}")
            scenario_ids.add(scenario_id)

        level = row.get("evidence_level")
        if level not in rows_by_level:
            # Shape validation reports the unsupported value.
            continue
        rows_by_level[level].append(row)

        row_effect_ids = _string_list(row.get("external_effect_ids"))
        row_operation_ids = _string_list(row.get("operation_ids"))
        for effect_id in row_effect_ids:
            if effect_id in effect_ids:
                errors.append(f"external effect id appears in multiple rows: {effect_id}")
            effect_ids.add(effect_id)
        for operation_id in row_operation_ids:
            if operation_id in operation_ids:
                errors.append(f"operation id appears in multiple rows: {operation_id}")
            operation_ids.add(operation_id)

        operations = _plain_int(row.get("operations"))
        if operations is not None and operations != len(row_operation_ids):
            errors.append(
                f"{scenario_id or f'results[{index}]'} operations does not match operation_ids"
            )

        if level == "modeled":
            for field in ("tool_calls", "operations"):
                if _plain_int(row.get(field)) != 0:
                    errors.append(f"{scenario_id} modeled row has nonzero {field}")
            if row_effect_ids or row_operation_ids:
                errors.append(f"{scenario_id} modeled row contains runtime evidence ids")
            continue

        semantic_effects = _plain_int(row.get("semantic_effects"))
        tool_calls = _plain_int(row.get("tool_calls"))
        if row.get("ok") is True:
            if semantic_effects is not None and tool_calls != semantic_effects:
                errors.append(
                    f"{scenario_id} passing native row must have one tool call per semantic effect"
                )
            if semantic_effects is not None and len(row_effect_ids) != semantic_effects:
                errors.append(
                    f"{scenario_id} passing native row must have one external effect id "
                    "per semantic effect"
                )
            if (
                semantic_effects is not None
                and operations is not None
                and operations < semantic_effects
            ):
                errors.append(
                    f"{scenario_id} passing native row must resolve at least one operation "
                    "per semantic effect"
                )

    expected_scenario_counts = {
        level: len(rows_by_level[level]) for level in _EVIDENCE_LEVELS
    }
    _compare_mapping(
        report.get("scenario_counts"),
        expected_scenario_counts,
        field="scenario_counts",
        errors=errors,
    )
    expected_effect_counts = {
        level: sum(
            _plain_int(row.get("semantic_effects")) or 0
            for row in rows_by_level[level]
        )
        for level in _EVIDENCE_LEVELS
    }
    _compare_mapping(
        report.get("semantic_effect_counts"),
        expected_effect_counts,
        field="semantic_effect_counts",
        errors=errors,
    )

    expected_tool_calls = sum(
        _plain_int(row.get("tool_calls")) or 0
        for row in rows_by_level["native-live"]
    )
    if report.get("native_tool_calls") != expected_tool_calls:
        errors.append(
            "native_tool_calls does not equal the sum of native-live result rows"
        )
    expected_operations = sum(
        _plain_int(row.get("operations")) or 0
        for row in rows_by_level["native-live"]
    )
    if report.get("native_operations") != expected_operations:
        errors.append(
            "native_operations does not equal the sum of native-live result rows"
        )
    if report.get("modeled_fallback") != 0:
        errors.append("modeled_fallback must be exactly zero in report schema v1")

    expected_native_gate = all(
        row.get("ok") is True for row in rows_by_level["native-live"]
    )
    if report.get("native_live_ok") is not expected_native_gate:
        errors.append("native_live_ok does not match native-live result rows")
    expected_modeled_gate = all(
        row.get("ok") is True for row in rows_by_level["modeled"]
    )
    if report.get("modeled_suite_ok") is not expected_modeled_gate:
        errors.append("modeled_suite_ok does not match modeled result rows")
    return errors


def _plain_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _compare_mapping(
    actual: Any,
    expected: dict[str, int],
    *,
    field: str,
    errors: list[str],
) -> None:
    if actual != expected:
        errors.append(f"{field} does not match results: expected {expected}, got {actual}")
