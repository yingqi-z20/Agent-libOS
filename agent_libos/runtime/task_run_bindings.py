from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

from agent_libos.models import canonical_task_run_json
from agent_libos.models.exceptions import ValidationError
from agent_libos.utils.serde import to_jsonable


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRE_ACTION_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "process_revision",
        "image_binding_hash",
        "tool_binding_hash",
        "provider_binding_hash",
        "tool_table",
        "model_tool_table",
        "loaded_skill_sha256",
    }
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_task_run_json(value).encode("utf-8")).hexdigest()


def loaded_skill_hashes(process: Any) -> dict[str, str]:
    loaded = getattr(process, "loaded_skills", None)
    if not isinstance(loaded, Mapping):
        raise ValidationError("TaskRun process loaded-Skill state is invalid")
    return {
        str(skill_id): canonical_sha256(to_jsonable(record))
        for skill_id, record in sorted(loaded.items(), key=lambda item: str(item[0]))
    }


def tool_binding_projection(process: Any) -> dict[str, Any]:
    tool_table = _text_map(
        getattr(process, "tool_table", None),
        "TaskRun process tool table",
    )
    model_tool_table = _text_map(
        getattr(process, "model_tool_table", None),
        "TaskRun process model tool table",
    )
    return {
        "tool_table": tool_table,
        "model_tool_table": model_tool_table,
        "loaded_skill_sha256": loaded_skill_hashes(process),
    }


def tool_binding_hash(process: Any) -> str:
    return canonical_sha256(tool_binding_projection(process))


def pre_action_binding(
    process: Any,
    *,
    image_binding_hash: str,
    provider_binding_hash: str,
) -> dict[str, Any]:
    projection = tool_binding_projection(process)
    return {
        "schema_version": 1,
        "process_revision": _nonnegative_int(
            getattr(process, "revision", None),
            "TaskRun process revision",
        ),
        "image_binding_hash": _hash(image_binding_hash, "TaskRun Image binding"),
        "tool_binding_hash": canonical_sha256(projection),
        "provider_binding_hash": _hash(
            provider_binding_hash,
            "TaskRun provider binding",
        ),
        **projection,
    }


def validate_pre_action_binding(
    value: Any,
    *,
    image_binding_hash: str,
    tool_binding_hash: str,
    provider_binding_hash: str,
    process_revision: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _PRE_ACTION_BINDING_KEYS:
        raise ValidationError("TaskRun pre-action binding has an invalid shape")
    selected = dict(value)
    if selected.get("schema_version") != 1 or type(selected.get("schema_version")) is not int:
        raise ValidationError("TaskRun pre-action binding schema must be 1")
    revision = _nonnegative_int(
        selected.get("process_revision"),
        "TaskRun pre-action process revision",
    )
    if process_revision is not None and revision != process_revision:
        raise ValidationError("TaskRun pre-action process revision changed")
    expected_hashes = (
        _hash(image_binding_hash, "TaskRun Image binding"),
        _hash(tool_binding_hash, "TaskRun tool binding"),
        _hash(provider_binding_hash, "TaskRun provider binding"),
    )
    actual_hashes = (
        _hash(selected.get("image_binding_hash"), "TaskRun pre-action Image binding"),
        _hash(selected.get("tool_binding_hash"), "TaskRun pre-action tool binding"),
        _hash(
            selected.get("provider_binding_hash"),
            "TaskRun pre-action provider binding",
        ),
    )
    if actual_hashes != expected_hashes:
        raise ValidationError("TaskRun pre-action binding changed")
    tool_projection = {
        "tool_table": _text_map(
            selected.get("tool_table"),
            "TaskRun pre-action tool table",
        ),
        "model_tool_table": _text_map(
            selected.get("model_tool_table"),
            "TaskRun pre-action model tool table",
        ),
        "loaded_skill_sha256": _hash_map(
            selected.get("loaded_skill_sha256"),
            "TaskRun pre-action loaded-Skill hashes",
        ),
    }
    if canonical_sha256(tool_projection) != actual_hashes[1]:
        raise ValidationError("TaskRun pre-action tool projection hash changed")
    normalized = {
        "schema_version": 1,
        "process_revision": revision,
        "image_binding_hash": actual_hashes[0],
        "tool_binding_hash": actual_hashes[1],
        "provider_binding_hash": actual_hashes[2],
        **tool_projection,
    }
    canonical_task_run_json(normalized)
    return normalized


def expected_activated_process_projection(
    pre_binding: Mapping[str, Any],
    *,
    skill_id: str,
    tool_ids: Mapping[str, str],
    jit_tool_ids: Mapping[str, str],
    loaded_skill_sha256: str,
) -> dict[str, Any]:
    if skill_id in pre_binding["loaded_skill_sha256"]:
        raise ValidationError(
            "TaskRun cannot certify replacement of an already-loaded Skill"
        )
    published = {**dict(tool_ids), **dict(jit_tool_ids)}
    expected_tools = dict(pre_binding["tool_table"])
    expected_model_tools = dict(pre_binding["model_tool_table"])
    expected_tools.update(published)
    expected_model_tools.update(published)
    expected_loaded = dict(pre_binding["loaded_skill_sha256"])
    expected_loaded[skill_id] = _hash(
        loaded_skill_sha256,
        "TaskRun activated Skill record",
    )
    return {
        "tool_table": expected_tools,
        "model_tool_table": expected_model_tools,
        "loaded_skill_sha256": expected_loaded,
    }


def require_exact_activated_projection(
    process: Any,
    expected: Mapping[str, Any],
) -> str:
    current = tool_binding_projection(process)
    if current != dict(expected):
        raise ValidationError(
            "TaskRun activate_skill changed an uncertified tool or Skill binding"
        )
    return canonical_sha256(current)


def _text_map(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{label} must be an object")
    selected: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key or not isinstance(item, str) or not item:
            raise ValidationError(f"{label} must contain non-empty text bindings")
        selected[key] = item
    return selected


def _hash_map(value: Any, label: str) -> dict[str, str]:
    selected = _text_map(value, label)
    if any(_SHA256.fullmatch(item) is None for item in selected.values()):
        raise ValidationError(f"{label} contains an invalid SHA-256")
    return selected


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValidationError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "canonical_sha256",
    "expected_activated_process_projection",
    "loaded_skill_hashes",
    "pre_action_binding",
    "require_exact_activated_projection",
    "tool_binding_hash",
    "tool_binding_projection",
    "validate_pre_action_binding",
]
