from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, LLMProfile
from agent_libos.config.defaults import (
    sanitize_llm_base_url_for_summary,
    validate_llm_base_url,
)
from agent_libos.models.exceptions import NotFound, ValidationError

SCHEMA_VERSION = 1
PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_STRING_LIMITS = {
    "profile_id": 128,
    "model": 256,
    "base_url": 2_048,
    "api_key_env": 128,
    "reasoning_effort": 64,
    "safety_identifier_env": 128,
}
_API_MODES = {"auto", "responses", "chat"}
_VERBOSITY = {"low", "medium", "high"}
_PROMPT_CACHE_MODES = {"provider_default", "implicit", "explicit"}
_USER_PROFILE_FIELDS = (
    "kind",
    "base_url",
    "model",
    "api_key_env",
    "api_mode",
    "timeout_s",
    "max_retries",
    "store",
    "reasoning_effort",
    "verbosity",
    "safety_identifier_env",
    "prompt_cache_retention",
    "prompt_cache_mode",
    "prompt_cache_ttl",
    "responses_previous_response_id",
    "parallel_tool_calls",
    "auto_wait_on_empty_tool_calls",
    "fallback_json_actions",
    "temperature",
    "max_tokens",
    "max_input_tokens_per_call",
    "max_total_tokens_per_call",
    "context_window_tokens",
    "allow_custom_base_url",
)
_CORE_ONLY_PROFILE_FIELDS = (
    "max_input_tokens_per_call",
    "max_total_tokens_per_call",
)


def default_user_llm_profiles_path(
    *,
    platform: str | None = None,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    selected_platform = sys.platform if platform is None else platform
    selected_env = os.environ if env is None else env
    selected_home = Path.home() if home is None else home
    if selected_platform.startswith("win"):
        base = Path(selected_env.get("APPDATA") or selected_home / "AppData" / "Roaming")
        return base / "Agent libOS" / "llm-profiles.json"
    if selected_platform == "darwin":
        return selected_home / "Library" / "Application Support" / "Agent libOS" / "llm-profiles.json"
    base = Path(selected_env.get("XDG_CONFIG_HOME") or selected_home / ".config")
    return base / "agent-libos" / "llm-profiles.json"


class UserLLMProfileStore:
    """User-level, host-only LLM profile storage for the local GUI."""

    def __init__(self, path: str | Path | None = None, *, config: AgentLibOSConfig | None = None):
        self.path = Path(path) if path is not None else default_user_llm_profiles_path()
        self.config = config or DEFAULT_CONFIG

    def load(self) -> dict[str, LLMProfile]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid LLM profiles JSON {self.path}: {exc}") from exc
        except OSError as exc:
            raise ValidationError(f"cannot read LLM profiles file {self.path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValidationError("LLM profiles file root must be a JSON object")
        if raw.get("schema_version") != SCHEMA_VERSION:
            raise ValidationError(f"unsupported LLM profiles schema_version: {raw.get('schema_version')!r}")
        profiles_raw = raw.get("profiles", {})
        if not isinstance(profiles_raw, dict):
            raise ValidationError("LLM profiles file 'profiles' must be a JSON object")
        profiles: dict[str, LLMProfile] = {}
        for profile_id, payload in profiles_raw.items():
            selected_id = normalize_user_llm_profile_id(profile_id)
            profiles[selected_id] = validate_user_llm_profile_payload(
                selected_id,
                payload,
                config=self.config,
            )
        return profiles

    def upsert(self, profile_id: str, payload: Mapping[str, Any]) -> LLMProfile:
        selected_id = normalize_user_llm_profile_id(profile_id)
        profiles = self.load()
        selected_payload = dict(payload)
        existing = profiles.get(selected_id)
        if existing is not None:
            # The bundled GUI intentionally has no controls for these core-only
            # limits. Preserve an existing override when an older/editor payload
            # omits it; an explicit null still clears the override.
            for field_name in _CORE_ONLY_PROFILE_FIELDS:
                if field_name not in selected_payload:
                    selected_payload[field_name] = getattr(existing, field_name)
        profile = validate_user_llm_profile_payload(
            selected_id,
            selected_payload,
            config=self.config,
        )
        profiles[selected_id] = profile
        self.save(profiles)
        return profile

    def delete(self, profile_id: str) -> None:
        selected_id = normalize_user_llm_profile_id(profile_id)
        profiles = self.load()
        if selected_id not in profiles:
            raise NotFound(f"user LLM profile not found: {selected_id}")
        profiles.pop(selected_id)
        self.save(profiles)

    def save(self, profiles: Mapping[str, LLMProfile]) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "profiles": {
                profile_id: serialize_user_llm_profile(profile)
                for profile_id, profile in sorted(profiles.items())
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, self.path)
        except OSError as exc:
            raise ValidationError(f"cannot write LLM profiles file {self.path}: {exc}") from exc
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def normalize_user_llm_profile_id(value: Any) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValidationError("LLM profile id must be a non-empty string")
    if len(selected) > _STRING_LIMITS["profile_id"] or not PROFILE_ID_RE.match(selected):
        raise ValidationError("LLM profile id may contain only letters, numbers, '.', '_', ':', and '-'")
    return selected


def validate_user_llm_profile_payload(
    profile_id: str,
    payload: Any,
    *,
    config: AgentLibOSConfig | None = None,
) -> LLMProfile:
    if not isinstance(payload, Mapping):
        raise ValidationError(f"LLM profile payload must be an object: {profile_id}")
    if "api_key" in payload:
        raise ValidationError("API keys are not accepted in GUI LLM profiles; set api_key_env instead")
    raw = dict(payload)
    embedded_id = raw.pop("profile_id", None)
    if embedded_id is not None and normalize_user_llm_profile_id(embedded_id) != profile_id:
        raise ValidationError("profile_id in body must match the route profile id")
    unknown = sorted(set(raw) - set(_USER_PROFILE_FIELDS))
    if unknown:
        raise ValidationError(f"unknown LLM profile fields: {', '.join(unknown)}")
    _normalize_user_llm_profile_fields(raw)
    cleaned = {key: value for key, value in raw.items() if value is not None}
    selected_config = config or DEFAULT_CONFIG
    _validate_user_llm_profile_token_bounds(raw, selected_config)
    _validate_user_llm_profile_cache(raw, selected_config)
    try:
        return LLMProfile(**cleaned)
    except (TypeError, ValueError, PydanticValidationError) as exc:
        raise ValidationError(f"invalid LLM profile {profile_id}: {exc}") from exc


def _normalize_user_llm_profile_fields(raw: dict[str, Any]) -> None:
    raw["kind"] = str(raw.get("kind") or "openai_compatible")
    if raw["kind"] != "openai_compatible":
        raise ValidationError(f"unsupported LLM profile kind: {raw['kind']}")
    raw["model"] = _required_string(raw.get("model"), "model", _STRING_LIMITS["model"])
    raw["api_key_env"] = _required_env_name(raw.get("api_key_env"), "api_key_env")
    raw["base_url"] = _optional_base_url(raw.get("base_url"))
    raw["api_mode"] = _optional_choice(raw.get("api_mode"), "api_mode", _API_MODES)
    raw["verbosity"] = _optional_choice(raw.get("verbosity"), "verbosity", _VERBOSITY)
    raw["reasoning_effort"] = _optional_string(raw.get("reasoning_effort"), "reasoning_effort")
    raw["safety_identifier_env"] = _optional_env_name(raw.get("safety_identifier_env"), "safety_identifier_env")
    raw["prompt_cache_retention"] = _optional_prompt_cache_retention(
        raw.get("prompt_cache_retention")
    )
    raw["prompt_cache_mode"] = _optional_choice(
        raw.get("prompt_cache_mode"),
        "prompt_cache_mode",
        _PROMPT_CACHE_MODES,
    )
    raw["prompt_cache_ttl"] = _optional_prompt_cache_ttl(
        raw.get("prompt_cache_ttl")
    )
    raw["timeout_s"] = _optional_positive_float(raw.get("timeout_s"), "timeout_s")
    raw["max_retries"] = _optional_nonnegative_int(raw.get("max_retries"), "max_retries")
    raw["max_tokens"] = _optional_positive_int(raw.get("max_tokens"), "max_tokens")
    raw["max_input_tokens_per_call"] = _optional_positive_int(
        raw.get("max_input_tokens_per_call"),
        "max_input_tokens_per_call",
    )
    raw["max_total_tokens_per_call"] = _optional_positive_int(
        raw.get("max_total_tokens_per_call"),
        "max_total_tokens_per_call",
    )
    raw["context_window_tokens"] = _optional_positive_int(
        raw.get("context_window_tokens"),
        "context_window_tokens",
    )
    raw["temperature"] = _optional_nonnegative_float(raw.get("temperature"), "temperature")
    for key in (
        "store",
        "responses_previous_response_id",
        "parallel_tool_calls",
        "auto_wait_on_empty_tool_calls",
        "fallback_json_actions",
        "allow_custom_base_url",
    ):
        raw[key] = _optional_bool(raw.get(key), key)


def _validate_user_llm_profile_token_bounds(
    raw: Mapping[str, Any],
    config: AgentLibOSConfig,
) -> None:
    effective_max_tokens = raw["max_tokens"] or config.llm.max_tokens
    effective_context_window = (
        raw["context_window_tokens"]
        or config.llm.context_window_tokens
    )
    effective_max_input_tokens = (
        raw["max_input_tokens_per_call"]
        or config.llm.max_input_tokens_per_call
    )
    effective_max_total_tokens = (
        raw["max_total_tokens_per_call"]
        or config.llm.max_total_tokens_per_call
    )
    if effective_max_tokens >= effective_context_window:
        raise ValidationError(
            "LLM profile max_tokens must be less than context_window_tokens"
        )
    if effective_max_input_tokens > effective_max_total_tokens:
        raise ValidationError(
            "LLM profile max_input_tokens_per_call must not exceed "
            "max_total_tokens_per_call"
        )
    if effective_max_tokens > effective_max_total_tokens:
        raise ValidationError(
            "LLM profile max_tokens must not exceed max_total_tokens_per_call"
        )


def _validate_user_llm_profile_cache(
    raw: Mapping[str, Any],
    config: AgentLibOSConfig,
) -> None:
    effective_cache_mode = (
        raw["prompt_cache_mode"] or config.llm.prompt_cache_mode
    )
    effective_cache_retention = (
        raw["prompt_cache_retention"]
        if raw["prompt_cache_retention"] is not None
        else config.llm.prompt_cache_retention
    )
    effective_cache_ttl = (
        raw["prompt_cache_ttl"]
        if raw["prompt_cache_ttl"] is not None
        else config.llm.prompt_cache_ttl
    )
    if effective_cache_retention is not None and effective_cache_ttl is not None:
        raise ValidationError(
            "LLM profile prompt_cache_retention and prompt_cache_ttl are mutually exclusive"
        )
    if (
        effective_cache_mode != "provider_default"
        and not str(config.llm.prompt_cache_key or "").strip()
    ):
        raise ValidationError(
            "LLM profile implicit/explicit prompt caching requires a Host prompt_cache_key"
        )
    if effective_cache_mode == "provider_default" and effective_cache_ttl is not None:
        raise ValidationError(
            "LLM profile prompt_cache_ttl requires implicit or explicit mode"
        )
    if effective_cache_mode != "provider_default" and effective_cache_retention is not None:
        raise ValidationError(
            "LLM profile legacy prompt_cache_retention cannot be combined with implicit/explicit mode"
        )


def serialize_user_llm_profile(profile: LLMProfile) -> dict[str, Any]:
    data = asdict(profile)
    serialized: dict[str, Any] = {}
    for key in _USER_PROFILE_FIELDS:
        value = data.get(key)
        if value is None:
            continue
        if key == "kind" and value == "openai_compatible":
            continue
        if key == "prompt_cache_retention":
            value = _optional_prompt_cache_retention(value)
        if key == "prompt_cache_ttl":
            value = _optional_prompt_cache_ttl(value)
        serialized[key] = value
    return serialized


def summarize_llm_profile(
    profile_id: str,
    profile: LLMProfile,
    *,
    source: str,
    editable: bool,
    default_profile_id: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected_env = os.environ if env is None else env
    api_key_env = profile.api_key_env
    return {
        "profile_id": profile_id,
        "model": profile.model,
        "base_url": sanitize_llm_base_url_for_summary(profile.base_url),
        "api_key_env": api_key_env,
        "api_key_env_present": bool(str(selected_env.get(api_key_env) or "").strip()),
        "api_mode": profile.api_mode,
        "timeout_s": profile.timeout_s,
        "max_retries": profile.max_retries,
        "store": profile.store,
        "reasoning_effort": profile.reasoning_effort,
        "verbosity": profile.verbosity,
        "safety_identifier_env": profile.safety_identifier_env,
        "prompt_cache_retention": _optional_prompt_cache_retention(
            profile.prompt_cache_retention
        ),
        "prompt_cache_mode": profile.prompt_cache_mode,
        "prompt_cache_ttl": _optional_prompt_cache_ttl(
            profile.prompt_cache_ttl
        ),
        "responses_previous_response_id": profile.responses_previous_response_id,
        "parallel_tool_calls": profile.parallel_tool_calls,
        "auto_wait_on_empty_tool_calls": profile.auto_wait_on_empty_tool_calls,
        "fallback_json_actions": profile.fallback_json_actions,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "max_input_tokens_per_call": profile.max_input_tokens_per_call,
        "max_total_tokens_per_call": profile.max_total_tokens_per_call,
        "context_window_tokens": profile.context_window_tokens,
        "allow_custom_base_url": profile.allow_custom_base_url,
        "source": source,
        "editable": editable,
        "is_default": profile_id == default_profile_id,
    }


def _required_string(value: Any, label: str, limit: int) -> str:
    selected = str(value or "").strip()
    if not selected:
        raise ValidationError(f"LLM profile {label} must be non-empty")
    if len(selected) > limit:
        raise ValidationError(f"LLM profile {label} exceeds {limit} characters")
    return selected


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    selected = str(value).strip()
    if not selected:
        return None
    limit = _STRING_LIMITS[label]
    if len(selected) > limit:
        raise ValidationError(f"LLM profile {label} exceeds {limit} characters")
    return selected


def _required_env_name(value: Any, label: str) -> str:
    selected = _required_string(value, label, _STRING_LIMITS[label])
    if not ENV_NAME_RE.match(selected):
        raise ValidationError(f"LLM profile {label} must be an environment variable name")
    return selected


def _optional_env_name(value: Any, label: str) -> str | None:
    selected = _optional_string(value, label)
    if selected is None:
        return None
    if not ENV_NAME_RE.match(selected):
        raise ValidationError(f"LLM profile {label} must be an environment variable name")
    return selected


def _optional_base_url(value: Any) -> str | None:
    selected = _optional_string(value, "base_url")
    if selected is None:
        return None
    try:
        validate_llm_base_url(selected)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return selected.rstrip("/")


def _optional_choice(value: Any, label: str, choices: set[str]) -> str | None:
    if value is None:
        return None
    selected = str(value).strip().lower()
    if not selected:
        return None
    if selected not in choices:
        raise ValidationError(f"LLM profile {label} must be one of {sorted(choices)}")
    return selected


def _optional_prompt_cache_retention(value: Any) -> str | None:
    if value is None:
        return None
    selected = str(value).strip().lower()
    if not selected:
        return None
    if selected == "in-memory":
        return "in_memory"
    if selected not in {"in_memory", "24h"}:
        raise ValidationError(
            "LLM profile prompt_cache_retention must be one of ['24h', 'in_memory']"
        )
    return selected


def _optional_prompt_cache_ttl(value: Any) -> str | None:
    if value is None:
        return None
    selected = str(value).strip().lower()
    if not selected:
        return None
    if selected != "30m":
        raise ValidationError("LLM profile prompt_cache_ttl must be '30m'")
    return selected


def _optional_bool(value: Any, label: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValidationError(f"LLM profile {label} must be boolean")
    return value


def _optional_float(value: Any, label: str) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValidationError(f"LLM profile {label} must be a finite number")
    try:
        selected = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"LLM profile {label} must be a finite number") from exc
    if not math.isfinite(selected):
        raise ValidationError(f"LLM profile {label} must be a finite number")
    return selected


def _optional_positive_float(value: Any, label: str) -> float | None:
    selected = _optional_float(value, label)
    if selected is not None and selected <= 0:
        raise ValidationError(f"LLM profile {label} must be positive")
    return selected


def _optional_nonnegative_float(value: Any, label: str) -> float | None:
    selected = _optional_float(value, label)
    if selected is not None and selected < 0:
        raise ValidationError(f"LLM profile {label} must be non-negative")
    return selected


def _optional_nonnegative_int(value: Any, label: str) -> int | None:
    selected = _optional_int(value, label)
    if selected is not None and selected < 0:
        raise ValidationError(f"LLM profile {label} must be non-negative")
    return selected


def _optional_positive_int(value: Any, label: str) -> int | None:
    selected = _optional_int(value, label)
    if selected is not None and selected <= 0:
        raise ValidationError(f"LLM profile {label} must be positive")
    return selected


def _optional_int(value: Any, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValidationError(f"LLM profile {label} must be an integer")
    try:
        selected = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"LLM profile {label} must be an integer") from exc
    if str(value).strip() not in {str(selected), f"{selected}.0"} and not isinstance(value, int):
        raise ValidationError(f"LLM profile {label} must be an integer")
    return selected
