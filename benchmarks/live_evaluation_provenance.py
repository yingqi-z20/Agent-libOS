from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, LLMProfile
from agent_libos.models.exceptions import GitError
from agent_libos.substrate.git import LocalGitProvider


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_MAX_UNTRACKED_FILE_BYTES = 16 * 1024 * 1024
_MAX_UNTRACKED_TOTAL_BYTES = 64 * 1024 * 1024
_DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class EvaluationProvenance(TypedDict):
    """A stable, redacted source and effective LLM-configuration envelope."""

    schema_version: int
    start: dict[str, Any]
    end: dict[str, Any]
    stable: bool


def capture_source_provenance() -> dict[str, Any]:
    """Bind a live report to one bounded Git working-tree identity."""

    provider = LocalGitProvider(REPOSITORY_ROOT, config=DEFAULT_CONFIG.git)
    try:
        for operation in ("repository_info", "list_refs", "status", "diff"):
            provider.validate_read_only_operation(operation)
        commit = _git(provider, "rev-parse", "HEAD").strip()
        status = _git(
            provider,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        diff = _git(
            provider,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "HEAD",
            "--",
        )
        untracked = _git(
            provider,
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        )
        digest = hashlib.sha256()
        digest.update(commit)
        digest.update(b"\0status\0")
        digest.update(status)
        digest.update(b"\0diff\0")
        digest.update(diff)
        total = 0
        for raw_name in sorted(item for item in untracked.split(b"\0") if item):
            relative = raw_name.decode("utf-8", errors="surrogateescape")
            source = REPOSITORY_ROOT / relative
            digest.update(b"\0untracked\0")
            digest.update(raw_name)
            if source.is_symlink():
                digest.update(b"\0symlink\0")
                digest.update(os.readlink(source).encode("utf-8", errors="surrogateescape"))
                continue
            if not source.is_file():
                raise RuntimeError("untracked provenance entry is not a regular file")
            size = source.stat().st_size
            if size > _MAX_UNTRACKED_FILE_BYTES:
                raise RuntimeError("untracked provenance file exceeds the size limit")
            total += size
            if total > _MAX_UNTRACKED_TOTAL_BYTES:
                raise RuntimeError("untracked provenance exceeds the aggregate size limit")
            digest.update(source.read_bytes())
        return {
            "schema_version": 1,
            "available": True,
            "commit": commit.decode("ascii") or None,
            "dirty": bool(status),
            "working_tree_sha256": digest.hexdigest(),
        }
    except (GitError, OSError, RuntimeError, UnicodeError):
        return {
            "schema_version": 1,
            "available": False,
            "commit": None,
            "dirty": None,
            "working_tree_sha256": None,
        }


def build_source_provenance(
    start: dict[str, Any],
    end: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "start": start,
        "end": end,
        "stable": start == end and start.get("available") is True,
    }


def valid_stable_source_provenance(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        return False
    start = value.get("start")
    end = value.get("end")
    return bool(
        value.get("stable") is True
        and isinstance(start, dict)
        and isinstance(end, dict)
        and start == end
        and start.get("available") is True
        and isinstance(start.get("commit"), str)
        and len(start["commit"]) in {40, 64}
        and isinstance(start.get("dirty"), bool)
        and isinstance(start.get("working_tree_sha256"), str)
        and len(start["working_tree_sha256"]) == 64
    )


def capture_evaluation_provenance(
    config: AgentLibOSConfig | None = None,
) -> dict[str, Any]:
    """Capture one source/config snapshot without retaining provider secrets.

    The exact model is publication metadata.  Provider endpoints are reduced
    to an official/custom classification and a hash of their normalized form;
    credentials, raw endpoints, cache keys, safety identifiers, organization,
    and project values are never included.
    """

    selected_config = config or DEFAULT_CONFIG
    return {
        "schema_version": 1,
        "source": capture_source_provenance(),
        "llm": _capture_safe_llm_config(selected_config),
    }


def build_evaluation_provenance(
    start: dict[str, Any],
    end: dict[str, Any],
) -> EvaluationProvenance:
    """Bind a report to one unchanged source and effective LLM identity."""

    return {
        "schema_version": 1,
        "start": start,
        "end": end,
        "stable": start == end and _valid_evaluation_snapshot(start),
    }


def valid_evaluation_provenance(value: Any) -> bool:
    """Validate an evaluation envelope and recompute its safe config digest."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "start",
        "end",
        "stable",
    }:
        return False
    start = value.get("start")
    end = value.get("end")
    return bool(
        value.get("schema_version") == 1
        and value.get("stable") is True
        and isinstance(start, dict)
        and isinstance(end, dict)
        and start == end
        and _valid_evaluation_snapshot(start)
        and _valid_evaluation_snapshot(end)
    )


def evaluation_provenance_identity(value: Any) -> dict[str, Any] | None:
    """Return the stable safe snapshot used for cross-report comparison."""

    if not valid_evaluation_provenance(value):
        return None
    return deepcopy(value["end"])


def live_evaluation_provenance_ready(value: Any) -> bool:
    """Require the extra model and credential facts needed by paid live gates."""

    identity = evaluation_provenance_identity(value)
    if identity is None:
        return False
    llm = identity.get("llm")
    return bool(
        isinstance(llm, dict)
        and isinstance(llm.get("model"), str)
        and bool(llm["model"].strip())
        and llm.get("credential_present") is True
    )


def _capture_safe_llm_config(config: AgentLibOSConfig) -> dict[str, Any]:
    try:
        profile_id = config.llm.default_profile_id
        profile = config.llm.profiles[profile_id]
        uses_legacy_environment = profile_id == config.llm.default_profile_id
        environment = dict(os.environ) if uses_legacy_environment else {}
        base_url = profile.base_url or _optional_env(environment, "OPENAI_BASE_URL")
        model = (
            profile.model
            or _optional_env(environment, "OPENAI_LANGUAGE_MODEL")
            or _optional_env(environment, "OPENAI_MODEL")
        )
        custom_endpoint_environment_opt_in = (
            _environment_bool_or_none(
                environment,
                "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL",
            )
            is True
        )
        allow_custom_base_url = bool(
            profile.allow_custom_base_url or custom_endpoint_environment_opt_in
        )
        endpoint = _safe_endpoint_identity(
            base_url or _DEFAULT_OPENAI_ENDPOINT,
            custom_endpoint_allowed=allow_custom_base_url,
        )
        api_mode = (
            profile.api_mode
            or _optional_env(environment, "OPENAI_API_MODE")
            or config.llm.api_mode
        ).strip().lower()
        if api_mode not in {"auto", "responses", "chat"}:
            raise ValueError("invalid API mode")
        timeout_s = (
            profile.timeout_s
            if profile.timeout_s is not None
            else _float_env(environment, "OPENAI_TIMEOUT", config.llm.timeout_s)
        )
        max_retries = (
            profile.max_retries
            if profile.max_retries is not None
            else _int_env(
                environment,
                "OPENAI_MAX_RETRIES",
                config.llm.max_retries,
            )
        )
        cache_retention = _normalize_cache_retention(
            profile.prompt_cache_retention
            if profile.prompt_cache_retention is not None
            else _optional_env(environment, "OPENAI_PROMPT_CACHE_RETENTION")
            or config.llm.prompt_cache_retention
        )
        cache_mode = _normalize_cache_mode(
            profile.prompt_cache_mode
            if profile.prompt_cache_mode is not None
            else _optional_env(environment, "OPENAI_PROMPT_CACHE_MODE")
            or config.llm.prompt_cache_mode
        )
        cache_ttl = _normalize_cache_ttl(
            profile.prompt_cache_ttl
            if profile.prompt_cache_ttl is not None
            else _optional_env(environment, "OPENAI_PROMPT_CACHE_TTL")
            or config.llm.prompt_cache_ttl
        )
        cache_key = (
            profile.prompt_cache_key
            if profile.prompt_cache_key is not None
            else _optional_env(environment, "OPENAI_PROMPT_CACHE_KEY")
            or config.llm.prompt_cache_key
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "available": True,
            "provider_kind": profile.kind,
            "profile_id": profile_id,
            "model": model,
            "endpoint": endpoint,
            "api_mode": api_mode,
            "credential_present": bool(
                _optional_env(environment, profile.api_key_env)
            ),
            "request": {
                "temperature": _profile_or_default(
                    profile,
                    config,
                    "temperature",
                ),
                "max_tokens": _profile_or_default(profile, config, "max_tokens"),
                "max_input_tokens_per_call": _profile_or_default(
                    profile,
                    config,
                    "max_input_tokens_per_call",
                ),
                "max_total_tokens_per_call": _profile_or_default(
                    profile,
                    config,
                    "max_total_tokens_per_call",
                ),
                "context_window_tokens": _profile_or_default(
                    profile,
                    config,
                    "context_window_tokens",
                ),
                "timeout_s": timeout_s,
                "max_retries": max_retries,
                "compatibility_retry_attempts": (
                    config.llm.compatibility_retry_attempts
                ),
                "action_repair_attempts": config.llm.action_repair_attempts,
                "parallel_tool_calls": _profile_bool_or_environment(
                    profile.parallel_tool_calls,
                    environment,
                    "OPENAI_PARALLEL_TOOL_CALLS",
                    config.llm.parallel_tool_calls,
                ),
                "auto_wait_on_empty_tool_calls": (
                    profile.auto_wait_on_empty_tool_calls
                    if profile.auto_wait_on_empty_tool_calls is not None
                    else config.llm.auto_wait_on_empty_tool_calls
                ),
                "enable_thinking": (
                    _environment_bool_or_none(
                        environment,
                        "OPENAI_ENABLE_THINKING",
                    )
                ),
            },
            "prompt": {
                "layout": config.llm.prompt_layout,
                "store": _profile_bool_or_environment(
                    profile.store,
                    environment,
                    "OPENAI_STORE",
                    config.llm.store,
                ),
                "cache_mode": cache_mode,
                "cache_retention": cache_retention,
                "cache_ttl": cache_ttl,
                "cache_key_configured": cache_key is not None,
                "responses_previous_response_id": _profile_bool_or_environment(
                    profile.responses_previous_response_id,
                    environment,
                    "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
                    config.llm.responses_previous_response_id,
                ),
                "fallback_json_actions": _profile_bool_or_environment(
                    profile.fallback_json_actions,
                    environment,
                    "OPENAI_FALLBACK_JSON_ACTIONS",
                    config.llm.fallback_json_actions,
                ),
                "reasoning_effort": (
                    profile.reasoning_effort
                    if profile.reasoning_effort is not None
                    else _optional_env(environment, "OPENAI_REASONING_EFFORT")
                ),
                "verbosity": (
                    profile.verbosity
                    if profile.verbosity is not None
                    else _lower_optional(
                        _optional_env(environment, "OPENAI_VERBOSITY")
                    )
                ),
            },
        }
        payload["config_sha256"] = _safe_llm_config_digest(payload)
        return payload
    except (KeyError, OSError, TypeError, ValueError):
        return _unavailable_llm_config()


def _valid_evaluation_snapshot(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "source",
        "llm",
    }:
        return False
    return bool(
        value.get("schema_version") == 1
        and _valid_source_identity(value.get("source"))
        and _valid_safe_llm_config(value.get("llm"))
    )


def _valid_source_identity(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value)
        == {
            "schema_version",
            "available",
            "commit",
            "dirty",
            "working_tree_sha256",
        }
        and value.get("schema_version") == 1
        and value.get("available") is True
        and _is_hex_digest(value.get("commit"), lengths={40, 64})
        and type(value.get("dirty")) is bool
        and _is_sha256(value.get("working_tree_sha256"))
    )


def _valid_safe_llm_config(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "available",
        "provider_kind",
        "profile_id",
        "model",
        "endpoint",
        "api_mode",
        "credential_present",
        "request",
        "prompt",
        "config_sha256",
    }:
        return False
    endpoint = value.get("endpoint")
    request = value.get("request")
    prompt = value.get("prompt")
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "classification",
        "normalized_sha256",
        "custom_endpoint_allowed",
    }:
        return False
    if not isinstance(request, dict) or set(request) != {
        "temperature",
        "max_tokens",
        "max_input_tokens_per_call",
        "max_total_tokens_per_call",
        "context_window_tokens",
        "timeout_s",
        "max_retries",
        "compatibility_retry_attempts",
        "action_repair_attempts",
        "parallel_tool_calls",
        "auto_wait_on_empty_tool_calls",
        "enable_thinking",
    }:
        return False
    if not isinstance(prompt, dict) or set(prompt) != {
        "layout",
        "store",
        "cache_mode",
        "cache_retention",
        "cache_ttl",
        "cache_key_configured",
        "responses_previous_response_id",
        "fallback_json_actions",
        "reasoning_effort",
        "verbosity",
    }:
        return False
    digest = value.get("config_sha256")
    without_digest = dict(value)
    without_digest.pop("config_sha256", None)
    return bool(
        value.get("schema_version") == 1
        and value.get("available") is True
        and value.get("provider_kind") == "openai_compatible"
        and isinstance(value.get("profile_id"), str)
        and bool(value["profile_id"].strip())
        and (
            value.get("model") is None
            or (
                isinstance(value.get("model"), str)
                and bool(value["model"].strip())
            )
        )
        and endpoint.get("classification") in {"official_openai", "custom"}
        and _is_sha256(endpoint.get("normalized_sha256"))
        and type(endpoint.get("custom_endpoint_allowed")) is bool
        and value.get("api_mode") in {"auto", "responses", "chat"}
        and type(value.get("credential_present")) is bool
        and _is_nonnegative_number(request.get("temperature"))
        and _is_positive_int(request.get("max_tokens"))
        and _is_positive_int(request.get("max_input_tokens_per_call"))
        and _is_positive_int(request.get("max_total_tokens_per_call"))
        and _is_positive_int(request.get("context_window_tokens"))
        and _is_positive_number(request.get("timeout_s"))
        and _is_nonnegative_int(request.get("max_retries"))
        and _is_positive_int(request.get("compatibility_retry_attempts"))
        and _is_positive_int(request.get("action_repair_attempts"))
        and request["max_tokens"] < request["context_window_tokens"]
        and request["max_input_tokens_per_call"]
        <= request["max_total_tokens_per_call"]
        and request["max_tokens"] <= request["max_total_tokens_per_call"]
        and type(request.get("parallel_tool_calls")) is bool
        and type(request.get("auto_wait_on_empty_tool_calls")) is bool
        and (
            request.get("enable_thinking") is None
            or type(request.get("enable_thinking")) is bool
        )
        and prompt.get("layout") in {"legacy_v1", "cache_optimized_v2"}
        and type(prompt.get("store")) is bool
        and prompt.get("cache_mode")
        in {"provider_default", "implicit", "explicit"}
        and prompt.get("cache_retention") in {None, "in_memory", "24h"}
        and prompt.get("cache_ttl") in {None, "30m"}
        and type(prompt.get("cache_key_configured")) is bool
        and type(prompt.get("responses_previous_response_id")) is bool
        and type(prompt.get("fallback_json_actions")) is bool
        and (
            prompt.get("reasoning_effort") is None
            or (
                isinstance(prompt.get("reasoning_effort"), str)
                and bool(prompt["reasoning_effort"].strip())
            )
        )
        and prompt.get("verbosity") in {None, "low", "medium", "high"}
        and isinstance(digest, str)
        and _is_sha256(digest)
        and digest == _safe_llm_config_digest(without_digest)
    )


def _safe_llm_config_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_endpoint_identity(
    raw_endpoint: str,
    *,
    custom_endpoint_allowed: bool,
) -> dict[str, Any]:
    parsed = urlsplit(raw_endpoint.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("invalid provider endpoint")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("provider endpoint must not contain credentials")
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower().rstrip(".")
    port = parsed.port
    default_port = (scheme == "https" and port == 443) or (
        scheme == "http" and port == 80
    )
    authority = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path.rstrip("/") or "/"
    normalized = urlunsplit((scheme, authority, path, parsed.query, ""))
    classification = (
        "official_openai"
        if scheme == "https" and hostname == "api.openai.com"
        else "custom"
    )
    return {
        "classification": classification,
        "normalized_sha256": hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        "custom_endpoint_allowed": custom_endpoint_allowed,
    }


def _profile_or_default(
    profile: LLMProfile,
    config: AgentLibOSConfig,
    name: str,
) -> Any:
    selected = getattr(profile, name)
    return getattr(config.llm, name) if selected is None else selected


def _profile_bool_or_environment(
    profile_value: bool | None,
    environment: dict[str, str],
    environment_key: str,
    default: bool,
) -> bool:
    if profile_value is not None:
        return profile_value
    value = _optional_env(environment, environment_key)
    return default if value is None else value.lower() in _TRUE_VALUES


def _environment_bool_or_none(
    environment: dict[str, str],
    key: str,
) -> bool | None:
    value = _optional_env(environment, key)
    return None if value is None else value.lower() in _TRUE_VALUES


def _lower_optional(value: str | None) -> str | None:
    return None if value is None else value.strip().lower()


def _normalize_cache_retention(value: str | None) -> str | None:
    selected = _lower_optional(value)
    if selected == "in-memory":
        return "in_memory"
    if selected not in {None, "in_memory", "24h"}:
        raise ValueError("invalid prompt cache retention")
    return selected


def _normalize_cache_mode(value: str) -> str:
    selected = value.strip().lower()
    if selected not in {"provider_default", "implicit", "explicit"}:
        raise ValueError("invalid prompt cache mode")
    return selected


def _normalize_cache_ttl(value: str | None) -> str | None:
    selected = _lower_optional(value)
    if selected not in {None, "30m"}:
        raise ValueError("invalid prompt cache TTL")
    return selected


def _optional_env(environment: Any, key: str) -> str | None:
    value = environment.get(key)
    if value is None:
        return None
    selected = str(value).strip()
    return selected or None


def _float_env(environment: dict[str, str], key: str, default: float) -> float:
    value = _optional_env(environment, key)
    return default if value is None else float(value)


def _int_env(environment: dict[str, str], key: str, default: int) -> int:
    value = _optional_env(environment, key)
    return default if value is None else int(value)


def _is_sha256(value: Any) -> bool:
    return _is_hex_digest(value, lengths={64})


def _is_hex_digest(value: Any, *, lengths: set[int]) -> bool:
    if not isinstance(value, str) or len(value) not in lengths:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value == value and value not in {float("inf"), float("-inf")}


def _is_positive_number(value: Any) -> bool:
    return _is_finite_number(value) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return _is_finite_number(value) and value >= 0


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _is_nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _unavailable_llm_config() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "available": False,
        "provider_kind": None,
        "profile_id": None,
        "model": None,
        "endpoint": {
            "classification": None,
            "normalized_sha256": None,
            "custom_endpoint_allowed": None,
        },
        "api_mode": None,
        "credential_present": False,
        "request": {
            "temperature": None,
            "max_tokens": None,
            "max_input_tokens_per_call": None,
            "max_total_tokens_per_call": None,
            "context_window_tokens": None,
            "timeout_s": None,
            "max_retries": None,
            "compatibility_retry_attempts": None,
            "action_repair_attempts": None,
            "parallel_tool_calls": None,
            "auto_wait_on_empty_tool_calls": None,
            "enable_thinking": None,
        },
        "prompt": {
            "layout": None,
            "store": None,
            "cache_mode": None,
            "cache_retention": None,
            "cache_ttl": None,
            "cache_key_configured": False,
            "responses_previous_response_id": None,
            "fallback_json_actions": None,
            "reasoning_effort": None,
            "verbosity": None,
        },
    }
    payload["config_sha256"] = _safe_llm_config_digest(payload)
    return payload


def _git(provider: LocalGitProvider, *args: str) -> bytes:
    result = provider.run(
        args,
        read_only=True,
        max_output_bytes=DEFAULT_CONFIG.git.output_hard_limit_bytes,
    )
    if result.returncode != 0:
        raise RuntimeError("Git provenance command failed")
    return result.stdout
