from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import threading
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig, LLMProfile
from agent_libos.llm.client import LLMClient, LLMError
from agent_libos.models.exceptions import NotFound, ValidationError
from agent_libos.ports.blocking_work import run_blocking_once
from agent_libos.storage import ProcessRepository
from agent_libos.utils.serde import dumps

_TRUE_VALUES = {"1", "true", "yes", "on"}
_API_MODES = {"auto", "responses", "chat"}
_LEGACY_PROFILE_ENV_KEYS = {
    "OPENAI_API_MODE",
    "OPENAI_BASE_URL",
    "OPENAI_ENABLE_THINKING",
    "OPENAI_FALLBACK_JSON_ACTIONS",
    "OPENAI_LANGUAGE_MODEL",
    "OPENAI_MAX_RETRIES",
    "OPENAI_MODEL",
    "OPENAI_ORGANIZATION",
    "OPENAI_ORG_ID",
    "OPENAI_PARALLEL_TOOL_CALLS",
    "OPENAI_PROMPT_CACHE_KEY",
    "OPENAI_PROMPT_CACHE_MODE",
    "OPENAI_PROMPT_CACHE_RETENTION",
    "OPENAI_PROMPT_CACHE_TTL",
    "OPENAI_PROJECT",
    "OPENAI_PROJECT_ID",
    "OPENAI_REASONING_EFFORT",
    "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
    "OPENAI_SAFETY_IDENTIFIER",
    "OPENAI_STORE",
    "OPENAI_TIMEOUT",
    "OPENAI_VERBOSITY",
}


@dataclass(frozen=True)
class ResolvedLLMProfile:
    profile_id: str
    profile: LLMProfile
    client: Any
    identity_sha256: str
    temperature: float
    max_tokens: int
    max_input_tokens_per_call: int
    max_total_tokens_per_call: int
    context_window_tokens: int
    parallel_tool_calls: bool
    auto_wait_on_empty_tool_calls: bool
    fallback_json_actions: bool


@dataclass(frozen=True)
class _ResolvedLLMPolicy:
    api_mode: str
    store: bool
    prompt_cache_retention: str | None
    prompt_cache_mode: str
    prompt_cache_ttl: str | None
    responses_previous_response_id: bool
    fallback_json_actions: bool


@dataclass(frozen=True)
class _LLMProfileSnapshot:
    profile_id: str
    profile: LLMProfile
    legacy_env: Mapping[str, str] = field(repr=False)
    client_env: Mapping[str, str] = field(repr=False)
    policy: _ResolvedLLMPolicy
    identity_sha256: str
    client_cache_sha256: str = field(repr=False)


class LLMProfileRegistry:
    """Host-side resolver for per-process LLM clients.

    Processes persist only a profile id. The profile registry owns the host
    client cache and reads secrets from environment variables when a real
    OpenAI-compatible client is first needed.
    """

    def __init__(
        self,
        processes: ProcessRepository,
        *,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self._processes = processes
        self.config = config or DEFAULT_CONFIG
        self._profiles: dict[str, LLMProfile] = {}
        self._clients: dict[str, Any] = {}
        self._client_cache_sha256: dict[str, str] = {}
        self._test_clients: dict[str, Any] = {}
        self._lock = threading.RLock()

    def register_profile(self, profile_id: str, profile: LLMProfile | dict[str, Any]) -> None:
        selected_id = self._normalize_profile_id(profile_id)
        selected_profile = profile if isinstance(profile, LLMProfile) else LLMProfile(**dict(profile))
        self._validate_profile(selected_id, selected_profile)
        with self._lock:
            self._profiles[selected_id] = selected_profile
            stale = self._clients.pop(selected_id, None)
            self._client_cache_sha256.pop(selected_id, None)
        self._shutdown_client(stale)

    def unregister_profile(self, profile_id: str) -> None:
        selected_id = self._normalize_profile_id(profile_id)
        with self._lock:
            if selected_id not in self._profiles:
                raise ValidationError(f"LLM profile is not dynamically registered: {selected_id}")
            self._profiles.pop(selected_id)
            stale_client = self._clients.pop(selected_id, None)
            self._client_cache_sha256.pop(selected_id, None)
            stale_test_client = self._test_clients.pop(selected_id, None)
        self._shutdown_client(stale_client)
        if stale_test_client is not stale_client:
            self._shutdown_client(stale_test_client)

    def set_test_client(self, profile_id: str, client: Any) -> None:
        selected_id = self.require_profile_id(profile_id)
        with self._lock:
            stale = self._test_clients.get(selected_id)
            self._test_clients[selected_id] = client
        if stale is not None and stale is not client:
            self._shutdown_client(stale)

    def clear_test_client(self, profile_id: str) -> None:
        selected_id = self.require_profile_id(profile_id)
        with self._lock:
            stale = self._test_clients.pop(selected_id, None)
        self._shutdown_client(stale)

    def resolve_for_process(self, pid: str) -> ResolvedLLMProfile:
        process = self._processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        profile_id = process.llm_profile_id or self.config.llm.default_profile_id
        return self.resolve(profile_id)

    def client_for_process(self, pid: str) -> Any:
        return self.resolve_for_process(pid).client

    def fallback_json_actions_for_process(self, pid: str) -> bool:
        """Resolve compatibility policy without constructing a provider client."""

        process = self._processes.get_process(pid)
        if process is None:
            raise NotFound(f"process not found: {pid}")
        profile_id = process.llm_profile_id or self.config.llm.default_profile_id
        return self.profile_snapshot(profile_id).policy.fallback_json_actions

    def resolve(
        self,
        profile_id: str,
        *,
        snapshot: _LLMProfileSnapshot | None = None,
    ) -> ResolvedLLMProfile:
        selected_id = self._normalize_profile_id(profile_id)
        with self._lock:
            profile = self._profile_unlocked(selected_id)
            if snapshot is None:
                snapshot = self._profile_snapshot(selected_id, profile)
            elif snapshot.profile_id != selected_id or snapshot.profile != profile:
                raise ValidationError(f"LLM profile changed after resolution snapshot: {selected_id}")

            client = self._test_clients.get(selected_id)
            if client is None:
                client = self._clients.get(selected_id)
                if (
                    client is not None
                    and self._client_cache_sha256.get(selected_id)
                    != snapshot.client_cache_sha256
                ):
                    stale = self._clients.pop(selected_id)
                    self._client_cache_sha256.pop(selected_id, None)
                    self._shutdown_client(stale)
                    client = None
                if client is None:
                    client = self._create_client(selected_id, profile, snapshot=snapshot)
                    self._clients[selected_id] = client
                    self._client_cache_sha256[selected_id] = snapshot.client_cache_sha256
            return ResolvedLLMProfile(
                profile_id=selected_id,
                profile=profile,
                client=client,
                identity_sha256=snapshot.identity_sha256,
                temperature=self.config.llm.temperature if profile.temperature is None else profile.temperature,
                max_tokens=self.config.llm.max_tokens if profile.max_tokens is None else profile.max_tokens,
                max_input_tokens_per_call=(
                    self.config.llm.max_input_tokens_per_call
                    if profile.max_input_tokens_per_call is None
                    else profile.max_input_tokens_per_call
                ),
                max_total_tokens_per_call=(
                    self.config.llm.max_total_tokens_per_call
                    if profile.max_total_tokens_per_call is None
                    else profile.max_total_tokens_per_call
                ),
                context_window_tokens=(
                    self.config.llm.context_window_tokens
                    if profile.context_window_tokens is None
                    else profile.context_window_tokens
                ),
                parallel_tool_calls=self._resolved_parallel_tool_calls(profile, client),
                auto_wait_on_empty_tool_calls=self._resolved_auto_wait_on_empty_tool_calls(profile),
                fallback_json_actions=snapshot.policy.fallback_json_actions,
            )

    @property
    def default_client(self) -> Any:
        return self.resolve(self.config.llm.default_profile_id).client

    def profile(self, profile_id: str) -> LLMProfile:
        selected_id = self._normalize_profile_id(profile_id)
        with self._lock:
            return self._profile_unlocked(selected_id)

    def _profile_unlocked(self, profile_id: str) -> LLMProfile:
        selected_id = self._normalize_profile_id(profile_id)
        if selected_id in self._profiles:
            return self._profiles[selected_id]
        profile = self.config.llm.profiles.get(selected_id)
        if profile is None:
            raise ValidationError(f"unknown LLM profile: {selected_id}")
        return profile

    def require_profile_id(self, profile_id: str | None) -> str:
        selected_id = self._normalize_profile_id(profile_id)
        self.profile(selected_id)
        return selected_id

    def profile_identity_sha256(self, profile_id: str) -> str:
        """Return the non-secret Host configuration identity for Sink trust."""

        return self.profile_snapshot(profile_id).identity_sha256

    def profile_snapshot(self, profile_id: str) -> _LLMProfileSnapshot:
        """Freeze one Host profile, policy, and client-environment resolution."""

        selected_id = self._normalize_profile_id(profile_id)
        with self._lock:
            profile = self._profile_unlocked(selected_id)
            return self._profile_snapshot(selected_id, profile)

    def _profile_snapshot(self, profile_id: str, profile: LLMProfile) -> _LLMProfileSnapshot:
        environment = dict(os.environ)
        legacy_env: Mapping[str, str] = (
            MappingProxyType(
                {
                    key: environment[key]
                    for key in _LEGACY_PROFILE_ENV_KEYS
                    if key in environment
                }
            )
            if self._uses_legacy_openai_env(profile_id)
            else MappingProxyType({})
        )
        client_env_keys = {
            profile.api_key_env,
            "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL",
        }
        if profile.safety_identifier_env is not None:
            client_env_keys.add(profile.safety_identifier_env)
        client_env: Mapping[str, str] = MappingProxyType(
            {
                key: environment[key]
                for key in client_env_keys
                if key in environment
            }
        )
        policy = self._resolved_policy(profile, legacy_env)
        identity_profile = asdict(profile)
        identity_profile["prompt_cache_retention"] = _normalize_prompt_cache_retention(
            profile.prompt_cache_retention,
            label=f"LLM profile {profile_id} prompt_cache_retention",
        )
        # Model-window sizing and per-call budget envelopes control only local
        # scheduling/admission. They do not identify the external Provider/Sink
        # and must not invalidate existing trust rules when hosts change them.
        for local_policy_field in (
            "context_window_tokens",
            "max_input_tokens_per_call",
            "max_total_tokens_per_call",
        ):
            identity_profile.pop(local_policy_field, None)
        identity = {
            "schema_version": 1,
            "profile_id": profile_id,
            "profile": identity_profile,
            "effective": {
                "base_url": profile.base_url or _optional_env(legacy_env, "OPENAI_BASE_URL"),
                "model": profile.model
                or _optional_env(legacy_env, "OPENAI_LANGUAGE_MODEL")
                or _optional_env(legacy_env, "OPENAI_MODEL"),
                "api_mode": policy.api_mode,
                "store": policy.store,
                "prompt_cache_retention": policy.prompt_cache_retention,
                "prompt_cache_mode": policy.prompt_cache_mode,
                "prompt_cache_ttl": policy.prompt_cache_ttl,
                "responses_previous_response_id": policy.responses_previous_response_id,
                "fallback_json_actions": policy.fallback_json_actions,
                "api_key_env": profile.api_key_env,
                "enable_thinking": (
                    _bool_env(legacy_env, "OPENAI_ENABLE_THINKING", False)
                    if "OPENAI_ENABLE_THINKING" in legacy_env
                    else None
                ),
                "organization": (
                    _optional_env(legacy_env, "OPENAI_ORGANIZATION")
                    or _optional_env(legacy_env, "OPENAI_ORG_ID")
                ),
                "project": (
                    _optional_env(legacy_env, "OPENAI_PROJECT")
                    or _optional_env(legacy_env, "OPENAI_PROJECT_ID")
                ),
            },
        }
        client_options = self._client_options(
            profile,
            legacy_env=legacy_env,
            client_env=client_env,
            policy=policy,
        )
        return _LLMProfileSnapshot(
            profile_id=profile_id,
            profile=profile,
            legacy_env=legacy_env,
            client_env=client_env,
            policy=policy,
            identity_sha256=hashlib.sha256(dumps(identity).encode("utf-8")).hexdigest(),
            client_cache_sha256=self._client_options_sha256(client_options),
        )

    def shutdown(self) -> None:
        """Close every cached client from synchronous host teardown.

        Synchronous method names are preferred; async-only close methods and
        awaitable close results are driven to completion by
        :meth:`_shutdown_client`. A client remains cached until its close method
        returns successfully, so a failed Runtime shutdown or failed assembly
        cleanup retains the exact Host resource for a later retry.
        """

        failures: list[BaseException] = []
        with self._lock:
            clients = self._unique_clients()
        for client in clients:
            try:
                self._shutdown_client(client)
            except BaseException as exc:
                failures.append(exc)
            else:
                self._forget_closed_client(client)
        self._raise_shutdown_failures(failures)

    async def ashutdown(self) -> None:
        """Close every cached client from asynchronous host teardown.

        Async-specific methods are preferred and synchronous methods are the
        fallback.  As with :meth:`shutdown`, only a successfully closed unique
        client is removed from every production/test cache entry that still
        refers to that exact object.
        """

        failures: list[BaseException] = []
        caller_interrupted = False
        with self._lock:
            clients = self._unique_clients()
        for client in clients:
            try:
                interrupted = await self._ashutdown_client(client)
            except BaseException as exc:
                failures.append(exc)
            else:
                self._forget_closed_client(client)
                caller_interrupted = caller_interrupted or interrupted
        if caller_interrupted:
            cancellation = asyncio.CancelledError()
            if failures:
                raise BaseExceptionGroup(
                    "LLM profile shutdown was cancelled after all clients were attempted",
                    [cancellation, *failures],
                )
            raise cancellation
        self._raise_shutdown_failures(failures)

    def _create_client(
        self,
        profile_id: str,
        profile: LLMProfile,
        *,
        snapshot: _LLMProfileSnapshot,
    ) -> LLMClient:
        if profile.kind != "openai_compatible":
            raise ValidationError(f"unsupported LLM profile kind for {profile_id}: {profile.kind}")
        options = self._client_options(
            profile,
            legacy_env=snapshot.legacy_env,
            client_env=snapshot.client_env,
            policy=snapshot.policy,
        )
        api_mode = str(options["api_mode"])
        if api_mode not in _API_MODES:
            raise LLMError(f"OPENAI_API_MODE must be one of {sorted(_API_MODES)}, got {api_mode!r}")
        return LLMClient(  # type: ignore[arg-type]
            **options,
            defaults=self.config.llm,
        )

    def _client_options(
        self,
        profile: LLMProfile,
        *,
        legacy_env: Mapping[str, str],
        client_env: Mapping[str, str],
        policy: _ResolvedLLMPolicy,
    ) -> dict[str, Any]:
        """Resolve every client-construction input from one environment snapshot."""

        return {
            "base_url": (
                profile.base_url
                if profile.base_url is not None
                else _optional_env(legacy_env, "OPENAI_BASE_URL")
            ),
            "model": (
                profile.model
                if profile.model is not None
                else _optional_env(legacy_env, "OPENAI_LANGUAGE_MODEL")
                or _optional_env(legacy_env, "OPENAI_MODEL")
            ),
            "api_key": client_env.get(profile.api_key_env),
            "api_key_env": profile.api_key_env,
            "timeout": (
                profile.timeout_s
                if profile.timeout_s is not None
                else _float_env(legacy_env, "OPENAI_TIMEOUT", self.config.llm.timeout_s)
            ),
            "max_retries": (
                profile.max_retries
                if profile.max_retries is not None
                else _int_env(legacy_env, "OPENAI_MAX_RETRIES", self.config.llm.max_retries)
            ),
            "api_mode": policy.api_mode,
            "store": policy.store,
            "reasoning_effort": (
                profile.reasoning_effort
                if profile.reasoning_effort is not None
                else _optional_env(legacy_env, "OPENAI_REASONING_EFFORT")
            ),
            "verbosity": (
                profile.verbosity
                if profile.verbosity is not None
                else _verbosity_env(legacy_env)
            ),
            "safety_identifier": self._profile_safety_identifier(
                profile,
                legacy_env,
                client_env,
            ),
            "prompt_cache_key": (
                profile.prompt_cache_key
                if profile.prompt_cache_key is not None
                else _optional_env(legacy_env, "OPENAI_PROMPT_CACHE_KEY")
                or self.config.llm.prompt_cache_key
            ),
            "prompt_cache_retention": policy.prompt_cache_retention,
            "prompt_cache_mode": policy.prompt_cache_mode,
            "prompt_cache_ttl": policy.prompt_cache_ttl,
            "responses_previous_response_id": policy.responses_previous_response_id,
            "fallback_json_actions": policy.fallback_json_actions,
            "parallel_tool_calls": (
                profile.parallel_tool_calls
                if profile.parallel_tool_calls is not None
                else _bool_env(
                    legacy_env,
                    "OPENAI_PARALLEL_TOOL_CALLS",
                    self.config.llm.parallel_tool_calls,
                )
            ),
            "enable_thinking": (
                _bool_env(legacy_env, "OPENAI_ENABLE_THINKING", False)
                if "OPENAI_ENABLE_THINKING" in legacy_env
                else None
            ),
            "organization": (
                _optional_env(legacy_env, "OPENAI_ORGANIZATION")
                or _optional_env(legacy_env, "OPENAI_ORG_ID")
            ),
            "project": (
                _optional_env(legacy_env, "OPENAI_PROJECT")
                or _optional_env(legacy_env, "OPENAI_PROJECT_ID")
            ),
            # Resolution has snapshotted the only ambient OPENAI_* values that
            # may apply. Prevent the SDK from re-reading a different account,
            # endpoint, or provider policy while constructing its client.
            "inherit_ambient_openai_sdk_config": False,
            "allow_custom_base_url": (
                profile.allow_custom_base_url
                or _bool_env(
                    client_env,
                    "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL",
                    False,
                )
            ),
        }

    @staticmethod
    def _client_options_sha256(options: Mapping[str, Any]) -> str:
        """Hash cache inputs without retaining or exposing their plaintext values."""

        return hashlib.sha256(
            dumps({"schema_version": 1, "options": dict(options)}).encode("utf-8")
        ).hexdigest()

    def _resolved_policy(
        self,
        profile: LLMProfile,
        legacy_env: Mapping[str, str],
    ) -> _ResolvedLLMPolicy:
        return _ResolvedLLMPolicy(
            api_mode=(
                profile.api_mode
                or _optional_env(legacy_env, "OPENAI_API_MODE")
                or self.config.llm.api_mode
            ).strip().lower(),
            store=(
                profile.store
                if profile.store is not None
                else _bool_env(legacy_env, "OPENAI_STORE", self.config.llm.store)
            ),
            prompt_cache_retention=(
                _normalize_prompt_cache_retention(
                    profile.prompt_cache_retention
                    if profile.prompt_cache_retention is not None
                    else _prompt_cache_retention_env(legacy_env)
                    or self.config.llm.prompt_cache_retention,
                    label="prompt_cache_retention",
                )
            ),
            prompt_cache_mode=(
                _normalize_prompt_cache_mode(
                    profile.prompt_cache_mode
                    if profile.prompt_cache_mode is not None
                    else _prompt_cache_mode_env(legacy_env)
                    or self.config.llm.prompt_cache_mode,
                    label="prompt_cache_mode",
                )
            ),
            prompt_cache_ttl=(
                _normalize_prompt_cache_ttl(
                    profile.prompt_cache_ttl
                    if profile.prompt_cache_ttl is not None
                    else _prompt_cache_ttl_env(legacy_env)
                    or self.config.llm.prompt_cache_ttl,
                    label="prompt_cache_ttl",
                )
            ),
            responses_previous_response_id=(
                profile.responses_previous_response_id
                if profile.responses_previous_response_id is not None
                else _bool_env(
                    legacy_env,
                    "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
                    self.config.llm.responses_previous_response_id,
                )
            ),
            fallback_json_actions=(
                profile.fallback_json_actions
                if profile.fallback_json_actions is not None
                else _bool_env(
                    legacy_env,
                    "OPENAI_FALLBACK_JSON_ACTIONS",
                    self.config.llm.fallback_json_actions,
                )
            ),
        )

    def _resolved_parallel_tool_calls(self, profile: LLMProfile, client: Any) -> bool:
        if profile.parallel_tool_calls is not None:
            return bool(profile.parallel_tool_calls)
        client_value = getattr(client, "parallel_tool_calls", None)
        if client_value is not None:
            return bool(client_value)
        return bool(self.config.llm.parallel_tool_calls)

    def _resolved_auto_wait_on_empty_tool_calls(self, profile: LLMProfile) -> bool:
        if profile.auto_wait_on_empty_tool_calls is not None:
            return bool(profile.auto_wait_on_empty_tool_calls)
        return bool(self.config.llm.auto_wait_on_empty_tool_calls)

    def _profile_safety_identifier(
        self,
        profile: LLMProfile,
        legacy_env: Mapping[str, str],
        env: Mapping[str, str],
    ) -> str | None:
        if profile.safety_identifier is not None:
            return profile.safety_identifier
        if profile.safety_identifier_env is not None:
            return _optional_env(env, profile.safety_identifier_env)
        return _optional_env(legacy_env, "OPENAI_SAFETY_IDENTIFIER") or self.config.llm.safety_identifier

    def _validate_profile(self, profile_id: str, profile: LLMProfile) -> None:
        if profile.kind != "openai_compatible":
            raise ValidationError(f"unsupported LLM profile kind for {profile_id}: {profile.kind}")
        if not profile.api_key_env.strip():
            raise ValidationError(f"LLM profile api_key_env must be non-empty: {profile_id}")
        for field_name in (
            "max_tokens",
            "max_input_tokens_per_call",
            "max_total_tokens_per_call",
            "context_window_tokens",
        ):
            value = getattr(profile, field_name)
            if value is not None and value <= 0:
                raise ValidationError(
                    f"LLM profile {field_name} must be positive: {profile_id}"
                )
        max_tokens = self.config.llm.max_tokens if profile.max_tokens is None else profile.max_tokens
        context_window = (
            self.config.llm.context_window_tokens
            if profile.context_window_tokens is None
            else profile.context_window_tokens
        )
        max_input_tokens = (
            self.config.llm.max_input_tokens_per_call
            if profile.max_input_tokens_per_call is None
            else profile.max_input_tokens_per_call
        )
        max_total_tokens = (
            self.config.llm.max_total_tokens_per_call
            if profile.max_total_tokens_per_call is None
            else profile.max_total_tokens_per_call
        )
        if max_tokens >= context_window:
            raise ValidationError(
                f"LLM profile max_tokens must be less than context_window_tokens: {profile_id}"
            )
        if max_input_tokens > max_total_tokens:
            raise ValidationError(
                "LLM profile max_input_tokens_per_call must not exceed "
                f"max_total_tokens_per_call: {profile_id}"
            )
        if max_tokens > max_total_tokens:
            raise ValidationError(
                "LLM profile max_tokens must not exceed "
                f"max_total_tokens_per_call: {profile_id}"
            )

    def _normalize_profile_id(self, profile_id: str | None) -> str:
        selected = str(profile_id or "").strip()
        if not selected:
            raise ValidationError("LLM profile id must be a non-empty string")
        return selected

    def _uses_legacy_openai_env(self, profile_id: str) -> bool:
        return profile_id == self.config.llm.default_profile_id

    def _unique_clients(self) -> list[Any]:
        clients: list[Any] = []
        seen: set[int] = set()
        for client in [*self._clients.values(), *self._test_clients.values()]:
            marker = id(client)
            if marker in seen:
                continue
            seen.add(marker)
            clients.append(client)
        return clients

    def _forget_closed_client(self, client: Any) -> None:
        """Forget every cache reference still bound to one closed identity."""

        with self._lock:
            for profile_id, cached in tuple(self._clients.items()):
                if cached is not client:
                    continue
                self._clients.pop(profile_id, None)
                self._client_cache_sha256.pop(profile_id, None)
            for profile_id, cached in tuple(self._test_clients.items()):
                if cached is client:
                    self._test_clients.pop(profile_id, None)

    @staticmethod
    def _raise_shutdown_failures(failures: list[BaseException]) -> None:
        if not failures:
            return
        if any(not isinstance(exc, Exception) for exc in failures):
            raise BaseExceptionGroup(
                "LLM profile shutdown was interrupted after all clients were attempted",
                failures,
            )
        errors = [f"{type(exc).__name__}: {exc}" for exc in failures]
        raise RuntimeError(f"LLM profile shutdown failed: {errors}")

    def _shutdown_client(self, client: Any | None) -> None:
        if client is None:
            return
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            result = shutdown()
            if inspect.isawaitable(result):
                _run_awaitable_sync(result)
            return
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                _run_awaitable_sync(result)
            return
        ashutdown = getattr(client, "ashutdown", None)
        if callable(ashutdown):
            result = ashutdown()
            if inspect.isawaitable(result):
                _run_awaitable_sync(result)
            return
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                _run_awaitable_sync(result)

    async def _ashutdown_client(self, client: Any | None) -> bool:
        if client is None:
            return False
        ashutdown = getattr(client, "ashutdown", None)
        if callable(ashutdown):
            return await self._ainvoke_shutdown_callback(
                ashutdown,
                native_async=True,
            )
        aclose = getattr(client, "aclose", None)
        if callable(aclose):
            return await self._ainvoke_shutdown_callback(
                aclose,
                native_async=True,
            )
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            return await self._ainvoke_shutdown_callback(
                shutdown,
                native_async=False,
            )
        close = getattr(client, "close", None)
        if callable(close):
            return await self._ainvoke_shutdown_callback(
                close,
                native_async=False,
            )
        return False

    @classmethod
    async def _ainvoke_shutdown_callback(
        cls,
        callback: Any,
        *,
        native_async: bool,
    ) -> bool:
        """Run one client close without blocking or abandoning Host work."""

        interrupted = False
        if native_async and (
            inspect.iscoroutinefunction(callback)
            or inspect.iscoroutinefunction(getattr(callback, "__call__", None))
        ):
            # Create native async close work on the owning caller loop.
            result = callback()
        else:
            result, interrupted = await cls._await_shutdown_step(
                run_blocking_once(callback)
            )
        if inspect.isawaitable(result):
            try:
                _, await_interrupted = await cls._await_shutdown_step(result)
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        "LLM client shutdown was cancelled before its awaitable failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            interrupted = interrupted or await_interrupted
        return interrupted

    @staticmethod
    async def _await_shutdown_step(awaitable: Any) -> tuple[Any, bool]:
        """Drain a client-close step before reporting caller cancellation."""

        task = asyncio.ensure_future(awaitable)
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled():
                    raise
                interrupted = True
                if task.done():
                    try:
                        return task.result(), True
                    except BaseException as exc:
                        raise BaseExceptionGroup(
                            "LLM client shutdown was cancelled while close failed",
                            [asyncio.CancelledError(), exc],
                        ) from None
                continue
            except BaseException as exc:
                if interrupted:
                    raise BaseExceptionGroup(
                        "LLM client shutdown was cancelled while close failed",
                        [asyncio.CancelledError(), exc],
                    ) from None
                raise
            return result, interrupted


def _optional_env(env: Mapping[str, str], key: str) -> str | None:
    value = env.get(key)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _bool_env(env: Mapping[str, str], key: str, default: bool) -> bool:
    value = env.get(key)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise LLMError(f"{key} must be a number") from exc


def _int_env(env: Mapping[str, str], key: str, default: int) -> int:
    value = env.get(key)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMError(f"{key} must be an integer") from exc


def _verbosity_env(env: Mapping[str, str]) -> str | None:
    value = _optional_env(env, "OPENAI_VERBOSITY")
    if value is None:
        return None
    selected = value.lower()
    if selected not in {"low", "medium", "high"}:
        raise LLMError("OPENAI_VERBOSITY must be one of low, medium, high")
    return selected


def _prompt_cache_retention_env(env: Mapping[str, str]) -> str | None:
    value = _optional_env(env, "OPENAI_PROMPT_CACHE_RETENTION")
    return _normalize_prompt_cache_retention(
        value,
        label="OPENAI_PROMPT_CACHE_RETENTION",
    )


def _prompt_cache_mode_env(env: Mapping[str, str]) -> str | None:
    value = _optional_env(env, "OPENAI_PROMPT_CACHE_MODE")
    if value is None:
        return None
    return _normalize_prompt_cache_mode(
        value,
        label="OPENAI_PROMPT_CACHE_MODE",
    )


def _prompt_cache_ttl_env(env: Mapping[str, str]) -> str | None:
    value = _optional_env(env, "OPENAI_PROMPT_CACHE_TTL")
    if value is None:
        return None
    return _normalize_prompt_cache_ttl(
        value,
        label="OPENAI_PROMPT_CACHE_TTL",
    )


def _normalize_prompt_cache_retention(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    selected = value.strip().lower()
    if selected == "in-memory":
        return "in_memory"
    if selected not in {"in_memory", "24h"}:
        raise LLMError(f"{label} must be one of in_memory, 24h")
    return selected


def _normalize_prompt_cache_mode(value: str, *, label: str) -> str:
    selected = str(value).strip().lower()
    if selected not in {"provider_default", "implicit", "explicit"}:
        raise LLMError(
            f"{label} must be one of provider_default, implicit, explicit"
        )
    return selected


def _normalize_prompt_cache_ttl(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    selected = str(value).strip().lower()
    if selected != "30m":
        raise LLMError(f"{label} must be 30m or unset")
    return selected


def _run_awaitable_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=runner, name="agent-libos-llm-profile-close", daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
