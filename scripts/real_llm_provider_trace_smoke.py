from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal, cast


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agent_libos import Runtime  # noqa: E402
from agent_libos.config import (  # noqa: E402
    DEFAULT_CONFIG,
    AgentLibOSConfig,
    LLMProfile,
)
from agent_libos.llm.client import LLMClient  # noqa: E402
from agent_libos.llm.provider_trace import (  # noqa: E402
    is_provider_trace,
    provider_reasoning_view,
    provider_trace_summary,
)
from agent_libos.models import (  # noqa: E402
    ProcessStatus,
    ResourceBudget,
    ResourceUsageReservationStatus,
)
from agent_libos.substrate import LocalResourceProviderSubstrate  # noqa: E402


PROJECT_ENV_FILE = PROJECT_ROOT / ".env"
TRACE_PROFILE_ID = "provider-trace-smoke"
MAX_QUANTA = 2
MAX_INPUT_TOKENS = 32_768
MAX_OUTPUT_TOKENS = 1_024
MAX_TOTAL_TOKENS = MAX_INPUT_TOKENS + MAX_OUTPUT_TOKENS

ALLOWED_DOTENV_KEYS = frozenset(
    {
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "OPENAI_LANGUAGE_MODEL",
        "OPENAI_API_MODE",
        "OPENAI_REASONING_EFFORT",
        "OPENAI_VERBOSITY",
        "OPENAI_ENABLE_THINKING",
    }
)

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_EXTERNAL_CAPABILITY_PREFIXES = (
    "filesystem:",
    "shell:",
    "mcp:",
    "mcp_server:",
    "jsonrpc:",
    "jsonrpc_endpoint:",
)
_TRACE_GOAL = (
    "Machine-only Provider trace smoke test. Make exactly one tool call: "
    "process_exit with the exact arguments "
    '{"payload":{"trace_smoke":true}}. Do not call human_output or any other '
    "tool. Do not inspect files, Skills, capabilities, MCP servers, JSON-RPC "
    "endpoints, or the shell."
)


class SmokeConfigurationError(RuntimeError):
    """The fixed project environment is insufficient for this smoke test."""


class SmokeVerificationError(RuntimeError):
    """The real run did not satisfy the bounded Provider-trace contract."""


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    api_key: str = field(repr=False)
    base_url: str | None = field(repr=False)
    model: str = field(repr=False)
    api_mode: Literal["auto", "responses", "chat"]
    reasoning_effort: str | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    enable_thinking: bool | None = None


@dataclass(frozen=True, slots=True)
class SmokeReport:
    provider_attempts: int
    reasoning_availability: Literal["returned", "not_returned"]
    charged_llm_calls: int = 1


RuntimeFactory = Callable[..., Any]


def read_allowlisted_project_env(path: Path = PROJECT_ENV_FILE) -> dict[str, str]:
    """Read only the Provider fields approved for this dedicated entrypoint."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SmokeConfigurationError("project .env is unavailable") from exc

    selected: dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in ALLOWED_DOTENV_KEYS:
            continue
        selected[key] = _dotenv_value(raw_value, key=key)
    return selected


def provider_settings_from_project_env(
    path: Path = PROJECT_ENV_FILE,
) -> ProviderSettings:
    env = read_allowlisted_project_env(path)
    api_key = _required_env(env, "OPENAI_API_KEY")
    model = str(
        env.get("OPENAI_LANGUAGE_MODEL") or env.get("OPENAI_MODEL") or ""
    ).strip()
    if not model:
        raise SmokeConfigurationError("project .env has no approved model setting")

    raw_api_mode = str(env.get("OPENAI_API_MODE") or "auto").strip().lower()
    if raw_api_mode not in {"auto", "responses", "chat"}:
        raise SmokeConfigurationError("project .env has an invalid OPENAI_API_MODE")
    api_mode = cast(Literal["auto", "responses", "chat"], raw_api_mode)

    raw_verbosity = str(env.get("OPENAI_VERBOSITY") or "").strip().lower()
    if raw_verbosity and raw_verbosity not in {"low", "medium", "high"}:
        raise SmokeConfigurationError("project .env has an invalid OPENAI_VERBOSITY")
    verbosity = cast(
        Literal["low", "medium", "high"] | None,
        raw_verbosity or None,
    )

    raw_enable_thinking = env.get("OPENAI_ENABLE_THINKING")
    enable_thinking = (
        _boolean_env(raw_enable_thinking, name="OPENAI_ENABLE_THINKING")
        if raw_enable_thinking is not None and raw_enable_thinking.strip()
        else None
    )

    return ProviderSettings(
        api_key=api_key,
        base_url=str(env.get("OPENAI_BASE_URL") or "").strip() or None,
        model=model,
        api_mode=api_mode,
        reasoning_effort=(
            str(env.get("OPENAI_REASONING_EFFORT") or "").strip() or None
        ),
        verbosity=verbosity,
        enable_thinking=enable_thinking,
    )


def build_smoke_config(settings: ProviderSettings) -> AgentLibOSConfig:
    profile = LLMProfile(
        base_url=settings.base_url,
        model=settings.model,
        api_key_env="AGENT_LIBOS_PROVIDER_TRACE_SMOKE_API_KEY",
        api_mode=settings.api_mode,
        timeout_s=60.0,
        max_retries=0,
        store=False,
        reasoning_effort=settings.reasoning_effort,
        verbosity=settings.verbosity,
        responses_previous_response_id=False,
        parallel_tool_calls=False,
        auto_wait_on_empty_tool_calls=False,
        fallback_json_actions=False,
        temperature=0.0,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_input_tokens_per_call=MAX_INPUT_TOKENS,
        max_total_tokens_per_call=MAX_TOTAL_TOKENS,
        allow_custom_base_url=True,
    )
    llm = replace(
        DEFAULT_CONFIG.llm,
        profiles={**DEFAULT_CONFIG.llm.profiles, TRACE_PROFILE_ID: profile},
        persist_full_io=True,
    )
    return replace(DEFAULT_CONFIG, llm=llm)


def build_smoke_client(
    settings: ProviderSettings,
    config: AgentLibOSConfig,
    *,
    client_factory: Callable[..., Any] = LLMClient,
) -> Any:
    """Construct a frozen client without exporting `.env` values to the process."""

    return client_factory(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        api_key_env="OPENAI_API_KEY",
        timeout=60.0,
        max_retries=0,
        api_mode=settings.api_mode,
        store=False,
        reasoning_effort=settings.reasoning_effort,
        verbosity=settings.verbosity,
        responses_previous_response_id=False,
        parallel_tool_calls=False,
        fallback_json_actions=False,
        enable_thinking=settings.enable_thinking,
        organization=None,
        project=None,
        inherit_ambient_openai_sdk_config=False,
        allow_custom_base_url=True,
        defaults=config.llm,
    )


def run_provider_trace_smoke(
    env_path: Path = PROJECT_ENV_FILE,
    *,
    runtime_factory: RuntimeFactory | None = None,
    temp_parent: Path | None = None,
) -> SmokeReport:
    """Run one bounded real Provider action and verify its durable trace."""

    settings = provider_settings_from_project_env(env_path)
    config = build_smoke_config(settings)
    client = build_smoke_client(settings, config)
    selected_runtime_factory = runtime_factory or _open_runtime
    temp_root: Path | None = None
    report: SmokeReport | None = None

    with tempfile.TemporaryDirectory(
        prefix="agent-libos-provider-trace-",
        dir=str(temp_parent) if temp_parent is not None else None,
    ) as raw_temp_root:
        temp_root = Path(raw_temp_root)
        workspace = temp_root / "workspace"
        workspace.mkdir(mode=0o700)
        database_path = temp_root / "runtime.sqlite3"
        runtime: Any | None = None
        try:
            runtime = selected_runtime_factory(
                database_path=database_path,
                workspace=workspace,
                config=config,
                client=client,
            )
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal=_TRACE_GOAL,
                capabilities=[],
                resource_budget=_smoke_budget(),
                working_directory=".",
                llm_profile_id=TRACE_PROFILE_ID,
            )
            _verify_no_external_authority(runtime, pid)
            results = runtime.run_process_until_idle(pid, max_quanta=MAX_QUANTA)
            report = _verify_run(runtime, pid, results)
        finally:
            if runtime is not None:
                runtime.close()
            else:
                _close_if_possible(client)

    _require(temp_root is not None and not temp_root.exists(), "temporary data cleanup")
    _require(report is not None, "smoke report creation")
    return report


def _open_runtime(
    *,
    database_path: Path,
    workspace: Path,
    config: AgentLibOSConfig,
    client: LLMClient,
) -> Runtime:
    runtime = Runtime.open(
        database_path,
        substrate=LocalResourceProviderSubstrate(workspace),
        config=config,
    )
    try:
        # The named non-default profile prevents ambient OPENAI_* compatibility
        # settings from participating in profile resolution. The real client is
        # already frozen from the allowlisted project `.env` values above.
        runtime.llms.set_test_client(TRACE_PROFILE_ID, client)
    except BaseException:
        runtime.close()
        raise
    return runtime


def _smoke_budget() -> ResourceBudget:
    return ResourceBudget(
        max_tool_calls=1,
        max_child_processes=0,
        max_llm_calls=1,
        max_llm_total_tokens=MAX_TOTAL_TOKENS,
        max_subprocess_wall_seconds=0.0,
        max_subprocess_cpu_seconds=0.0,
        max_subprocess_memory_bytes=0,
        max_external_read_bytes=0,
        max_external_write_bytes=0,
        max_jsonrpc_bytes=0,
        max_mcp_bytes=0,
        max_deno_syscalls=0,
    )


def _verify_no_external_authority(runtime: Any, pid: str) -> None:
    manifest = runtime.authority_manifests.get_for_process(pid)
    _require(manifest is not None, "authority manifest persistence")
    _require(
        list(manifest.authorized_capabilities) == [],
        "empty launch authority",
    )
    capabilities = list(runtime.capability.capabilities_for(pid))
    forbidden = [
        capability
        for capability in capabilities
        if str(getattr(capability, "resource", "")).startswith(
            _EXTERNAL_CAPABILITY_PREFIXES
        )
    ]
    _require(not forbidden, "no external capability grants")


def _verify_run(runtime: Any, pid: str, results: list[Any]) -> SmokeReport:
    process = runtime.process.get(pid)
    _require(process.status is ProcessStatus.EXITED, "terminal process status")

    actions = [
        result["action"]
        for result in results
        if isinstance(result, dict) and isinstance(result.get("action"), dict)
    ]
    _require(len(actions) == 1, "single selected action")
    action = actions[0]
    _require(action.get("action") == "process_exit", "selected process_exit action")
    action_arguments = {key: value for key, value in action.items() if key != "action"}

    calls = list(runtime.store.list_llm_calls(pid=pid))
    _require(len(calls) == 1, "single logical LLMCallRecord")
    call = calls[0]
    _require(call.status == "ok", "successful LLMCallRecord")
    _require(call.purpose == "action_selection", "action-selection LLMCallRecord")
    trace = call.reasoning
    _require(is_provider_trace(trace), "Provider trace schema")
    _require(trace.get("coverage") == "complete", "complete Provider trace coverage")
    _require(not trace.get("limited"), "unlimited Provider trace")

    summary = provider_trace_summary(trace)
    _require(summary["attempt_count"] >= 1, "recorded Provider attempt")
    _require(
        call.request_options.get("provider_trace_summary") == summary,
        "Provider trace summary projection",
    )
    selected_sequence = trace.get("selected_attempt")
    _require(type(selected_sequence) is int, "selected Provider attempt")
    selected_attempts = [
        attempt
        for attempt in trace["attempts"]
        if isinstance(attempt, dict) and attempt.get("sequence") == selected_sequence
    ]
    _require(len(selected_attempts) == 1, "selected Provider attempt relation")
    selected_attempt = selected_attempts[0]
    _require(selected_attempt.get("status") == "ok", "selected Provider response")
    _require(
        selected_attempt.get("output") == call.response_content,
        "selected output projection",
    )

    persisted_tool_calls = _semantic_tool_calls(call.tool_calls)
    selected_tool_calls = _semantic_tool_calls(selected_attempt.get("tool_calls"))
    _require(
        persisted_tool_calls == selected_tool_calls,
        "selected tool-call projection",
    )
    _require(
        len(persisted_tool_calls) == 1
        and persisted_tool_calls[0][0] == "process_exit"
        and persisted_tool_calls[0][1] == action_arguments,
        "selected action relation",
    )

    selected_reasoning = selected_attempt.get("reasoning")
    _require(
        isinstance(selected_reasoning, dict)
        and provider_reasoning_view(selected_reasoning) == selected_reasoning,
        "selected reasoning projection",
    )
    availability = (
        selected_reasoning.get("availability")
        if isinstance(selected_reasoning, dict)
        else None
    )
    _require(
        availability in {"returned", "not_returned"},
        "explicit reasoning availability",
    )
    if availability == "returned":
        _require(bool(selected_reasoning.get("blocks")), "returned reasoning blocks")
    else:
        _require(selected_reasoning.get("blocks") == [], "not_returned reasoning")

    _require(process.resource_usage.llm_calls == 1, "single charged LLM call")
    total_tokens = call.usage.get("total_tokens")
    _require(
        type(total_tokens) is int and 0 < total_tokens <= MAX_TOTAL_TOKENS,
        "bounded final Provider usage",
    )
    _require(
        process.resource_usage.llm_total_tokens == total_tokens,
        "final Provider usage settlement",
    )
    reservations = list(
        runtime.uow.resources.list_resource_usage_reservations(pid=pid)
    )
    _require(len(reservations) == 1, "single LLM usage reservation")
    reservation = reservations[0]
    _require(reservation.reason == "llm.request", "LLM usage reservation reason")
    _require(
        reservation.status is ResourceUsageReservationStatus.SETTLED,
        "settled LLM usage reservation",
    )
    _require(
        reservation.settled_usage is not None
        and reservation.settled_usage.llm_calls == 1
        and reservation.settled_usage.llm_total_tokens == total_tokens,
        "single settled LLM charge",
    )

    return SmokeReport(
        provider_attempts=int(summary["attempt_count"]),
        reasoning_availability=cast(
            Literal["returned", "not_returned"],
            availability,
        ),
    )


def _semantic_tool_calls(value: Any) -> list[tuple[str, dict[str, Any]]]:
    _require(isinstance(value, list), "tool-call list")
    result: list[tuple[str, dict[str, Any]]] = []
    for item in value:
        _require(isinstance(item, dict), "tool-call entry")
        name = item.get("name")
        arguments = item.get("arguments")
        _require(isinstance(name, str) and bool(name), "tool-call name")
        _require(isinstance(arguments, str), "tool-call arguments")
        try:
            decoded = json.loads(arguments)
        except (TypeError, ValueError) as exc:
            raise SmokeVerificationError("smoke invariant failed: tool-call JSON") from exc
        _require(isinstance(decoded, dict), "tool-call argument object")
        result.append((name, decoded))
    return result


def _dotenv_value(raw_value: str, *, key: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    if value[0] in {"'", '"'}:
        quote = value[0]
        escaped = False
        for index in range(1, len(value)):
            character = value[index]
            if quote == '"' and character == "\\" and not escaped:
                escaped = True
                continue
            if character == quote and not escaped:
                trailing = value[index + 1 :].strip()
                if trailing and not trailing.startswith("#"):
                    raise SmokeConfigurationError(
                        f"project .env has an invalid {key} entry"
                    )
                return value[1:index]
            escaped = False
        raise SmokeConfigurationError(f"project .env has an invalid {key} entry")

    for index, character in enumerate(value):
        if character == "#" and index > 0 and value[index - 1].isspace():
            return value[:index].rstrip()
    return value


def _required_env(env: dict[str, str], name: str) -> str:
    value = str(env.get(name) or "").strip()
    if not value:
        raise SmokeConfigurationError(f"project .env has no approved {name}")
    return value


def _boolean_env(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise SmokeConfigurationError(f"project .env has an invalid {name}")


def _close_if_possible(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise SmokeVerificationError(f"smoke invariant failed: {label}")


def main() -> int:
    # Provider/transport exceptions can contain endpoint details. Keep the CLI
    # surface fixed and discard all nested output; callers receive only a safe
    # pass/fail line and the non-sensitive reasoning availability state.
    previous_logging_disable = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                report = run_provider_trace_smoke()
    except BaseException:
        print("real LLM provider trace smoke failed", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_logging_disable)

    print(
        "real LLM provider trace smoke passed "
        f"(reasoning={report.reasoning_availability})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
