from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_libos.llm.client import LLMClient
from agent_libos.llm.provider_trace import (
    provider_reasoning_view,
    provider_trace_summary,
)
from agent_libos.models import (
    ProcessStatus,
    ResourceUsage,
    ResourceUsageReservationStatus,
)
from scripts import real_llm_provider_trace_smoke as smoke


def test_project_env_loader_retains_only_the_explicit_provider_allowlist(
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY='key#literal' # retained comment",
                'OPENAI_BASE_URL="https://provider.invalid/v1"',
                "OPENAI_MODEL=model-fallback",
                "OPENAI_LANGUAGE_MODEL=model-selected",
                "OPENAI_API_MODE=responses",
                "OPENAI_REASONING_EFFORT=low",
                "OPENAI_VERBOSITY=medium",
                "export OPENAI_ENABLE_THINKING=off",
                "OPENAI_ORGANIZATION=must-not-be-read",
                "OPENAI_TIMEOUT=999",
                "AGENT_LIBOS_ALLOW_CUSTOM_LLM_BASE_URL=false",
                "UNRELATED_SECRET='must-not-be-retained'",
            ]
        ),
        encoding="utf-8",
    )

    selected = smoke.read_allowlisted_project_env(env_path)
    settings = smoke.provider_settings_from_project_env(env_path)

    assert set(selected) == smoke.ALLOWED_DOTENV_KEYS
    assert selected["OPENAI_API_KEY"] == "key#literal"
    assert "must-not-be-read" not in repr(selected)
    assert "must-not-be-retained" not in repr(selected)
    assert settings.model == "model-selected"
    assert settings.api_mode == "responses"
    assert settings.reasoning_effort == "low"
    assert settings.verbosity == "medium"
    assert settings.enable_thinking is False


@pytest.mark.parametrize(
    ("line", "message", "forbidden_value"),
    [
        ("OPENAI_API_KEY=key\n", "model", "key"),
        ("OPENAI_MODEL=model\n", "OPENAI_API_KEY", "model"),
        (
            "OPENAI_API_KEY=key\nOPENAI_MODEL=model\nOPENAI_API_MODE=bogus-mode\n",
            "OPENAI_API_MODE",
            "bogus-mode",
        ),
        (
            "OPENAI_API_KEY=key\nOPENAI_MODEL=model\nOPENAI_ENABLE_THINKING=maybe\n",
            "OPENAI_ENABLE_THINKING",
            "maybe",
        ),
    ],
)
def test_project_env_validation_fails_without_echoing_values(
    tmp_path: Path,
    line: str,
    message: str,
    forbidden_value: str,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(line, encoding="utf-8")

    with pytest.raises(smoke.SmokeConfigurationError, match=message) as raised:
        smoke.provider_settings_from_project_env(env_path)

    assert forbidden_value not in str(raised.value)


def test_client_and_named_profile_freeze_safety_controls_without_network() -> None:
    settings = _settings()
    config = smoke.build_smoke_config(settings)
    captured: dict[str, Any] = {}

    def client_factory(**kwargs: Any) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    client = smoke.build_smoke_client(
        settings,
        config,
        client_factory=client_factory,
    )
    profile = config.llm.profiles[smoke.TRACE_PROFILE_ID]

    assert profile.allow_custom_base_url is True
    assert profile.max_retries == 0
    assert profile.max_tokens == smoke.MAX_OUTPUT_TOKENS
    assert profile.max_input_tokens_per_call == smoke.MAX_INPUT_TOKENS
    assert profile.max_total_tokens_per_call == smoke.MAX_TOTAL_TOKENS
    assert profile.parallel_tool_calls is False
    assert profile.fallback_json_actions is False
    assert captured["allow_custom_base_url"] is True
    assert captured["max_retries"] == 0
    assert captured["inherit_ambient_openai_sdk_config"] is False
    assert captured["api_key"] == settings.api_key
    assert captured["enable_thinking"] is True
    assert client.base_url == settings.base_url


@pytest.mark.parametrize(
    ("reasoning_text", "expected_availability"),
    [
        ("selected the requested terminal action", "returned"),
        (None, "not_returned"),
    ],
)
def test_runner_is_ephemeral_authority_free_and_charges_once(
    tmp_path: Path,
    reasoning_text: str | None,
    expected_availability: str,
) -> None:
    env_path = _write_env(tmp_path)
    captured: dict[str, Any] = {}

    def runtime_factory(**kwargs: Any) -> _FakeRuntime:
        database_path = kwargs["database_path"]
        workspace = kwargs["workspace"]
        captured.update(kwargs)
        captured["temp_root"] = workspace.parent
        assert database_path.parent == workspace.parent
        assert workspace.is_dir()
        assert list(workspace.iterdir()) == []
        database_path.touch()
        runtime = _FakeRuntime(reasoning_text=reasoning_text)
        captured["runtime"] = runtime
        return runtime

    report = smoke.run_provider_trace_smoke(
        env_path,
        runtime_factory=runtime_factory,
        temp_parent=tmp_path,
    )

    runtime = captured["runtime"]
    spawn = runtime.spawn_kwargs
    budget = spawn["resource_budget"]
    assert isinstance(captured["client"], LLMClient)
    assert captured["config"].llm.profiles[smoke.TRACE_PROFILE_ID].max_retries == 0
    assert spawn["image"] == "base-agent:v0"
    assert spawn["capabilities"] == []
    assert spawn["llm_profile_id"] == smoke.TRACE_PROFILE_ID
    assert spawn["working_directory"] == "."
    assert "exactly one tool call" in spawn["goal"]
    assert budget.max_llm_calls == 1
    assert budget.max_llm_total_tokens == smoke.MAX_TOTAL_TOKENS
    assert budget.max_tool_calls == 1
    assert budget.max_child_processes == 0
    assert budget.max_external_read_bytes == 0
    assert budget.max_external_write_bytes == 0
    assert budget.max_jsonrpc_bytes == 0
    assert budget.max_mcp_bytes == 0
    assert runtime.max_quanta == smoke.MAX_QUANTA == 2
    assert runtime.closed
    assert not captured["temp_root"].exists()
    assert report.provider_attempts == 1
    assert report.reasoning_availability == expected_availability
    assert report.charged_llm_calls == 1


def test_runner_contract_uses_the_real_runtime_without_network(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path)
    response = _AttrMapping(
        _request_id="request-runtime",
        id="response-runtime",
        model="test-model",
        choices=[
            _AttrMapping(
                finish_reason="tool_calls",
                message=_AttrMapping(
                    content="",
                    reasoning_content="selected the terminal smoke action",
                    tool_calls=[
                        _AttrMapping(
                            id="call-runtime-exit",
                            function=_AttrMapping(
                                name="process_exit",
                                arguments=json.dumps(
                                    {"payload": {"trace_smoke": True}},
                                    separators=(",", ":"),
                                ),
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=_AttrMapping(
            prompt_tokens=10,
            completion_tokens=2,
            total_tokens=12,
        ),
    )
    completions = _FakeAsyncCompletions(response)

    def runtime_factory(**kwargs: Any) -> Any:
        client = kwargs["client"]
        client._async_client = SimpleNamespace(  # noqa: SLF001 - injected Provider double
            chat=SimpleNamespace(completions=completions)
        )
        return smoke._open_runtime(**kwargs)

    report = smoke.run_provider_trace_smoke(
        env_path,
        runtime_factory=runtime_factory,
        temp_parent=tmp_path,
    )

    assert report == smoke.SmokeReport(
        provider_attempts=1,
        reasoning_availability="returned",
    )
    assert len(completions.requests) == 1
    assert completions.requests[0]["parallel_tool_calls"] is False
    assert (
        completions.requests[0].get("max_tokens")
        or completions.requests[0].get("max_completion_tokens")
    ) == smoke.MAX_OUTPUT_TOKENS


def test_cli_failure_output_never_echoes_provider_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "sk-private-value"
    endpoint = "https://private-provider.invalid/v1"
    trace = '{"kind":"provider_trace","output":"private"}'

    def fail() -> smoke.SmokeReport:
        raise RuntimeError(f"{secret} {endpoint} {trace}")

    monkeypatch.setattr(smoke, "run_provider_trace_smoke", fail)

    assert smoke.main() == 1
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert rendered == "real LLM provider trace smoke failed\n"
    assert secret not in rendered
    assert endpoint not in rendered
    assert trace not in rendered


def test_cli_success_output_contains_only_safe_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        smoke,
        "run_provider_trace_smoke",
        lambda: smoke.SmokeReport(
            provider_attempts=3,
            reasoning_availability="not_returned",
        ),
    )

    assert smoke.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "real LLM provider trace smoke passed (reasoning=not_returned)\n"
    )
    assert "provider_attempts" not in captured.out


def _settings() -> smoke.ProviderSettings:
    return smoke.ProviderSettings(
        api_key="sk-unit-test",
        base_url="https://provider.invalid/v1",
        model="test-model",
        api_mode="chat",
        reasoning_effort="low",
        verbosity="low",
        enable_thinking=True,
    )


def _write_env(tmp_path: Path) -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=sk-unit-test",
                "OPENAI_BASE_URL=https://provider.invalid/v1",
                "OPENAI_MODEL=test-model",
                "OPENAI_API_MODE=chat",
                "OPENAI_REASONING_EFFORT=low",
                "OPENAI_ENABLE_THINKING=true",
            ]
        ),
        encoding="utf-8",
    )
    return env_path


class _FakeRuntime:
    def __init__(self, *, reasoning_text: str | None) -> None:
        arguments = json.dumps(
            {"payload": {"trace_smoke": True}},
            separators=(",", ":"),
        )
        tool_calls = [
            {
                "id": "call_exit",
                "name": "process_exit",
                "arguments": arguments,
            }
        ]
        reasoning = provider_reasoning_view(
            {"reasoning_content": reasoning_text}
            if reasoning_text is not None
            else None
        )
        trace = {
            "kind": "provider_trace",
            "schema_version": 1,
            "coverage": "complete",
            "selected_attempt": 1,
            "limited": False,
            "omitted_attempts": 0,
            "attempts": [
                {
                    "sequence": 1,
                    "kind": "initial",
                    "api": "chat",
                    "status": "ok",
                    "reasoning": reasoning,
                    "output": "",
                    "tool_calls": tool_calls,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                    },
                    "model": "test-model",
                    "request_id": "request-unit",
                    "response_id": "response-unit",
                    "started_at": "2026-08-03T00:00:00+00:00",
                    "completed_at": "2026-08-03T00:00:01+00:00",
                    "duration_ms": 1,
                    "error": None,
                }
            ],
        }
        raw_message: dict[str, Any] = {"tool_calls": []}
        if reasoning_text is not None:
            raw_message["reasoning_content"] = reasoning_text
        self.call = SimpleNamespace(
            status="ok",
            purpose="action_selection",
            api="chat",
            response_content="",
            tool_calls=[{**tool_calls[0], "call_id": None}],
            reasoning=trace,
            raw_response={"choices": [{"message": raw_message}]},
            request_options={"provider_trace_summary": provider_trace_summary(trace)},
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            },
        )
        self.process_row = SimpleNamespace(
            status=ProcessStatus.RUNNABLE,
            resource_usage=ResourceUsage(
                llm_calls=1,
                llm_prompt_tokens=10,
                llm_completion_tokens=2,
                llm_total_tokens=12,
            ),
        )
        self.reservation = SimpleNamespace(
            reason="llm.request",
            status=ResourceUsageReservationStatus.SETTLED,
            settled_usage=ResourceUsage(
                llm_calls=1,
                llm_prompt_tokens=10,
                llm_completion_tokens=2,
                llm_total_tokens=12,
            ),
        )
        self.spawn_kwargs: dict[str, Any] = {}
        self.max_quanta: int | None = None
        self.closed = False
        self.process = SimpleNamespace(spawn=self._spawn, get=self._get_process)
        self.store = SimpleNamespace(list_llm_calls=self._list_llm_calls)
        self.capability = SimpleNamespace(capabilities_for=lambda _pid: [])
        self.authority_manifests = SimpleNamespace(
            get_for_process=lambda _pid: SimpleNamespace(
                authorized_capabilities=[]
            )
        )
        self.uow = SimpleNamespace(
            resources=SimpleNamespace(
                list_resource_usage_reservations=lambda **_kwargs: [
                    self.reservation
                ]
            )
        )

    def _spawn(self, **kwargs: Any) -> str:
        self.spawn_kwargs = kwargs
        return "pid_trace_smoke"

    def _get_process(self, _pid: str) -> SimpleNamespace:
        return self.process_row

    def _list_llm_calls(self, **_kwargs: Any) -> list[SimpleNamespace]:
        return [self.call]

    def run_process_until_idle(
        self,
        _pid: str,
        *,
        max_quanta: int,
    ) -> list[dict[str, Any]]:
        self.max_quanta = max_quanta
        self.process_row.status = ProcessStatus.EXITED
        return [
            {
                "ok": True,
                "action": {
                    "action": "process_exit",
                    "payload": {"trace_smoke": True},
                },
                "result": {"ok": True},
            }
        ]

    def close(self) -> None:
        self.closed = True


class _FakeAsyncCompletions:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.response


class _AttrMapping(dict[str, Any]):
    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
