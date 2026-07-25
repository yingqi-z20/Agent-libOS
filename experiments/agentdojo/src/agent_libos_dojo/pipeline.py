from __future__ import annotations

import copy
import hashlib
import json
import os
from ast import literal_eval
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop,
    ToolsExecutor,
    is_string_list,
    tool_result_to_str,
)
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatToolResultMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.llm import client as llm_client_module
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.models import (
    PROMPT_MODE_IMAGE_ONLY,
    PROMPT_MODES,
    AgentImage,
    ProcessStatus,
)
from agent_libos.models.exceptions import NotFound
from agent_libos.storage import SQLiteStore
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.base import SyncAgentTool, ToolContext, ToolPolicy, ToolResult
from agent_libos.utils.serde import to_jsonable
from agent_libos import Runtime


HIDDEN_TERMINAL_TOOL = "agentdojo_runtime_submit_final"
DEFAULT_MAX_TOOL_ITERATIONS = 15
_IMAGE_ID = "agentdojo-native-semantics:v0"


class PipelineRunError(RuntimeError):
    """A trajectory did not produce oracle-usable evidence."""


@dataclass(frozen=True)
class ExplicitDotenvSnapshot:
    """Immutable provider configuration captured from one explicit dotenv file."""

    path: Path
    file_sha256: str
    _dotenv_items: tuple[tuple[str, str], ...] = field(repr=False)
    _client_init_items: tuple[tuple[str, Any], ...] = field(repr=False)

    def new_client(self) -> LLMClient:
        """Build an independent client without consulting the file or environment."""

        return LLMClient(
            **{
                name: copy.deepcopy(value)
                for name, value in self._client_init_items
            }
        )

    def assert_unchanged(self) -> None:
        """Fail closed if the selected dotenv bytes changed after capture."""

        try:
            current_sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PipelineRunError(
                "selected dotenv file became unavailable during the evaluation run"
            ) from exc
        if current_sha256 != self.file_sha256:
            raise PipelineRunError(
                "selected dotenv file changed during the evaluation run"
            )

    def redactions(self) -> dict[str, str]:
        """Return exact run-start values that must never enter artifacts."""

        dotenv = dict(self._dotenv_items)
        return {
            value: replacement
            for value, replacement in (
                (dotenv.get("OPENAI_API_KEY"), "[redacted-api-key]"),
                (dotenv.get("OPENAI_BASE_URL"), "[redacted-endpoint]"),
            )
            if value
        }


@dataclass
class RunRecorder:
    """In-memory, secret-free projection of one provider/tool trajectory."""

    events: list[ChatMessage] = field(default_factory=list)
    provider_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_requests: list[dict[str, Any]] = field(default_factory=list)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str | None = None
    _pending_calls: list[FunctionCall] = field(default_factory=list, repr=False)

    def record_provider_request(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> None:
        rendered_tools = to_jsonable(list(tools))
        canonical_tools = json.dumps(
            rendered_tools,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.provider_requests.append(
            {
                "capture_stage": "llm_client_input_before_provider_normalization",
                "messages": to_jsonable(list(messages)),
                "message_roles": [str(message.get("role") or "") for message in messages],
                "tools": rendered_tools,
                "tool_names": [_openai_tool_name(tool) for tool in tools],
                "tools_sha256": hashlib.sha256(
                    canonical_tools.encode("utf-8")
                ).hexdigest(),
            }
        )

    def record_assistant(self, completion: LLMCompletion) -> None:
        tool_calls = [_function_call(value) for value in completion.tool_calls]
        content = (
            [text_content_block_from_string(completion.content)]
            if completion.content
            else None
        )
        self.events.append(
            ChatAssistantMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
            )
        )
        self._pending_calls.extend(tool_calls)
        call_index = len(self.provider_calls)
        provider_call = {
            "api": completion.api,
            "model": completion.model,
            "response_id": completion.response_id,
            "request_id": completion.request_id,
            "usage": to_jsonable(completion.usage),
            "content": completion.content,
            "tool_calls": [call.model_dump(mode="json") for call in tool_calls],
            "fallback_json_action_used": completion.fallback_json_action_used,
            "compatibility_removed_options": list(
                completion.compatibility_removed_options
            ),
            "provider_request_options": to_jsonable(
                completion.provider_request_options
            ),
        }
        if call_index < len(self.provider_requests):
            provider_call["request"] = self.provider_requests[call_index]
        self.provider_calls.append(provider_call)

    def record_tool(
        self,
        *,
        function: str,
        args: Mapping[str, Any],
        formatted_result: str,
        raw_result: Any,
        error: str | None,
    ) -> None:
        call = self._take_pending(function, args)
        self.events.append(
            ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(formatted_result)],
                tool_call_id=call.id,
                tool_call=call,
                error=error,
            )
        )
        self.tool_executions.append(
            {
                "function": function,
                "args": to_jsonable(dict(args)),
                "result": to_jsonable(raw_result),
                "formatted_result": formatted_result,
                "error": error,
                "provider_tool_call_id": call.id,
            }
        )

    def usage(self) -> dict[str, int]:
        totals: Counter[str] = Counter()
        for call in self.provider_calls:
            usage = call.get("usage")
            if not isinstance(usage, Mapping):
                continue
            for key, value in usage.items():
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    continue
                totals[str(key)] += value
        if "total_tokens" not in totals:
            prompt = totals.get("prompt_tokens", totals.get("input_tokens", 0))
            completion = totals.get(
                "completion_tokens", totals.get("output_tokens", 0)
            )
            if prompt or completion:
                totals["total_tokens"] = prompt + completion
        return dict(sorted(totals.items()))

    def _take_pending(
        self, function: str, args: Mapping[str, Any]
    ) -> FunctionCall:
        selected_args = to_jsonable(dict(args))
        for index, call in enumerate(self._pending_calls):
            if call.function == function and to_jsonable(dict(call.args)) == selected_args:
                return self._pending_calls.pop(index)
        for index, call in enumerate(self._pending_calls):
            if call.function == function:
                return self._pending_calls.pop(index)
        return FunctionCall(function=function, args=dict(args), id=None)


def evaluation_config(*, max_output_tokens: int) -> AgentLibOSConfig:
    """Return the fixed model/runtime controls shared by both arms."""

    llm = replace(
        DEFAULT_CONFIG.llm,
        temperature=0.0,
        max_tokens=max_output_tokens,
        parallel_tool_calls=False,
        auto_wait_on_empty_tool_calls=False,
    )
    return replace(DEFAULT_CONFIG, llm=llm)


def client_from_env(
    env_file: str | Path,
    *,
    config: AgentLibOSConfig,
) -> LLMClient:
    snapshot = capture_explicit_dotenv_environment(env_file, config=config)
    return snapshot.new_client()


def capture_explicit_dotenv_environment(
    env_file: str | Path,
    *,
    config: AgentLibOSConfig,
) -> ExplicitDotenvSnapshot:
    """Resolve one stable dotenv/client snapshot before any artifact is created."""

    env_path = Path(env_file).absolute()
    try:
        initial_bytes = env_path.read_bytes()
    except OSError as exc:
        raise PipelineRunError(f"dotenv file does not exist: {env_path}") from exc
    initial_sha256 = hashlib.sha256(initial_bytes).hexdigest()
    dotenv = _read_dotenv_bytes(initial_bytes)
    _validate_explicit_dotenv_values(dotenv)
    resolved_client = _client_from_captured_dotenv(dotenv, config=config)
    try:
        client_init_items = tuple(
            (item.name, copy.deepcopy(getattr(resolved_client, item.name)))
            for item in fields(LLMClient)
            if item.init
        )
    finally:
        if resolved_client is not None:
            resolved_client.close()
    return ExplicitDotenvSnapshot(
        path=env_path,
        file_sha256=initial_sha256,
        _dotenv_items=tuple(sorted(dotenv.items())),
        _client_init_items=client_init_items,
    )


def validate_explicit_dotenv_environment(env_file: str | Path) -> dict[str, str]:
    """Require the selected dotenv file to be the effective OpenAI config.

    ``LLMClient.from_env`` intentionally gives the process environment precedence
    over a dotenv file.  Evaluation runs need a stronger provenance contract: an
    ambient setting may be present only when it is byte-for-byte identical to the
    selected file.  Reject conflicts before constructing a provider client, and
    never include secret values in the diagnostic.
    """

    env_path = Path(env_file)
    try:
        dotenv = _read_dotenv_bytes(env_path.read_bytes())
    except OSError as exc:
        raise PipelineRunError(f"dotenv file does not exist: {env_path}") from exc
    _validate_explicit_dotenv_values(dotenv)
    return dotenv


def _read_dotenv_bytes(raw: bytes) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in raw.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _validate_explicit_dotenv_values(dotenv: Mapping[str, str]) -> None:
    conflicts = sorted(
        key
        for key, value in os.environ.items()
        if key.startswith("OPENAI_") and dotenv.get(key) != value
    )
    if conflicts:
        rendered = ", ".join(conflicts)
        raise PipelineRunError(
            "ambient OpenAI configuration conflicts with the selected dotenv "
            f"file for: {rendered}"
        )


def _client_from_captured_dotenv(
    dotenv: Mapping[str, str],
    *,
    config: AgentLibOSConfig,
) -> LLMClient:
    """Mirror ``LLMClient.from_env`` without re-reading mutable process state."""

    env = dict(dotenv)
    defaults = config.llm
    api_mode = env.get("OPENAI_API_MODE", defaults.api_mode).strip().lower()
    if api_mode not in {"auto", "responses", "chat"}:
        raise llm_client_module.LLMError(
            "OPENAI_API_MODE must be one of ['auto', 'chat', 'responses'], "
            f"got {api_mode!r}"
        )
    return LLMClient(
        base_url=env.get("OPENAI_BASE_URL"),
        model=env.get("OPENAI_LANGUAGE_MODEL") or env.get("OPENAI_MODEL"),
        api_key=env.get("OPENAI_API_KEY"),
        api_key_env="OPENAI_API_KEY",
        timeout=llm_client_module._float_env_from(
            env,
            "OPENAI_TIMEOUT",
            default=defaults.timeout_s,
        ),
        max_retries=llm_client_module._int_env_from(
            env,
            "OPENAI_MAX_RETRIES",
            default=defaults.max_retries,
        ),
        api_mode=api_mode,  # type: ignore[arg-type]
        store=llm_client_module._bool_env_from(
            env,
            "OPENAI_STORE",
            default=defaults.store,
        ),
        reasoning_effort=llm_client_module._optional_env_from(
            env,
            "OPENAI_REASONING_EFFORT",
        ),
        verbosity=llm_client_module._verbosity_env_from(env, "OPENAI_VERBOSITY"),
        safety_identifier=(
            llm_client_module._optional_env_from(env, "OPENAI_SAFETY_IDENTIFIER")
            or defaults.safety_identifier
        ),
        prompt_cache_key=(
            llm_client_module._optional_env_from(env, "OPENAI_PROMPT_CACHE_KEY")
            or defaults.prompt_cache_key
        ),
        prompt_cache_retention=(
            llm_client_module._prompt_cache_retention_env_from(
                env,
                "OPENAI_PROMPT_CACHE_RETENTION",
            )
            or defaults.prompt_cache_retention
        ),
        responses_previous_response_id=llm_client_module._bool_env_from(
            env,
            "OPENAI_RESPONSES_PREVIOUS_RESPONSE_ID",
            default=defaults.responses_previous_response_id,
        ),
        parallel_tool_calls=llm_client_module._bool_env_from(
            env,
            "OPENAI_PARALLEL_TOOL_CALLS",
            default=defaults.parallel_tool_calls,
        ),
        fallback_json_actions=llm_client_module._bool_env_from(
            env,
            "OPENAI_FALLBACK_JSON_ACTIONS",
            default=defaults.fallback_json_actions,
        ),
        enable_thinking=(
            llm_client_module._bool_env_from(
                env,
                "OPENAI_ENABLE_THINKING",
                default=False,
            )
            if "OPENAI_ENABLE_THINKING" in env
            else None
        ),
        organization=(
            llm_client_module._optional_env_from(env, "OPENAI_ORGANIZATION")
            or llm_client_module._optional_env_from(env, "OPENAI_ORG_ID")
        ),
        project=(
            llm_client_module._optional_env_from(env, "OPENAI_PROJECT")
            or llm_client_module._optional_env_from(env, "OPENAI_PROJECT_ID")
        ),
        inherit_ambient_openai_sdk_config=False,
        allow_custom_base_url=True,
        defaults=defaults,
    )


class SharedModelElement(BasePipelineElement):
    """AgentDojo model element backed by Agent libOS's provider client."""

    def __init__(
        self,
        client: LLMClient,
        *,
        recorder: RunRecorder,
        max_output_tokens: int,
    ) -> None:
        self.client = client
        self.recorder = recorder
        self.max_output_tokens = max_output_tokens
        self.name = str(client.model or "agent-libos-llm")

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        provider_messages = _dojo_messages_to_openai(messages)
        provider_tools = _dojo_tools_to_openai(runtime)
        self.recorder.record_provider_request(
            messages=provider_messages,
            tools=provider_tools,
        )
        completion = self.client.complete_action(
            messages=provider_messages,
            tools=provider_tools,
            temperature=0.0,
            max_tokens=self.max_output_tokens,
            parallel_tool_calls=False,
        )
        self.recorder.record_assistant(completion)
        return query, runtime, env, [*messages, self.recorder.events[-1]], extra_args


class ControlPipeline(AgentPipeline):
    """AgentDojo-native tool loop using the same provider client as libOS."""

    def __init__(
        self,
        *,
        client: LLMClient,
        system_message: str,
        max_output_tokens: int,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
    ) -> None:
        self.recorder = RunRecorder()
        self.model_element = SharedModelElement(
            client,
            recorder=self.recorder,
            max_output_tokens=max_output_tokens,
        )
        loop = ToolsExecutionLoop(
            [ToolsExecutor(tool_result_to_str), self.model_element],
            max_iters=max_tool_iterations,
        )
        super().__init__([SystemMessage(system_message), InitQuery(), self.model_element, loop])
        self.name = f"{client.model or 'unknown-model'}-upstream-control"
        self.last_run: dict[str, Any] = {}

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        returned_messages: list[ChatMessage] = list(messages)
        try:
            outcome = super().query(query, runtime, env, messages, extra_args)
            returned_messages = list(outcome[3])
            return outcome
        finally:
            self.last_run = {
                "arm": "upstream_control",
                "messages": to_jsonable(returned_messages),
                "provider_calls": list(self.recorder.provider_calls),
                "usage": self.recorder.usage(),
                "provider_call_count": len(self.recorder.provider_calls),
                "tool_call_count": sum(
                    len(message.get("tool_calls") or [])
                    for message in returned_messages
                    if message.get("role") == "assistant"
                ),
            }

    def close(self) -> None:
        self.model_element.client.close()


@dataclass
class TerminalCaptureLLMClient(LLMClient):
    """Hide the runtime-only terminal tool and capture natural final text."""

    recorder: RunRecorder = field(default_factory=RunRecorder, repr=False)
    terminal_tool_name: str = HIDDEN_TERMINAL_TOOL

    @classmethod
    def from_client(
        cls,
        client: LLMClient,
        *,
        recorder: RunRecorder,
    ) -> "TerminalCaptureLLMClient":
        kwargs = {
            item.name: getattr(client, item.name)
            for item in fields(LLMClient)
            if item.init
        }
        return cls(**kwargs, recorder=recorder)

    async def acomplete_action(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        previous_response_id: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> LLMCompletion:
        visible_tools = [
            tool
            for tool in tools
            if _openai_tool_name(tool) != self.terminal_tool_name
        ]
        self.recorder.record_provider_request(
            messages=messages,
            tools=visible_tools,
        )
        completion = await super().acomplete_action(
            messages,
            visible_tools,
            temperature=temperature,
            max_tokens=max_tokens,
            previous_response_id=previous_response_id,
            parallel_tool_calls=False,
        )
        self.recorder.record_assistant(copy.deepcopy(completion))
        if not completion.tool_calls:
            self.recorder.final_answer = completion.content
            completion.tool_calls = [
                {
                    "id": f"agentdojo-final-{len(self.recorder.provider_calls)}",
                    "name": self.terminal_tool_name,
                    "arguments": json.dumps(
                        {"content": completion.content},
                        ensure_ascii=False,
                    ),
                }
            ]
        return completion


class _TerminalArgs(BaseModel):
    content: str = Field(description="The assistant's natural final response.")


class HiddenTerminalTool(SyncAgentTool[_TerminalArgs]):
    name = HIDDEN_TERMINAL_TOOL
    description = "Runtime-internal terminal carrier; never exposed to the model."
    args_schema = _TerminalArgs
    policy = ToolPolicy(side_effects=False, idempotent=True)
    tags = ["evaluation", "internal"]

    def __init__(self, recorder: RunRecorder) -> None:
        self.recorder = recorder

    def run(self, args: _TerminalArgs, ctx: ToolContext) -> ToolResult:
        self.recorder.final_answer = args.content
        return ToolResult.success(
            data={"captured": True},
            model_data="",
        )


class AgentDojoFunctionTool(SyncAgentTool[BaseModel]):
    """Exact-name/schema wrapper around an AgentDojo Function."""

    name = "agentdojo_placeholder"
    description = "AgentDojo function bridge placeholder."
    args_schema = BaseModel
    policy = ToolPolicy(side_effects=False, idempotent=False)
    tags = ["evaluation", "agentdojo", "native-semantics"]

    def __init__(
        self,
        function: Function,
        *,
        dojo_runtime: FunctionsRuntime,
        env: Any,
        recorder: RunRecorder,
    ) -> None:
        self.function = function
        self.name = function.name
        self.description = function.description
        self.args_schema = _agentdojo_compatible_parameters(function)
        self.dojo_runtime = dojo_runtime
        self.env = env
        self.recorder = recorder
        self.metadata = {
            "evaluation_semantics": "ambient_native_semantics",
            "agentdojo_function": function.name,
        }

    def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult:
        arguments = args.model_dump(mode="python")
        raw_result, error = self.dojo_runtime.run_function(
            self.env,
            self.function.name,
            arguments,
        )
        formatted = error or tool_result_to_str(raw_result)
        self.recorder.record_tool(
            function=self.function.name,
            args=arguments,
            formatted_result=formatted,
            raw_result=raw_result,
            error=error,
        )
        return ToolResult.success(
            content=formatted,
            data={
                "agentdojo_result": to_jsonable(raw_result),
                "agentdojo_error": error,
            },
            model_data=formatted,
            metadata={
                "agentdojo_error": error,
                "evaluation_semantics": "ambient_native_semantics",
            },
        )


def _agentdojo_compatible_parameters(function: Function) -> type[BaseModel]:
    """Preserve AgentDojo's pre-validation string-list coercion.

    AgentDojo's native ``ToolsExecutor`` converts arguments such as
    ``'["a@example.com"]'`` to a Python list before ``FunctionsRuntime`` runs
    Pydantic validation.  Agent libOS validates a tool call before invoking the
    wrapper, so the adapter must perform the same conversion in its argument
    model.  Retaining the original schema title keeps the provider-visible JSON
    schema byte-for-byte equivalent to the upstream function schema.
    """

    parameters = function.parameters

    def normalize_string_lists(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            key: (
                literal_eval(item)
                if isinstance(item, str) and is_string_list(item)
                else item
            )
            for key, item in value.items()
        }

    schema_title = str(
        parameters.model_json_schema().get("title") or parameters.__name__
    )
    return create_model(
        f"{parameters.__name__}AgentLibOSAdapter",
        __base__=parameters,
        __config__=ConfigDict(title=schema_title),
        __validators__={
            "normalize_agentdojo_string_lists": model_validator(mode="before")(
                normalize_string_lists
            )
        },
    )


ClientFactory = Callable[[RunRecorder], TerminalCaptureLLMClient]


class AgentLibOSAmbientPipeline(BasePipelineElement):
    """Run AgentDojo functions through the native Agent libOS loop.

    This arm intentionally grants ambient suite-wide tool authority. It tests
    integration and behavior parity, not capability/approval containment.
    """

    def __init__(
        self,
        *,
        client_factory: ClientFactory,
        system_message: str,
        runtime_dir: str | Path,
        config: AgentLibOSConfig,
        max_quanta: int = DEFAULT_MAX_TOOL_ITERATIONS + 1,
        prompt_mode: str = PROMPT_MODE_IMAGE_ONLY,
    ) -> None:
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"unknown Agent libOS prompt mode: {prompt_mode}")
        self.client_factory = client_factory
        self.system_message = system_message
        self.runtime_dir = Path(runtime_dir)
        self.config = config
        self.max_quanta = max_quanta
        self.prompt_mode = prompt_mode
        self.name = "agent-libos-ambient-native-semantics"
        self.last_run: dict[str, Any] = {}

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = (),
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        recorder = RunRecorder()
        client = self.client_factory(recorder)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        workspace = self.runtime_dir / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        store = SQLiteStore(self.runtime_dir / "runtime.sqlite")
        host: Runtime | None = None
        pid: str | None = None
        error: BaseException | None = None
        results: list[Any] = []
        replaced_tool_names: list[str] = []
        try:
            host = Runtime(
                store,
                llm_client=client,
                substrate=LocalResourceProviderSubstrate(workspace),
                config=self.config,
            )
            function_tools = [
                AgentDojoFunctionTool(
                    function,
                    dojo_runtime=runtime,
                    env=env,
                    recorder=recorder,
                )
                for function in runtime.functions.values()
            ]
            for tool in function_tools:
                try:
                    existing = host.tools.resolve(tool.name)
                except NotFound:
                    continue
                if not host.tools.unregister_tool(existing):
                    raise PipelineRunError(
                        f"failed to replace colliding runtime tool: {tool.name}"
                    )
                replaced_tool_names.append(tool.name)
            for tool in [*function_tools, HiddenTerminalTool(recorder)]:
                host.tools.register_tool(
                    tool,
                    registered_by="agentdojo-evaluation",
                    ephemeral=True,
                )
            host.register_image(
                AgentImage(
                    image_id=_IMAGE_ID,
                    name="agentdojo-native-semantics",
                    system_prompt=self.system_message,
                    prompt_mode=self.prompt_mode,
                    default_tools=[
                        *(function.name for function in runtime.functions.values()),
                        HIDDEN_TERMINAL_TOOL,
                    ],
                    metadata={
                        "evaluation": "agentdojo",
                        "semantics": "ambient_native_semantics",
                        "prompt_mode": self.prompt_mode,
                        "hidden_terminal_tool": HIDDEN_TERMINAL_TOOL,
                    },
                ),
                actor="agentdojo-evaluation",
            )
            pid = host.process.spawn(image=_IMAGE_ID, goal=query)
            for _ in range(self.max_quanta):
                results.extend(host.run_process_until_idle(pid, max_quanta=1))
                if recorder.final_answer is not None:
                    break
                if host.process.get(pid).status in host.process.TERMINAL_STATUSES:
                    break
            if recorder.final_answer is None:
                raise PipelineRunError(
                    f"Agent libOS reached max_quanta={self.max_quanta} without a final response"
                )
            process = host.process.get(pid)
            status_before_host_exit = process.status.value
            if (
                process.status in host.process.TERMINAL_STATUSES
                and process.status != ProcessStatus.EXITED
            ):
                raise PipelineRunError(
                    "Agent libOS terminated before the host could commit the captured "
                    f"final response: status={process.status.value}"
                )
            if process.status not in host.process.TERMINAL_STATUSES:
                host.process.exit(
                    pid,
                    message=recorder.final_answer,
                )
            process = host.process.get(pid)
            if process.status != ProcessStatus.EXITED:
                raise PipelineRunError(
                    f"Agent libOS final process status is not exited: {process.status.value}"
                )
            returned_messages: list[ChatMessage] = [
                {
                    "role": "system",
                    "content": [text_content_block_from_string(self.system_message)],
                },
                {
                    "role": "user",
                    "content": [text_content_block_from_string(query)],
                },
                *recorder.events,
            ]
            if returned_messages[-1]["role"] != "assistant":
                raise PipelineRunError("Agent libOS trace did not end with an assistant response")
            self.last_run = _ambient_run_snapshot(
                host,
                pid=pid,
                results=results,
                recorder=recorder,
                status_before_host_exit=status_before_host_exit,
                replaced_tool_names=replaced_tool_names,
                prompt_mode=self.prompt_mode,
                error=None,
            )
            returned_extra = dict(extra_args)
            returned_extra["agent_libos_dojo"] = {
                "arm": "libos_ambient",
                "pid": pid,
                "usage": recorder.usage(),
            }
            return query, runtime, env, returned_messages, returned_extra
        except BaseException as exc:
            error = exc
            if host is not None and pid is not None:
                self.last_run = _ambient_run_snapshot(
                    host,
                    pid=pid,
                    results=results,
                    recorder=recorder,
                    status_before_host_exit=None,
                    replaced_tool_names=replaced_tool_names,
                    prompt_mode=self.prompt_mode,
                    error=exc,
                )
            else:
                self.last_run = {
                    "arm": "libos_ambient",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "provider_calls": list(recorder.provider_calls),
                    "usage": recorder.usage(),
                }
            raise
        finally:
            if host is not None:
                host.close()
            else:
                store.close()
            if error is not None and not self.last_run:
                self.last_run = {
                    "arm": "libos_ambient",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }


def make_terminal_client_factory(
    snapshot: ExplicitDotenvSnapshot,
) -> ClientFactory:
    def factory(recorder: RunRecorder) -> TerminalCaptureLLMClient:
        base = snapshot.new_client()
        return TerminalCaptureLLMClient.from_client(base, recorder=recorder)

    return factory


def _ambient_run_snapshot(
    host: Runtime,
    *,
    pid: str,
    results: Sequence[Any],
    recorder: RunRecorder,
    status_before_host_exit: str | None,
    replaced_tool_names: Sequence[str],
    prompt_mode: str,
    error: BaseException | None,
) -> dict[str, Any]:
    process = host.process.get(pid)
    calls = sorted(
        host.store.list_llm_calls(
            pid=pid,
            limit=host.config.llm.call_record_hard_limit,
        ),
        key=lambda call: (call.created_at, call.call_id),
    )
    audit = host.audit.trace(actor=pid)
    audit_counts = Counter(record.action for record in audit)
    return {
        "arm": "libos_ambient",
        "semantics": "ambient_native_semantics",
        "prompt_mode": prompt_mode,
        "pid": pid,
        "process_status": process.status.value,
        "status_before_host_exit": status_before_host_exit,
        "status_message": process.status_message,
        "scheduler_result_count": len(results),
        "provider_call_count": len(recorder.provider_calls),
        "llm_call_record_count": len(calls),
        "tool_call_count": len(recorder.tool_executions),
        "hidden_terminal_tool_calls": 1 if recorder.final_answer is not None else 0,
        "replaced_runtime_tool_names": sorted(replaced_tool_names),
        "usage": recorder.usage(),
        "provider_calls": list(recorder.provider_calls),
        "tool_executions": list(recorder.tool_executions),
        "messages": to_jsonable(recorder.events),
        "audit_action_counts": dict(sorted(audit_counts.items())),
        "llm_call_records": [
            {
                "call_id": call.call_id,
                "status": call.status,
                "api": call.api,
                "model": call.model,
                "usage": to_jsonable(call.usage),
                "error": call.error,
            }
            for call in calls
        ],
        "external_effect_count": len(host.store.list_external_effects()),
        "external_effect_scope": (
            "Includes protected LLM provider calls; bridged AgentDojo tools are "
            "deliberately not classified as protected effects in this ambient pilot."
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
    }


def _dojo_messages_to_openai(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message["role"]
        content_value = message.get("content")
        content = (
            get_text_content_as_str(content_value)
            if isinstance(content_value, list)
            else ""
        )
        if role in {"system", "user"}:
            converted.append({"role": role, "content": content})
            continue
        if role == "assistant":
            assistant: dict[str, Any] = {"role": "assistant", "content": content}
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function,
                            "arguments": json.dumps(call.args, ensure_ascii=False),
                        },
                    }
                    for call in tool_calls
                ]
            converted.append(assistant)
            continue
        if role == "tool":
            converted.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id"),
                    "name": message["tool_call"].function,
                    "content": message.get("error") or content,
                }
            )
            continue
        raise ValueError(f"unsupported AgentDojo message role: {role}")
    return converted


def _dojo_tools_to_openai(runtime: FunctionsRuntime) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": function.parameters.model_json_schema(),
            },
        }
        for function in runtime.functions.values()
    ]


def _function_call(value: Mapping[str, Any]) -> FunctionCall:
    raw_args = value.get("arguments", {})
    if isinstance(raw_args, str):
        parsed = json.loads(raw_args or "{}")
    else:
        parsed = raw_args
    if not isinstance(parsed, Mapping):
        raise ValueError("provider tool arguments must decode to an object")
    return FunctionCall(
        function=str(value.get("name") or ""),
        args=dict(parsed),
        id=(str(value["id"]) if value.get("id") is not None else None),
    )


def _openai_tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")
