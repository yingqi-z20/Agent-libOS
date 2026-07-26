from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from typing import Any, Protocol

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.llm.client import LLMClient, LLMCompletion
from agent_libos.llm.records import observable_llm_call_fields
from agent_libos.models import AgentImage, LLMCallRecord, ProcessStatus, ResourceBudget
from agent_libos.utils.ids import new_id, utc_now
from agent_libos.utils.public_errors import (
    internal_exception_observation,
    public_error_envelope,
)
from agent_libos.utils.serde import to_jsonable

if __package__:  # pragma: no branch - depends on module versus file execution
    from scripts.llm_context_probe import last_tool_result, recent_events
    from scripts.runtime_assembly import aopen_runtime
else:  # pragma: no cover - exercised by direct-entrypoint subprocess tests
    from llm_context_probe import last_tool_result, recent_events
    from runtime_assembly import aopen_runtime

_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts

CHAT_IMAGE_ID = "chat-image:v0"
CHAT_IMAGE_NAME = "ChatImage"
DEFAULT_EXIT_WORDS = ("/exit", "/quit", "quit", "退出", "再见")
DEFAULT_SYSTEM_PROMPT = ("You are a helpful assistant in a terminal chat. Reply to the user's latest message directly. "
                         "Keep answers concise unless the user asks for detail.")
CHAT_PROCESS_GOAL = (
    "You are an AI assistant interacting via a terminal interface. To ensure a smooth and efficient conversation, "
    "please adhere to the following rules: "
    "1. Do not repeat the same text, greetings, or explanations in one turn. Keep responses concise and strictly "
    "relevant to the new context. "
    "2. Every turn MUST conclude with a tool call to either `ask_human` (to receive the next user message) or "
    "`process_exit` (when the user wants to exit or the task is done). Never end a turn without calling one of these "
    "tools."
)


class ChatResponder(Protocol):
    def reply(self, history: list[dict[str, str]], user_message: str) -> str:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Run a traditional human/LLM chat loop through Agent libOS HumanObject tools: "
                     "ask_human for user input and human_output for assistant replies."))
    parser.add_argument(
        "--db",
        default=_RUNTIME_DEFAULTS.local_store_target,
        help=f"Runtime SQLite database path, or '{_RUNTIME_DEFAULTS.local_store_target}' for in-memory.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=_SCRIPT_DEFAULTS.chat_max_turns,
        help="Maximum user turns before the chat process exits.",
    )
    parser.add_argument("--max-quanta", type=int, default=None, help="Maximum Agent execution quanta.")
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt for the chat model.")
    parser.add_argument(
        "--exit-word",
        action="append",
        default=None,
        help="Exit word. Can be repeated. Defaults include /exit, quit, 退出, 再见.",
    )
    parser.add_argument(
        "--auto-message",
        action="append",
        default=None,
        help="Non-interactive human message. Repeat for multiple turns; /exit is used when exhausted.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic local echo responder instead of calling the configured LLM provider.",
    )
    args = parser.parse_args()

    responder: ChatResponder = EchoResponder() if args.mock else ModelResponder(system_prompt=args.system)
    report = asyncio.run(
        run_chat(
            db=args.db,
            responder=responder,
            max_turns=args.max_turns,
            max_quanta=args.max_quanta,
            exit_words=args.exit_word or DEFAULT_EXIT_WORDS,
            auto_messages=args.auto_message,
        )
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))


async def run_chat(
    *,
    db: str = _RUNTIME_DEFAULTS.local_store_target,
    responder: ChatResponder | None = None,
    max_turns: int = _SCRIPT_DEFAULTS.chat_max_turns,
    max_quanta: int | None = None,
    exit_words: Sequence[str] | None = DEFAULT_EXIT_WORDS,
    auto_messages: Sequence[str] | None = None,
    echo: bool = True,
) -> dict[str, Any]:
    if exit_words is None or len(exit_words) == 0:
        exit_words = DEFAULT_EXIT_WORDS
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    runtime = await aopen_runtime(db)
    outputs: list[str] = []
    client = HumanChatActionClient(
        responder=responder or ModelResponder(system_prompt=DEFAULT_SYSTEM_PROMPT),
        max_turns=max_turns,
        exit_words=exit_words,
    )
    runtime.llm.client = client

    def output_sink(message: str) -> None:
        outputs.append(message)
        if echo:
            print(message, flush=True)

    input_fn = _auto_input_fn(auto_messages, echo=echo) if auto_messages is not None else None
    runtime.substrate.human.output_sink = output_sink
    if input_fn is not None:
        runtime.substrate.human.input_reader = input_fn
    try:
        # A persistent Runtime reloads this fixed image identity. Replacing the
        # same script-owned definition makes reruns idempotent while still
        # exercising the normal image validation and publication path.
        runtime.register_image(chat_image(), replace=True)
        pid = runtime.process.spawn(
            image=CHAT_IMAGE_ID,
            goal=CHAT_PROCESS_GOAL,
            resource_budget=ResourceBudget(max_context_materialization_tokens=_SCRIPT_DEFAULTS.chat_context_tokens),
            authority_manifest={
                "authorized_capabilities": [
                    {
                        "resource": _RUNTIME_DEFAULTS.default_human_resource,
                        "rights": ["write"],
                    }
                ],
                "permitted_effects": ["llm.*", "human.*"],
                "metadata": {"provided_by": "human_llm_chat"},
            },
        )
        client.bind_runtime(runtime, pid)
        default_max_quanta = max_turns * _SCRIPT_DEFAULTS.chat_quanta_per_turn + _SCRIPT_DEFAULTS.chat_quanta_overhead
        results = await runtime.arun_until_idle(max_quanta=max_quanta or default_max_quanta)
        process = runtime.process.get(pid)
        report = {"pid": pid, "turns": client.turns, "process_status": process.status.value,
            "actions": [_action_name(result) for result in results], "outputs": outputs, "history": client.history,
            "model_calls": client.calls, "results": to_jsonable(results), }
        if process.status != ProcessStatus.EXITED:
            raise RuntimeError(f"chat process did not exit; status={process.status.value}")
        return report
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


class HumanChatActionClient:
    def __init__(self, *, responder: ChatResponder, max_turns: int, exit_words: Sequence[str], ):
        self.responder = responder
        self.max_turns = max_turns
        self.exit_words = {word.strip().lower() for word in exit_words if word.strip()}
        self.calls = 0
        self.turns = 0
        self.history: list[dict[str, str]] = []
        self._waiting_for_answer = False
        self._exit_after_output = False

    def bind_runtime(self, runtime: Runtime, pid: str) -> None:
        binder = getattr(self.responder, "bind_runtime", None)
        if callable(binder):
            binder(runtime, pid)

    def complete_action(self, messages: list[dict[str, str]], tools: list[dict[str, object]]) -> LLMCompletion:
        self.calls += 1
        if self._exit_after_output:
            return self._completion("process_exit", {"payload": {"turns": self.turns, "reason": "chat_finished"}}, )
        if not self._waiting_for_answer:
            # The script uses ask_human/human_output as the terminal transport.
            # This local client only decides which libOS tool action comes next.
            if self.turns >= self.max_turns:
                return self._completion("process_exit",
                    {"payload": {"turns": self.turns, "reason": "max_turns_reached"}}, )
            self._waiting_for_answer = True
            return self._completion("ask_human", {"question": "Human:"}, )

        user_message = _last_human_answer(messages).strip()
        self._waiting_for_answer = False
        if user_message.lower() in self.exit_words:
            self._exit_after_output = True
            return self._completion("human_output", {"message": "Assistant: goodbye."})

        reply = self.responder.reply(list(self.history), user_message).strip()
        if not reply:
            reply = "(empty response)"
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        self.turns += 1
        if self.turns >= self.max_turns:
            self._exit_after_output = True
        return self._completion("human_output", {"message": f"Assistant: {reply}"})

    def _completion(self, name: str, args: dict[str, Any]) -> LLMCompletion:
        return LLMCompletion(content="",
            tool_calls=[{"id": f"human_chat_{self.calls}", "name": name, "arguments": json.dumps(args)}], )


class ModelResponder:
    def __init__(self, *, system_prompt: str):
        self.system_prompt = system_prompt
        self.client = LLMClient.from_env()
        self._runtime: Runtime | None = None
        self._pid: str | None = None

    def bind_runtime(self, runtime: Runtime, pid: str) -> None:
        self._runtime = runtime
        self._pid = pid

    def reply(self, history: list[dict[str, str]], user_message: str) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        call_id = new_id("llmcall")
        created_at = utc_now()
        try:
            completion = self.client.complete_with_metadata(messages, json_mode=False)
        except Exception as exc:
            public_error = public_error_envelope(
                exc,
                code="llm_provider_error",
            )
            self._record_llm_call(
                call_id=call_id,
                messages=messages,
                created_at=created_at,
                error=public_error["message"],
                internal_error=internal_exception_observation(
                    exc,
                    correlation_id=public_error["correlation_id"],
                ),
            )
            raise
        self._record_llm_call(call_id=call_id, messages=messages, created_at=created_at, completion=completion)
        return completion.content

    def _record_llm_call(
        self,
        *,
        call_id: str,
        messages: list[dict[str, str]],
        created_at: str,
        completion: LLMCompletion | None = None,
        error: str | None = None,
        internal_error: dict[str, Any] | None = None,
    ) -> None:
        if self._runtime is None:
            return
        observable_fields = observable_llm_call_fields(
            messages=messages,
            tools=[],
            response_content=completion.content if completion else "",
            tool_calls=completion.tool_calls if completion else [],
            reasoning=completion.reasoning if completion else None,
            raw_response=completion.raw if completion else None,
            error=error,
            config=self._runtime.config,
        )
        observability = dict(observable_fields["observability"])
        if internal_error is not None:
            observability["failure"] = {"internal_error": internal_error}
        self._runtime.store.insert_llm_call(
            LLMCallRecord(
                call_id=call_id,
                pid=self._pid,
                image_id=CHAT_IMAGE_ID,
                purpose="script_human_chat_reply",
                status="error" if error else "ok",
                api=completion.api if completion else None,
                model=completion.model if completion else self.client.model,
                request_id=completion.request_id if completion else None,
                response_id=completion.response_id if completion else None,
                messages=observable_fields["messages"],
                tools=observable_fields["tools"],
                request_options={
                    "json_mode": False,
                    "client_class": type(self.client).__name__,
                    "real_llm_client": True,
                },
                response_content=observable_fields["response_content"],
                tool_calls=observable_fields["tool_calls"],
                reasoning=observable_fields["reasoning"],
                usage=completion.usage if completion else {},
                raw_response=observable_fields["raw_response"],
                observability=observability,
                error=observable_fields["error"],
                created_at=created_at,
                completed_at=utc_now(),
            )
        )


class EchoResponder:
    def reply(self, history: list[dict[str, str]], user_message: str) -> str:
        return f"Echo: {user_message}"


def chat_image() -> AgentImage:
    return AgentImage(image_id=CHAT_IMAGE_ID, name=CHAT_IMAGE_NAME, version="v0",
        system_prompt="Traditional human/LLM chat image with only human I/O and process exit tools.",
        default_tools=["ask_human", "human_output", "process_exit"], context_policy="recency_first",
        required_capabilities=[{"resource": _RUNTIME_DEFAULTS.default_human_resource, "rights": ["write"]}], )


def _auto_input_fn(messages: Sequence[str], *, echo: bool):
    remaining = list(messages)

    def input_fn(prompt: str) -> str:
        answer = remaining.pop(0) if remaining else "/exit"
        if echo:
            print(f"{prompt}{answer}", flush=True)
        return answer

    return input_fn


def _last_tool_result(messages: list[dict[str, str]], tool_name: str) -> dict[str, Any]:
    result = last_tool_result(messages, tool_name)
    if result is not None:
        return result
    raise AssertionError(f"no visible result for {tool_name}")


def _last_human_answer(messages: list[dict[str, str]]) -> str:
    # Prefer recent events over Object Memory payload order; the latter can
    # include older tool results after context sorting.
    for event in reversed(recent_events(messages)):
        if event.get("type") != "human_response":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        decision = payload.get("decision")
        if isinstance(decision, dict) and isinstance(decision.get("answer"), str):
            return decision["answer"]
    return str(_last_tool_result(messages, "ask_human").get("answer", ""))


def _action_name(result: object) -> str | None:
    if not isinstance(result, dict):
        return None
    action = result.get("action")
    if isinstance(action, dict):
        return action.get("action")
    return None


if __name__ == "__main__":
    main()
