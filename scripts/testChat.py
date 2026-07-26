from __future__ import annotations

import asyncio

from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import AgentImage, ProcessStatus, ResourceBudget


_RUNTIME_DEFAULTS = DEFAULT_CONFIG.runtime
_SCRIPT_DEFAULTS = DEFAULT_CONFIG.scripts

CHAT_IMAGE_ID = "chat-image:v0"
CHAT_IMAGE_NAME = "ChatImage"
CHAT_GOAL = (
    "You are an AI assistant interacting via a terminal interface. To ensure "
    "a smooth and efficient conversation, follow these rules: "
    "1. Do not repeat the same text, greetings, or explanations in one turn. "
    "Keep responses concise and relevant to the new context. "
    "2. Every turn must conclude with a tool call to either `ask_human` "
    "(to receive the next user message) or `process_exit` (when the user wants "
    "to exit or the task is done). Never end a turn without one of those calls."
)


def chat_image() -> AgentImage:
    return AgentImage(
        image_id=CHAT_IMAGE_ID,
        name=CHAT_IMAGE_NAME,
        version="v0",
        system_prompt=(
            "Traditional human/LLM chat image with only human I/O and process "
            "exit tools."
        ),
        default_tools=["ask_human", "human_output", "process_exit"],
        context_policy="recency_first",
        required_capabilities=[
            {
                "resource": _RUNTIME_DEFAULTS.default_human_resource,
                "rights": ["write"],
            }
        ],
    )


async def run_chat() -> str:
    runtime = await Runtime.aopen(_RUNTIME_DEFAULTS.local_store_target)
    try:
        runtime.register_image(chat_image())
        pid = runtime.process.spawn(
            image=CHAT_IMAGE_ID,
            goal=CHAT_GOAL,
            resource_budget=ResourceBudget(
                max_context_materialization_tokens=_SCRIPT_DEFAULTS.chat_context_tokens
            ),
        )
        await runtime.arun_until_idle()
        process = runtime.process.get(pid)
        if process.status != ProcessStatus.EXITED:
            raise RuntimeError(
                f"chat process did not exit; status={process.status.value}"
            )
        return pid
    finally:
        await runtime.ashutdown(actor="script", reason="script.complete")


def main() -> None:
    asyncio.run(run_chat())


if __name__ == "__main__":
    main()
