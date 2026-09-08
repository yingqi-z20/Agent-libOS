"""Explicit paid provider-tool smoke cases; skipped in deterministic test lanes.

Requires --run-real-llm, an exact case in AGENT_LIBOS_REAL_PROVIDER_TOOLS_CASES,
and dedicated AGENT_LIBOS_REAL_PROVIDER_TOOLS_{OPENAI|ALIYUN}_{API_KEY,MODEL}.
Aliyun additionally requires that prefix's BASE_URL. No .env or ambient SDK
settings are inherited. Select individual cases to control provider charges.
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Literal

import pytest

from agent_libos.config import ProviderToolsConfig
from agent_libos.llm.client import LLMClient


@dataclass(frozen=True)
class _Case:
    provider: Literal["openai", "aliyun"]
    api_mode: Literal["responses", "chat"]
    feature: Literal["web_search", "web_extractor", "code_interpreter"]

    @property
    def prefix(self) -> str:
        return f"AGENT_LIBOS_REAL_PROVIDER_TOOLS_{self.provider.upper()}"


_CASES = {
    "openai-search": _Case("openai", "responses", "web_search"),
    "openai-code": _Case("openai", "responses", "code_interpreter"),
    "aliyun-search": _Case("aliyun", "responses", "web_search"),
    "aliyun-extractor": _Case("aliyun", "responses", "web_extractor"),
    "aliyun-code": _Case("aliyun", "responses", "code_interpreter"),
    "aliyun-chat-search": _Case("aliyun", "chat", "web_search"),
}
_PROMPTS = {
    "web_search": "Use the provided web search tool to find the current title of the official Python documentation homepage. Return the title and source URL. Perform the search before answering.",
    "web_extractor": "Use web search and then the provided web_extractor tool to read https://example.com and report its heading. Actually invoke the extractor before answering.",
    "code_interpreter": "Use the provided code interpreter to calculate 6 * 7 in Python and print its result. Actually execute the code before answering. Do not create any files.",
}


@pytest.mark.parametrize("case_id", [
    pytest.param(name, marks=pytest.mark.real_llm(host_env_prefix=case.prefix))
    for name, case in _CASES.items()
])
def test_provider_tools_real(case_id: str) -> None:
    selected = {item.strip() for item in os.getenv("AGENT_LIBOS_REAL_PROVIDER_TOOLS_CASES", "").split(",")}
    if case_id not in selected:
        pytest.skip("Select this exact case in AGENT_LIBOS_REAL_PROVIDER_TOOLS_CASES")
    case = _CASES[case_id]
    model = os.getenv(f"{case.prefix}_MODEL")
    api_key = os.getenv(f"{case.prefix}_API_KEY")
    base_url = os.getenv(f"{case.prefix}_BASE_URL")
    if not model or not api_key:
        pytest.skip("Dedicated provider-tools Host credentials and model are required")
    if case.provider == "aliyun" and not base_url:
        pytest.skip("Dedicated Aliyun Host BASE_URL is required")
    config = ProviderToolsConfig(
        provider=case.provider,
        web_search=case.feature in {"web_search", "web_extractor"},
        web_extractor=case.feature == "web_extractor",
        code_interpreter=case.feature == "code_interpreter",
    )
    client = LLMClient(
        model=model, api_key=api_key, base_url=base_url,
        api_mode=case.api_mode, provider_tools=config,
        store=False, max_retries=0, timeout=45, logical_call_timeout_s=60,
        responses_replay=False, responses_previous_response_id=False,
        prompt_cache_mode="provider_default", fallback_json_actions=False,
        inherit_ambient_openai_sdk_config=False,
        allow_custom_base_url=base_url is not None,
    )

    async def run() -> None:
        try:
            completion = await client.acomplete_action(
                [{"role": "user", "content": _PROMPTS[case.feature]}], [], max_tokens=4096,
            )
            assert completion.api == case.api_mode
            assert completion.tool_calls == []
            observation = completion.provider_request_options["provider_tools"]
            assert case.feature in observation["effective"]
            assert completion.content or completion.provider_tool_activities
            if case.api_mode == "responses":
                assert any(activity["type"] == f"{case.feature}_call" for activity in completion.provider_tool_activities)
                assert observation["observed"] == "returned"
            else:
                # Chat may not return execution evidence. Enabling search does
                # not justify claiming that the provider actually performed it.
                assert observation["observed"] in {"returned", "unknown"}
            assert completion.response_items == []
        finally:
            await client.aclose()

    asyncio.run(run())
