from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent_libos.tools.builtin.process import (
    CompactProcessCompletionEvidence,
    ProcessExitArgs,
    ProcessExitTool,
)
from agent_libos.utils.openai_schema import openai_responses_tool_schema


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_exit_result_schema_explains_null_without_accepting_empty_string(
    layout: str, api: str
) -> None:
    chat = ProcessExitTool().to_openai_chat_tool(prompt_layout=layout)
    if api == "responses":
        tool = openai_responses_tool_schema(chat)
        assert tool is not None
    else:
        tool = chat["function"]
    field = tool["parameters"]["properties"]["result_oid"]

    assert "pass JSON null" in field["description"]
    assert "omission is also valid when allowed by the call schema" in field["description"]
    assert "Never pass an empty string" in field["description"]
    assert field["anyOf"] == [
        {"minLength": 1, "type": "string"},
        {"type": "null"},
    ]


@pytest.mark.parametrize("result_fields", [{}, {"result_oid": None}])
def test_missing_exit_result_preserves_stringified_evidence_compatibility(
    result_fields: dict[str, object],
) -> None:
    evidence = {
        "acceptance_checks": [
            {
                "status": "completed",
                "evidence_tool_calls": ["read_text_file"],
                "evidence_summary": "Readback matches the intended content.",
            }
        ],
        "final_verification": ["read_text_file"],
    }

    parsed = ProcessExitArgs.model_validate(
        {
            **result_fields,
            "payload": json.dumps({"summary": "Verified the edit."}),
            "completion_evidence": json.dumps(evidence),
        }
    )

    assert parsed.result_oid is None
    assert parsed.payload == {"summary": "Verified the edit."}
    assert isinstance(parsed.completion_evidence, CompactProcessCompletionEvidence)
    assert parsed.completion_evidence.model_dump() == evidence


def test_empty_exit_result_remains_a_validation_error() -> None:
    with pytest.raises(ValidationError) as caught:
        ProcessExitArgs.model_validate(
            {"result_oid": "", "payload": {"must_not_commit": True}}
        )

    errors = caught.value.errors(include_url=False, include_input=False)
    assert len(errors) == 1
    assert errors[0]["loc"] == ("result_oid",)
    assert errors[0]["type"] == "string_too_short"
