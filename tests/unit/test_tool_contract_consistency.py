"""Contract checks must detect drift, not merely regenerate matching prose."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import Field, field_validator

from agent_libos.llm.client import LLMClient
from agent_libos.skills.builtin_catalog import BuiltinSkillCatalog
from agent_libos.tools.builtin.checkpoint import CreateCheckpointArgs, CreateCheckpointTool
from agent_libos.tools.builtin.memory import CreateMemoryObjectArgs, CreateMemoryObjectTool, DirectJsonValue
from agent_libos.tools.contracts import CURRENT_PROCESS, RESULT_CONTRACTS, declared_field_contracts
from scripts import check_tool_contracts as checker
from scripts.tool_contract_support import assert_wire_schema, builtin_tools


def test_all_builtin_contracts_pass_without_rewriting_files() -> None:
    paths = [checker.ROOT / checker.REFERENCE, *(
        checker.ROOT / "agent_libos/skills/builtin"
    ).glob("*/SKILL.md")]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}

    report = checker.audit_contracts()

    assert report.violations == []
    assert report.tools == 101
    assert report.schema_cases == 808
    assert report.mcp_cases == 202
    assert (report.contracted_tools, report.contracted_fields) == (12, 14)
    assert (report.canonical_cases, report.rejection_cases, report.result_cases) == (184, 212, 32)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}


@pytest.mark.parametrize("drift", ["description", "default", "nullability", "coercion"])
def test_checker_rejects_argument_contract_drift(drift: str) -> None:
    class DriftArgs(CreateCheckpointArgs):
        pid: str | None = CURRENT_PROCESS.field()

        @field_validator("pid", mode="before")
        @classmethod
        def drift_alias(cls, value: Any) -> Any:
            return None if drift == "coercion" and value == "self" else value

    field = DriftArgs.model_fields["pid"]
    if drift == "description":
        field.description = "Use self instead of null"
    elif drift == "default":
        field.default = "pid_wrong_default"
    elif drift == "nullability":
        field.annotation = str
    DriftArgs.model_rebuild(force=True)

    class DriftTool(CreateCheckpointTool):
        args_schema = DriftArgs

    report = checker.ContractReport()
    checker.audit_tool(DriftTool(), report)
    assert report.violations
    assert all("create_checkpoint" in item for item in report.violations)


def test_removing_one_contract_from_a_multi_contract_tool_cannot_reduce_coverage() -> None:
    class MissingPayloadContract(CreateMemoryObjectArgs):
        payload: DirectJsonValue = Field()

    class DriftTool(CreateMemoryObjectTool):
        args_schema = MissingPayloadContract

    assert set(declared_field_contracts(DriftTool.args_schema)) == {"namespace"}
    report = checker.ContractReport()
    checker.audit_tool(DriftTool(), report)
    assert "create_memory_object: semantic field coverage drift" in report.violations


@pytest.mark.parametrize("drift", ["required", "closed", "constraint"])
def test_checker_detects_partial_strict_fallback_and_constraint_changes(drift: str) -> None:
    source = {
        "type": "object", "properties": {
            "pid": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "payload": {"type": "object", "additionalProperties": True},
        }, "required": ["payload"],
    }
    wire = deepcopy(source)
    if drift == "required":
        wire["required"] = ["pid", "payload"]
    elif drift == "closed":
        wire["additionalProperties"] = False
    else:
        wire["properties"]["pid"] = {"type": "string"}
    with pytest.raises(ValueError, match="changed constraints|partial strict"):
        assert_wire_schema(source, wire, False)
    with pytest.raises(ValueError, match="intentionally open"):
        assert_wire_schema(source, wire, True)


@pytest.mark.parametrize("raw", [
    "",
    "<!-- tool-contract: field:current_process -->\nx\n<!-- /tool-contract -->\n" * 2,
    "<!-- tool-contract: field:unknown -->\nx\n<!-- /tool-contract -->",
    "<!-- tool-contract: field:current_process -->\nx",
])
def test_generated_blocks_reject_missing_duplicate_unknown_and_malformed_markers(raw: str) -> None:
    with pytest.raises(ValueError, match="block declarations differ"):
        checker.render_skill_blocks(raw, {"field:current_process": CURRENT_PROCESS.description})


@pytest.fixture
def documentation_copy(tmp_path: Path) -> Path:
    shutil.copytree(
        checker.ROOT / "agent_libos/skills/builtin", tmp_path / "agent_libos/skills/builtin",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / checker.REFERENCE).write_text("old reference", encoding="utf-8")
    return tmp_path


def test_generator_repairs_stale_guidance_with_the_actual_catalog_loader(documentation_copy: Path) -> None:
    path = documentation_copy / "agent_libos/skills/builtin/agent-libos-checkpoints/SKILL.md"
    canonical = path.read_text()
    path.write_text(canonical.replace(CURRENT_PROCESS.description, "stale guidance"))
    tools = builtin_tools()
    stale = checker.ContractReport()
    checker.check_artifacts(tools, stale, root=documentation_copy, write=False)
    assert any("guidance is stale" in item for item in stale.violations)
    repaired = checker.ContractReport()
    checker.check_artifacts(tools, repaired, root=documentation_copy, write=True)
    assert repaired.violations == []
    assert path.read_text() == canonical
    fresh = checker.ContractReport()
    checker.check_artifacts(tools, fresh, root=documentation_copy, write=False)
    assert fresh.violations == []


@pytest.mark.parametrize("drift", ["budget", "host_budget", "unexpected_block", "semantic_error"])
def test_invalid_generation_does_not_write_any_staged_file(
    documentation_copy: Path, monkeypatch: pytest.MonkeyPatch, drift: str,
) -> None:
    path = documentation_copy / "agent_libos/skills/builtin/agent-libos-checkpoints/SKILL.md"
    path.write_text(path.read_text().replace(CURRENT_PROCESS.description, "stale guidance"))
    if drift == "budget":
        original = checker.contract_blocks

        def oversized(*args: Any) -> dict[str, dict[str, str]]:
            blocks = original(*args)
            blocks["agent-libos-runtime-session"]["result:process_exit"] = "x" * 17000
            return blocks

        monkeypatch.setattr(checker, "contract_blocks", oversized)
    elif drift == "host_budget":
        config = checker.contract_configs()[0][1]
        small = replace(config, skills=replace(config.skills, package_max_bytes=512, resource_read_max_bytes=256))
        monkeypatch.setattr(checker, "contract_configs", lambda: (("small", small), ("small_second", small)))
    elif drift == "unexpected_block":
        unexpected = documentation_copy / "agent_libos/skills/builtin/agent-libos-workspace-navigation/SKILL.md"
        unexpected.write_text(unexpected.read_text() + "\n<!-- tool-contract: field:current_process -->\nx\n<!-- /tool-contract -->\n")
    report = checker.ContractReport(violations=["injected semantic error"] if drift == "semantic_error" else [])
    before = {path: path.read_bytes() for path in documentation_copy.rglob("*.md")}

    checker.check_artifacts(builtin_tools(), report, root=documentation_copy, write=True)

    assert report.violations
    assert before == {path: path.read_bytes() for path in documentation_copy.rglob("*.md")}


@pytest.mark.parametrize("drift", ["missing_failure", "raw_data", "business_payload", "null_payload", "terminal_status"])
def test_checker_detects_broken_result_projection(monkeypatch: pytest.MonkeyPatch, drift: str) -> None:
    original = checker._compact_tool_result_payload

    def broken(carrier: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        result = deepcopy(original(carrier, **kwargs))
        if drift == "missing_failure" and carrier.get("ok") is False:
            result["result"] = None
        elif drift == "raw_data" and carrier.get("ok") is False:
            result["content"] = carrier.get("content")
        elif (drift == "business_payload" and carrier["tool_name"] == "read_memory_object"
              and isinstance(result.get("result", {}).get("payload"), dict)):
            result["result"]["payload"]["namespace"] = None
        elif (drift == "null_payload" and carrier["tool_name"] == "read_memory_object"
              and result.get("result", {}).get("representation") == "json_value"):
            result["result"].pop("payload", None)
        elif drift == "terminal_status" and carrier["tool_name"] == "process_exit" and "result" in carrier:
            result["result"]["status"] = "exited"
        return result

    monkeypatch.setattr(checker, "_compact_tool_result_payload", broken)
    report = checker.ContractReport()
    checker.audit_results(report)
    assert report.violations


def test_failure_checks_cannot_be_disabled_by_changing_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    contracts = dict(RESULT_CONTRACTS)
    contracts["append_memory_object"] = replace(contracts["append_memory_object"], preserve_public_failure=False)
    monkeypatch.setattr(checker, "RESULT_CONTRACTS", contracts)
    report = checker.ContractReport()
    checker.audit_results(report)
    assert "append_memory_object: failure diagnostics contract removed" in report.violations


@pytest.mark.parametrize("layout", ["legacy_v1", "cache_optimized_v2"])
@pytest.mark.parametrize("api", ["chat", "responses"])
def test_actual_client_request_preserves_every_builtin_wire_contract(layout: str, api: str) -> None:
    captured: list[dict[str, Any]] = []

    async def create(**payload: Any) -> Any:
        # JSON serialization also rejects accidental local metadata in wire data.
        captured.append(json.loads(json.dumps(payload)))
        return SimpleNamespace(
            id="contract_response", model="contract-test", output_text="", output=[],
            choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="", tool_calls=[]))],
        )

    endpoint = SimpleNamespace(create=create)
    client = LLMClient(model="contract-test", api_key="test-only", api_mode=api)
    client._async_client = SimpleNamespace(responses=endpoint, chat=SimpleNamespace(completions=endpoint))
    tools = builtin_tools()
    schemas = [tool.to_openai_chat_tool(prompt_layout=layout) for tool in tools]
    before = deepcopy(schemas)

    completion = asyncio.run(client.acomplete_action(
        messages=[{"role": "user", "content": "Contract transport check"}], tools=schemas,
    ))

    assert completion.api == api
    assert schemas == before
    assert len(captured) == 1
    sent = captured[0]["tools"]
    assert len(sent) == len(tools)
    for tool, definition in zip(tools, sent, strict=True):
        wire = definition if api == "responses" else definition["function"]
        assert wire["name"] == tool.name
        source = tool.spec(model_visible=True, prompt_layout=layout).input_schema
        assert_wire_schema(source, wire["parameters"], wire["strict"])
    assert set(BuiltinSkillCatalog().skill_for_tool(tool.name) for tool in tools) >= {
        "agent-libos-checkpoints", "agent-libos-object-memory", "agent-libos-runtime-session",
    }
