"""Check native/wire/Skill/result contracts without LLM calls or tool effects.

Use --write to refresh existing generated Skill blocks and the reference.
The default command is read-only and exits nonzero for any mismatch.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from agent_libos.config import AgentLibOSConfig
from agent_libos.llm.prompt import _compact_tool_result_payload
from agent_libos.models.exceptions import ValidationError
from agent_libos.skills.builtin_catalog import BuiltinSkillCatalog
from agent_libos.tools.base import BaseAgentTool
from agent_libos.tools.contracts import (
    FIELD_CONTRACTS, RESULT_CONTRACTS, FieldContract, compact_checkpoint_created, declared_field_contracts,
)
from agent_libos.utils.openai_schema import openai_responses_tool_schema
from scripts.tool_contract_support import (
    CANONICAL_CALLS, REQUIRED_FIELD_CONTRACTS, REQUIRED_RESULT_FAMILIES,
    assert_wire_schema, builtin_tools, contract_configs, success_projection_cases, validate_field_value,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path("docs/tool_contracts.md")
_BLOCK = re.compile(
    r"<!-- tool-contract: ([a-z_]+:[a-z_]+) -->\n.*?\n<!-- /tool-contract -->",
    re.DOTALL,
)


@dataclass
class ContractReport:
    tools: int = 0
    schema_cases: int = 0
    mcp_cases: int = 0
    contracted_tools: int = 0
    contracted_fields: int = 0
    canonical_cases: int = 0
    rejection_cases: int = 0
    result_cases: int = 0
    violations: list[str] = field(default_factory=list)

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.violations.append(message)


def audit_field(
    tool: BaseAgentTool, name: str, contract: FieldContract,
    config: AgentLibOSConfig, layout: str, report: ContractReport,
) -> None:
    where = f"{tool.name}.{name} [{layout}]"
    field_info = tool.args_schema.model_fields[name]
    report.require(FIELD_CONTRACTS.get(contract.key) == contract, f"{where}: unknown or divergent contract")
    report.require(field_info.description == contract.description, f"{where}: field guidance drift")
    report.require(field_info.is_required() != contract.default_is_null, f"{where}: required/default drift")
    if contract.default_is_null:
        report.require(field_info.default is None, f"{where}: default must remain JSON null")
    seed = CANONICAL_CALLS.get(tool.name)
    if seed is None:
        report.violations.append(f"{where}: add an executable canonical call fixture")
        return
    native = tool.spec(config=config).input_schema
    source = tool.spec(config=config, model_visible=True, prompt_layout=layout).input_schema
    wire = tool.to_openai_chat_tool(config=config, prompt_layout=layout)["function"]["parameters"]
    for label, schema in (("native", native), ("model", source), ("wire", wire)):
        report.require(name in schema.get("properties", {}), f"{where}: field missing from {label}")
        if name not in schema.get("properties", {}):
            return
        report.require(schema["properties"][name].get("description") == contract.description,
                       f"{where}: {label} guidance drift")
        report.require(validate_field_value(schema, name, None) == contract.nullable,
                       f"{where}: {label} null semantics drift")
    if contract.default_is_null:
        omitted = {key: value for key, value in seed.items() if key != name}
        report.require(getattr(tool.parse_args(omitted, config=config), name) is None,
                       f"{where}: omission no longer resolves to null")
    for encoded in contract.canonical_values:
        value = json.loads(encoded)
        args = {**deepcopy(seed), name: value}
        parsed = tool.parse_args(args, config=config)
        actual = getattr(parsed, name)
        report.require(type(actual) is type(value) and actual == value,
                       f"{where}: canonical value coerced: {encoded}")
        full_call = parsed.model_dump(mode="json")
        full_call = {key: item for key, item in full_call.items() if key in source["properties"]}
        report.require(Draft202012Validator(wire).is_valid(full_call),
                       f"{where}: runtime-accepted canonical call rejected by wire: {encoded}")
        report.canonical_cases += 1
    for encoded in contract.rejected_values:
        value = json.loads(encoded)
        try:
            tool.parse_args({**deepcopy(seed), name: value}, config=config)
        except (PydanticValidationError, SchemaValidationError):
            pass
        else:
            report.violations.append(f"{where}: parser accepted forbidden value: {encoded}")
        for label, schema in (("native", native), ("model", source), ("wire", wire)):
            report.require(not validate_field_value(schema, name, value),
                           f"{where}: {label} accepts forbidden value: {encoded}")
        report.rejection_cases += 1


def audit_tool(tool: BaseAgentTool, report: ContractReport) -> None:
    try:
        contracts = declared_field_contracts(tool.args_schema)
    except ValueError as exc:
        report.violations.append(f"{tool.name}: {exc}")
        return
    report.require(
        {name: contract.key for name, contract in contracts.items()}
        == REQUIRED_FIELD_CONTRACTS.get(tool.name, {}),
        f"{tool.name}: semantic field coverage drift",
    )
    report.contracted_tools += bool(contracts)
    report.contracted_fields += len(contracts)
    for config_name, config in contract_configs():
        for layout in ("legacy_v1", "cache_optimized_v2"):
            where = f"{tool.name} [{config_name}/{layout}]"
            try:
                native = tool.spec(config=config)
                Draft202012Validator.check_schema(native.input_schema)
                Draft202012Validator.check_schema(native.output_schema)
                source = tool.spec(config=config, model_visible=True, prompt_layout=layout).input_schema
                chat = tool.to_openai_chat_tool(config=config, prompt_layout=layout)["function"]
                chat_before = deepcopy(chat)
                responses = openai_responses_tool_schema({"type": "function", "function": chat})
                report.require(responses is not None, f"{where}: Responses schema missing")
                if responses is None:
                    continue
                report.require(chat == chat_before, f"{where}: Responses conversion mutated Chat schema")
                for api, definition in (("chat", chat), ("responses", responses)):
                    assert_wire_schema(source, definition["parameters"], definition["strict"])
                    report.require(definition["name"] == tool.name, f"{where}/{api}: wrong tool identity")
                    report.schema_cases += 1
                report.require(chat["parameters"] == responses["parameters"] and chat["strict"] == responses["strict"],
                               f"{where}: Chat/Responses semantics differ")
                if layout == "legacy_v1":
                    mcp = tool.to_mcp_tool(config=config)
                    report.require(mcp["name"] == tool.name and mcp["inputSchema"] == source,
                                   f"{where}: MCP schema diverged from its model-visible schema")
                    report.mcp_cases += 1
                for name, contract in contracts.items():
                    audit_field(tool, name, contract, config, layout, report)
            except Exception as exc:
                report.violations.append(f"{where}: {type(exc).__name__}: {exc}")


def audit_results(report: ContractReport) -> None:
    """Check failure recovery, business data, exact targets, and terminal receipts."""

    report.require({name: contract.family for name, contract in RESULT_CONTRACTS.items()} == REQUIRED_RESULT_FAMILIES,
                   "result contract family coverage drift")
    hint = "target_payload_is_not_a_container_recreate_with_object_or_array"
    for name, family in REQUIRED_RESULT_FAMILIES.items():
        if family not in {"memory", "process_exit"}:
            continue
        contract = RESULT_CONTRACTS.get(name)
        report.require(contract is not None and contract.preserve_public_failure,
                       f"{name}: failure diagnostics contract removed")
        error = {"code": "validation_error", "type": "ValidationError",
                 "message": "PRIVATE-CONTRACT-SENTINEL", "correlation_id": "PRIVATE-CONTRACT-SENTINEL",
                 "details": {"hint": hint, "payload": "PRIVATE-CONTRACT-SENTINEL"}}
        for inner in ({"failure": {"error": error}}, {"error": error}, {"failure_omitted": True}):
            carrier = {"tool_name": name, "ok": False, **inner,
                       "content": "PRIVATE-CONTRACT-SENTINEL", "artifacts": ["PRIVATE-CONTRACT-SENTINEL"]}
            projection = _compact_tool_result_payload(carrier)
            result = projection.get("result")
            report.require(isinstance(result, dict) and result.get("ok") is False,
                           f"{name}: failure disappeared from model projection")
            if isinstance(result, dict) and not inner.get("failure_omitted"):
                report.require(result.get("error", {}).get("details", {}).get("hint") == hint,
                               f"{name}: public recovery hint disappeared")
            report.require("PRIVATE-CONTRACT-SENTINEL" not in json.dumps(projection),
                           f"{name}: raw failure data leaked")
            report.result_cases += 1
    for carrier, expected in success_projection_cases():
        before = deepcopy(carrier)
        projected = _compact_tool_result_payload(carrier, current_namespace="tenant:pid_contract_caller")
        report.require(projected == {"tool_name": carrier["tool_name"], "result": expected},
                       f"{carrier['tool_name']}: successful/terminal replay contract drift")
        report.require(carrier == before, f"{carrier['tool_name']}: replay mutated durable source data")
        report.result_cases += 1
    report.require(compact_checkpoint_created("Verified milestone") == {"created": True, "reason": "Verified milestone"},
                   "create_checkpoint: compact creation receipt drift")
    report.result_cases += 1


def contract_blocks(tools: list[BaseAgentTool], catalog: BuiltinSkillCatalog) -> dict[str, dict[str, str]]:
    """Derive required documentation blocks from the actual owning Skill."""

    blocks: dict[str, dict[str, str]] = {}
    for tool in tools:
        owner = catalog.skill_for_tool(tool.name)
        if owner is None:
            raise ValueError(f"{tool.name}: no owning builtin Skill")
        selected = blocks.setdefault(owner, {})
        for contract in declared_field_contracts(tool.args_schema).values():
            selected[f"field:{contract.key}"] = contract.description
        result = RESULT_CONTRACTS.get(tool.name)
        if result is not None:
            selected[f"result:{result.family}"] = result.guidance
    return {key: value for key, value in blocks.items() if value}


def render_skill_blocks(raw: str, expected: dict[str, str]) -> str:
    found = _BLOCK.findall(raw)
    if (len(found) != len(set(found)) or set(found) != set(expected)
            or raw.count("<!-- tool-contract:") != len(found)
            or raw.count("<!-- /tool-contract -->") != len(found)):
        raise ValueError(f"generated block declarations differ: expected={sorted(expected)}, found={sorted(found)}")
    return _BLOCK.sub(lambda match: (
        f"<!-- tool-contract: {match.group(1)} -->\n"
        f"{expected[match.group(1)]}\n<!-- /tool-contract -->"
    ), raw)


def render_reference(tools: list[BaseAgentTool], report: ContractReport) -> str:
    small_skills = contract_configs()[1][1].skills
    rows = []
    for tool in tools:
        for name, contract in declared_field_contracts(tool.args_schema).items():
            rows.append(f"| `{tool.name}.{name}` | `{contract.key}` | {contract.description} |")
    results = [f"| `{name}` | `{value.family}` | {value.guidance} |"
               for name, value in RESULT_CONTRACTS.items()]
    return "\n".join([
        "# Tool contracts", "", "Generated by `uv run python scripts/check_tool_contracts.py --write`.", "",
        "The checker inventories the core module without opening a Runtime or dispatching tools.",
        "It checks native input/output schemas, model-visible schemas, Chat/Responses strict conversion and non-strict fallback,",
        "MCP schema parity, canonical argument preservation/rejection, generated Skill guidance, and result replay.", "",
        f"Coverage: {report.tools} builtin tools; {report.contracted_tools} tools / {report.contracted_fields} fields have explicit semantic declarations.",
        f"Matrix: {report.schema_cases} Chat/Responses cases (two layouts, two Host configurations); {report.mcp_cases} MCP cases;",
        f"{report.canonical_cases} accepted argument cases, {report.rejection_cases} rejected argument cases, and {report.result_cases} result cases.",
        f"The second Host configuration raises selected tool bounds and sets a {small_skills.package_max_bytes // 1024} KiB Skill package / {small_skills.resource_read_max_bytes // 1024} KiB resource-read budget.",
        "All builtin tools receive schema checks. Unlisted fields and business effects still require their domain tests.",
        "Canonical examples are parser/schema evidence, not proof of authority or execution. Literal targets are never aliases.",
        "Descriptions and transport shapes may change; this declares no new authority, coercion, or provider capability.",
        "Passing this deterministic check does not satisfy the paired multi-provider prompt-cache release gate.", "",
        "## Check and regenerate", "",
        "```sh", "uv run python scripts/check_tool_contracts.py",
        "uv run python scripts/check_tool_contracts.py --write", "```", "",
        "The default command is read-only, requires no credentials or network, and exits nonzero on drift; CI runs it before test lanes.",
        "`--write` refreshes only declared Skill blocks and this reference. It validates all staged Skills using the actual catalog loader",
        "and the catalog plus matrix Host size limits before writing. Semantic/schema errors or a malformed/duplicate/missing block prevent all writes.",
        "Filesystem errors during writing are not a cross-file transaction. Free-form Skill prose remains hand-maintained.", "",
        "## Field declarations", "", "| Field | Contract | Generated guidance |", "|---|---|---|", *rows, "",
        "## Result declarations", "", "| Tool | Family | Generated guidance |", "|---|---|---|", *results, "",
        "## Extending coverage", "",
        "1. Define or reuse an immutable FieldContract in `agent_libos/tools/contracts.py` and assign its fresh `.field()` to the Pydantic argument.",
        "2. Add a canonical call and independent REQUIRED_FIELD_CONTRACTS coverage entry to `scripts/tool_contract_support.py`; retain literal/null/omitted and denial-path Runtime tests for actual target semantics.",
        "3. Add the required generated block markers to the owning Skill, then run the generator and review the diff.",
        "4. Result declarations drive specialized replay handling; extend REQUIRED_RESULT_FAMILIES, independent success/failure oracles, and actual Runtime output/projection tests when adding a family.",
        "5. Run this checker, the contract tests, and affected runtime/security tests. Never delete a contract to make drift disappear.", "",
        "Generated blocks govern only their declared facts. Free-form workflow prose, provider support, effects, and authority are not proven by string checks.",
        "Contracts are local Pydantic metadata; custom contract keywords are not sent to providers and are not a model-facing policy surface.", "",
        "[Documentation home](index.md) · [Tools and JIT](tools_and_jit.md)", "",
    ])


@dataclass
class _RenderedSkillFile:
    """Feed staged text through the actual catalog loader before writing."""

    text: str

    def open(self, mode: str) -> BytesIO:
        assert mode == "rb"
        return BytesIO(self.text.encode("utf-8"))


def check_artifacts(tools: list[BaseAgentTool], report: ContractReport, *, root: Path, write: bool) -> None:
    try:
        catalog = BuiltinSkillCatalog()
        blocks = contract_blocks(tools, catalog)
    except (OSError, ValueError, ValidationError) as exc:
        report.violations.append(f"builtin Skill catalog: {exc}")
        return
    report.require({tool.name for tool in tools} == {name for skill in catalog.list() for name in skill.allowed_tools},
                   "core tool registration and Skill ownership differ")
    staged: dict[Path, str] = {}
    for package in catalog.list():
        skill = package.skill_id
        expected = blocks.get(skill, {})
        path = root / "agent_libos/skills/builtin" / skill / "SKILL.md"
        try:
            before = path.read_text(encoding="utf-8")
            after = render_skill_blocks(before, expected)
            loaded = catalog._load_package(_RenderedSkillFile(after), skill)  # type: ignore[arg-type]
            report.require(loaded.allowed_tools == package.allowed_tools, f"{skill}: owning tools drift")
            for config_name, config in contract_configs():
                report.require(sum(resource.size_bytes for resource in loaded.resources) <= config.skills.package_max_bytes,
                               f"{skill} [{config_name}]: exceeds Host package_max_bytes={config.skills.package_max_bytes}")
                report.require(len(loaded.instructions) <= config.skills.max_prompt_instruction_chars,
                               f"{skill} [{config_name}]: exceeds Host max_prompt_instruction_chars")
        except (OSError, ValueError, ValidationError) as exc:
            report.violations.append(f"{skill}: {exc}")
            continue
        if before != after:
            staged[path] = after
        if not write:
            report.require(before == after, f"{skill}: generated tool contract guidance is stale")
    expected_reference = render_reference(tools, report)
    target = root / REFERENCE
    staged[target] = expected_reference
    if not write:
        report.require(target.is_file() and target.read_text(encoding="utf-8") == expected_reference,
                       f"{REFERENCE}: generated reference is stale")
    elif not report.violations:
        # Validation errors must not leave a partially generated repository.
        # This is a local documentation update, not a cross-file transaction.
        for path, text in staged.items():
            path.write_text(text, encoding="utf-8")


def audit_contracts(*, root: Path = ROOT, write: bool = False) -> ContractReport:
    report = ContractReport()
    tools = builtin_tools()
    report.tools = len(tools)
    for tool in tools:
        audit_tool(tool, report)
    report.require(set(CANONICAL_CALLS) == set(REQUIRED_FIELD_CONTRACTS),
                   "canonical call inventory and semantic field declarations differ")
    report.require(set(REQUIRED_FIELD_CONTRACTS) <= {tool.name for tool in tools},
                   "field contract references an absent tool")
    report.require(set(RESULT_CONTRACTS) <= {tool.name for tool in tools}, "result contract references an absent tool")
    audit_results(report)
    check_artifacts(tools, report, root=root, write=write)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Refresh existing generated blocks and reference.")
    args = parser.parse_args(argv)
    report = audit_contracts(write=args.write)
    for violation in report.violations:
        print(violation)
    print(json.dumps({**vars(report), "passed": not report.violations}, ensure_ascii=False))
    return 1 if report.violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
