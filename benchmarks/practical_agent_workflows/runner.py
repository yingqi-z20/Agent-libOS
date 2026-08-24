from __future__ import annotations

import json
import socket
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

from agent_libos import Runtime
from agent_libos.models import (
    CapabilityRight,
    ExternalEffectClassification,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    JsonRpcTransportResult,
)
from agent_libos.substrate import LocalResourceProviderSubstrate
from benchmarks.practical_agent_workflows.models import (
    EvidenceLevel,
    NativeToolAction,
    PracticalRunReport,
    PracticalScenario,
    PracticalScenarioResult,
    SemanticEffect,
)
from benchmarks.practical_agent_workflows.catalog import build_modeled_scenarios
from benchmarks.practical_agent_workflows.oracle import validate_modeled_scenario
from benchmarks.practical_agent_workflows.validation import validate_practical_report


_NATIVE_DNS_PATCH_LOCK = threading.RLock()
_BENCHMARK_CONNECTOR_HOST = "connector.example.test"


def _benchmark_getaddrinfo(
    original: Any,
    host: object,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if host == _BENCHMARK_CONNECTOR_HOST:
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ]
    return original(host, *args, **kwargs)


class StatefulConnectorProvider:
    """Deterministic stateful provider used by native-live workflow scenarios."""

    def __init__(self) -> None:
        self.state: dict[str, dict[str, dict[str, Any]]] = {
            "mail": {},
            "crm": {},
            "calendar": {},
        }
        self.receipts: dict[str, dict[str, str]] = {}

    def call(self, _endpoint: Any, _method: Any, request_body: bytes, **_kwargs: Any) -> JsonRpcTransportResult:
        request = json.loads(request_body.decode("utf-8"))
        method = str(request["method"])
        params = dict(request.get("params") or {})
        collection, action = method.split(".", 1)
        key = str(params["id"])
        if key in self.state[collection]:
            result = {"ok": True, "id": key, "deduplicated": True}
        else:
            self.state[collection][key] = params
            result = {"ok": True, "id": key, "deduplicated": False}
        self.receipts[str(request.get("id") or "")] = {
            "status": "accepted",
            "semantic_effect_class": f"connector.{collection}.{action}",
            "semantic_target": key,
        }
        payload = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}).encode("utf-8")
        return JsonRpcTransportResult(
            status_code=200,
            body=payload,
            elapsed_s=0.001,
            response_bytes=len(payload),
        )

    def classify_external_effect(
        self,
        operation: str,
        context: dict[str, Any],
        result: Any,
    ) -> ExternalEffectClassification:
        receipt = dict(self.receipts.get(str(context.get("request_id") or ""), {}))
        return ExternalEffectClassification(
            rollback_class=ExternalEffectRollbackClass.IRREVERSIBLE,
            rollback_status=ExternalEffectRollbackStatus.NOT_SUPPORTED,
            state_mutation=True,
            information_flow=True,
            metadata={
                "operation": operation,
                "endpoint_id": context.get("endpoint_id"),
                "provider_receipt": receipt,
            },
        )


def default_scenarios() -> list[PracticalScenario]:
    return [
        _connector_scenario(
            "mail-send-001",
            "Send one email through a stateful connector",
            "mail",
            "send",
            {"id": "mail-001", "to": "reviewer@example.test", "subject": "Evaluation"},
        ),
        _connector_scenario(
            "crm-create-001",
            "Create one CRM contact through a stateful connector",
            "crm",
            "create",
            {"id": "crm-001", "name": "Ada", "stage": "new"},
        ),
        _connector_scenario(
            "calendar-create-001",
            "Create one calendar event through a stateful connector",
            "calendar",
            "create",
            {"id": "calendar-001", "title": "Review", "starts_at": "2030-01-01T09:00:00Z"},
        ),
        *build_modeled_scenarios(),
    ]


def run_practical_evaluation(
    scenarios: Iterable[PracticalScenario] | None = None,
    *,
    work_dir: str | Path | None = None,
) -> PracticalRunReport:
    selected = default_scenarios() if scenarios is None else list(scenarios)
    results: list[PracticalScenarioResult] = []
    for scenario in selected:
        if scenario.evidence_level == EvidenceLevel.MODELED:
            errors = validate_modeled_scenario(scenario)
            results.append(
                PracticalScenarioResult(
                    scenario_id=scenario.scenario_id,
                    evidence_level=scenario.evidence_level,
                    ok=not errors,
                    semantic_effects=len(scenario.effects),
                    tool_calls=0,
                    operations=0,
                    errors=errors,
                )
            )
            continue
        results.append(_run_native_scenario(scenario, work_dir=work_dir))
    scenario_counts = {
        level.value: sum(item.evidence_level == level for item in results)
        for level in EvidenceLevel
    }
    semantic_effect_counts = {
        level.value: sum(
            item.semantic_effects for item in results if item.evidence_level == level
        )
        for level in EvidenceLevel
    }
    native = [item for item in results if item.evidence_level == EvidenceLevel.NATIVE_LIVE]
    # There is intentionally no fallback path in the native executor. Any
    # absent tool/effect/operation evidence makes that scenario fail.
    report = PracticalRunReport(
        schema_version=1,
        results=results,
        scenario_counts=scenario_counts,
        semantic_effect_counts=semantic_effect_counts,
        native_tool_calls=sum(item.tool_calls for item in native),
        native_operations=sum(item.operations for item in native),
        modeled_fallback=0,
        native_live_ok=all(item.ok for item in native),
        modeled_suite_ok=all(
            item.ok for item in results if item.evidence_level == EvidenceLevel.MODELED
        ),
    )
    invariant_errors = validate_practical_report(report.to_dict())
    if invariant_errors:
        raise AssertionError(
            "practical report invariant violation: " + "; ".join(invariant_errors)
        )
    return report


def _run_native_scenario(
    scenario: PracticalScenario,
    *,
    work_dir: str | Path | None,
) -> PracticalScenarioResult:
    if scenario.evidence_level != EvidenceLevel.NATIVE_LIVE:
        raise ValueError("native executor accepts only native-live scenarios")
    errors: list[str] = []
    effect_ids: list[str] = []
    operation_ids: set[str] = set()
    tool_calls = 0
    unsupported_outcomes = [
        f"effects[{index}]={effect.expected_outcome!r}"
        for index, effect in enumerate(scenario.effects)
        if effect.expected_outcome != "committed"
    ]
    if unsupported_outcomes:
        return PracticalScenarioResult(
            scenario_id=scenario.scenario_id,
            evidence_level=scenario.evidence_level,
            ok=False,
            semantic_effects=len(scenario.effects),
            tool_calls=0,
            operations=0,
            errors=[
                "native-live supports only expected_outcome='committed': "
                + ", ".join(unsupported_outcomes)
            ],
        )
    owner = tempfile.TemporaryDirectory(dir=str(work_dir) if work_dir is not None else None)
    try:
        root = Path(owner.name)
        provider = StatefulConnectorProvider()
        workspace = root / "workspace"
        workspace.mkdir()
        substrate = LocalResourceProviderSubstrate(workspace)
        substrate.jsonrpc = provider
        runtime = Runtime.open(root / "runtime.sqlite", substrate=substrate)
        try:
            runtime.jsonrpc.register_endpoint_from_yaml_text(
                _connector_manifest(),
                actor="benchmark.host",
                require_capability=False,
            )
            # Use the exact method ids rather than inferring authority from the
            # image. The host-authored manifest is the native evaluation input.
            authorized = [
                {
                    "resource": f"jsonrpc:practical-connectors:{action.arguments['method_id']}",
                    "rights": [CapabilityRight.WRITE.value],
                }
                for action in scenario.native_actions
            ]
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal=scenario.title,
                authority_manifest={
                    "authorized_capabilities": authorized,
                    "permitted_effects": ["llm.*", "jsonrpc.call"],
                    "metadata": {"benchmark_scenario_id": scenario.scenario_id},
                },
            )
            with _NATIVE_DNS_PATCH_LOCK:
                original_getaddrinfo = socket.getaddrinfo

                def benchmark_getaddrinfo(
                    host: object,
                    *args: Any,
                    **kwargs: Any,
                ) -> Any:
                    return _benchmark_getaddrinfo(
                        original_getaddrinfo,
                        host,
                        *args,
                        **kwargs,
                    )

                with patch(
                    "agent_libos.primitives.jsonrpc.socket.getaddrinfo",
                    side_effect=benchmark_getaddrinfo,
                ):
                    for index, action in enumerate(scenario.native_actions):
                        expected_semantic_effect = scenario.effects[index]
                        before = dict(provider.state[action.oracle_collection])
                        prior_effects = {
                            item.effect_id
                            for item in runtime.store.list_external_effects(pid=pid)
                        }
                        tool_calls += 1
                        result = runtime.tools.call(
                            pid,
                            action.tool,
                            dict(action.arguments),
                        )
                        if not result.ok:
                            errors.append(f"tool {action.tool} failed: {result.error}")
                            continue
                        after = provider.state[action.oracle_collection]
                        if action.oracle_key in before or action.oracle_key not in after:
                            errors.append(
                                "state oracle failed for "
                                f"{action.oracle_collection}:{action.oracle_key}"
                            )
                        new_effects = [
                            item
                            for item in runtime.store.list_external_effects(pid=pid)
                            if item.effect_id not in prior_effects
                        ]
                        if len(new_effects) != 1:
                            errors.append(
                                "expected one native external effect, "
                                f"found {len(new_effects)}"
                            )
                            continue
                        effect = new_effects[0]
                        effect_ids.append(effect.effect_id)
                        if (
                            effect.transaction_state
                            != expected_semantic_effect.expected_outcome
                        ):
                            errors.append(
                                f"effect {effect.effect_id} is "
                                f"{effect.transaction_state}, not "
                                f"{expected_semantic_effect.expected_outcome}"
                            )
                        actual_semantic_effect = (
                            str(
                                effect.provider_receipt.get(
                                    "semantic_effect_class"
                                )
                                or ""
                            ),
                            str(
                                effect.provider_receipt.get("semantic_target")
                                or ""
                            ),
                        )
                        expected_semantic_pair = (
                            expected_semantic_effect.effect_class,
                            expected_semantic_effect.target,
                        )
                        if actual_semantic_effect != expected_semantic_pair:
                            errors.append(
                                "semantic effect mismatch: "
                                f"expected {expected_semantic_pair[0]}:"
                                f"{expected_semantic_pair[1]}, provider recorded "
                                f"{actual_semantic_effect[0]}:"
                                f"{actual_semantic_effect[1]}"
                            )
                        resolved = runtime.explain.resolve(
                            "effect",
                            effect.effect_id,
                        )
                        effect_operations = resolved.get("operations", [])
                        if not effect_operations:
                            errors.append(
                                f"effect {effect.effect_id} has no explicit "
                                "operation link"
                            )
                        operation_ids.update(
                            str(item["operation_id"])
                            for item in effect_operations
                        )
                        call_resolved = runtime.explain.resolve(
                            "call",
                            result.call_id,
                        )
                        operation_ids.update(
                            str(item["operation_id"])
                            for item in call_resolved.get("operations", [])
                        )
            if len(effect_ids) != len(scenario.effects):
                errors.append(
                    f"semantic/native effect mismatch: expected {len(scenario.effects)}, found {len(effect_ids)}"
                )
        finally:
            runtime.close()
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        owner.cleanup()
    return PracticalScenarioResult(
        scenario_id=scenario.scenario_id,
        evidence_level=scenario.evidence_level,
        ok=not errors,
        semantic_effects=len(scenario.effects),
        tool_calls=tool_calls,
        operations=len(operation_ids),
        external_effect_ids=effect_ids,
        operation_ids=sorted(operation_ids),
        errors=errors,
    )


def _connector_scenario(
    scenario_id: str,
    title: str,
    collection: str,
    action: str,
    params: dict[str, Any],
) -> PracticalScenario:
    method_id = f"{collection}-{action}"
    return PracticalScenario(
        scenario_id=scenario_id,
        title=title,
        evidence_level=EvidenceLevel.NATIVE_LIVE,
        effects=(SemanticEffect(f"connector.{collection}.{action}", str(params["id"])),),
        native_actions=(
            NativeToolAction(
                tool="call_jsonrpc_method",
                arguments={
                    "endpoint_id": "practical-connectors",
                    "method_id": method_id,
                    "params": params,
                },
                oracle_collection=collection,
                oracle_key=str(params["id"]),
            ),
        ),
    )


def _connector_manifest() -> str:
    methods = []
    for method_id, rpc_method in (
        ("mail-send", "mail.send"),
        ("crm-create", "crm.create"),
        ("calendar-create", "calendar.create"),
    ):
        methods.append(
            "\n".join(
                [
                    f"  - method_id: {method_id}",
                    f"    rpc_method: {rpc_method}",
                    "    right: write",
                    "    rollback_class: irreversible",
                    "    state_mutation: true",
                    "    information_flow: true",
                ]
            )
        )
    return "\n".join(
        [
            "schema_version: 1",
            "endpoint_id: practical-connectors",
            "url: https://connector.example.test/jsonrpc",
            "methods:",
            *methods,
            "timeout_s: 5",
            "max_request_bytes: 65536",
            "max_response_bytes: 1048576",
            "",
        ]
    )
