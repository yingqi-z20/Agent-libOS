from __future__ import annotations

import json
import time
from contextlib import nullcontext
from typing import Any

import pytest
from pydantic import BaseModel

from agent_libos import Runtime
from agent_libos.capability.evaluator import CapabilityEvaluator
from agent_libos.capability.resources import ResourceAuthority
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, ShellRuleEngine
from agent_libos.capability.transaction import AuthorityTransaction
from agent_libos.models import (
    AuthorityRisk,
    AuthorityRule,
    CapabilityDecision,
    CapabilityEffect,
    CapabilityLease,
    CapabilityRight,
    CapabilitySpec,
)
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.tools.base import SyncAgentTool, ToolContext


class _NoopArgs(BaseModel):
    pass


class _EphemeralNoopTool(SyncAgentTool[_NoopArgs]):
    name = "repair_ephemeral_noop"
    description = "Exercise exact ephemeral registry rollback."
    args_schema = _NoopArgs

    def run(self, args: _NoopArgs, ctx: ToolContext) -> dict[str, bool]:
        return {"ok": True}


class _TransactionStore:
    def transaction(self, *, include_object_payloads: bool = False):
        del include_object_payloads
        return nullcontext()


def _finite_decision() -> CapabilityDecision:
    return CapabilityDecision(
        subject="pid-subject",
        resource="object:finite",
        right=CapabilityRight.READ.value,
        allowed=True,
        effect=CapabilityEffect.ALLOW,
        reason="test",
        matched_capability_ids=["cap-finite"],
        selected_capability_id="cap-finite",
        consume_capability_id="cap-finite",
    )


def test_authority_transaction_fails_closed_when_reservation_is_missing() -> None:
    entered = False
    transaction = AuthorityTransaction(
        _TransactionStore(),
        [_finite_decision()],
        actor="pid-subject",
        operation="test mutation",
        reauthorize=lambda decision: decision,
        reserve=lambda *args, **kwargs: None,
        commit=lambda *args, **kwargs: True,
    )

    with pytest.raises(CapabilityDenied, match="reservation was not created"):
        with transaction:
            entered = True

    assert entered is False


@pytest.mark.parametrize("malformed", ("*", "untyped", "object:bad*tail"))
def test_resource_coverage_rejects_equal_malformed_patterns(malformed: str) -> None:
    assert ResourceAuthority().covers(malformed, malformed) is False


def test_authority_rule_match_modifier_requires_argv() -> None:
    rule = AuthorityRule(
        rule_id="bad.match.without.argv",
        operation="shell.run",
        effect=CapabilityEffect.ALLOW,
        risk=AuthorityRisk.HARMLESS,
        conditions={"match": "prefix"},
    )

    with pytest.raises(ValidationError, match="malformed conditions: match"):
        ShellRuleEngine([rule])


def test_regex_rule_has_admission_and_total_match_time_bounds() -> None:
    evaluator = CapabilityEvaluator()
    overlong = AuthorityRule(
        rule_id="overlong.regex",
        operation="shell.run",
        effect=CapabilityEffect.ALLOW,
        risk=AuthorityRisk.HARMLESS,
        conditions={"regex_token": "x" * 1_025},
    )
    catastrophic = AuthorityRule(
        rule_id="bounded.regex",
        operation="shell.run",
        effect=CapabilityEffect.ALLOW,
        risk=AuthorityRisk.HARMLESS,
        conditions={"regex_token": r"^(a|aa)+$"},
    )

    assert evaluator.malformed_authority_rule_conditions(overlong) == [
        "regex_token"
    ]
    started = time.perf_counter()
    assert evaluator.authority_rule_matches(
        catastrophic,
        {"operation": "shell.run", "argv": ["a" * 4_000 + "!"]},
    ) is False
    assert time.perf_counter() - started < 0.5


def test_regex_deny_rule_fails_closed_when_match_budget_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator = CapabilityEvaluator()
    deny = AuthorityRule(
        rule_id="bounded.deny",
        operation="shell.run",
        effect=CapabilityEffect.DENY,
        risk=AuthorityRisk.HIGH,
        conditions={"regex_token": r"--safe"},
    )
    ticks = iter((10.0, 11.0))
    monkeypatch.setattr(
        "agent_libos.capability.evaluator.time.monotonic",
        lambda: next(ticks),
    )

    assert evaluator.authority_rule_matches(
        deny,
        {"operation": "shell.run", "argv": ["--unknown"]},
    ) is True


def test_typed_capability_spec_rejects_conflicting_structured_aliases() -> None:
    runtime = Runtime.open("local")
    try:
        requested = {
            "resource": "object:typed-conflict",
            "rights": [CapabilityRight.READ.value],
        }
        with pytest.raises(ValidationError, match="conflicting lease"):
            runtime.capability.spec_covers(
                CapabilitySpec(
                    resource="object:typed-conflict",
                    rights={CapabilityRight.READ.value},
                    lease=CapabilityLease(uses_remaining=None),
                    uses_remaining=1,
                ),
                requested,
            )
        with pytest.raises(ValidationError, match="conflicting delegation"):
            runtime.capability.spec_covers(
                CapabilitySpec(
                    resource="object:typed-conflict",
                    rights={CapabilityRight.READ.value},
                    delegation={"delegable": False, "revocable": True},
                    delegable=True,
                ),
                requested,
            )
    finally:
        runtime.close()


@pytest.mark.parametrize("effect", (CapabilityEffect.DENY, CapabilityEffect.ASK))
def test_restrictive_spec_never_covers_allow_authority(
    effect: CapabilityEffect,
) -> None:
    runtime = Runtime.open("local")
    try:
        parent = CapabilitySpec(
            resource="object:effect-ceiling",
            rights={CapabilityRight.READ.value},
            effect=effect,
        )
        requested = CapabilitySpec(
            resource="object:effect-ceiling",
            rights={CapabilityRight.READ.value},
            effect=CapabilityEffect.ALLOW,
        )

        assert runtime.capability.spec_covers(parent, requested) is False
    finally:
        runtime.close()


def test_grant_transfer_uses_later_valid_covering_parent() -> None:
    runtime = Runtime.open("local")
    try:
        actor = runtime.process.spawn(goal="transfer actor")
        child = runtime.process.spawn(goal="transfer recipient")
        specific = runtime.capability.issue_trusted(
            actor,
            "object:transfer-target",
            [CapabilityRight.READ],
            issued_by="test.host",
            uses_remaining=1,
        )
        broad = runtime.capability.issue_trusted(
            actor,
            "object:*",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        runtime.capability.issue_trusted(
            actor,
            "object:transfer-target",
            [CapabilityRight.GRANT],
            issued_by="test.host",
        )

        child_capability = runtime.capability.issue(
            actor,
            child,
            CapabilitySpec(
                resource="object:transfer-target",
                rights={CapabilityRight.READ.value},
            ),
        )

        assert child_capability.parent_cap_id == broad.cap_id
        assert child_capability.parent_cap_id != specific.cap_id
    finally:
        runtime.close()


def test_stale_tool_handle_cannot_mutate_process_tables() -> None:
    runtime = Runtime.open("local")
    try:
        pid = runtime.process.spawn(goal="reject stale tool handle")
        handle = runtime.tools.register_tool(
            _EphemeralNoopTool(),
            registered_by="test",
            ephemeral=True,
        )
        assert runtime.tools.unregister_tool(handle, registered_by="test") is True
        before = runtime.process.get(pid)

        with pytest.raises(NotFound, match="tool not found"):
            runtime.tools.configure_process_tools(pid, [handle], assigned_by="test")

        after = runtime.process.get(pid)
        assert after.tool_table == before.tool_table
        assert after.model_tool_table == before.model_tool_table
        assert after.revision == before.revision
    finally:
        runtime.close()


def test_failed_ephemeral_unregister_restores_exact_name_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = Runtime.open("local")
    try:
        handle = runtime.tools.register_tool(
            _EphemeralNoopTool(),
            registered_by="test",
            ephemeral=True,
        )
        original_record = runtime.audit.record

        def fail_unregister_audit(**kwargs: Any) -> Any:
            record = original_record(**kwargs)
            if kwargs.get("action") == "tool.unregister":
                raise RuntimeError("injected unregister audit failure")
            return record

        with monkeypatch.context() as scoped:
            scoped.setattr(runtime.audit, "record", fail_unregister_audit)
            with pytest.raises(RuntimeError, match="unregister audit"):
                runtime.tools.unregister_tool(handle, registered_by="test")

        assert runtime.tools.resolve(handle.name) == handle
        assert runtime.tools.registry.implementation(handle.tool_id) is not None
    finally:
        runtime.close()


def test_capability_tools_preserve_large_receipts_and_page_large_inventory() -> None:
    runtime = Runtime.open("local")
    try:
        parent = runtime.process.spawn(goal="capability projection parent")
        child = runtime.process.spawn_child(
            parent,
            goal="capability projection child",
        )
        runtime.capability.issue_trusted(
            parent,
            "object:large-delegation",
            [CapabilityRight.READ],
            issued_by="test.host",
            delegable=True,
            max_delegation_depth=4,
        )
        runtime.tools.configure_process_tools(
            parent,
            ["delegate_capability"],
            assigned_by="test",
        )

        delegated = runtime.tools.call(
            parent,
            "delegate_capability",
            {
                "child_pid": child,
                "resource": "object:large-delegation",
                "rights": [CapabilityRight.READ.value],
                "metadata": {"large": "m" * 220_000},
            },
        )

        assert delegated.ok is True
        assert isinstance(delegated.payload, dict)
        receipt = delegated.payload["capability"]
        assert receipt["cap_id"]
        assert receipt["metadata"] == {}
        assert receipt["metadata_projection"]["omitted"] is True
        assert "result_omitted" not in delegated.payload

        expected_ids = {receipt["cap_id"]}
        for index in range(13):
            capability = runtime.capability.issue_trusted(
                child,
                f"object:large-list-{index:02d}",
                [CapabilityRight.READ],
                issued_by="test.host",
                constraints={
                    "git_allowed_refs": ["refs/heads/" + "c" * 14_980]
                },
            )
            expected_ids.add(capability.cap_id)

        runtime.tools.configure_process_tools(
            child,
            ["inspect_capability", "list_capabilities"],
            assigned_by="test",
        )
        inspected = runtime.tools.call(
            child,
            "inspect_capability",
            {"cap_id": receipt["cap_id"]},
        )
        assert inspected.ok is True
        assert inspected.payload["capability"]["cap_id"] == receipt["cap_id"]

        found: set[str] = set()
        cursor: str | None = None
        pages = 0
        while True:
            listed = runtime.tools.call(
                child,
                "list_capabilities",
                {"limit": 100, "after_cap_id": cursor},
            )
            assert listed.ok is True
            assert isinstance(listed.payload, dict)
            assert len(json.dumps(listed.payload).encode("utf-8")) < min(
                runtime.config.tools.tool_result_payload_hard_limit_bytes,
                runtime.config.tools.memory_payload_hard_limit_bytes,
            )
            found.update(
                item["cap_id"]
                for item in listed.payload["capabilities"]
                if item["cap_id"] in expected_ids
            )
            pages += 1
            if not listed.payload["has_more"]:
                assert listed.payload["next_cursor"] is None
                break
            cursor = listed.payload["next_cursor"]
            assert cursor
            assert pages < 30

        assert expected_ids <= found
        assert pages > 1
    finally:
        runtime.close()


def test_authority_rule_shape_is_rejected_before_capability_persistence() -> None:
    runtime = Runtime.open("local")
    try:
        subject = runtime.process.spawn(goal="reject malformed durable rule")
        with pytest.raises(ValidationError, match="malformed conditions: match"):
            runtime.capability.issue_trusted(
                subject,
                "shell:*",
                [CapabilityRight.EXECUTE],
                issued_by="test.host",
                constraints={
                    AUTHORITY_RULES_KEY: [
                        {
                            "rule_id": "bad.persisted.match",
                            "operation": "shell.run",
                            "effect": "allow",
                            "risk": "harmless",
                            "conditions": {"match": "exact"},
                        }
                    ]
                },
            )

        assert not [
            cap
            for cap in runtime.capability.capabilities_for(subject)
            if cap.resource == "shell:*"
        ]
    finally:
        runtime.close()
