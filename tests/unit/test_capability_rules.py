from __future__ import annotations

import pytest

from agent_libos.capability.resources import ResourceAuthority
from agent_libos.capability.rules import ShellRuleEngine
from agent_libos.models import AuthorityRisk, AuthorityRule, CapabilityEffect
from agent_libos.models.exceptions import CapabilityDenied


class TestCapabilityResources:
    def test_typed_resource_canonicalization_and_subsumption(self) -> None:
        resources = ResourceAuthority()

        assert resources.canonical("filesystem:workspace:src//main.py/") == "filesystem:workspace:src/main.py"
        assert resources.canonical("filesystem:workspace\\src//main.py/") == "filesystem:workspace\\src/main.py"
        assert resources.matches("filesystem:workspace:src/*", "filesystem:workspace:src/main.py")
        assert not resources.matches("filesystem:workspace:src/*", "filesystem:workspace:src2/main.py")

        with pytest.raises(CapabilityDenied):
            resources.parse("*")
        with pytest.raises(CapabilityDenied):
            resources.parse("filesystem:workspace:src*")


class TestShellRuleEngine:
    def test_classifies_default_shell_risk_levels(self) -> None:
        engine = ShellRuleEngine()

        assert engine.classify(["git", "status", "--short"]).rule.risk == AuthorityRisk.HARMLESS
        assert engine.classify(["git", "diff"]).rule.risk == AuthorityRisk.LOW
        assert engine.classify(["pytest"]).rule.risk == AuthorityRisk.MEDIUM
        assert engine.classify(["pytest", "--collect-only"]).rule.risk == AuthorityRisk.MEDIUM
        assert engine.classify(["python", "-m", "compileall", "agent_libos"]).rule.risk == AuthorityRisk.HIGH
        assert engine.classify(["curl", "https://example.test"]).rule.risk == AuthorityRisk.HIGH
        assert engine.classify(["rm", "-rf", "build"]).rule.risk == AuthorityRisk.DESTRUCTIVE

    def test_path_qualified_binary_does_not_match_harmless_bare_rule(self) -> None:
        engine = ShellRuleEngine()
        match = engine.classify(["./git", "status", "--short"])

        assert match.rule.effect == CapabilityEffect.ASK
        assert match.rule.risk == AuthorityRisk.MEDIUM

    def test_custom_authority_rule_can_match_argv_tokens(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.tool.safe",
                    operation="shell.run",
                    effect=CapabilityEffect.ALLOW,
                    risk=AuthorityRisk.LOW,
                    conditions={"argv": ["tool", "inspect"], "match": "prefix"},
                )
            ]
        )

        match = engine.classify(["tool", "inspect", "--json"])

        assert match.rule.rule_id == "custom.tool.safe"
        assert match.rule.effect == CapabilityEffect.ALLOW
