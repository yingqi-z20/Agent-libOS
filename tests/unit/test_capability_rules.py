from __future__ import annotations

import pytest

from agent_libos.capability.resources import ResourceAuthority
from agent_libos.capability.rules import AuthorityRuleCodec, ShellRuleEngine
from agent_libos.models import AuthorityRisk, AuthorityRule, CapabilityEffect
from agent_libos.models.exceptions import CapabilityDenied, ValidationError


class TestAuthorityRuleCodec:
    @pytest.mark.parametrize("conditions", [[], "", 0, False])
    def test_falsey_non_mapping_conditions_are_rejected(self, conditions: object) -> None:
        with pytest.raises(ValidationError, match="conditions must be a mapping"):
            AuthorityRuleCodec().coerce(
                {
                    "rule_id": "test.strict.conditions",
                    "operation": "shell.run",
                    "conditions": conditions,
                }
            )

    @pytest.mark.parametrize("payload", [{}, {"conditions": None}])
    def test_missing_or_none_conditions_are_normalized_to_empty_mapping(
        self,
        payload: dict[str, object],
    ) -> None:
        rule = AuthorityRuleCodec().coerce(
            {
                "rule_id": "test.optional.conditions",
                "operation": "shell.run",
                **payload,
            }
        )

        assert rule.conditions == {}


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

    @pytest.mark.parametrize(
        "argv",
        [
            ["curl", "https://example.test"],
            ["pytest"],
            ["npm", "test"],
            ["bash", "script.sh"],
        ],
    )
    def test_custom_deny_overrides_builtin_ask_rules(self, argv: list[str]) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.command.deny",
                    operation="shell.run",
                    effect=CapabilityEffect.DENY,
                    risk=AuthorityRisk.HIGH,
                    conditions={"argv": argv, "match": "exact"},
                )
            ]
        )

        assert engine.classify(argv).rule.rule_id == "custom.command.deny"

    def test_builtin_destructive_deny_remains_stronger_than_custom_rule(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.rm.allow",
                    operation="shell.run",
                    effect=CapabilityEffect.ALLOW,
                    risk=AuthorityRisk.HARMLESS,
                    conditions={"argv": ["rm", "-rf", "build"], "match": "exact"},
                )
            ]
        )

        assert engine.classify(["rm", "-rf", "build"]).rule.rule_id == "shell.destructive.default"

    def test_custom_deny_precedence_does_not_depend_on_rule_order(self) -> None:
        allow = AuthorityRule(
            rule_id="custom.tool.allow",
            operation="shell.run",
            effect=CapabilityEffect.ALLOW,
            risk=AuthorityRisk.LOW,
            conditions={"argv": ["tool", "inspect"], "match": "exact"},
        )
        deny = AuthorityRule(
            rule_id="custom.tool.deny",
            operation="shell.run",
            effect=CapabilityEffect.DENY,
            risk=AuthorityRisk.HIGH,
            conditions={"argv": ["tool", "inspect"], "match": "exact"},
        )

        for rules in ([allow, deny], [deny, allow]):
            assert ShellRuleEngine(rules).classify(["tool", "inspect"]).rule.rule_id == deny.rule_id

    def test_path_qualified_custom_rule_matches_full_executable_identity(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.absolute-tool.deny",
                    operation="shell.run",
                    effect=CapabilityEffect.DENY,
                    risk=AuthorityRisk.HIGH,
                    conditions={"argv": ["/usr/bin/trusted-tool"], "match": "exact"},
                )
            ]
        )

        assert engine.classify(["/usr/bin/trusted-tool"]).rule.rule_id == "custom.absolute-tool.deny"
        assert engine.classify(["./trusted-tool"]).rule.rule_id != "custom.absolute-tool.deny"

    def test_windows_path_identity_normalizes_drive_case_and_executable_suffix_conservatively(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.windows-tool.deny",
                    operation="shell.run",
                    effect=CapabilityEffect.DENY,
                    risk=AuthorityRisk.HIGH,
                    conditions={
                        "argv": [r"C:\Trusted\Bin\Tool.EXE"],
                        "match": "exact",
                    },
                )
            ]
        )

        assert engine.classify(["c:/trusted/bin/tool"]).rule.rule_id == "custom.windows-tool.deny"
        assert engine.classify(["d:/trusted/bin/tool"]).rule.rule_id != "custom.windows-tool.deny"
        assert engine.classify(["c:/other/bin/tool"]).rule.rule_id != "custom.windows-tool.deny"

    def test_custom_rule_conditions_are_conjunctive(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.context.deny",
                    operation="shell.run",
                    effect=CapabilityEffect.DENY,
                    risk=AuthorityRisk.HIGH,
                    conditions={
                        "argv": ["tool", "inspect"],
                        "match": "prefix",
                        "regex_token": r"--safe",
                        "cwd": ".",
                        "timeout_max_s": 1.0,
                        "resource": "shell:tool",
                        "right": "execute",
                    },
                )
            ]
        )
        context = {
            "operation": "shell.run",
            "authority_operation": "shell.run",
            "argv": ["tool", "inspect", "--safe"],
            "cwd": ".",
            "timeout_s": 0.5,
            "resource": "shell:tool",
            "right": "execute",
        }

        assert engine.classify(context["argv"], context=context).rule.rule_id == "custom.context.deny"
        for key, value in (
            ("cwd", "subdir"),
            ("timeout_s", 2.0),
            ("resource", "shell:other"),
        ):
            mismatched = {**context, key: value}
            assert engine.classify(context["argv"], context=mismatched).rule.rule_id != "custom.context.deny"
        mismatched_argv = ["tool", "inspect", "--unsafe-value"]
        assert engine.classify(
            mismatched_argv,
            context={**context, "argv": mismatched_argv},
        ).rule.rule_id != "custom.context.deny"

    def test_context_dependent_custom_rule_without_context_fails_closed(self) -> None:
        engine = ShellRuleEngine(
            [
                AuthorityRule(
                    rule_id="custom.context.allow",
                    operation="shell.run",
                    effect=CapabilityEffect.ALLOW,
                    risk=AuthorityRisk.LOW,
                    conditions={
                        "argv": ["tool", "inspect"],
                        "match": "exact",
                        "cwd": ".",
                    },
                )
            ]
        )

        match = engine.classify(["tool", "inspect"])

        assert match.rule.rule_id == "shell.custom.context-required"
        assert match.rule.effect == CapabilityEffect.DENY
