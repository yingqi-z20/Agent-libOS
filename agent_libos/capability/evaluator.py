from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import regex as bounded_regex

from agent_libos.capability.effect_binding import (
    APPROVAL_BINDING_KEY,
    canonical_effect_hash,
    normalize_approval_binding,
)
from agent_libos.capability.rules import AUTHORITY_RULES_KEY, AuthorityRuleCodec
from agent_libos.config import DEFAULT_CONFIG, AgentLibOSConfig
from agent_libos.models import (
    Capability,
    CapabilityDecision,
    CapabilityEffect,
    DataReleaseBinding,
    OperationContext,
)
from agent_libos.models.exceptions import ValidationError


DATA_RELEASE_BINDING_KEY = "data_release_binding"

KNOWN_CONSTRAINT_KEYS = frozenset(
    {
        "shell_policy_level",
        "inherited_from",
        AUTHORITY_RULES_KEY,
        APPROVAL_BINDING_KEY,
        DATA_RELEASE_BINDING_KEY,
        "git_remote",
        "git_allowed_refs",
        "git_url_fingerprint",
        "git_expected_state_token",
        "git_old_oid",
    }
)

_STRING_RULE_CONDITIONS = frozenset(
    {
        "operation",
        "authority_operation",
        "argv_sha256",
        "cwd",
        "path",
        "resource",
        "right",
        "endpoint_id",
        "method_id",
        "rpc_method",
        "params_sha256",
        "server_id",
        "transport",
        "tool_id",
        "mcp_name",
        "arguments_sha256",
        "registry_spec_sha256",
        "content_sha256",
        "network",
        "filesystem_intent",
        "image_id",
        "git_remote",
        "git_remote_ref",
        "git_url_fingerprint",
        "git_expected_state_token",
        "git_old_oid",
    }
)

_INTEGER_RULE_CONDITIONS = frozenset({"registry_generation"})

_BOOLEAN_RULE_CONDITIONS = frozenset(
    {"continuous_session", "recursive", "missing_ok", "overwrite", "parents", "exist_ok"}
)

_DIRECT_RULE_CONDITIONS = tuple(
    sorted(
        (_STRING_RULE_CONDITIONS - {"operation", "authority_operation"})
        | _BOOLEAN_RULE_CONDITIONS
        | _INTEGER_RULE_CONDITIONS
    )
)

_ALLOWED_RULE_CONDITIONS = frozenset(
    {
        *_STRING_RULE_CONDITIONS,
        *_BOOLEAN_RULE_CONDITIONS,
        *_INTEGER_RULE_CONDITIONS,
        "argv",
        "match",
        "regex_token",
        "timeout_s",
        "timeout_max_s",
    }
)


class CapabilityEvaluator:
    """Side-effect-free capability constraint and precedence evaluator."""

    def __init__(
        self,
        rule_codec: AuthorityRuleCodec | None = None,
        *,
        config: AgentLibOSConfig | None = None,
    ) -> None:
        self.rule_codec = rule_codec or AuthorityRuleCodec()
        self.config = config or DEFAULT_CONFIG

    def decide(
        self,
        *,
        subject: str,
        resource: str,
        requested_right: str,
        matches: Sequence[Capability],
        context: Mapping[str, Any] | None = None,
        issuer_chains: Mapping[str, Sequence[str]] | None = None,
    ) -> CapabilityDecision:
        selected_context = dict(context or {})
        chains = issuer_chains or {}
        matched_ids = [cap.cap_id for cap in matches]
        failed_constraints: list[tuple[Capability, dict[str, Any]]] = []
        decisions: list[tuple[Capability, CapabilityDecision]] = []
        for cap in matches:
            results = self.evaluate_constraints(cap, selected_context)
            decision = self._decide_match(
                cap=cap,
                subject=subject,
                resource=resource,
                requested_right=requested_right,
                matched_ids=matched_ids,
                context=selected_context,
                constraint_results=results,
                issuer_chain=list(chains.get(cap.cap_id, ())),
            )
            if decision is not None:
                decisions.append((cap, decision))
            else:
                failed_constraints.append((cap, results))
        denied = next(
            (decision for _cap, decision in decisions if decision.effect == CapabilityEffect.DENY),
            None,
        )
        if denied is not None:
            return denied
        approved_once = next(
            (
                decision
                for cap, decision in decisions
                if self._is_matched_one_shot_approval(cap, decision)
            ),
            None,
        )
        if approved_once is not None:
            return approved_once
        for effect in (CapabilityEffect.ASK, CapabilityEffect.ALLOW):
            selected = next(
                (decision for _cap, decision in decisions if decision.effect == effect),
                None,
            )
            if selected is not None:
                return selected
        if failed_constraints:
            cap, results = failed_constraints[0]
            return self._decision(
                cap=cap,
                subject=subject,
                resource=resource,
                right=requested_right,
                allowed=False,
                effect=None,
                reason=f"capability constraints rejected {requested_right} on {resource}",
                matched_ids=matched_ids,
                issuer_chain=list(chains.get(cap.cap_id, ())),
                constraint_results=results,
                context=selected_context,
            )
        return CapabilityDecision(
            subject=subject,
            resource=resource,
            right=requested_right,
            allowed=False,
            effect=None,
            reason=f"{subject} lacks {requested_right} on {resource}",
            matched_capability_ids=matched_ids,
            context=selected_context,
        )

    @staticmethod
    def _is_matched_one_shot_approval(
        cap: Capability,
        decision: CapabilityDecision,
    ) -> bool:
        binding_result = decision.constraint_results.get(APPROVAL_BINDING_KEY)
        return (
            decision.effect == CapabilityEffect.ALLOW
            and cap.uses_remaining == 1
            and decision.consume_capability_id == cap.cap_id
            and isinstance(binding_result, dict)
            and binding_result.get("ok") is True
        )

    def _decide_match(
        self,
        *,
        cap: Capability,
        subject: str,
        resource: str,
        requested_right: str,
        matched_ids: list[str],
        context: dict[str, Any],
        constraint_results: dict[str, Any],
        issuer_chain: list[str],
    ) -> CapabilityDecision | None:
        constraint_effect = self.constraint_effect(constraint_results)
        constraints_ok = all(bool(item.get("ok")) for item in constraint_results.values())
        restrictive_failure = (
            not constraints_ok
            and cap.effect in {CapabilityEffect.DENY, CapabilityEffect.ASK}
            and not self.constraint_failure_is_scoped_miss(constraint_results)
        )
        if constraint_effect == CapabilityEffect.DENY or restrictive_failure:
            return self._decision(
                cap=cap,
                subject=subject,
                resource=resource,
                right=requested_right,
                allowed=False,
                effect=CapabilityEffect.DENY,
                reason=f"capability constraints denied {requested_right} on {resource}",
                matched_ids=matched_ids,
                issuer_chain=issuer_chain,
                constraint_results=constraint_results,
                context=context,
            )
        if not constraints_ok:
            return None
        effect = cap.effect
        if effect == CapabilityEffect.DENY:
            reason = f"{subject} denied {requested_right} on {resource}"
            allowed = False
        elif effect == CapabilityEffect.ASK or constraint_effect == CapabilityEffect.ASK:
            effect = CapabilityEffect.ASK
            reason = f"{subject} requires human approval for {requested_right} on {resource}"
            allowed = False
        else:
            reason = "capability allowed operation"
            allowed = True
        return self._decision(
            cap=cap,
            subject=subject,
            resource=resource,
            right=requested_right,
            allowed=allowed,
            effect=effect,
            reason=reason,
            matched_ids=matched_ids,
            issuer_chain=issuer_chain,
            constraint_results=constraint_results,
            context=context,
        )

    @staticmethod
    def _decision(
        *,
        cap: Capability,
        subject: str,
        resource: str,
        right: str,
        allowed: bool,
        effect: CapabilityEffect | None,
        reason: str,
        matched_ids: list[str],
        issuer_chain: list[str],
        constraint_results: dict[str, Any],
        context: dict[str, Any],
    ) -> CapabilityDecision:
        return CapabilityDecision(
            subject=subject,
            resource=resource,
            right=right,
            allowed=allowed,
            effect=effect,
            reason=reason,
            matched_capability_ids=matched_ids,
            selected_capability_id=cap.cap_id,
            consume_capability_id=cap.cap_id if allowed and cap.uses_remaining is not None else None,
            issuer_chain=issuer_chain,
            constraint_results=constraint_results,
            context=context,
        )

    def evaluate_constraints(self, cap: Capability, context: Mapping[str, Any]) -> dict[str, Any]:
        selected_context = dict(context)
        results: dict[str, Any] = {}
        for key, value in cap.constraints.items():
            if key not in KNOWN_CONSTRAINT_KEYS:
                results[key] = {"ok": False, "reason": "unknown constraint key"}
            elif key == AUTHORITY_RULES_KEY:
                results[key] = self._evaluate_rule_constraint(value, selected_context)
            elif key == APPROVAL_BINDING_KEY:
                results[key] = self._evaluate_approval_binding(value, selected_context)
            elif key == DATA_RELEASE_BINDING_KEY:
                results[key] = self._evaluate_data_release_binding(value, selected_context)
            elif key == "git_remote":
                results[key] = self._evaluate_git_equal(
                    value,
                    selected_context.get("git_remote"),
                    "remote",
                )
            elif key == "git_url_fingerprint":
                results[key] = self._evaluate_git_equal(
                    value,
                    selected_context.get("git_url_fingerprint"),
                    "remote URL fingerprint",
                )
            elif key == "git_expected_state_token":
                results[key] = self._evaluate_git_equal(
                    value,
                    selected_context.get("git_expected_state_token"),
                    "state token",
                )
            elif key == "git_old_oid":
                results[key] = self._evaluate_git_equal(
                    value,
                    selected_context.get("git_old_oid"),
                    "old object id",
                )
            elif key == "git_allowed_refs":
                allowed = value if isinstance(value, (list, tuple, set, frozenset)) else ()
                actual = selected_context.get("git_remote_ref")
                matched = isinstance(actual, str) and actual in {str(item) for item in allowed}
                results[key] = {
                    "ok": matched,
                    "reason": (
                        "Git ref matched"
                        if matched
                        else "Git ref is outside the capability allowlist"
                    ),
                }
            else:
                results[key] = {"ok": True, "value": value}
        return results

    @staticmethod
    def _evaluate_git_equal(expected: Any, actual: Any, label: str) -> dict[str, Any]:
        matched = isinstance(expected, str) and expected == actual
        return {
            "ok": matched,
            "reason": (
                f"Git {label} matched"
                if matched
                else f"Git {label} did not match"
            ),
        }

    def _evaluate_rule_constraint(self, value: Any, context: dict[str, Any]) -> dict[str, Any]:
        try:
            rules = self.rule_codec.coerce_many(value)
        except ValidationError as exc:
            return {"ok": False, "reason": str(exc)}
        return self._evaluate_authority_rules(rules, context)

    @staticmethod
    def _evaluate_approval_binding(value: Any, context: dict[str, Any]) -> dict[str, Any]:
        try:
            binding = normalize_approval_binding(value)
        except ValidationError as exc:
            return {"ok": False, "reason": str(exc)}
        actual_hash = canonical_effect_hash(context)
        expected_version = binding.get("target_state_version")
        actual_version = context.get("target_state_version")
        matched = binding["canonical_args_hash"] == actual_hash and (
            expected_version is None or expected_version == actual_version
        )
        return {
            "ok": matched,
            "reason": (
                "approval binding matched"
                if matched
                else "approved effect arguments or target state changed"
            ),
            "effect_id": binding["effect_id"],
            "canonical_args_hash": actual_hash,
            "target_state_version": actual_version,
        }

    @staticmethod
    def _evaluate_data_release_binding(value: Any, context: dict[str, Any]) -> dict[str, Any]:
        try:
            expected = DataReleaseBinding.normalize(value)
            actual = DataReleaseBinding.normalize(context.get(DATA_RELEASE_BINDING_KEY))
        except (TypeError, ValueError) as exc:
            return {"ok": False, "reason": str(exc)}
        matched = expected == actual
        return {
            "ok": matched,
            "reason": (
                "data release binding matched"
                if matched
                else "data release Sink, source, payload, policy, or operation changed"
            ),
            "sink": actual["sink"],
            "registry_generation": actual["registry_generation"],
            "payload_hash": actual["payload_hash"],
        }

    @staticmethod
    def constraint_effect(constraint_results: Mapping[str, Any]) -> CapabilityEffect | None:
        effects = {
            str(result.get("effect"))
            for result in constraint_results.values()
            if result.get("effect") is not None
        }
        for effect in (CapabilityEffect.DENY, CapabilityEffect.ASK, CapabilityEffect.ALLOW):
            if effect.value in effects:
                return effect
        return None

    @staticmethod
    def constraint_failure_is_scoped_miss(constraint_results: Mapping[str, Any]) -> bool:
        failed = {
            key: result
            for key, result in constraint_results.items()
            if not bool(result.get("ok"))
        }
        return (
            set(failed) == {AUTHORITY_RULES_KEY}
            and failed[AUTHORITY_RULES_KEY].get("reason") == "no authority rule matched operation context"
        )

    def _evaluate_authority_rules(self, rules: list[Any], context: dict[str, Any]) -> dict[str, Any]:
        operation = str(context.get("authority_operation") or context.get("operation") or "")
        if not operation:
            return {"ok": False, "reason": "authority rule requires operation context"}
        matched = []
        for rule in (item for item in rules if item.operation == operation):
            condition_error = self._rule_condition_error(rule, operation)
            if condition_error is not None:
                return condition_error
            if self._authority_rule_matches(rule, context):
                matched.append(rule)
        if not matched:
            return {
                "ok": False,
                "reason": "no authority rule matched operation context",
                "operation": operation,
                "rule_ids": [rule.rule_id for rule in rules],
            }
        selected = next((rule for rule in matched if rule.effect == CapabilityEffect.DENY), None)
        selected = selected or next((rule for rule in matched if rule.effect == CapabilityEffect.ASK), None)
        selected = selected or matched[0]
        denied = selected.effect == CapabilityEffect.DENY
        result = {
            "ok": not denied,
            "effect": selected.effect.value,
            "rule_id": selected.rule_id,
            "risk": selected.risk.value,
            "operation": operation,
        }
        if denied:
            result["reason"] = "authority rule denied operation"
        return result

    def _rule_condition_error(self, rule: Any, operation: str) -> dict[str, Any] | None:
        unknown = self._unknown_authority_rule_conditions(rule)
        malformed = self._malformed_authority_rule_conditions(rule)
        if not unknown and not malformed:
            return None
        detail_key = "unknown_conditions" if unknown else "malformed_conditions"
        return {
            "ok": False,
            "effect": CapabilityEffect.DENY.value,
            "reason": "malformed authority rule condition",
            "operation": operation,
            "rule_id": rule.rule_id,
            detail_key: unknown or malformed,
        }

    @staticmethod
    def _unknown_authority_rule_conditions(rule: Any) -> list[str]:
        return sorted(key for key in dict(rule.conditions or {}) if key not in _ALLOWED_RULE_CONDITIONS)

    def unknown_authority_rule_conditions(self, rule: Any) -> list[str]:
        return self._unknown_authority_rule_conditions(rule)

    def _malformed_authority_rule_conditions(self, rule: Any) -> list[str]:
        conditions = dict(rule.conditions or {})
        malformed = self._malformed_typed_authority_conditions(conditions)
        if "argv" in conditions and (
            not isinstance(conditions["argv"], list)
            or not all(isinstance(item, str) for item in conditions["argv"])
        ):
            malformed.append("argv")
        if "match" in conditions and (
            "argv" not in conditions
            or conditions["match"] not in {"exact", "prefix"}
        ):
            malformed.append("match")
        if "regex_token" in conditions and not self._valid_regex(conditions["regex_token"]):
            malformed.append("regex_token")
        for key in ("timeout_s", "timeout_max_s"):
            if key in conditions and self._finite_nonnegative_timeout(conditions[key]) is None:
                malformed.append(key)
        return sorted(set(malformed))

    @staticmethod
    def _malformed_typed_authority_conditions(
        conditions: dict[str, Any],
    ) -> list[str]:
        malformed = [
            key
            for key in _STRING_RULE_CONDITIONS
            if key in conditions and not isinstance(conditions[key], str)
        ]
        malformed.extend(
            key
            for key in _BOOLEAN_RULE_CONDITIONS
            if key in conditions and not isinstance(conditions[key], bool)
        )
        malformed.extend(
            key
            for key in _INTEGER_RULE_CONDITIONS
            if key in conditions
            and (
                isinstance(conditions[key], bool)
                or not isinstance(conditions[key], int)
                or conditions[key] < 0
            )
        )
        return malformed

    def malformed_authority_rule_conditions(self, rule: Any) -> list[str]:
        return self._malformed_authority_rule_conditions(rule)

    def _authority_rule_matches(self, rule: Any, context: dict[str, Any]) -> bool:
        conditions = dict(rule.conditions or {})
        regex_unavailable = False
        if "operation" in conditions and str(context.get("operation")) != str(conditions["operation"]):
            return False
        if "authority_operation" in conditions and (
            str(context.get("authority_operation")) != str(conditions["authority_operation"])
        ):
            return False
        if "argv" in conditions and not self._argv_condition_matches(conditions, context):
            return False
        regex = conditions.get("regex_token")
        argv = context.get("argv")
        if isinstance(regex, str):
            if not self._valid_regex(regex) or not isinstance(argv, list):
                return False
            regex_match = self._regex_matches_any_token(regex, argv)
            if regex_match is False:
                return False
            regex_unavailable = regex_match is None
        if any(key in conditions and context.get(key) != conditions[key] for key in _DIRECT_RULE_CONDITIONS):
            return False
        if not self._timeout_conditions_match(conditions, context):
            return False
        if regex_unavailable:
            # A bounded regex timeout must never turn a deny/ask rule into an
            # implicit allow. Auto-allow rules still require a positive match.
            return rule.effect in {CapabilityEffect.DENY, CapabilityEffect.ASK}
        return True

    def authority_rule_matches(self, rule: Any, context: dict[str, Any]) -> bool:
        return self._authority_rule_matches(rule, context)

    def _valid_regex(self, value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            encoded = value.encode("utf-8")
        except UnicodeEncodeError:
            return False
        if len(encoded) > self.config.capability.regex_pattern_max_bytes:
            return False
        try:
            bounded_regex.compile(value)
        except bounded_regex.error:
            return False
        return True

    def _regex_matches_any_token(
        self,
        pattern: str,
        argv: list[Any],
    ) -> bool | None:
        selected: list[str] = []
        for token in argv:
            if not isinstance(token, str):
                return False
            try:
                encoded = token.encode("utf-8")
            except UnicodeEncodeError:
                return False
            if len(encoded) > self.config.capability.regex_token_max_bytes:
                return False
            selected.append(token)

        deadline = time.monotonic() + self.config.capability.regex_match_timeout_s
        try:
            for token in selected:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                if bounded_regex.fullmatch(
                    pattern,
                    token,
                    timeout=remaining,
                ) is not None:
                    return True
        except (TimeoutError, bounded_regex.error):
            return None
        return False

    def _timeout_conditions_match(self, conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        if "timeout_s" in conditions:
            actual = self._finite_nonnegative_timeout(context.get("timeout_s"))
            expected = self._finite_nonnegative_timeout(conditions["timeout_s"])
            if actual is None or expected is None or actual != expected:
                return False
        if "timeout_max_s" in conditions:
            actual = self._finite_nonnegative_timeout(context.get("timeout_s"))
            maximum = self._finite_nonnegative_timeout(conditions["timeout_max_s"])
            if actual is None or maximum is None or actual > maximum:
                return False
        return True

    @staticmethod
    def _finite_nonnegative_timeout(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            selected = float(value)
        except (TypeError, ValueError):
            return None
        return selected if math.isfinite(selected) and selected >= 0 else None

    @staticmethod
    def _argv_condition_matches(conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        expected = conditions.get("argv")
        actual = context.get("argv")
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            return False
        if not isinstance(actual, list) or not all(isinstance(item, str) for item in actual):
            return False
        match = str(conditions.get("match", "exact"))
        return actual == expected if match == "exact" else (
            match == "prefix" and len(actual) >= len(expected) and actual[: len(expected)] == expected
        )

    def argv_condition_matches(self, conditions: dict[str, Any], context: dict[str, Any]) -> bool:
        return self._argv_condition_matches(conditions, context)

    @staticmethod
    def context_dict(context: OperationContext | Mapping[str, Any] | None) -> dict[str, Any]:
        if context is None:
            return {}
        if isinstance(context, OperationContext):
            return {
                "primitive": context.primitive,
                "operation": context.operation,
                **context.metadata,
            }
        return dict(context)

    @staticmethod
    def is_expired(capability: Capability, *, now: datetime | None = None) -> bool:
        if capability.expires_at is None:
            return False
        try:
            expires_at = CapabilityEvaluator.expires_at_datetime(capability.expires_at)
        except ValidationError:
            return True
        return expires_at <= (now or datetime.now(timezone.utc))

    @staticmethod
    def expires_at_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            selected = value
        elif isinstance(value, str):
            raw = value.strip()
            if not raw:
                raise ValidationError("capability expires_at must be a non-empty ISO timestamp")
            try:
                selected = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValidationError("capability expires_at must be an ISO timestamp") from exc
        else:
            raise ValidationError("capability expires_at must be an ISO timestamp")
        if selected.tzinfo is None:
            selected = selected.replace(tzinfo=timezone.utc)
        return selected.astimezone(timezone.utc)

    @staticmethod
    def sort_matching_capabilities(capabilities: Iterable[Capability]) -> list[Capability]:
        matches = list(capabilities)
        matches.sort(key=lambda cap: cap.cap_id)
        matches.sort(key=lambda cap: cap.issued_at, reverse=True)
        matches.sort(key=lambda cap: len(cap.resource), reverse=True)
        matches.sort(key=lambda cap: 0 if cap.effect == CapabilityEffect.DENY else 1)
        return matches
