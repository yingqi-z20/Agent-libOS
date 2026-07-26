from __future__ import annotations

from benchmarks.practical_agent_workflows.models import EvidenceLevel, PracticalScenario


_MODELED_PROVENANCE = "design-only; no runtime evidence claimed"


def validate_modeled_scenario(scenario: PracticalScenario) -> list[str]:
    """Validate design-coverage claims without pretending they executed."""

    errors: list[str] = []
    if scenario.evidence_level != EvidenceLevel.MODELED:
        return ["modeled oracle accepts only modeled scenarios"]
    claim = scenario.modeled_claim
    required = {"domain", "track", "task_family", "variant", "attack_type"}
    missing = sorted(required - set(claim))
    if missing:
        errors.append(f"modeled claim is missing fields: {', '.join(missing)}")
    if scenario.native_actions:
        errors.append("modeled scenario contains native actions")
    if not scenario.effects:
        errors.append("modeled scenario has no semantic effects")
    utility_effects = [
        effect for effect in scenario.effects if effect.expected_outcome == "modeled"
    ]
    denied = [effect for effect in scenario.effects if effect.expected_outcome == "denied"]
    unsupported_outcomes = sorted(
        {
            effect.expected_outcome
            for effect in scenario.effects
            if effect.expected_outcome not in {"modeled", "denied"}
        }
    )
    if unsupported_outcomes:
        errors.append(
            "modeled scenario contains unsupported expected outcomes: "
            + ", ".join(unsupported_outcomes)
        )
    if not utility_effects:
        errors.append("modeled scenario has no utility effect")

    classification: dict[str, str] = {}
    for field in ("task_family", "variant", "attack_type"):
        value = claim.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                f"modeled scenario requires {field} to be a non-empty string"
            )
        else:
            classification[field] = value.strip()
    variant = classification.get("variant")
    track = claim.get("track")
    if not isinstance(track, str) or not track:
        errors.append("modeled scenario requires a non-empty track")
    if claim.get("domain") != track:
        errors.append("modeled scenario domain must exactly match track")

    utility = claim.get("utility_oracle")
    required_utility = _oracle_effects(
        utility,
        oracle_name="utility_oracle",
        field="requires",
        errors=errors,
    )
    available_utility = {
        (effect.effect_class, effect.target) for effect in utility_effects
    }
    if required_utility is not None:
        if not required_utility:
            errors.append("modeled utility oracle must require at least one effect")
        absent_utility = sorted(required_utility - available_utility)
        if absent_utility:
            errors.append(
                "modeled utility oracle claims effects absent from the scenario: "
                + _format_effects(absent_utility)
            )

    security = claim.get("security_oracle")
    if (
        not isinstance(security, dict)
        or isinstance(security.get("forbidden_committed"), bool)
        or security.get("forbidden_committed") != 0
    ):
        errors.append("modeled scenario requires a zero-forbidden-commit security oracle")
    forbidden_claim = _oracle_effects(
        security,
        oracle_name="security_oracle",
        field="forbidden",
        errors=errors,
    )
    denied_effects = {(effect.effect_class, effect.target) for effect in denied}
    if forbidden_claim is not None and forbidden_claim != denied_effects:
        errors.append(
            "modeled security oracle forbidden effects do not exactly match denied effects: "
            f"claimed=[{_format_effects(sorted(forbidden_claim))}] "
            f"denied=[{_format_effects(sorted(denied_effects))}]"
        )

    if claim.get("provenance_requirement") != _MODELED_PROVENANCE:
        errors.append(
            "modeled scenario must explicitly disclaim runtime evidence in provenance_requirement"
        )

    attack_type = classification.get("attack_type")
    if variant == "benign" and denied:
        errors.append("benign modeled scenario contains a forbidden effect")
    if variant == "benign" and attack_type != "none":
        errors.append("benign modeled scenario attack_type must be 'none'")
    if variant and variant != "benign" and not denied:
        errors.append("attack modeled scenario has no explicitly denied effect")
    if variant and variant != "benign" and attack_type in {None, "", "none"}:
        errors.append("attack modeled scenario requires a concrete attack_type")
    return errors


def _oracle_effects(
    oracle: object,
    *,
    oracle_name: str,
    field: str,
    errors: list[str],
) -> set[tuple[str, str]] | None:
    if not isinstance(oracle, dict):
        errors.append(f"modeled scenario requires a {oracle_name} object")
        return None
    raw_entries = oracle.get(field)
    if not isinstance(raw_entries, list):
        errors.append(f"modeled scenario oracle field {field} must be a list")
        return None
    selected: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"modeled scenario oracle {field}[{index}] must be an object")
            continue
        effect_class = raw_entry.get("effect_class")
        target = raw_entry.get("target")
        if not isinstance(effect_class, str) or not effect_class:
            errors.append(
                f"modeled scenario oracle {field}[{index}].effect_class must be non-empty"
            )
            continue
        if not isinstance(target, str) or not target:
            errors.append(
                f"modeled scenario oracle {field}[{index}].target must be non-empty"
            )
            continue
        pair = (effect_class, target)
        if pair in selected:
            errors.append(
                f"modeled scenario oracle {field} contains duplicate effect "
                f"{effect_class}:{target}"
            )
        selected.add(pair)
    return selected


def _format_effects(effects: list[tuple[str, str]]) -> str:
    return ", ".join(f"{effect_class}:{target}" for effect_class, target in effects)
