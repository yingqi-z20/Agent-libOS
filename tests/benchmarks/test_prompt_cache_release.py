from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_libos.llm.prompt_cache_gate import (
    PromptCachePricing,
    evaluate_prompt_cache_release_gate,
)
from benchmarks.prompt_cache_evidence import aggregate_prompt_cache_run_evidence
from benchmarks.prompt_cache_release import (
    ProviderPromptCacheArmInput,
    build_prompt_cache_arm_report,
)
from scripts import build_prompt_cache_arm_report as arm_cli


def test_build_prompt_cache_arm_aggregates_providers_and_known_cost() -> None:
    report = build_prompt_cache_arm_report(
        [
            ProviderPromptCacheArmInput(
                provider_id="custom",
                model_id="custom-model",
                repetitions=3,
                report=_provider_report(),
            ),
            ProviderPromptCacheArmInput(
                provider_id="openai",
                model_id="gpt-test",
                repetitions=3,
                report=_provider_report(),
                pricing=_pricing(),
            ),
        ],
        security_invariants_passed=True,
    )

    assert report["prompt_layout"] == "cache_optimized_v2"
    assert report["metrics"]["runs"] == 12
    assert report["metrics"]["successful_runs"] == 12
    assert report["metrics"]["total_input_tokens"] == 2_000
    assert report["release_gates"] == {
        "all_oracles_passed": True,
        "completion_evidence_passed": True,
        "security_invariants_passed": True,
        "workflow_count": 12,
    }
    assert report["pricing_known"] is False
    assert "cost" not in report
    official = report["providers"][1]
    assert official["pricing_known"] is True
    assert official["cost"]["net_cost"] > 0


def test_build_prompt_cache_arm_fails_closed_on_unknown_write_cost() -> None:
    provider_report = _provider_report()
    provider_report["metrics"]["cache_write_tokens"] = None

    with pytest.raises(ValueError, match="cache_write_tokens must be reported"):
        build_prompt_cache_arm_report(
            [
                ProviderPromptCacheArmInput(
                    provider_id="openai",
                    model_id="gpt-test",
                    repetitions=3,
                    report=provider_report,
                    pricing=_pricing(),
                )
            ],
            security_invariants_passed=True,
        )


def test_build_prompt_cache_arm_accepts_complete_zero_leak_measurement() -> None:
    report = build_prompt_cache_arm_report(
        [
            ProviderPromptCacheArmInput(
                provider_id="custom",
                model_id="custom-model",
                repetitions=3,
                report=_provider_report(),
            )
        ],
        security_invariants_passed=True,
    )

    assert report["metrics"]["forbidden_internal_id_leaks"] == 0
    assert report["metrics"]["forbidden_internal_id_leak_call_count"] == 0
    assert (
        report["metrics"]["forbidden_internal_id_leak_evidence_complete"]
        is True
    )
    assert report["providers"][0]["forbidden_internal_id_leaks"] == 0


def test_missing_raw_run_leak_evidence_stays_unknown_and_arm_rejects() -> None:
    aggregated = aggregate_prompt_cache_run_evidence([{}])

    assert aggregated["forbidden_internal_id_leak_evidence_complete"] is False
    assert aggregated["forbidden_internal_id_leaks"] is None
    assert aggregated["forbidden_internal_id_leaks_by_category"] is None
    assert aggregated["forbidden_internal_id_leak_call_count"] is None

    provider_report = _provider_report()
    metrics = provider_report["metrics"]
    assert isinstance(metrics, dict)
    metrics.update(aggregated)
    with pytest.raises(
        ValueError,
        match="forbidden_internal_id_leak_evidence_complete must be true",
    ):
        build_prompt_cache_arm_report(
            [
                ProviderPromptCacheArmInput(
                    provider_id="custom",
                    model_id="custom-model",
                    repetitions=3,
                    report=provider_report,
                )
            ],
            security_invariants_passed=True,
        )


def test_build_prompt_cache_arm_accepts_legacy_redacted_leak_details() -> None:
    provider_report = _provider_report()
    metrics = provider_report["metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("forbidden_internal_id_leak_call_count")
    metrics["forbidden_internal_id_leak_calls"] = []

    report = build_prompt_cache_arm_report(
        [
            ProviderPromptCacheArmInput(
                provider_id="custom",
                model_id="custom-model",
                repetitions=3,
                report=provider_report,
            )
        ],
        security_invariants_passed=True,
    )

    assert report["metrics"]["forbidden_internal_id_leak_call_count"] == 0


@pytest.mark.parametrize(
    ("removed", "message"),
    [
        (
            (
                "forbidden_internal_id_leaks",
                "forbidden_internal_id_leaks_by_category",
                "forbidden_internal_id_leak_call_count",
            ),
            "forbidden_internal_id_leaks",
        ),
        (("forbidden_internal_id_leaks",), "forbidden_internal_id_leaks"),
        (
            ("forbidden_internal_id_leaks_by_category",),
            "forbidden_internal_id_leaks_by_category",
        ),
        (
            ("forbidden_internal_id_leak_call_count",),
            "forbidden_internal_id_leak_call_count",
        ),
    ],
)
def test_build_prompt_cache_arm_rejects_missing_leak_measurement(
    removed: tuple[str, ...],
    message: str,
) -> None:
    provider_report = _provider_report()
    metrics = provider_report["metrics"]
    assert isinstance(metrics, dict)
    for key in removed:
        metrics.pop(key)

    with pytest.raises(ValueError, match=message):
        build_prompt_cache_arm_report(
            [
                ProviderPromptCacheArmInput(
                    provider_id="custom",
                    model_id="custom-model",
                    repetitions=3,
                    report=provider_report,
                )
            ],
            security_invariants_passed=True,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("total_type", "non-negative integer"),
        ("missing_category", "closed category set"),
        ("unknown_category", "closed category set"),
        ("category_total", "equal the category total"),
        ("call_count_type", "non-negative integer"),
        ("call_count", "reconcile with the leak total"),
    ],
)
def test_build_prompt_cache_arm_rejects_inconsistent_leak_measurement(
    mutation: str,
    message: str,
) -> None:
    provider_report = _provider_report()
    metrics = provider_report["metrics"]
    assert isinstance(metrics, dict)
    categories = metrics["forbidden_internal_id_leaks_by_category"]
    assert isinstance(categories, dict)
    if mutation == "total_type":
        metrics["forbidden_internal_id_leaks"] = "0"
    elif mutation == "missing_category":
        categories.pop("terminal_host_identifiers")
    elif mutation == "unknown_category":
        categories["unknown"] = 0
    elif mutation == "category_total":
        categories["host_contract_fields"] = 1
    elif mutation == "call_count_type":
        metrics["forbidden_internal_id_leak_call_count"] = False
    else:
        metrics["forbidden_internal_id_leak_call_count"] = 1

    with pytest.raises(ValueError, match=message):
        build_prompt_cache_arm_report(
            [
                ProviderPromptCacheArmInput(
                    provider_id="custom",
                    model_id="custom-model",
                    repetitions=3,
                    report=provider_report,
                )
            ],
            security_invariants_passed=True,
        )


@pytest.mark.parametrize("release_gate", [None, {}, {"passed": "true"}])
def test_build_prompt_cache_arm_fails_closed_without_valid_provider_gate(
    release_gate: object,
) -> None:
    provider_report = _provider_report()
    if release_gate is None:
        provider_report.pop("release_gate")
    else:
        provider_report["release_gate"] = release_gate

    report = build_prompt_cache_arm_report(
        [
            ProviderPromptCacheArmInput(
                provider_id="custom",
                model_id="custom-model",
                repetitions=3,
                report=provider_report,
            )
        ],
        security_invariants_passed=True,
    )

    assert report["providers"][0]["all_oracles_passed"] is False
    assert report["release_gates"]["all_oracles_passed"] is False


def test_prompt_cache_arm_cli_reads_relative_reports(tmp_path: Path) -> None:
    provider_report = tmp_path / "provider.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    provider_report.write_text(json.dumps(_provider_report()), encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "security_invariants_passed": True,
                "providers": [
                    {
                        "provider_id": "custom",
                        "model_id": "custom-model",
                        "repetitions": 3,
                        "report": provider_report.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert arm_cli.main(["--manifest", str(manifest), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["providers"][0]["provider_id"] == "custom"
    assert payload["metrics"]["runs"] == 6


@pytest.mark.parametrize("security_value", [None, "true", 1])
def test_prompt_cache_arm_cli_requires_explicit_boolean_security_evidence(
    tmp_path: Path,
    security_value: object,
) -> None:
    provider_report = tmp_path / "provider.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    provider_report.write_text(json.dumps(_provider_report()), encoding="utf-8")
    payload: dict[str, object] = {
        "providers": [
            {
                "provider_id": "custom",
                "model_id": "custom-model",
                "repetitions": 3,
                "report": provider_report.name,
            }
        ]
    }
    if security_value is not None:
        payload["security_invariants_passed"] = security_value
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="manifest.security_invariants_passed must be an explicit boolean",
    ):
        arm_cli.main(["--manifest", str(manifest), "--output", str(output)])

    assert not output.exists()


@pytest.mark.parametrize(
    "manifest_text",
    [
        '{"security_invariants_passed":false,'
        '"security_invariants_passed":true,"providers":[]}',
        '{"security_invariants_passed":true,"providers":[],'
        '"measurement":NaN}',
    ],
)
def test_prompt_cache_arm_cli_rejects_ambiguous_or_nonfinite_manifest_json(
    tmp_path: Path,
    manifest_text: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    manifest.write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError):
        arm_cli.main(["--manifest", str(manifest), "--output", str(output)])

    assert not output.exists()


def test_prompt_cache_arm_cli_rejects_oversized_manifest_without_output(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    manifest.write_bytes(b" " * (arm_cli._MAX_MANIFEST_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        arm_cli.main(["--manifest", str(manifest), "--output", str(output)])

    assert not output.exists()


def test_prompt_cache_arm_cli_rejects_duplicate_provider_report_keys(
    tmp_path: Path,
) -> None:
    provider_report = tmp_path / "provider.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    encoded_report = json.dumps(_provider_report(), separators=(",", ":"))
    provider_report.write_text(
        '{"prompt_layout":"legacy_v1",' + encoded_report[1:],
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps(
            {
                "security_invariants_passed": True,
                "providers": [
                    {
                        "provider_id": "custom",
                        "model_id": "custom-model",
                        "repetitions": 3,
                        "report": provider_report.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        arm_cli.main(["--manifest", str(manifest), "--output", str(output)])

    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("provider_id", 7, "provider_id must be a non-empty string"),
        ("model_id", False, "model_id must be a non-empty string"),
        ("pricing", [], "provider pricing must be an object when present"),
    ],
)
def test_prompt_cache_arm_cli_rejects_wrong_typed_provider_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    provider_report = tmp_path / "provider.json"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "arm.json"
    provider_report.write_text(json.dumps(_provider_report()), encoding="utf-8")
    provider: dict[str, object] = {
        "provider_id": "custom",
        "model_id": "custom-model",
        "repetitions": 3,
        "report": provider_report.name,
    }
    provider[field] = value
    manifest.write_text(
        json.dumps(
            {
                "security_invariants_passed": True,
                "providers": [provider],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        arm_cli.main(["--manifest", str(manifest), "--output", str(output)])

    assert not output.exists()


def test_built_multi_provider_arms_pass_the_strict_paired_gate() -> None:
    legacy = _arm(
        layout="legacy_v1",
        total_input=1_000,
        cache_read=500,
        uncached=500,
    )
    candidate = _arm(
        layout="cache_optimized_v2",
        total_input=850,
        cache_read=475,
        uncached=375,
    )

    result = evaluate_prompt_cache_release_gate(legacy, candidate)

    assert result["passed"] is True
    assert all(result["checks"].values())


def _arm(
    *,
    layout: str,
    total_input: int,
    cache_read: int,
    uncached: int,
) -> dict[str, object]:
    inputs = [
        ProviderPromptCacheArmInput(
            provider_id="custom",
            model_id="custom-model",
            repetitions=3,
            report=_provider_report(
                layout=layout,
                total_input=total_input,
                cache_read=cache_read,
                uncached=uncached,
            ),
        ),
        ProviderPromptCacheArmInput(
            provider_id="openai",
            model_id="gpt-test",
            repetitions=3,
            report=_provider_report(
                layout=layout,
                total_input=total_input,
                cache_read=cache_read,
                uncached=uncached,
            ),
            pricing=_pricing(),
        ),
    ]
    return build_prompt_cache_arm_report(
        inputs,
        security_invariants_passed=True,
    )


def _provider_report(
    *,
    layout: str = "cache_optimized_v2",
    total_input: int = 1_000,
    cache_read: int = 600,
    uncached: int = 400,
) -> dict[str, object]:
    return {
        "prompt_layout": layout,
        "release_gate": {"passed": True},
        "metrics": {
            "runs": 6,
            "successful_runs": 6,
            "completion_evidence_successful_runs": 6,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": 100,
            "cache_total_calls": 1,
            "cache_reported_calls": 1,
            "cache_read_reported_calls": 1,
            "cache_write_reported_calls": 1,
            "cache_metric_reported_calls": 1,
            "cache_metric_input_tokens": total_input,
            "uncached_input_tokens": uncached,
            "cache_hit_rate": cache_read / total_input,
            "total_input_tokens": total_input,
            "total_output_tokens": 50,
            "forbidden_internal_id_leak_evidence_complete": True,
            "forbidden_internal_id_leaks": 0,
            "forbidden_internal_id_leaks_by_category": {
                "host_contract_fields": 0,
                "materialization_fields": 0,
                "completion_binding_fields": 0,
                "current_process_ids": 0,
                "terminal_host_identifiers": 0,
            },
            "forbidden_internal_id_leak_call_count": 0,
        },
    }


def _pricing() -> PromptCachePricing:
    return PromptCachePricing(
        input_per_million=5.0,
        cached_input_per_million=0.5,
        cache_write_input_per_million=6.25,
        output_per_million=30.0,
    )
