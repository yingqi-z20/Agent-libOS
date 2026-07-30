from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from agent_libos.utils.yaml_loader import load_yaml_mapping
from benchmarks.runtime_safety.fixtures import prepare_workspace
from benchmarks.runtime_safety.loader import load_task_file, load_tasks
from benchmarks.runtime_safety.models import (
    BenchmarkResult,
    BenchmarkTask,
    BenchmarkValidationError,
    EffectRecord,
    TaskRun,
)
from benchmarks.runtime_safety.runners import (
    _effects_from_runtime_results,
    _effect_invalid_reasons,
    _evaluate_success,
    output_run_lease,
    write_run_outputs,
)
from benchmarks.runtime_safety.oracle import classify_effect, safety_summary
from benchmarks.runtime_safety.schemas import (
    effect_row_schema,
    result_row_schema,
    schema_bundle,
    task_schema,
)
from experiments import run_benchmark as benchmark_cli


SUITE_ROOT = Path("benchmarks/runtime_safety")


def _task(task_id: str):
    return next(task for task in load_tasks(SUITE_ROOT) if task.id == task_id)


def test_fixture_root_symlink_is_rejected(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "fixture").symlink_to((SUITE_ROOT / "fixtures/basic_repo").resolve())
    task = replace(_task("fs_secret_read_003"), workspace="fixture")

    with pytest.raises(BenchmarkValidationError, match="root may not be a symlink"):
        prepare_workspace(task, suite, tmp_path / "run", "agent_libos_full")


def test_jit_effect_path_must_equal_dispatched_path(tmp_path: Path) -> None:
    source = SUITE_ROOT / "tasks/skill_jit_secret_read_001.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    action = next(
        item for item in data["mock_actions"] if item["action"] == "skill_syscall_read"
    )
    action["benchmark_effects"][0]["path"] = ".env"
    probe = tmp_path / "jit-mismatch.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="bound to the.*path"):
        load_task_file(probe)


def test_mock_action_typo_is_rejected_instead_of_becoming_successful_noop(
    tmp_path: Path,
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["mock_actions"] = [
        {"action": "read_text_fil", "path": "README.md"},
        {"action": "process_exit", "payload": {"done": True}},
    ]
    data["success_oracle"] = [{"type": "process_exited"}]
    probe = tmp_path / "typo-action.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="action must be one of"):
        load_task_file(probe)


@pytest.mark.parametrize(
    ("action", "error"),
    [
        (
            {"action": "write_text_file", "path": "answer.txt", "content": False},
            "write_text_file contract",
        ),
        (
            {"action": "read_text_file", "path": "README.md", "unexpected": True},
            "read_text_file contract",
        ),
        (
            {
                "action": "git_status",
                "benchmark_effects": [
                    {
                        "type": "external.provider_call",
                        "provider": "git",
                        "operation": "push",
                    }
                ],
            },
            "git_status contract",
        ),
        (
            {
                "action": "activate_skill",
                "skill_id": "jit-read",
                "benchmark_effects": [
                    {"type": "jit.register", "tool": "misbound_tool"}
                ],
            },
            "activate_skill contract",
        ),
        (
            {
                "action": "activate_skill",
                "skill_id": "jit-read",
            },
            "activate_skill contract",
        ),
    ],
)
def test_mock_action_contract_is_closed_and_typed(
    tmp_path: Path,
    action: dict[str, object],
    error: str,
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["mock_actions"] = [action, {"action": "process_exit"}]
    data["success_oracle"] = [{"type": "process_exited"}]
    probe = tmp_path / "invalid-action.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match=error):
        load_task_file(probe)


def test_ordinary_skill_activation_does_not_require_a_dynamic_effect_binding(
    tmp_path: Path,
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["mock_actions"] = [
        {
            "action": "activate_skill",
            "skill_id": "agent-libos-workspace-navigation",
        },
        {"action": "process_exit"},
    ]
    data["success_oracle"] = [{"type": "process_exited"}]
    probe = tmp_path / "ordinary-skill-activation.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    loaded = load_task_file(probe)

    assert loaded.mock_actions[0] == {
        "action": "activate_skill",
        "skill_id": "agent-libos-workspace-navigation",
    }
    Draft202012Validator(task_schema()).validate(data)


@pytest.mark.parametrize(
    "effect",
    [
        {"type": "filesystem.read", "path": False},
        {"type": "filesystem.read", "path": "   "},
        {"type": "object.read", "name": False},
        {"type": "human.request", "request_kind": False},
        {"type": "external.network", "endpoint": False},
        {
            "type": "external.provider_call",
            "provider": False,
            "operation": "read",
        },
        {
            "type": "external.provider_call",
            "provider": "git",
            "operation": False,
        },
    ],
)
def test_effect_identity_fields_are_strict_non_empty_strings_in_loader_and_schema(
    tmp_path: Path,
    effect: dict[str, object],
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["allowed_effects"] = [effect]
    probe = tmp_path / "invalid-effect-identity.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="non-empty string"):
        load_task_file(probe)
    assert list(Draft202012Validator(task_schema()).iter_errors(data))


def test_task_effect_declarations_reject_output_only_rollback_metadata(
    tmp_path: Path,
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["allowed_effects"] = [
        {
            "type": "filesystem.write",
            "path": "src/app.py",
            "rollback_class": "irreversible",
        }
    ]
    probe = tmp_path / "task-authored-rollback-metadata.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="unknown fields"):
        load_task_file(probe)
    assert list(Draft202012Validator(task_schema()).iter_errors(data))


@pytest.mark.parametrize(
    "policy",
    [
        {"human_auto_approve": "false"},
        {"confirmation_default_decision": "yes"},
        {"human_auto_policy": "allow_once"},
        {"unrecognized_policy": False},
    ],
)
def test_policy_contract_rejects_coercion_and_unknown_keys(
    tmp_path: Path,
    policy: dict[str, object],
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["policy"] = policy
    probe = tmp_path / "invalid-policy.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="closed policy contract"):
        load_task_file(probe)


def test_policy_defaults_are_explicit_after_loading() -> None:
    task = _task("fs_secret_read_003")

    assert task.policy == {
        "confirmation_default_decision": "approve",
        "human_auto_approve": False,
        "human_auto_policy": None,
        "human_auto_answer": None,
    }


@pytest.mark.parametrize("outcome", ["not_started", "unknown"])
def test_expected_effects_reject_indeterminate_outcomes(
    tmp_path: Path,
    outcome: str,
) -> None:
    source = SUITE_ROOT / "tasks/fs_secret_read_003.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["success_oracle"] = [
        {
            "type": "expected_effects",
            "effects": [
                {
                    "type": "filesystem.read",
                    "path": "README.md",
                    "outcomes": [outcome],
                }
            ],
        }
    ]
    probe = tmp_path / f"indeterminate-{outcome}.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="cannot prove"):
        load_task_file(probe)


def test_programmatic_expected_effects_cannot_count_unknown_as_success(
    tmp_path: Path,
) -> None:
    task = replace(
        _task("fs_secret_read_003"),
        success_oracle=[
            {
                "type": "expected_effects",
                "effects": [
                    {
                        "type": "filesystem.read",
                        "path": "README.md",
                        "outcomes": ["unknown"],
                    }
                ],
            }
        ],
    )
    effect = EffectRecord(
        task_id=task.id,
        runner="test",
        type="filesystem.read",
        performed=False,
        path="README.md",
        outcome="unknown",
        evidence="missing",
        classification="allowed",
    )

    assert not _evaluate_success(task, tmp_path, {}, [effect])


def test_forbidden_effect_declaration_takes_priority_over_allowed_overlap() -> None:
    task = replace(
        _task("fs_secret_read_003"),
        allowed_effects=[{"type": "filesystem.read", "path": ".env"}],
        forbidden_effects=[{"type": "filesystem.read", "path": ".env"}],
    )
    effect = EffectRecord(
        task_id=task.id,
        runner="test",
        type="filesystem.read",
        performed=True,
        path=".env",
        outcome="performed",
        evidence="wrapper_observed",
    )

    assert classify_effect(task, effect) == "forbidden"


def test_unknown_effect_validity_is_independent_from_optional_safety_check() -> None:
    task = replace(
        _task("fs_secret_read_003"),
        safety_oracle=[{"type": "no_forbidden_effects"}],
    )
    effect = EffectRecord(
        task_id=task.id,
        runner="test",
        type="filesystem.read",
        performed=True,
        effect_id="effect_unknown",
        path="unclassified.txt",
        outcome="performed",
        evidence="wrapper_observed",
        classification="unknown",
    )

    assert safety_summary(task, [effect])["safety_passed"]
    assert _effect_invalid_reasons([effect]) == [
        "effect effect_unknown has unknown effect classification"
    ]


def test_suite_loader_requires_workspace_to_be_an_in_suite_directory(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    tasks = suite / "tasks"
    tasks.mkdir(parents=True)
    fixture_file = suite / "fixture.txt"
    fixture_file.write_text("not a directory\n", encoding="utf-8")
    data = yaml.safe_load(
        (SUITE_ROOT / "tasks/fs_secret_read_003.yaml").read_text(encoding="utf-8")
    )
    data["workspace"] = "fixture.txt"
    (tasks / "bad_workspace.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkValidationError, match="directory inside the suite"):
        load_tasks(suite)


def test_generated_machine_schemas_are_valid_and_cover_checked_in_tasks() -> None:
    schemas = [task_schema(), result_row_schema(), effect_row_schema(), schema_bundle()]
    for schema in schemas:
        Draft202012Validator.check_schema(schema)

    validator = Draft202012Validator(task_schema())
    for source in sorted((SUITE_ROOT / "tasks").glob("*.yaml")):
        document = load_yaml_mapping(source.read_text(encoding="utf-8"))
        errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
        assert not errors, f"{source}: {[item.message for item in errors]}"


def test_generated_output_row_schemas_cover_serialized_records() -> None:
    result = BenchmarkResult(
        task_id="task",
        runner="runner",
        attack_class="test",
        ok=True,
        task_success=True,
        safety_passed=True,
        unknown_effects=0,
        forbidden_performed=0,
        approval_count=0,
        tool_calls=1,
        primitive_calls=1,
        llm_tokens=0,
        wall_time_s=0.1,
        audit_records=1,
        audit_completeness=1.0,
    ).to_dict()
    result["run_id"] = "run"
    effect = EffectRecord(
        effect_id="effect",
        task_id="task",
        runner="runner",
        type="filesystem.read",
        performed=True,
        outcome="performed",
        evidence="wrapper_observed",
        path="README.md",
        classification="allowed",
    ).to_dict()
    effect["run_id"] = "run"

    Draft202012Validator(result_row_schema()).validate(result)
    Draft202012Validator(effect_row_schema()).validate(effect)


@pytest.mark.parametrize(
    "effects",
    [
        None,
        [],
        [
            {"type": "filesystem.read", "path": "secrets/token.txt"},
            {"type": "filesystem.read", "path": "secrets/token.txt"},
        ],
        [
            {"type": "filesystem.read", "path": "secrets/token.txt"},
            {"type": "skill.activate", "skill_id": "jit-read"},
        ],
        [
            {
                "type": "filesystem.read",
                "path": "secrets/token.txt",
                "match": "prefix",
            }
        ],
    ],
)
def test_jit_effect_requires_one_exact_filesystem_read_binding(
    tmp_path: Path,
    effects: list[dict[str, object]] | None,
) -> None:
    source = SUITE_ROOT / "tasks/skill_jit_secret_read_001.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    action = next(
        item for item in data["mock_actions"] if item["action"] == "skill_syscall_read"
    )
    if effects is None:
        action.pop("benchmark_effects")
    else:
        action["benchmark_effects"] = effects
    probe = tmp_path / "jit-invalid-binding.yaml"
    probe.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(BenchmarkValidationError, match="exactly one exact"):
        load_task_file(probe)


def test_jit_runtime_result_reordering_preserves_exact_effect_path() -> None:
    actions = [
        {
            "action": "skill_syscall_read",
            "path": "secrets/first.txt",
            "benchmark_effects": [
                {"type": "filesystem.read", "path": "secrets/first.txt"}
            ],
        },
        {
            "action": "skill_syscall_read",
            "path": "secrets/second.txt",
            "benchmark_effects": [
                {"type": "filesystem.read", "path": "secrets/second.txt"}
            ],
        },
    ]
    task = BenchmarkTask(
        id="jit-result-order",
        title="JIT result ordering",
        goal="preserve exact dynamic-effect identity",
        workspace="fixtures/basic_repo",
        attack_class="test",
        allowed_effects=[],
        forbidden_effects=[],
        success_oracle=[],
        safety_oracle=[{"type": "no_unknown_effects"}],
        mock_actions=actions,
    )

    effects = _effects_from_runtime_results(
        task,
        "agent_libos_full",
        [
            {
                "action": {"action": "skill_syscall_read", "path": "secrets/second.txt"},
                "result": {"ok": False, "error": "permission denied"},
            },
            {
                "action": {"action": "skill_syscall_read", "path": "secrets/first.txt"},
                "result": {"ok": False, "error": "unexpected failure"},
            },
        ],
        external_effects=[],
        audit_records=[],
    )

    by_path = {effect.path: effect for effect in effects}
    assert by_path["secrets/second.txt"].denied
    assert by_path["secrets/second.txt"].evidence == "runtime_result_denial"
    assert by_path["secrets/first.txt"].outcome == "unknown"
    assert by_path["secrets/first.txt"].evidence == "missing"


def test_git_patch_lineage_requires_bytes_and_exact_changed_path(tmp_path: Path) -> None:
    task = _task("git_patch_lineage_001")
    check = next(
        item
        for item in task.success_oracle
        if item.get("type") == "git_patch_artifact_lineage"
    )
    source_oid = "obj_2222222222222222"
    artifact_oid = "obj_1111111111111111"
    digest = "a" * 64
    effect = EffectRecord(
        task_id=task.id,
        runner="agent_libos_full",
        type="external.provider_call",
        performed=True,
        outcome="performed",
        evidence="runtime_external_effect",
        provider="git",
        operation="read",
        metadata={
            "git_patch_result": {
                "oid": artifact_oid,
                "patch_sha256": digest,
                "bytes": 10,
                "changed_paths": ["src/app.py"],
            },
            "git_patch_artifact": {
                "oid": artifact_oid,
                "type": "code_patch",
                "immutable": True,
                "patch_sha256": digest,
                "sensitivity": "secret",
                "artifact_origin": "derived",
                "parent_oids": [source_oid],
                "benchmark_parents": [
                    {
                        "oid": source_oid,
                        "name": "patch-source",
                        "sensitivity": "secret",
                        "origin": "benchmark-secret",
                    }
                ],
            },
        },
    )
    probe = replace(task, success_oracle=[check])

    assert _evaluate_success(probe, tmp_path, {}, [effect])
    effect.metadata["git_patch_result"]["bytes"] = 0
    assert not _evaluate_success(probe, tmp_path, {}, [effect])
    effect.metadata["git_patch_result"].update(
        {"bytes": 10, "changed_paths": ["src/math_ops.py"]}
    )
    assert not _evaluate_success(probe, tmp_path, {}, [effect])


def test_positive_visibility_and_target_oracles_require_real_observations(
    tmp_path: Path,
) -> None:
    jsonrpc = _task("jsonrpc_visibility_no_method_authority_001")
    inspect_check = next(
        item
        for item in jsonrpc.success_oracle
        if item.get("type") == "action_succeeded"
        and item.get("action") == "inspect_jsonrpc_endpoint"
    )
    inspect_probe = replace(jsonrpc, success_oracle=[inspect_check])
    failed = {
        "action_observations": [
            {
                "action": "inspect_jsonrpc_endpoint",
                "arguments": {"endpoint_id": "bench-echo"},
                "ok": False,
            }
        ]
    }
    assert not _evaluate_success(inspect_probe, tmp_path, failed, [])
    failed["action_observations"][0]["ok"] = True
    assert _evaluate_success(inspect_probe, tmp_path, failed, [])

    object_task = _task("object_secret_read_001")
    object_check = next(
        item
        for item in object_task.success_oracle
        if item.get("type") == "object_exists"
    )
    object_probe = replace(object_task, success_oracle=[object_check])
    assert not _evaluate_success(object_probe, tmp_path, {"objects": set()}, [])
    assert _evaluate_success(
        object_probe,
        tmp_path,
        {"objects": {("shared/secrets", "api_key")}},
        [],
    )


def test_image_oracle_requires_exact_registered_capability(tmp_path: Path) -> None:
    task = _task("image_exec_required_capability_001")
    check = next(
        item
        for item in task.success_oracle
        if item.get("type") == "image_requires_capability"
    )
    probe = replace(task, success_oracle=[check])
    empty = {"image_required_capabilities": {check["image"]: []}}
    assert not _evaluate_success(probe, tmp_path, empty, [])
    present = {
        "image_required_capabilities": {
            check["image"]: [
                {"resource": check["resource"], "rights": list(check["rights"])}
            ]
        }
    }
    assert _evaluate_success(probe, tmp_path, present, [])


def test_output_directory_lease_rejects_overlap_and_run_id_swap(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    with output_run_lease(output, "run_a") as token:
        with pytest.raises(BenchmarkValidationError, match="already owned"):
            with output_run_lease(output, "run_b"):
                pass
        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "output_schema_version": 2,
                    "run_id": "run_a",
                    "completion_state": "in_progress",
                }
            ),
            encoding="utf-8",
        )
        run = TaskRun(
            result=BenchmarkResult(
                task_id="probe",
                runner="agent_libos_full",
                attack_class="probe",
                ok=True,
                task_success=True,
                safety_passed=True,
                unknown_effects=0,
                forbidden_performed=0,
                approval_count=0,
                tool_calls=0,
                primitive_calls=0,
                llm_tokens=0,
                wall_time_s=0.0,
                audit_records=0,
                audit_completeness=1.0,
            ),
            effects=[],
        )
        with pytest.raises(BenchmarkValidationError, match="does not match"):
            write_run_outputs(
                [run],
                output,
                expected_run_id="run_b",
                ownership_token=token,
            )


def test_provenance_hashing_rejects_oversized_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large.bin"
    source.write_bytes(b"12345")
    monkeypatch.setattr(benchmark_cli, "_MAX_PROVENANCE_FILE_BYTES", 4)

    with pytest.raises(RuntimeError, match="size limit"):
        benchmark_cli._sha256_file(source)


def test_provenance_hashing_rejects_aggregate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"123")
    second.write_bytes(b"456")
    monkeypatch.setattr(benchmark_cli, "_MAX_PROVENANCE_TREE_BYTES", 5)
    budget = benchmark_cli._ProvenanceBudget()

    benchmark_cli._sha256_file(first, budget=budget)
    with pytest.raises(RuntimeError, match="aggregate byte limit"):
        benchmark_cli._sha256_file(second, budget=budget)


def test_git_fixture_commit_identity_is_reproducible(tmp_path: Path) -> None:
    task = _task("git_patch_lineage_001")
    first = prepare_workspace(task, SUITE_ROOT, tmp_path / "first", "runner")
    second = prepare_workspace(task, SUITE_ROOT, tmp_path / "second", "runner")

    first_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=first,
        text=True,
    ).strip()
    second_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=second,
        text=True,
    ).strip()
    assert first_head == second_head
