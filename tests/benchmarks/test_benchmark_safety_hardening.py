from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

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
    _evaluate_success,
    output_run_lease,
    write_run_outputs,
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
