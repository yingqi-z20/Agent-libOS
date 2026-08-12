from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ObjectType
from agent_libos.models.exceptions import NotFound


PACKAGE_ROOT = Path("skills/task-plan")
JIT_MANIFEST = PACKAGE_ROOT / "references/agent-libos/jit-tools.json"
TOOL_NAMES = {
    "create_task_plan",
    "read_task_plan",
    "update_task_plan",
}
STATUSES = {
    "pending",
    "in_progress",
    "blocked",
    "completed",
    "cancelled",
}


def _specs() -> dict[str, dict[str, Any]]:
    return {
        str(item["name"]): item
        for item in json.loads(JIT_MANIFEST.read_text(encoding="utf-8"))
    }


def _register_and_activate(runtime: Runtime, *pids: str) -> dict[str, Any]:
    registered = runtime.register_skill_from_path(
        PACKAGE_ROOT,
        actor="cli",
        source_type="workspace",
    )
    assert registered["skill_id"] == "task-plan"
    loaded: dict[str, Any] = {}
    for pid in pids:
        runtime.capability.grant(
            pid,
            "skill:task-plan",
            [CapabilityRight.EXECUTE],
            issued_by="test",
        )
        loaded = runtime.skills.activate_skill(pid, "task-plan", actor=pid)
        assert set(loaded["jit_tool_ids"]) == TOOL_NAMES
    return loaded


def _namespace_capabilities(runtime: Runtime, pid: str) -> set[tuple[str, tuple[str, ...]]]:
    return {
        (
            capability.resource,
            tuple(
                sorted(
                    str(getattr(right, "value", right))
                    for right in capability.rights
                )
            ),
        )
        for capability in runtime.capability.list_subject(pid)
        if capability.resource.startswith("object_namespace:")
    }


def _call(runtime: Runtime, pid: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = runtime.tools.call(pid, name, args)
    assert result.ok, result.error
    assert isinstance(result.payload, dict)
    return result.payload


class TestTaskPlanSkill:
    def test_package_declares_closed_jit_contracts_and_behavioral_tests(self) -> None:
        runtime = Runtime.open("local")
        try:
            validation = runtime.skills.validate_package_path(PACKAGE_ROOT)
        finally:
            runtime.close()

        assert validation["valid"] is True
        assert validation["skill_id"] == "task-plan"
        assert validation["allowed_tools"] == []
        assert set(validation["jit_tools"]) == TOOL_NAMES
        assert set(validation["resources"]) == {
            "SKILL.md",
            "references/agent-libos/jit-tools.json",
            "scripts/create_task_plan.ts",
            "scripts/read_task_plan.ts",
            "scripts/update_task_plan.ts",
        }

        skill_md = PACKAGE_ROOT.joinpath("SKILL.md").read_text(encoding="utf-8")
        skills_doc = Path("docs/skills.md").read_text(encoding="utf-8")
        openai_yaml = PACKAGE_ROOT.joinpath("agents/openai.yaml").read_text(
            encoding="utf-8"
        )
        for tool_name in TOOL_NAMES:
            assert tool_name in skill_md
        assert "run_jit_tool" in skill_md
        assert "not an atomic compare-and-swap" in skill_md
        assert "Do not issue parallel `update_task_plan`" in skill_md
        assert "## Task Plan Skill" in skills_doc
        assert "caller-supplied compare-and-swap token" in skills_doc
        assert 'display_name: "Task Plan"' in openai_yaml
        assert "$task-plan" in openai_yaml

        specs = _specs()
        assert set(specs) == TOOL_NAMES
        for name, spec in specs.items():
            input_schema = spec["input_schema"]
            output_schema = spec["output_schema"]
            namespace_description = input_schema["properties"]["namespace"][
                "description"
            ]
            assert "JSON null" in namespace_description
            assert "process-default namespace" in namespace_description
            assert input_schema["additionalProperties"] is False
            assert output_schema["additionalProperties"] is False
            assert set(input_schema["required"]) == set(input_schema["properties"])
            assert set(output_schema["required"]) == set(output_schema["properties"])
            assert spec["tests"], f"{name} must ship behavioral tests"

            input_validator = Draft202012Validator(input_schema)
            output_validator = Draft202012Validator(output_schema)
            for case in spec["tests"]:
                input_validator.validate(case["args"])
                output_validator.validate(case["expected"])
                with pytest.raises(JsonSchemaValidationError):
                    input_validator.validate(
                        {**case["args"], "unexpected": True}
                    )
                with pytest.raises(JsonSchemaValidationError):
                    output_validator.validate(
                        {**case["expected"], "unexpected": True}
                    )

        for name in ("create_task_plan", "update_task_plan"):
            plan_schema = specs[name]["input_schema"]["properties"]["plan"]
            assert set(plan_schema["items"]["properties"]["status"]["enum"]) == STATUSES
            assert plan_schema["maxContains"] == 1
            validator = Draft202012Validator(specs[name]["input_schema"])
            valid = dict(specs[name]["tests"][0]["args"])
            with pytest.raises(JsonSchemaValidationError):
                validator.validate(
                    {
                        **valid,
                        "plan": [
                            {"step": "first", "status": "in_progress"},
                            {"step": "second", "status": "in_progress"},
                        ],
                    }
                )
            with pytest.raises(JsonSchemaValidationError):
                validator.validate(
                    {
                        **valid,
                        "plan": [{"step": "invalid", "status": "paused"}],
                    }
                )

    @pytest.mark.real_deno
    def test_lifecycle_flexible_revisions_and_retry_safety(self) -> None:
        runtime = Runtime.open("local")
        try:
            pid = runtime.process.spawn(
                image="base-agent:v0",
                goal="maintain a revisioned implementation plan",
            )
            _register_and_activate(runtime, pid)

            initial_plan = [
                {"step": "Inspect", "status": "in_progress"},
                {"step": "Implement", "status": "pending"},
                {"step": "Wait for access", "status": "blocked"},
                {"step": "Verify", "status": "pending"},
                {"step": "Remove obsolete work", "status": "cancelled"},
            ]
            create_args = {
                "name": "implementation-plan",
                "namespace": None,
                "explanation": "Initial plan.",
                "plan": initial_plan,
            }
            created = _call(
                runtime,
                pid,
                "create_task_plan",
                create_args,
            )

            assert created["created"] is True
            assert created["revision"] == 1
            assert created["plan"] == initial_plan
            assert created["status_counts"] == {
                "pending": 2,
                "in_progress": 1,
                "blocked": 1,
                "completed": 0,
                "cancelled": 1,
            }
            stored = runtime.memory.get_object_by_name(
                pid,
                "implementation-plan",
            )
            assert stored.type == ObjectType.PLAN
            assert stored.immutable is False
            assert stored.payload == {
                "schema_version": "task-plan/v1",
                "entries": [
                    {
                        "revision": 1,
                        "explanation": "Initial plan.",
                        "plan": initial_plan,
                    }
                ],
            }

            read = _call(
                runtime,
                pid,
                "read_task_plan",
                {"name": "implementation-plan", "namespace": None},
            )
            assert read["memory_version"] == stored.version
            assert read["revision"] == 1
            assert read["plan"] == initial_plan

            second_plan = [
                {"step": "Inspect", "status": "completed"},
                {"step": "Implement", "status": "in_progress"},
                {"step": "Wait for access", "status": "blocked"},
                {"step": "Verify", "status": "pending"},
                {"step": "Remove obsolete work", "status": "cancelled"},
            ]
            update_args = {
                "name": "implementation-plan",
                "namespace": None,
                "expected_revision": 1,
                "explanation": "Inspection finished.",
                "plan": second_plan,
            }
            updated = _call(
                runtime,
                pid,
                "update_task_plan",
                update_args,
            )
            assert updated["changed"] is True
            assert updated["revision"] == 2
            after_update = runtime.memory.get_object_by_name(
                pid,
                "implementation-plan",
            )
            assert updated["memory_version"] == after_update.version
            assert len(after_update.payload["entries"]) == 2

            ordinary_noop = _call(
                runtime,
                pid,
                "update_task_plan",
                {**update_args, "expected_revision": 2},
            )
            retry_noop = _call(
                runtime,
                pid,
                "update_task_plan",
                update_args,
            )
            assert ordinary_noop["changed"] is False
            assert retry_noop["changed"] is False
            assert ordinary_noop["revision"] == retry_noop["revision"] == 2
            after_noops = runtime.memory.get_object_by_name(
                pid,
                "implementation-plan",
            )
            assert after_noops.version == after_update.version
            assert len(after_noops.payload["entries"]) == 2

            stale = runtime.tools.call(
                pid,
                "update_task_plan",
                {
                    **update_args,
                    "explanation": "A conflicting stale update.",
                },
            )
            assert not stale.ok
            after_stale = runtime.memory.get_object_by_name(
                pid,
                "implementation-plan",
            )
            assert after_stale.version == after_update.version
            assert len(after_stale.payload["entries"]) == 2

            reopened_plan = [
                {"step": "Inspect", "status": "pending"},
                {"step": "Implement", "status": "completed"},
                {"step": "Wait for access", "status": "pending"},
                {"step": "Verify", "status": "in_progress"},
                {"step": "Remove obsolete work", "status": "pending"},
            ]
            reopened = _call(
                runtime,
                pid,
                "update_task_plan",
                {
                    "name": "implementation-plan",
                    "namespace": None,
                    "expected_revision": 2,
                    "explanation": "Reopen and reorder work as reality changes.",
                    "plan": reopened_plan,
                },
            )
            assert reopened["changed"] is True
            assert reopened["revision"] == 3
            assert reopened["status_counts"] == {
                "pending": 3,
                "in_progress": 1,
                "blocked": 0,
                "completed": 1,
                "cancelled": 0,
            }

            duplicate = runtime.tools.call(
                pid,
                "create_task_plan",
                create_args,
            )
            assert not duplicate.ok
            assert (
                runtime.memory.get_object_by_name(
                    pid,
                    "implementation-plan",
                ).payload["entries"][-1]["revision"]
                == 3
            )

            two_active = runtime.tools.call(
                pid,
                "create_task_plan",
                {
                    "name": "invalid-active-plan",
                    "namespace": None,
                    "explanation": None,
                    "plan": [
                        {"step": "one", "status": "in_progress"},
                        {"step": "two", "status": "in_progress"},
                    ],
                },
            )
            whitespace = runtime.tools.call(
                pid,
                "create_task_plan",
                {
                    "name": "invalid-whitespace-plan",
                    "namespace": None,
                    "explanation": None,
                    "plan": [{"step": "   ", "status": "pending"}],
                },
            )
            assert not two_active.ok
            assert not whitespace.ok
            with pytest.raises(NotFound):
                runtime.memory.get_object_by_name(pid, "invalid-active-plan")
            with pytest.raises(NotFound):
                runtime.memory.get_object_by_name(
                    pid,
                    "invalid-whitespace-plan",
                )
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_malformed_ledgers_and_foreign_namespaces_fail_closed(self) -> None:
        runtime = Runtime.open("local")
        try:
            owner = runtime.process.spawn(
                image="base-agent:v0",
                goal="own a private task plan",
            )
            outsider = runtime.process.spawn(
                image="base-agent:v0",
                goal="attempt an unauthorized plan read",
            )
            runtime.register_skill_from_path(
                PACKAGE_ROOT,
                actor="cli",
                source_type="workspace",
            )
            for pid in (owner, outsider):
                runtime.capability.grant(
                    pid,
                    "skill:task-plan",
                    [CapabilityRight.EXECUTE],
                    issued_by="test",
                )
                before_namespaces = _namespace_capabilities(runtime, pid)
                loaded = runtime.skills.activate_skill(
                    pid,
                    "task-plan",
                    actor=pid,
                )
                assert set(loaded["jit_tool_ids"]) == TOOL_NAMES
                assert _namespace_capabilities(runtime, pid) == before_namespaces

            created = _call(
                runtime,
                owner,
                "create_task_plan",
                {
                    "name": "private-plan",
                    "namespace": None,
                    "explanation": None,
                    "plan": [{"step": "Keep private", "status": "in_progress"}],
                },
            )
            denied = runtime.tools.call(
                outsider,
                "read_task_plan",
                {
                    "name": "private-plan",
                    "namespace": created["namespace"],
                },
            )
            assert not denied.ok
            owner_read = _call(
                runtime,
                owner,
                "read_task_plan",
                {"name": "private-plan", "namespace": None},
            )
            assert owner_read["revision"] == 1

            runtime.memory.create_object(
                owner,
                ObjectType.PLAN,
                {
                    "schema_version": "task-plan/v1",
                    "entries": [
                        {
                            "revision": 2,
                            "explanation": None,
                            "plan": [
                                {"step": "Malformed", "status": "pending"}
                            ],
                        }
                    ],
                },
                name="malformed-plan",
                immutable=False,
            )
            malformed = runtime.tools.call(
                owner,
                "read_task_plan",
                {"name": "malformed-plan", "namespace": None},
            )
            assert not malformed.ok
        finally:
            runtime.close()
