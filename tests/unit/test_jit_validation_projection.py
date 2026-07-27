from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models import ValidationResult
from agent_libos.tools.base import ToolContext
from agent_libos.tools.builtin.jit import ValidateJitTool
from agent_libos.tools.sandbox import DenoTypescriptSandbox


SOURCE = "export function run(args, libos) { return {}; }"


def _sandbox(monkeypatch: Any, result: Any) -> DenoTypescriptSandbox:
    sandbox = DenoTypescriptSandbox(deno_executable="deno")
    monkeypatch.setattr(sandbox, "deno_version", lambda: "deno test")
    monkeypatch.setattr(
        sandbox,
        "run_source",
        lambda *_args, **_kwargs: result,
    )
    return sandbox


def _test_record(logs: str) -> dict[str, Any]:
    return next(
        json.loads(line)
        for line in logs.splitlines()
        if line.startswith("{")
    )


def test_successful_jit_result_uses_deterministic_bounded_summary(
    monkeypatch: Any,
) -> None:
    result = {"blob": "x" * 16_384, "count": 7}
    sandbox = _sandbox(monkeypatch, result)

    first = sandbox.run_tests(SOURCE, [{"args": {}, "expected": result}])
    second = sandbox.run_tests(SOURCE, [{"args": {}, "expected": result}])
    record = _test_record(first.logs)
    encoded = json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert first.ok
    assert first.logs == second.logs
    assert len(first.logs) < 2_048
    assert "x" * 512 not in first.logs
    assert record["status"] == "passed"
    assert record["result"] == {
        "lossy": False,
        "preview": record["result"]["preview"],
        "preview_truncated": True,
        "serialized_size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "type": "object",
    }


def test_mismatch_keeps_bounded_head_and_tail_without_credentials(
    monkeypatch: Any,
) -> None:
    secret = "SECRET_JIT_MODEL_RESULT"
    result = {
        "blob": "HEAD" + ("x" * 4_096) + "TAIL",
        "token": secret,
    }
    sandbox = _sandbox(monkeypatch, result)

    validation = sandbox.run_tests(
        SOURCE,
        [{"args": {}, "expected": {"blob": "expected", "token": secret}}],
    )
    error = json.loads(validation.errors[0])

    assert not validation.ok
    assert error["status"] == "failed"
    assert error["reason"] == "result_mismatch"
    assert error["actual"]["type"] == "object"
    assert error["actual"]["serialized_size"] > 4_096
    assert "HEAD" in error["actual"]["preview"]
    assert "TAIL" in error["actual"]["preview"]
    assert "[truncated chars=" in error["actual"]["preview"]
    assert secret not in validation.logs
    assert secret not in validation.errors[0]
    assert "[redacted]" in validation.errors[0]


def test_execution_failure_omits_stack_host_path_and_credentials(
    monkeypatch: Any,
) -> None:
    secret = "SECRET_JIT_EXECUTION_TOKEN"
    sandbox = _sandbox(monkeypatch, None)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Authorization: Bearer "
            f"{secret}\n"
            "    at run (/Users/alice/private/tool.ts:9:3)\n"
            "safe actionable tail"
        )

    monkeypatch.setattr(sandbox, "run_source", fail)
    validation = sandbox.run_tests(SOURCE, [{"args": {}, "expected": {}}])
    error = json.loads(validation.errors[0])

    assert not validation.ok
    assert error["error"]["type"] == "RuntimeError"
    assert error["reason"] == "execution_error"
    assert "[redacted]" in error["error"]["message"]
    assert "[stack omitted]" in error["error"]["message"]
    assert "safe actionable tail" in error["error"]["message"]
    assert secret not in validation.errors[0]
    assert "/Users/alice" not in validation.errors[0]


def test_non_json_result_summary_is_stable_and_never_uses_repr_address(
    monkeypatch: Any,
) -> None:
    first = _sandbox(monkeypatch, object()).run_tests(
        SOURCE, [{"args": {}, "expected": None}]
    )
    second = _sandbox(monkeypatch, object()).run_tests(
        SOURCE, [{"args": {}, "expected": None}]
    )
    first_record = _test_record(first.logs)
    second_record = _test_record(second.logs)

    assert first_record == second_record
    assert first_record["result"]["type"] == "non-json:object"
    assert first_record["result"]["lossy"] is True
    assert "0x" not in first.logs


def test_candidate_test_requires_expected_without_executing_source(
    monkeypatch: Any,
) -> None:
    sandbox = _sandbox(monkeypatch, {"ok": True})
    calls = 0

    def run_source(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return {"ok": True}

    monkeypatch.setattr(sandbox, "run_source", run_source)
    validation = sandbox.run_tests(SOURCE, [{"args": {}}])

    assert not validation.ok
    assert validation.errors == ["JIT test 1 must include expected"]
    assert calls == 0


def test_candidate_expected_comparison_is_json_type_sensitive(
    monkeypatch: Any,
) -> None:
    sandbox = _sandbox(monkeypatch, {"value": True})

    validation = sandbox.run_tests(
        SOURCE,
        [{"args": {}, "expected": {"value": 1}}],
    )

    assert not validation.ok
    assert json.loads(validation.errors[0])["reason"] == "result_mismatch"


def test_candidate_expected_comparison_treats_json_numbers_as_one_type(
    monkeypatch: Any,
) -> None:
    sandbox = _sandbox(
        monkeypatch,
        {"top": 1, "nested": [2.0, {"value": 3}]},
    )

    validation = sandbox.run_tests(
        SOURCE,
        [
            {
                "args": {},
                "expected": {
                    "top": 1.0,
                    "nested": [2, {"value": 3.0}],
                },
            }
        ],
    )

    assert validation.ok, validation.errors


def test_validate_jit_tool_bounds_and_redacts_untrusted_backend_diagnostics() -> None:
    secret = "SECRET_JIT_BACKEND_CREDENTIAL"
    validation = ValidationResult(
        ok=False,
        errors=[
            "small actionable error",
            (
                "password="
                    + secret
                    + "\nTraceback (most recent call last):\n"
                    + '  File "/Users/alice/work/tool.py", line 9, in run\n'
                    + "ValueError: "
                    + ("middle " * 1_000)
                    + "actionable tail"
                ),
        ],
        warnings=[],
        logs=(
            "Authorization: Bearer "
            + secret
            + "\n"
            + "OPENAI_API_KEY=sk-proj-abcdefghijklmnop\n"
            + ("log-data " * 10_000)
            + "\nfinal status"
        ),
    )
    runtime = SimpleNamespace(
        config=DEFAULT_CONFIG,
        tools=SimpleNamespace(validate=lambda *_args, **_kwargs: validation),
    )

    result = ValidateJitTool().invoke(
        {"candidate_id": "jit_candidate"},
        ToolContext(
            trace_id="trace",
            call_id="call",
            pid="pid_test",
            runtime=runtime,
        ),
    )

    assert result.ok
    assert result.data is not None
    assert result.data["ok"] is False
    assert result.data["errors"][0] == "small actionable error"
    rendered = json.dumps(result.data, ensure_ascii=False)
    assert secret not in rendered
    assert "sk-proj-" not in rendered
    assert "/Users/alice" not in rendered
    assert "Traceback" not in rendered
    assert "[stack omitted]" in rendered
    assert "actionable tail" in rendered
    assert "final status" in rendered
    assert len(result.data["logs"]) <= (
        DEFAULT_CONFIG.tools.tool_observability_preview_chars * 4
    )
