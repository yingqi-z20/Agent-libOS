from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("script", "function", "expected_pids"),
    [
        ("ask_file_then_show.py", "run_file_viewer", ("pid",)),
        (
            "async_clock_interleave_smoke.py",
            "run_interleaved_clock_demo",
            ("pid_a", "pid_b"),
        ),
        ("run_coding_agent.py", "amain", ("pid",)),
    ],
)
def test_scenario_scheduler_run_is_scoped_to_invocation_owned_pids(
    script: str,
    function: str,
    expected_pids: tuple[str, ...],
) -> None:
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    selected = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function
    )
    calls = [
        node
        for node in ast.walk(selected)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "arun_until_idle"
    ]

    assert len(calls) == 1
    pids = next(
        keyword.value for keyword in calls[0].keywords if keyword.arg == "pids"
    )
    assert isinstance(pids, ast.Tuple)
    assert tuple(
        element.id for element in pids.elts if isinstance(element, ast.Name)
    ) == expected_pids
