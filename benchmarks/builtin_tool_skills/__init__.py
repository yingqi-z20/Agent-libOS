"""Opt-in real-LLM evaluation for built-in Tool Skill routing."""

from benchmarks.builtin_tool_skills.runner import (
    EVALUATION_REPETITIONS,
    EVALUATION_VARIANTS,
    HELD_OUT_SCENARIOS,
    WITH_SKILLS,
    WITHOUT_SKILLS,
    aggregate_runs,
    report_all_correct,
    run_evaluation,
)

__all__ = [
    "EVALUATION_REPETITIONS",
    "EVALUATION_VARIANTS",
    "HELD_OUT_SCENARIOS",
    "WITH_SKILLS",
    "WITHOUT_SKILLS",
    "aggregate_runs",
    "report_all_correct",
    "run_evaluation",
]
