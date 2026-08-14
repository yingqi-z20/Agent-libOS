"""Opt-in real-LLM evaluation for built-in Tool Skill routing."""

from benchmarks.builtin_tool_skills.runner import (
    EVALUATION_REPETITIONS,
    EVALUATION_VARIANTS,
    HELD_OUT_SCENARIOS,
    REAL_LLM_ROUTING_CATALOG,
    SkillRoutingCase,
    WITH_SKILLS,
    WITHOUT_SKILLS,
    aggregate_runs,
    evaluation_pair_plan,
    report_all_correct,
    report_publication_ready,
    run_evaluation,
)

__all__ = [
    "EVALUATION_REPETITIONS",
    "EVALUATION_VARIANTS",
    "HELD_OUT_SCENARIOS",
    "REAL_LLM_ROUTING_CATALOG",
    "SkillRoutingCase",
    "WITH_SKILLS",
    "WITHOUT_SKILLS",
    "aggregate_runs",
    "evaluation_pair_plan",
    "report_all_correct",
    "report_publication_ready",
    "run_evaluation",
]
