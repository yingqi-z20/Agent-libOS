"""Real-LLM evaluation for durable, multi-stage Agent libOS work."""

from benchmarks.long_horizon_agent.runner import (
    DEFAULT_MAX_QUANTA,
    DEFAULT_PHASE_ONE_QUANTA,
    DEFAULT_SCENARIO,
    SCENARIO_ID,
    SCENARIOS,
    HostOracleRunner,
    LongHorizonScenario,
    evaluate_run,
    prepare_workspace,
    report_all_successful,
    run_evaluation,
)
from benchmarks.long_horizon_agent.ledgerctl_scenario import LEDGERCTL_SCENARIO

# Registered scenarios beyond the default.  Importing the package (which any
# ``benchmarks.long_horizon_agent.*`` import does first) completes the registry.
SCENARIOS.setdefault(LEDGERCTL_SCENARIO.scenario_id, LEDGERCTL_SCENARIO)

__all__ = [
    "DEFAULT_MAX_QUANTA",
    "DEFAULT_PHASE_ONE_QUANTA",
    "DEFAULT_SCENARIO",
    "SCENARIO_ID",
    "SCENARIOS",
    "HostOracleRunner",
    "LongHorizonScenario",
    "evaluate_run",
    "prepare_workspace",
    "report_all_successful",
    "run_evaluation",
]
