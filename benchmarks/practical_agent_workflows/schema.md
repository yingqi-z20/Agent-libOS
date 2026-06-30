# Practical Agent Workflow Benchmark Schema v2

This benchmark is the paper-facing end-to-end evaluation suite. It is separate
from `benchmarks/runtime_safety`, which remains a primitive-level microbenchmark
for capability and approval primitives.

The default catalog expands into 80 practical scenarios:

- 5 tracks: coding, research, enterprise, devops, self_evolution.
- 8 task families per track: core_task, diagnostic_tool_output,
  external_context, state_update, tool_extension, remote_action,
  capability_transfer, delayed_trigger.
- 2 variants per family: one benign scenario plus one attack, adaptive, or
  long_horizon scenario.

Each scenario has:

- `id`, `title`, `domain`, `track`, `task_family`, `workflow`, `variant`, and
  `attack_type`.
- a workspace fixture.
- `trusted_sources` and `untrusted_sources`.
- `allowed_effects` and `forbidden_effects`.
- `quality_oracle`, `attack_oracle`, `utility_oracle`, `security_oracle`, and
  `state_diff_oracle`.
- `runtime_calls` for scenarios that can be executed through the live Agent
  libOS tool interface.
- `expected_provenance` requirements and `evidence_level`.
- deterministic actions used for modeled evaluation and replay.
- mock service state before and after execution.

Evidence levels:

- `modeled`: replayable effect simulation only.
- `modeled+live-runtime`: modeled evaluation plus compatible runtime calls.
- `real-llm-selection`: real LLM selected actions, guarded by
  `--allow-token-spend`.

Runner outputs:

- `results.jsonl`
- `effects.jsonl`
- `audit_trace.jsonl`
- `external_effects.jsonl`
- `llm_calls.jsonl`
- `human_requests.jsonl`
- `replay_trace.jsonl`
- `service_state_before_after.json`
- `failure_cases.json`
- `metrics.json` and `metrics.csv`
- `practical_eval_summary.md`
- `live_runtime_summary.md`
- `case_studies.md`
- `failure_taxonomy.md`

Modes:

- `deterministic`: use the scenario's planned action sequence.
- `replay`: use a prior `replay_trace.jsonl`.
- `real`: ask a configured LLM to choose action ids, guarded by
  `--allow-token-spend`, then write a replay trace.

Primary evidence slices:

- Modeled matrix: all 80 scenarios across external baselines, Agent libOS, and
  Agent libOS ablations.
- Live runtime slice: at least 40 scenarios through `agent_libos_live`, which
  writes a runtime SQLite DB, runtime tool results, external-effect rows, and
  audit records.
- Real LLM pilot: a guarded, smaller stress run whose action traces are replayed
  deterministically.
