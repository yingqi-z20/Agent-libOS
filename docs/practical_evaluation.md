# Practical Agent Workflow Evaluation

`benchmarks/practical_agent_workflows` is the paper-facing evaluation suite for
realistic end-to-end agent workflows. The legacy `runtime_safety` benchmark
remains useful as a primitive-level regression suite, but it is no longer the
main evaluation story.

Run a deterministic practical evaluation:

```powershell
.\.venv\Scripts\python.exe experiments\run_practical_workflows.py --runner direct_tool_agent --runner confirmation_agent --runner sandbox_agent --runner prompt_defense_agent --runner agent_libos --runner agent_libos_no_audit --runner agent_libos_no_fork_attenuation --runner agent_libos_no_human_approval --runner agent_libos_no_remote_method_caps --output .benchmark_runs\practical_eval_v2_modeled
```

Run a live-runtime slice:

```powershell
.\.venv\Scripts\python.exe experiments\run_practical_workflows.py --runner agent_libos_live --output .benchmark_runs\practical_eval_v2_live
```

Run guarded real-model action selection:

```powershell
.\.venv\Scripts\python.exe experiments\run_practical_workflows.py --mode real --allow-token-spend --runner agent_libos --limit 3 --output .benchmark_runs\practical-real-pilot
```

Generated reports:

- `practical_eval_summary.md`
- `live_runtime_summary.md`
- `case_studies.md`
- `failure_taxonomy.md`
- `metrics.json`
- `replay_trace.jsonl`

Reviewer-facing metrics include benign success, attack-setting task success,
state-diff success, attack-success-blocked rate, forbidden committed effects,
false denials, human approvals, pass^k, and trace coverage.

## Benchmark v2 Tracks

The v2 suite contains 80 main scenarios: 5 tracks x 8 task families x 2
variants. The tracks are coding, research/RAG, stateful enterprise tools,
devops/secops, and self-evolution/capability dynamics. See
`docs/benchmark_landscape.md` for the literature mapping and design rationale.

Each scenario records `trusted_sources`, `untrusted_sources`, `allowed_effects`,
`forbidden_effects`, `utility_oracle`, `security_oracle`, `state_diff_oracle`,
`runtime_calls`, and `evidence_level`. The old `runtime_safety` YAML suite is
kept as primitive microbenchmarks, not the main result table.

## Baseline Design

The practical suite separates runner categories rather than treating every
runner as the same kind of baseline:

- `agent_libos` is the primary system.
- `agent_libos_live` is the primary live-runtime evidence runner. It executes
  compatible scenario actions through real Agent libOS tools, capabilities,
  provider state, and audit.
- `direct_tool_agent`, `confirmation_agent`, `sandbox_agent`, and
  `prompt_defense_agent` are external deployment baselines. They represent
  common agent hardening patterns: raw tool access, coarse human confirmation,
  host sandboxing, and prompt-only defense.
- `agent_libos_no_*` runners are internal ablations. They should not be
  described as competing systems; they test whether audit linkage, fork
  attenuation, human approval, and remote method capabilities are necessary.

The baselines are intentionally mechanism-specific. `confirmation_agent` asks
for human review on coarse high-risk tool classes and records explicit approved
or rejected decisions, but it lacks provenance and capability narrowing.
`sandbox_agent` applies host-style containment to deletion, external network
access, and obvious network shell commands; it does not use the oracle to know
which in-workspace reads or agent-level API calls are harmful. `prompt_defense`
models a strong instruction-only defense for direct untrusted-source prompt
injection, but it has no runtime enforcement against tool escalation,
capability laundering, or remote method abuse.

## Evidence Slices

The reports are intentionally separated:

- Modeled matrix: all 80 scenarios across external baselines and ablations.
- Live runtime slice: at least 40 scenarios through `agent_libos_live`.
- Real LLM pilot: guarded stress runs; unknown oracle cases are reported
  separately and replay traces are retained.

Security claims should be based on committed effects. The reports also list
model-requested harmful actions and runtime-denied harmful actions so reviewers
can separate planner failures from enforcement failures.

## Current v2 Evidence

Generated on 2026-07-01:

- Modeled matrix: `.benchmark_runs/practical_eval_v2_modeled`, covering 80
  scenarios x 9 modeled systems/ablations = 720 result rows.
- Live runtime slice: `.benchmark_runs/practical_eval_v2_live`, covering all 80
  scenarios through `agent_libos_live`, with per-scenario runtime SQLite DBs,
  tool results, external-effect rows, and linked audit traces.
- Real LLM smoke pilot: `.benchmark_runs/practical_eval_v2_real_pilot_smoke`,
  covering 3 scenarios with 1,690 LLM tokens, plus deterministic replay at
  `.benchmark_runs/practical_eval_v2_real_pilot_replay`.

Modeled primary results:

| System | Scenarios | Benign Success | State Diff | Attack Blocked | Forbidden Committed | False Denials | Human Approvals | Trace Coverage | pass^k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `agent_libos` | 80 | 100.0% | 100.0% | 100.0% | 0 | 0 | 5 | 100.0% | 100.0% |
| `direct_tool_agent` | 80 | 100.0% | 50.0% | 0.0% | 100 | 0 | 0 | 0.0% | 50.0% |
| `confirmation_agent` | 80 | 100.0% | 55.0% | 10.0% | 67 | 0 | 114 | 0.0% | 55.0% |
| `sandbox_agent` | 80 | 100.0% | 50.0% | 0.0% | 83 | 0 | 0 | 0.0% | 50.0% |
| `prompt_defense_agent` | 80 | 100.0% | 50.0% | 0.0% | 100 | 0 | 0 | 0.0% | 50.0% |

Live runtime result:

| System | Scenarios | Benign Success | State Diff | Attack Blocked | Forbidden Committed | False Denials | Human Approvals | Trace Coverage | pass^k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `agent_libos_live` | 80 | 100.0% | 100.0% | 100.0% | 0 | 0 | 32 | 100.0% | 100.0% |

The live failure taxonomy reports 100 requested forbidden attempts, 100
runtime-denied forbidden attempts, and 0 committed forbidden effects.

Real LLM smoke result: 3/3 scenarios passed, 0 forbidden committed effects,
100.0% trace coverage, and 100.0% replay pass rate. This is only a guarded
smoke pilot; the larger multi-model/multi-repeat stress run should be launched
as a separate token-budgeted experiment.
