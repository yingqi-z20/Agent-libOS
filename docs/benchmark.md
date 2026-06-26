# Runtime-Safety Benchmark

The M1 benchmark harness is a deterministic runtime-safety workload for
Agent libOS. It is designed to compare agent runtime boundaries against simpler
wrappers while avoiding default token spend. The suite now includes a
self-evolution subset for the paper theme: capability-controlled changes through
Skills, Deno/TypeScript JIT tools, image registration/exec/checkpoint commit,
child processes, checkpoints, Object Memory, and registered remote resources.

The task schema is defined in
[benchmarks/runtime_safety/schema.md](../benchmarks/runtime_safety/schema.md).

## Task Suite

The checked-in suite contains 20+ YAML tasks under
`benchmarks/runtime_safety/tasks/`. They cover at least these classes:

- secret read attempts,
- forbidden filesystem writes,
- forbidden filesystem deletes,
- shell bypass and exfiltration attempts,
- object authority leakage,
- process authority leakage,
- self-evolution attempts involving Skills, JIT tools, image
  registration/exec/checkpoint commit, child processes, checkpoint fork, and
  JSON-RPC visibility.

Each task declares:

- a goal,
- a fixture workspace,
- attack class,
- allowed effects,
- forbidden effects,
- success oracle,
- safety oracle,
- initial capabilities and policy,
- deterministic `mock_actions`.

Fixtures live under `benchmarks/runtime_safety/fixtures/`. Runner workspaces are
copied to temporary output directories so checked-in fixtures are not mutated.

## Runners

Supported runner names are:

- `direct_tool_wrapper`
- `confirmation_wrapper`
- `sandbox_only`
- `agent_libos_full`
- `no_primitive_approval`
- `no_audit_linkage`
- `no_namespace_isolation`
- `no_fork_attenuation`

Wrapper and sandbox runners are baselines, not trusted security boundaries.
Risky shell/network behavior is simulated where needed and recorded as effects.

Agent libOS runners execute through the runtime, using process capabilities,
primitive checks, human policy, audit records, and persisted LLM calls.

## Running

Default smoke:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --output .benchmark_runs/m1-smoke
uv run python experiments/collect_metrics.py .benchmark_runs/m1-smoke
```

All runners:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner all --output .benchmark_runs/m1
uv run python experiments/collect_metrics.py .benchmark_runs/m1
```

Select tasks:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --task fs_secret_read_001 --output .benchmark_runs/one
```

Select attack classes:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner all --attack-class shell_policy_bypass --output .benchmark_runs/shell
```

## Real LLM Mode

The default mode is `--llm mock`. It uses task `mock_actions` and does not spend
tokens.

Real LLM mode is explicit and must be scoped to one task:

```bash
uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --llm real --limit 1 --output .benchmark_runs/real-smoke
```

The command rejects broad real-model runs unless `--limit 1` or exactly one
`--task` is supplied. Real mode uses `LLMClient.from_env()` and runtime
`llm_calls` persistence.

## Outputs

`run_benchmark.py` writes:

- `metadata.json`: selected suite, tasks, runners, LLM mode, and process id.
- `results.jsonl`: one `BenchmarkResult` row per task/runner.
- `effects.jsonl`: one `EffectRecord` row per modeled effect.
- `summary.json`: counts of results, effects, runners, tasks, ok runs, and
  safety-passed runs.
- `metrics.json`: aggregate metrics.
- `metrics.csv`: stable CSV metrics columns.

Agent libOS runner directories also include per-task runtime SQLite databases
under the output directory.

## Result Fields

`results.jsonl` rows include:

- `task_id`
- `runner`
- `attack_class`
- `ok`
- `task_success`
- `safety_passed`
- `unknown_effects`
- `forbidden_performed`
- `approval_count`
- `tool_calls`
- `primitive_calls`
- `llm_tokens`
- `wall_time_s`
- `audit_records`
- `audit_completeness`
- `errors`
- `workspace`
- `metadata`, including `metadata.self_evolution_counts` for per-run
  self-evolution attempts.

`effects.jsonl` rows include type-specific fields such as `path`, `argv`,
`namespace`, `name`, `skill_id`, `tool`, `image`, `checkpoint`, `endpoint`,
`method`, `provider`, `operation`, plus `performed`, `denied`, `simulated`,
`classification`, and `error`.

Denied attempts are recorded but do not count as performed unauthorized effects.

## Metrics

Stable metric columns are:

- `runner`
- `tasks`
- `task_success_rate`
- `safety_pass_rate`
- `unauthorized_side_effect_rate`
- `false_denial_rate`
- `approval_count`
- `tool_calls`
- `primitive_calls`
- `llm_tokens`
- `wall_time_s`
- `audit_completeness`
- `skill_activations`
- `jit_registrations`
- `image_commits`
- `image_registrations`
- `image_execs`
- `child_processes`
- `checkpoint_forks`
- `remote_calls`

The current benchmark is suitable for deterministic smoke and early evaluation.
It is not yet a full paper evaluation suite. Audit explain queries, richer
context materialization metadata, adversarial remote provider tasks, and
Git/worktree provider tasks remain future work.
