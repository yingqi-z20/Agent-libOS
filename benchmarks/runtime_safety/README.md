# Runtime-Safety Benchmark

This directory contains the deterministic runtime-safety task suite. Task YAML
lives in `tasks/`; immutable seed workspaces live in `fixtures/`; `loader.py`
enforces the closed task contract; `runners.py` executes Agent libOS and
baseline variants; `oracle.py` classifies effects; and `metrics.py` validates
completed output artifacts before aggregation.

The normative task and evidence semantics are in [schema.md](schema.md). The
human-facing commands and interpretation guidance are in
[docs/benchmark.md](../../docs/benchmark.md).

Run the deterministic suite with:

```bash
uv run python experiments/run_benchmark.py \
  --suite benchmarks/runtime_safety \
  --runner agent_libos_full \
  --require-all-passed \
  --require-release-evidence \
  --output .benchmark_runs/runtime-safety
```

`--require-all-passed` gates task success and safety oracles.
`--require-release-evidence` additionally gates complete audit evidence and
zero false denials; the checked-in deterministic release job requires both.

Run the benchmark regression lane with:

```bash
uv run python scripts/test_matrix.py --lane benchmark
```

Machine-readable JSON Schema 2020-12 contracts are generated from the same
closed definitions used by the loader:

```bash
uv run python benchmarks/runtime_safety/schemas.py task
uv run python benchmarks/runtime_safety/schemas.py bundle
```

Unknown mock actions, action fields, policy keys, and action-specific
`benchmark_effects` bindings fail before execution. A new action therefore
requires an explicit schema entry, runner effect normalization, documentation,
and a regression test; adding a tool name only to a task file is insufficient.

Fixture workspaces must be non-symlink directories inside the suite. Runs copy
them into the selected output root before applying setup. Never place secrets,
credentials, real remotes, generated run outputs, or pre-existing Git metadata
inside a checked-in fixture.
