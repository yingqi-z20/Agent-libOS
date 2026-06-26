# Repository Guidelines

## Project Structure & Module Organization

Agent libOS is a Python runtime with an optional Electron GUI. Core runtime code
lives in `agent_libos/`, organized by subsystem: `runtime/`, `primitives/`,
`capability/`, `memory/`, `skills/`, `modules/`, `tools/`, `substrate/`, and
`api/` for CLI/GUI server entrypoints. Pytest tests live in `tests/` and map to
lanes: `unit`, `runtime`, `security`, `self-evolution`, `providers`, and
`benchmark`; shared helpers live in `tests/support/`. Benchmark code and
fixtures are under `benchmarks/runtime_safety/`, with runners in `experiments/`.
Documentation is in `docs/`; the Electron/React frontend is in `gui/`; example
skills live in `skills/`.

## Build, Test, and Development Commands

- `uv sync --frozen --all-groups`: install the locked Python environment,
  including pytest tooling.
- `uv run python -m compileall agent_libos tests scripts experiments benchmarks`:
  catch syntax/import errors.
- `uv run python scripts/test_matrix.py --lane unit`: run fast pure-Python tests.
- `uv run python scripts/test_matrix.py --lane security`: run capability,
  approval, filesystem, shell, and JIT containment tests.
- `uv run python scripts/test_matrix.py --lane all`: run all deterministic
  Python pytest lanes.
- `uv run python scripts/check_test_invariants.py`: verify the invariant
  coverage manifest.
- `uv run agent-libos --help`: inspect CLI commands.
- `uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --output .benchmark_runs/smoke`: run a deterministic benchmark smoke.
- `uv run python scripts/test_matrix.py --lane gui`: run GUI Vitest,
  typecheck, and build.

## Coding Style & Naming Conventions

Use Python 3.11+ with 4-space indentation, type hints for public interfaces, and
dataclasses or Pydantic models for structured data. Keep runtime defaults in
`agent_libos.config.DEFAULT_CONFIG`; do not scatter magic numbers. Preserve the
core boundary: tools and Skills affect visibility, while primitives enforce
Capability, human approval, provider policy, events, and audit. TypeScript in
`gui/` should use strict component and API types.

## Testing Guidelines

Add or update tests with each behavior change. Name Python tests
`tests/<lane>/test_<feature>.py` and test methods `test_<expected_behavior>`.
Security or authority changes need denial-path tests, audit/event assertions,
and an entry in `tests/invariants.yaml` when they protect a runtime invariant.
Real LLM paths must remain opt-in through pytest markers and `--run-real-llm`.
Real Deno tests run by default when `deno` is installed and can be excluded with
`--skip-real-deno`; default tests should remain deterministic and token-free.

## Commit & Pull Request Guidelines

Recent commits are short topic summaries such as `GUI` or
`checkpoint commit to image`; prefer concise imperative subjects with a clear
scope, for example `harden checkpoint fork authority`. PRs should describe the
runtime invariant affected, list tests run, link issues or design notes, and
include GUI screenshots for visible frontend changes.

## Security & Configuration Tips

Never commit `.env`, credentials, benchmark outputs, or GUI build artifacts.
Remote access must go through registered JSON-RPC endpoints, not model-supplied
URLs. Checkpoint restore and image commit do not roll back or package external
provider state; provider-classified effects remain append-only audit records.
