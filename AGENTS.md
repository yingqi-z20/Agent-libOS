# Repository Guidelines

## Project Structure & Module Organization

Agent libOS is a Python runtime with an optional Electron GUI. Core runtime code
lives in `agent_libos/`, organized by subsystems including `runtime/`,
`primitives/`, `capability/`, `memory/`, `skills/`, `modules/`, `tools/`,
`substrate/`, `config/`, `evidence/`, `human/`, `images/`, `llm/`, `models/`,
`ports/`, `sdk/`, `semantic/`, `storage/`, `utils/`, and `api/` for CLI/GUI server
entrypoints. Pytest tests live in `tests/` and map to the six Python
test matrix lanes: `unit`, `runtime`, `security`, `self-evolution`,
`providers`, and `benchmark`; some lane names differ from directory
names, for example `self-evolution` maps to `tests/self_evolution` and
`benchmark` maps to `tests/benchmarks`. The `gui` lane is not a pytest
lane and has no `tests/` directory; it runs the frontend Vitest,
typecheck, and build tooling in `gui/`. Shared helpers live in
`tests/support/`. Runtime-safety task fixtures, runner implementations, oracle,
and metrics live under `benchmarks/runtime_safety/`; practical evidence-level
scenarios live under `benchmarks/practical_agent_workflows/`. User-facing
benchmark entrypoints are in `experiments/`.
Documentation is in `docs/`; the Electron/React frontend is in `gui/`; example
skills live in `skills/`.

## Build, Test, and Development Commands

- `uv sync --frozen`: install the locked default development environment,
  including pytest tooling.
- `uv run python -m compileall agent_libos tests scripts experiments benchmarks modules`:
  catch syntax/import errors.
- `uv run python scripts/test_matrix.py --lane unit`: run fast pure-Python tests.
- `uv run python scripts/test_matrix.py --lane security`: run capability,
  approval, filesystem, shell, and JIT containment tests.
- `uv run python scripts/test_matrix.py --lane all`: run all deterministic
  Python pytest lanes.
- `uv run python scripts/check_test_invariants.py`: verify the invariant
  coverage manifest.
- `uv run agent-libos --help`: inspect CLI commands.
- `uv run python experiments/run_benchmark.py --suite benchmarks/runtime_safety --runner agent_libos_full --limit 3 --require-all-passed --output .benchmark_runs/smoke`: run a deterministic benchmark smoke and fail if an oracle fails.
- `uv run python experiments/run_practical_evaluation.py --output .benchmark_runs/practical/report.json`: run practical workflows while preserving `native-live` and `modeled` evidence labels.
- `uv run python scripts/test_matrix.py --lane gui`: run GUI Vitest,
  typecheck, and build. Run `npm --prefix gui install` first in a fresh
  checkout.

## Coding Style & Naming Conventions

Use Python 3.11+ with 4-space indentation, type hints for public interfaces, and
dataclasses or Pydantic models for structured data. Keep runtime defaults in
`agent_libos.config.DEFAULT_CONFIG`; do not scatter magic numbers. Preserve the
core boundary: tools and Skills affect visibility, while primitives enforce
the Capability, human-approval, provider-policy, data-flow, resource, effect,
event, and audit boundaries applicable to their effect class. TypeScript in
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

Never commit local/real `.env`, credentials, benchmark outputs, generated
`agent_outputs/`, or GUI build artifacts.
Remote access must use Host-configured primitives or providers, such as
registered JSON-RPC/MCP endpoints, named LLM profiles, or configured Git
remotes; models must not supply ad hoc URLs, credentials, or transport commands.
Ordinary data-flow Sink-registry writes require admin authority; only trusted
Host bootstrap/reconciliation before Runtime OPEN may bypass a process
Capability, and neither path becomes a model tool. Semantic policy control is a
Host-composition surface reachable from the local Runtime Python object, not a
CLI, HTTP, GUI, process, Skill, JIT, Module, or model-facing surface.
Prompt caching defaults to the legacy prompt layout and provider-default cache
policy. Treat prompt-cache v2 as an explicit Host opt-in: preserve its stable
and dynamic prompt split, Host privacy-domain-derived cache key, same-logical-call
compatibility downgrade, configured-versus-effective evidence, and paired
multi-provider release gate when changing LLM prompts or transports.
Startup recovery must validate recoverable TaskRun payloads before durable
recovery effects, reconcile prepared/pending effects before abandoning stale
Capability reservations, and finish semantic authority, resource/publication,
root-goal/payload, JIT/stale-work, ObjectTask/cleanup, and TaskRun recovery in the
builder-defined order. Provider reconciliation may restore an effect-bound
Capability reservation, so do not move stale-reservation abandonment earlier.
Checkpoint restore and image commit do not roll back or package external
provider state. Audit and effect-transition history are append-only through the
RuntimeStore API, while guarded external-effect rows and retained payload
projections may be updated in place. None of this evidence is tamper-proof
against a direct database administrator.
