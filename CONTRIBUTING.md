# Contributing

Thanks for helping improve Agent libOS.

Use [docs/development.md](docs/development.md) for environment setup, standard
checks, optional provider gates, and release-artifact validation. A Git checkout
also contains [AGENTS.md](AGENTS.md), which gives coding agents additional
repository-local operating instructions. Source distributions intentionally do
not include that automation-specific file; this contributing guide and the
checked-in `docs/` pages are sufficient for human contributors using an sdist.

Core Python code lives in `agent_libos/`, deterministic pytest coverage in
`tests/`, runtime-safety fixtures and evaluators in `benchmarks/`, user-facing
entrypoints in `experiments/`, and the Electron/React application in `gui/`.
Use Python 3.11 or newer with four-space indentation and type hints on public
interfaces. Keep defaults in `agent_libos.config.DEFAULT_CONFIG`, and preserve
the boundary that Tools and Skills affect visibility while primitives enforce
authority, approval, data-flow, resource, effect, event, and audit rules.

In a Git checkout, install the reviewed lock with `uv sync --frozen`. The source
distribution intentionally omits the repository `uv.lock`; from an unpacked
sdist, `uv sync` can resolve a new local environment, but that is not a
frozen-lock reproduction or release receipt. The normal complete deterministic
Python check is `uv run python scripts/test_matrix.py --lane all`. Use the
narrower lane that matches the change while iterating, then run the complete set
before claiming a full local validation. Run
`uv run python scripts/check_test_invariants.py` for invariant changes and
`uv run python -m compileall agent_libos tests scripts experiments benchmarks
modules` for syntax/import coverage. GUI sources are absent from the Python
sdist; GUI changes require a checkout and the separate commands in the
development guide.

Keep changes focused and add or update tests for behavior changes. Security or
authority changes need denial-path coverage, audit/event assertions, and an
invariant-manifest update when they protect a runtime invariant. Run the checks
appropriate to the affected lanes and report the exact commands and relevant
environment gates in the pull request.

A pull request should explain the behavior and runtime invariant affected, link
the relevant issue or design note when one exists, and include screenshots for
visible GUI changes. Do not commit credentials, real `.env` files, benchmark
outputs, generated `agent_outputs/`, or GUI build artifacts. Only contribute
material you are authorized to share under the repository's Apache-2.0 license.

Ordinary bugs and feature proposals may use GitHub issues. Suspected security
vulnerabilities, exploit details, secrets, and private data must follow the
reporting guidance in [SECURITY.md](SECURITY.md), not a public issue or pull
request. Use the private-report form linked from that policy. If the form is
unavailable, a public issue may contain only the non-sensitive coordination
request that policy permits.
