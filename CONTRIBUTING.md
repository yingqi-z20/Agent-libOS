# Contributing

Thanks for helping improve Agent libOS.

Before changing code, read [AGENTS.md](AGENTS.md) for repository boundaries,
test expectations, and security constraints. Use
[docs/development.md](docs/development.md) for environment setup, standard
checks, optional provider gates, and release-artifact validation.

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
request. While no confidential intake is enabled, a public issue may contain
only the non-sensitive coordination request that policy permits.
