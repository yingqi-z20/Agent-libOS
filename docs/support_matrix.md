# Support and Validation Matrix

This is the living distinction between code that exists, package versions the
project intends to support, and environments exercised on every change. A
feature is not “CI-covered” merely because it has a platform branch or mock
test. The historical prelaunch report is not the current status source.

Legend:

- **CI**: exercised by the checked-in GitHub Actions workflow.
- **Deterministic**: token-free/local test coverage, possibly with a fake or
  loopback provider.
- **Environment gate**: requires a real OS, desktop, service, SDK, or credential
  and is intentionally outside the default deterministic matrix.
- **Not implemented**: documentation must not present the surface as current.

## Runtime and platform coverage

| Surface | Declared/current implementation | Per-change CI | Remaining boundary |
| --- | --- | --- | --- |
| Python | Package declares 3.11–3.14 | Ubuntu on 3.11 and 3.14 for Python lanes | 3.12/3.13 are inside the declared range but are not separate per-change jobs |
| Python release artifacts | Core-package wheel plus Python source distribution | CI builds both artifacts, validates their contents, clean-installs and dependency-checks both, runs all three CLI entrypoints and the deterministic-demo smoke from both installations, and preserves the validated distributions | Repository-level PTY/Skill/Image assets are source-distribution assets; Electron sources remain repository-checkout assets |
| SQLite RuntimeStore | Default local backend, file and in-memory targets | Ubuntu deterministic lanes | macOS/Windows filesystem ACL and locking behavior need native release-gate runs |
| PostgreSQL RuntimeStore | Optional `postgres` extra | PostgreSQL 17 service on Ubuntu/Python 3.11 | Other server versions and deployment TLS/auth topology are unvalidated operator gates |
| Core process/shell containment | POSIX process groups plus platform-specific fallbacks | Ubuntu security/runtime/provider lanes | macOS and Windows native process-tree/resource behavior are not CI-covered |
| Deno/TypeScript JIT | Deno required for real JIT; deterministic benchmark also has an explicit fake backend | Deno LTS on Ubuntu; real-Deno tests run when installed | Windows parent-death Job Object and macOS native behavior need release-gate runs |
| PTY Runtime Module | POSIX PTY plus optional Windows `pywinpty`/ConPTY session I/O; source checkout/source distribution only, not the core wheel | POSIX paths on Ubuntu plus deterministic branch tests | Current Windows backend has no Job Object, parent-death containment, or wall/CPU/RSS supervision; budgeted `SubprocessLimits` spawns fail closed. Real Windows PTY/resize/close I/O still needs a native release-gate run |
| Typed Git provider | System Git 2.26+, fixed non-bare workspace repository; local operations, managed worktrees, patch Objects, existing remotes, and repository-local simulated PRs | Deterministic provider/security/runtime tests use temporary SHA-1/SHA-256 repositories and local bare remotes; Shell/PTY/provenance hardening is parameterized | Native Windows locking/path/credential-manager and real HTTPS/OpenSSH authentication require release-gate runs; GitHub/GitLab APIs are not implemented |
| JSON-RPC client | Registered HTTP endpoints only | Deterministic loopback/provider tests | Real network proxy/TLS/DNS policy is deployment-specific |
| MCP client | Tools-only v1 over Streamable HTTP or stdio | Deterministic primitive/provider tests | Real MCP SDK/server integration uses the optional `mcp` extra and `--run-mcp` environment gate; Resources/Prompts are not implemented |
| Real LLM | OpenAI Responses and OpenAI-compatible Chat profiles | Mock/action-selection paths only | Credentials and token-spending smoke are opt-in with `--run-real-llm`; run one scoped task/profile per release target |
| Data-label egress enforcement | Host Sink registry and a unified gate cover LLM, Human, JSON-RPC, MCP, typed Git, filesystem writes, Shell/PTY, and internal process handoff | Deterministic unit/runtime/security/provider/benchmark tests, including pre-provider denial and exact conditional release | The guarantee covers runtime-mediated payloads; trusted modules/providers, native child I/O, Sink re-forwarding, and direct store administration remain operator trust boundaries |

## GUI and API coverage

| Surface | Per-change CI | Environment gate |
| --- | --- | --- |
| React/Vitest | Ubuntu, Node 24, source tests | Browser accessibility and operator usability study are not automated |
| Web and Electron TypeScript | Typecheck and production build on Ubuntu | Native Electron packaging/signing/notarization are not configured release jobs |
| Python GUI HTTP/SSE server | Providers lane exercises auth, route validation, bounded event windows, shutdown, CORS, and snapshots | Native desktop process lifecycle remains platform-specific |
| Headless Electron main-process smoke | Not in the default GUI lane | Run `AGENT_LIBOS_GUI_SMOKE=1 npm --prefix gui run electron:dev` |
| Production-build custom-protocol BrowserWindow smoke | Not in CI | On a desktop/GPU runner use `AGENT_LIBOS_GUI_SMOKE=1 AGENT_LIBOS_GUI_SMOKE_WINDOW=1 npm --prefix gui run electron:dev`; this does not package, sign, or notarize an Electron application |
| Local GUI API compatibility | Server and renderer tests cover the matching checkout | The unversioned `/api` surface is an internal same-build contract, not a stable third-party REST API |

## Evaluation coverage

| Suite | Default evidence | Boundary |
| --- | --- | --- |
| `benchmarks/runtime_safety` | 32 deterministic schema-v1 tasks, including data-label exfiltration plus Git worktree containment, malicious config, remote misuse, and patch lineage; fail-closed metrics and provenance-bearing CLI metadata | Early runtime-safety workload, not a complete paper evaluation or formal proof; Git network tasks use controlled local state rather than a hosted provider |
| `benchmarks/practical_agent_workflows` | Exactly two labels: `native-live` and `modeled`; native has no modeled fallback | Checked-in scenarios do not imply a real GitHub/provider integration |
| `benchmarks/external_effect_recovery` | 100k-record `ci` profile on each change; one-million-record `million` profile in the manual/nightly workflow | Structural paging/index/convergence checks are gates; elapsed times are diagnostic, not SLAs |
| `benchmarks/runtime_publication_recovery` | 10k terminal publications with 1,001 unreconciled rows in the only named profile, `ci` | No one-million-publication profile is currently implemented; custom sizes are explicit CLI overrides |
| `experiments/agentdojo` | Deterministic harness tests run in a separate Ubuntu matrix on Python 3.11 and 3.12 from the subproject's own frozen lock; `release-artifacts` waits for both matrix runs | The root lock and `scripts/test_matrix.py` do not include this environment. CI does not make provider calls or claim real-model AgentDojo utility/security results; those runs remain explicit credential/token gates |
| Real-model benchmark | One explicitly selected task with real LLM profile | Token/credential gate; results must retain model/profile/environment provenance |

## Release-gate policy

The static compile, architecture/blocking-work, protected-operation, and
invariant/collection checks; per-lane deterministic matrix on every CI Python
version; GUI lane; isolated AgentDojo Python 3.11/3.12 harness; PostgreSQL job;
runtime-safety release smoke; 100k external-effect recovery gate;
10k runtime-publication recovery gate; and release-artifact build and dual
clean-install smoke are all necessary but not sufficient for a cross-platform
release. `release-artifacts` waits for every required workflow job before it
builds the distributions. `scripts/test_matrix.py --lane all` remains the local
aggregate command for the root environment; it neither reproduces those static,
service, GUI, AgentDojo, scale, and artifact gates nor repeats the equivalent
per-lane CI matrix. Before advertising a platform or provider configuration as
release-validated, record a fresh native run for the corresponding
environment-gate cells above. Do not copy counts or “remaining gates” from
`docs/prelaunch_hardening_report.md`; that file is bound to its historical
commit.

When a new environment becomes CI-covered, update this matrix and
`docs/invariants.md` Known Test Gaps in the same change.
