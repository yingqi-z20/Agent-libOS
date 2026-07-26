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
| Python | Package declares 3.11–3.14 | Ubuntu on 3.11 and 3.14 for Python lanes; Windows on 3.11 for the complete deterministic matrix split into per-lane jobs; a configured Ubuntu/macOS 14 Python 3.11 matrix runs the platform-marked host-filesystem-identity gate with skips forbidden | 3.12/3.13 are inside the declared range but are not separate root-runtime jobs; macOS behavior outside the targeted filesystem-identity gate remains an environment gate |
| Python release artifacts | Core-package wheel plus Python source distribution | CI builds one canonical wheel/source pair once with the frozen release tool group, rejects extra or non-regular files, records its exact checksum manifest, runs `twine check` and `check-wheel-contents`, then downloads and verifies that same pair before hash-installing locked dependencies and smoking it on Python 3.11–3.14 | Repository-level PTY/Skill/Image assets are source-distribution assets; Electron sources remain repository-checkout assets; publication is a separate explicitly authorized operation |
| SQLite RuntimeStore | Default local backend, file and in-memory targets | Ubuntu deterministic lanes plus the Windows 3.11 complete deterministic matrix | macOS filesystem ACL and locking behavior still needs a native release-gate run; platform CI is not a proof against hostile local administrators |
| PostgreSQL RuntimeStore | Optional `postgres` extra | Digest-pinned PostgreSQL 17.10 Bookworm service on Ubuntu/Python 3.11 | Other server versions and deployment TLS/auth topology are unvalidated operator gates |
| Core process/shell containment | POSIX process groups plus platform-specific fallbacks | Ubuntu security/runtime/provider lanes plus the Windows 3.11 complete deterministic matrix | macOS native behavior and Windows guarantees beyond the implemented fallbacks remain environment gates; the Windows PTY backend still has no Job Object or wall/CPU/RSS supervisor |
| Deno/TypeScript JIT | Deno required for real JIT; deterministic benchmark also has an explicit fake backend | Deno 2.9.4 on Ubuntu and Windows; real-Deno tests run when installed | Windows parent-death Job Object support is not implemented, and macOS native behavior remains a release gate |
| PTY Runtime Module | POSIX PTY plus optional Windows `pywinpty`/ConPTY session I/O; source checkout/source distribution only, not the core wheel | POSIX paths on Ubuntu; Windows 3.11 installs the `pty` extra before running the complete deterministic matrix | Current Windows backend has no Job Object, parent-death containment, or wall/CPU/RSS supervision; budgeted `SubprocessLimits` spawns fail closed. Native CI coverage does not expand those implementation guarantees |
| Typed Git provider | System Git 2.26+, fixed non-bare workspace repository; local operations, managed worktrees, patch Objects, existing remotes, and repository-local simulated PRs | Deterministic provider/security/runtime tests use temporary SHA-1/SHA-256 repositories and local bare remotes on Ubuntu, with the complete deterministic matrix also running on Windows 3.11; Shell/PTY/provenance hardening is parameterized | Credential-manager integrations and real HTTPS/OpenSSH authentication require environment-gated runs; GitHub/GitLab APIs are not implemented |
| JSON-RPC client | Registered HTTP endpoints only | Deterministic loopback/provider tests | Real network proxy/TLS/DNS policy is deployment-specific |
| MCP client | Tools-only v1 over Streamable HTTP or stdio | Deterministic primitive/provider tests plus the complete MCP SDK integration file on Ubuntu from the frozen `mcp` extra | Real remote-server identity, proxy, and TLS topology remain deployment gates; Resources/Prompts are not implemented |
| Real LLM | OpenAI Responses and OpenAI-compatible Chat profiles | Mock/action-selection paths only | Credentials and token-spending smoke are opt-in with `--run-real-llm`; run one scoped task/profile per release target |
| Data-label egress enforcement | Host Sink registry and a unified gate cover LLM, Human, JSON-RPC, MCP, typed Git, filesystem writes, Shell/PTY, and internal process handoff | Deterministic unit/runtime/security/provider/benchmark tests, including pre-provider denial and exact conditional release | The guarantee covers runtime-mediated payloads; trusted modules/providers, native child I/O, Sink re-forwarding, and direct store administration remain operator trust boundaries |

## GUI and API coverage

| Surface | Per-change CI | Environment gate |
| --- | --- | --- |
| React/Vitest | Ubuntu, Node 24 with the npm version supplied by that toolchain, source tests | The package declares Node `>=22.12.0` and npm `>=8`, but those lower compatibility bounds are not separate per-change CI jobs; browser accessibility and operator usability studies are not automated |
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
version, including Windows Python 3.11 with Deno and the PTY extra; GUI lane;
isolated AgentDojo Python 3.11/3.12 harness; PostgreSQL job;
runtime-safety release smoke; 100k external-effect recovery gate;
10k runtime-publication recovery gate; Ubuntu MCP SDK integration; and the
single release-artifact build plus clean-install smoke on Python 3.11–3.14 are
all necessary but not sufficient for a cross-platform release. The
`release-artifacts` build waits for the pre-build jobs listed in its `needs` and
then preserves the canonical pair; the downstream `release-artifact-smoke`
matrix downloads that pair and must also pass before the candidate is treated
as release-validated. `scripts/test_matrix.py --lane all` remains the local
aggregate command for the root environment; it neither reproduces those static,
service, GUI, AgentDojo, scale, and artifact gates nor repeats the equivalent
per-lane CI matrix. Before advertising a platform or provider configuration as
release-validated, record a fresh native run for the corresponding
environment-gate cells above. Do not copy counts or “remaining gates” from
`docs/prelaunch_hardening_report.md`; that file is bound to its historical
commit.

The checked-in Windows 3.11 job is CI evidence for the deterministic paths it
actually executes; it is not a claim of a separate local Windows run and cannot
establish unimplemented Job Object/resource-supervision guarantees or real Git
credential-manager interoperability. Those boundaries remain exactly as stated
in the platform rows above.

The checked-in Ubuntu/macOS host-filesystem-identity matrix is a configured CI
gate for the exact manifest v2 platform nodes it selects. This repository record
does not claim that a separate local macOS CI run was performed, and it does not
expand the broader macOS process, PTY, locking, ACL, or packaging boundaries
listed above.

The checked-in remote Actions are pinned to reviewed full commit SHAs, uv and
Deno use exact versions, and the PostgreSQL service image is bound to a reviewed
digest. The workflow builds one canonical wheel/source pair with frozen release
tools, binds every install smoke to its exact checksum manifest, and installs
artifact dependencies from hash-bearing root-lock exports. Hosted runner images
plus the Python and Node compatibility selectors can still move, and there is
no signed attestation, so this is not a bit-for-bit reproducible publication
chain. It remains a release-readiness gate and performs no PyPI upload or
external release mutation; publication still requires an explicit environment
and credential authorization.

When a new environment becomes CI-covered, update this matrix and
`docs/invariants.md` Known Test Gaps in the same change.
