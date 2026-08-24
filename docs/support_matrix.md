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
| Python | Package declares 3.11–3.14 | Ubuntu on 3.11 and 3.14 for Python lanes; Windows on 3.11 for the complete deterministic matrix split into per-lane jobs, with four deterministic file-weighted runtime shards, three provider shards, and two benchmark shards; a configured Ubuntu, macOS 14, and Windows Python 3.11 matrix runs the platform-marked host-filesystem-identity gate with skips forbidden | 3.12/3.13 are inside the declared range but are not separate root-runtime jobs; macOS behavior outside the targeted filesystem-identity gate remains an environment gate |
| Python release artifacts | Core-package wheel plus Python source distribution | CI builds one canonical wheel/source pair once with the frozen release tool group, rejects extra or non-regular files, records its exact checksum manifest, runs `twine check` and `check-wheel-contents`, then downloads and verifies that same pair before hash-installing locked dependencies and smoking it on Python 3.11–3.14 | Repository-level PTY/Skill/Image assets are source-distribution assets; Electron sources remain repository-checkout assets; publication is a separate explicitly authorized operation |
| SQLite RuntimeStore | Default `user` target at `~/.agent-libos/runtime/agent-libos.sqlite`, explicit external file targets, and explicit in-memory targets; persistent stores cannot overlap the effective local workspace | Ubuntu deterministic lanes plus the Windows 3.11 complete deterministic matrix; native Windows identity gate rejects non-canonical live/restorable persisted filesystem authority | macOS filesystem ACL and locking behavior still needs a native release-gate run; legacy project-root stores and Windows alias state require explicit offline operator handling; platform CI is not a proof against hostile local administrators |
| Local filesystem identity | Descriptor/handle-derived canonical authority, lock, evidence, and file-label keys; Darwin case/Unicode aliases and Windows case/DOS 8.3 aliases cannot split one Host entry | Native Ubuntu, macOS 14, and Windows Python 3.11 host-filesystem-identity matrix with `--fail-on-skip` | Network filesystems, unusual mount policies, and hostile local administrators remain deployment boundaries |
| PostgreSQL RuntimeStore | Optional `postgres` extra; Runtime startup accepts server major 17 only | Digest-pinned PostgreSQL 17.10 Bookworm service on Ubuntu/Python 3.11 | Other server major versions are rejected by the fixed schema catalog; deployment TLS/auth topology remains an operator gate |
| Core process/shell containment | POSIX process groups plus platform-specific fallbacks | Ubuntu security/runtime/provider lanes plus the Windows 3.11 complete deterministic matrix | macOS native behavior and Windows guarantees beyond the implemented fallbacks remain environment gates; the Windows PTY backend still has no Job Object or wall/CPU/RSS supervisor |
| Deno/TypeScript JIT | Deno required for real JIT; deterministic benchmark also has an explicit fake backend | Deno 2.9.5 on Ubuntu and Windows; real-Deno tests run when installed, including the native Windows `KILL_ON_JOB_CLOSE` containment path | Broader native macOS process behavior remains a release gate |
| PTY Runtime Module | POSIX PTY plus optional Windows `pywinpty`/ConPTY session I/O; source checkout/source distribution only, not the core wheel | POSIX paths on Ubuntu; Windows 3.11 installs the `pty` extra before running the complete deterministic matrix | Current Windows backend has no Job Object, parent-death containment, or wall/CPU/RSS supervision; budgeted `SubprocessLimits` spawns fail closed. Native CI coverage does not expand those implementation guarantees |
| Typed Git provider | System Git 2.26+, fixed non-bare workspace repository; local operations, managed worktrees, patch Objects, existing remotes, and repository-local simulated PRs | Deterministic provider/security/runtime tests use temporary SHA-1/SHA-256 repositories and local bare remotes on Ubuntu, with the complete deterministic matrix also running on Windows 3.11; Shell/PTY/provenance hardening is parameterized | Credential-manager integrations and real HTTPS/OpenSSH authentication require environment-gated runs; GitHub/GitLab APIs are not implemented |
| JSON-RPC client | Registered HTTP endpoints only | Deterministic loopback/provider tests | Real network proxy/TLS/DNS policy is deployment-specific |
| MCP client | Client-only Manifest v1/v2 governed Tools compatibility and exact-`2026-07-28` Manifest v3 over Streamable HTTP or stdio; see the detail below | Deterministic, real loopback, conformance, native-platform, and clean-artifact gates described below | Remote deployment topology remains an operator gate; excluded protocol/product surfaces are listed below |
| Real LLM | OpenAI Responses and OpenAI-compatible Chat profiles | Mock/action-selection paths only | Credentials and token-spending smoke are opt-in with `--run-real-llm`; run one scoped task/profile per release target |
| Semantic approval/data identification | Default-off Phase 2–4 plane: payload-free FlowGraph, closed Host hard denial, and exact-once canary authority for frozen low-risk actions under a static immutable policy epoch | Deterministic unit/runtime/security/provider tests cover strict models, monotonic labels, coverage, authority ceiling, shared request CAS, exact binding/epoch revocation, budgets, privacy, read-only HTTP/GUI, Host-only review import, and fake provider failures | Label writeback, declassification/endorsement, high-risk/write/network auto approval, remote policy mutation, and automatic cohort expansion are not implemented. Real classifier smoke is opt-in and never a safety oracle; production calibration and native environment evidence remain operator gates |
| Data-label egress enforcement | Host Sink registry and a unified gate cover LLM, Human, JSON-RPC, MCP, typed Git, filesystem writes, Shell/PTY, and internal process handoff | Deterministic unit/runtime/security/provider/benchmark tests, including pre-provider denial and exact conditional release | The guarantee covers runtime-mediated payloads; trusted modules/providers, native child I/O, Sink re-forwarding, and direct store administration remain operator trust boundaries |

### MCP validation detail

- **Implemented surface:** Client-only Manifest v1/v2 governed Tools
  compatibility plus exact-`2026-07-28` Manifest v3 Tools, Resources,
  Resource Templates, Prompts, Completion, MRTR,
  bounded subscriptions, Host-preconfigured OAuth, and an optional
  digest-pinned Tasks extension.
- **Checked evidence:** deterministic primitive/provider/security/DX tests;
  real loopback stdio, HTTP, and pinned-TLS OAuth/PKCE/Bearer Runtime gates;
  frozen Python/TypeScript SDK integration; reviewed fixed-upstream Tools and
  HTTP-schema, MRTR, and Host-pinned pre-registration/CIMD OAuth conformance on
  Ubuntu Python 3.11 and 3.14; native stdio/HTTP smokes on Windows and macOS;
  and clean-installed wheel/sdist coverage for modern read/call surfaces,
  subscriptions, OAuth, durable MRTR/Tasks reopen, and Store v6-to-v7 migration
  on Python 3.11–3.14.
- **Remaining gates and exclusions:** real remote identity, proxy, TLS, and
  OAuth topology remain deployment-specific. Conformance never enables DCR or
  promotes PRM/401 discovery into authority; authorization-code scenarios
  without a pre-Runtime Host issuer pin are reviewed-but-unavailable. Apps,
  Roots, Sampling, Logging, OTel product integration, DCR,
  client-credentials/enterprise-managed/DPoP/workload-identity OAuth,
  `2025-03-26` OAuth backcompat, standalone SSE, and an MCP server surface are
  excluded.

### MCP OAuth default credential-backend support

The default `SystemKeyringMcpCredentialBroker` is intentionally narrower than
the Python keyring plugin ecosystem. It accepts only `keyring==25.7.0` and the
following exact distribution-owned implementation classes after verifying the
official source path and reviewed file digest:

| OS facility | Accepted exact backend identity |
| --- | --- |
| macOS Keychain | `keyring.backends.macOS.Keyring` |
| Windows Credential Manager | `keyring.backends.Windows.WinVaultKeyring` |
| Linux Secret Service (SecretStorage) | `keyring.backends.SecretService.Keyring` |
| Linux Secret Service (libsecret) | `keyring.backends.libsecret.Keyring` |
| Linux KWallet 5 | `keyring.backends.kwallet.DBusKeyring` |
| Linux KWallet 4 | `keyring.backends.kwallet.DBusKeyringKWallet4` |

`keyring.backends.chainer.ChainerBackend`, every plugin/third-party backend,
subclasses and identity lookalikes, and any unreviewed keyring version fail
closed. Selecting a configured exact backend is still not proof that the OS
service is unlocked or usable; each actual read/write/delete also fails closed.
Tests attest the real packaged class objects via `object.__new__` and fake only
the module-level storage calls, so deterministic CI never reads or writes a
developer's real keychain. Deployments using a different reviewed facility
must explicitly inject a complete Host-owned `McpCredentialBroker`; an upgrade
to keyring requires a new implementation/source-digest review and allowlist.

## GUI and API coverage

| Surface | Per-change CI | Environment gate |
| --- | --- | --- |
| React/Vitest | Ubuntu, Node 24 LTS with the npm version supplied by that toolchain, source tests | The package declares Node `^24.15.0 || >=26.0.0` and npm `>=11`; Node 26 Current satisfies the engine contract but is not a separate per-change CI job |
| Chromium GUI end-to-end | Ubuntu, version-matched Playwright Chromium, desktop/mobile Provider-trace journeys, real GUI HTTP/SSE authentication, loopback fake Provider retries/fallbacks, bounded pagination/content reads, retention races, keyboard operation, and feature-scoped axe checks | Other browser engines, native desktop accessibility APIs, and operator usability studies remain environment gates |
| Web and Electron TypeScript | Typecheck and production build on Ubuntu; manual native internal workflow builds Electron 43.2.0 + CPython 3.11.15/PyInstaller 6.21.0 + Deno 2.9.5 packages on macOS arm64, Windows x64, and Ubuntu 24.04 x64, with frozen-backend, renderer/preload, reopen, installer, SBOM/license, and artifact scans | Desktop packages are `internal-unsigned`: macOS is ad-hoc signed and not notarized; Windows/Linux are unsigned. Public distribution, code-signing identity management, Apple notarization, auto-update, and native accessibility/usability evidence remain separate gates |
| Python GUI HTTP/SSE server | Providers lane exercises auth, route validation, bounded event windows, shutdown, CORS, schema-v3 snapshots, Durable Task Run pagination/mutations, stable 409 conflicts, confirmation gates, and redacted monotonic SSE summaries | Native desktop process lifecycle remains platform-specific |
| Semantic GUI panel | Same-build strict decoder/component/client tests cover GET-only schema-v3 status, FlowGraph, settlements, policy/control history, health/metrics, bounded keyset pages, process filters, nullable review-rate denominators, and payload rejection | The panel has no policy/control or review-import mutation. Usability/calibration interpretation and native accessibility review remain operator/environment gates; semantic history is not part of the snapshot |
| Headless Electron main-process smoke | Development path remains outside the default GUI lane; native internal workflow runs the packaged path | Run `AGENT_LIBOS_GUI_SMOKE=1 npm --prefix gui run electron:dev` for source development, or `scripts/smoke_desktop_bundle.py desktop-dist` after a native package build |
| Production custom-protocol/installer smoke | Manual `desktop-internal.yml` runs the unpacked app twice and verifies renderer origin, preload, authenticated health, graceful shutdown, SQLite reopen, bundled Deno JIT, frozen MCP surfaces, and each native installer/portable form | It is an internal environment gate, not proof of formal signing/notarization or broad end-user compatibility outside macOS arm64, Windows x64, and Ubuntu 24.04/glibc x64 |
| Local GUI API compatibility | Server and renderer tests cover the matching checkout | The unversioned `/api` surface is an internal same-build contract, not a stable third-party REST API |

## Evaluation coverage

| Suite | Default evidence | Boundary |
| --- | --- | --- |
| `benchmarks/runtime_safety` | 33 deterministic schema-v1 tasks, including data-label exfiltration, Git worktree containment, malicious config, remote misuse, patch lineage, and Semantic Shadow authority injection; fail-closed metrics and provenance-bearing CLI metadata | Bounded runtime-safety workload, not a comprehensive production evaluation or formal proof; Git network tasks use controlled local state rather than a hosted provider |
| `benchmarks/practical_agent_workflows` | Exactly two labels: `native-live` and `modeled`; native has no modeled fallback | Checked-in scenarios do not imply a real GitHub/provider integration |
| `benchmarks/external_effect_recovery` | 100k-record `ci` profile on each change; one-million-record `million` profile in the manual/nightly workflow | Structural paging/index/convergence checks are gates; elapsed times are diagnostic, not SLAs |
| `benchmarks/runtime_publication_recovery` | 10k terminal publications with 1,001 unreconciled rows in the only named profile, `ci` | No one-million-publication profile is currently implemented; custom sizes are explicit CLI overrides |
| `benchmarks/durable_task_runs` crash matrix | Per-change deterministic release gate covers all six commit/effect barriers plus an unpaired durable provider result after the safe point; it requires safe convergence, provider-ledger idempotency, and a stable second reopen | The independently fsynced JSONL provider ledger is test evidence, not a production provider; timing is diagnostic only |
| `benchmarks/durable_task_runs` recovery scale | Per-change `ci` profile seeds 100,000 Task Runs, including 1,000 recoverable Runs, with page size 500; query/index bounds, complete ordered convergence, bounded diagnostics, and zero model dispatch are hard gates | The named profile is a bounded structural gate rather than a startup-latency SLA or proof for arbitrary larger histories |
| `experiments/agentdojo` | Deterministic harness tests run in a separate Ubuntu matrix on Python 3.11 and 3.12 from the subproject's own frozen lock; `release-artifacts` waits for both matrix runs | The root lock and `scripts/test_matrix.py` do not include this environment. CI does not make provider calls or claim real-model AgentDojo utility/security results; those runs remain explicit credential/token gates |
| Prompt-cache paired release qualification | Matching `legacy_v1` and `cache_optimized_v2` arms require at least two provider/model pairs, strict completion/security/leak evidence, three repetitions and six workflows per provider, token-reduction and hit-rate thresholds, and non-increasing known-price cost per successful task | Explicit credential/token environment gate required before changing the default layout; it is not a per-change CI job, and an ordinary CI receipt is not evidence that it passed |
| Live Durable Task Run release gate | Three real-LLM repository-maintenance runs, three real-LLM/Chromium customer-operation runs, and three repetitions each of real-LLM research and analysis; the canonical combiner requires safety 12/12, utility at least 10/12, every family gate, and matching stable clean-source provenance | Explicit credential, token, and browser environment gate; deterministic injections can test the evaluator but cannot satisfy the release gate |

## Release-gate policy

The static compile, architecture/blocking-work, protected-operation, and
invariant/collection checks; per-lane deterministic matrix on every CI Python
version, including Windows Python 3.11 with Deno and the PTY extra; GUI lane;
isolated AgentDojo Python 3.11/3.12 harness; PostgreSQL job;
runtime-safety release smoke; 100k external-effect recovery gate;
10k runtime-publication recovery gate; Durable TaskRun six-barrier crash and
100k-history recovery-scale gates; Ubuntu MCP SDK integration on Python 3.11
and 3.14; and the
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
prior revisions of the [historical prelaunch notice](prelaunch_hardening_report.md);
that path now contains
only a retirement notice for the old commit-bound report.

The checked-in Windows 3.11 jobs are CI evidence for the deterministic paths
they actually execute; they are not a claim of a separate local Windows run and
cannot establish the unimplemented ConPTY Job Object/resource-supervision
guarantees or real Git credential-manager interoperability. Those boundaries
remain exactly as stated in the platform rows above.

The checked-in Ubuntu/macOS/Windows host-filesystem-identity matrix is a configured CI
gate for the exact manifest v2 platform nodes it selects. This repository record
does not claim that separate local macOS or Windows CI runs were performed, and
it does not expand the broader macOS or Windows process, PTY, locking, ACL, or
packaging boundaries listed above.

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
the [Runtime Invariants: Known Test Gaps](invariants.md#known-test-gaps) in the
same change.
