# Agent libOS 1.0.0 Status

Agent libOS 1.0.0 is a release candidate for the core Python runtime scope
defined in `docs/support_matrix.md`. Release-ready status for any source tree is
conditional on that exact tree passing the checked-in CI workflow; local
deterministic results do not substitute for its Python-version, PostgreSQL, and
artifact gates. This is not a claim that every platform, desktop package, or
real external-provider configuration has been release-validated.

## Closed release blockers and P1 architecture debt

- `Runtime.git` is a typed system-Git provider pinned to the workspace root.
  It validates repository/config/executable identity, uses state-token CAS and
  a cross-process lock, requires affected filesystem plus Git authority, and
  separates local mutation, fetch, push, and simulated-PR evidence. Managed
  checkouts, immutable patch Objects, existing Host-configured remotes, and
  repository-local simulated PRs are implemented without arbitrary Git argv,
  model URLs, executable hooks/helpers, or a Git hosting dependency.
- Publication-owned launch and exec artifacts use exact durable receipts,
  atomic state transitions, idempotent compensation, recovery claims, and a
  fail-closed recovery fence. Recovery precedes global JIT rehydration.
- Authority-changing operations use one transaction boundary that revalidates
  allow and deny rules, authority revision, resource generation, finite-use
  reservations, business state, and evidence settlement together.
- Runtime public mutations are lifecycle-admitted. Worker mutations require a
  full process execution token, Host mutations use explicit status and revision
  fences, and an active process-exec publication rejects non-owner Host writes.
  Trusted emergency controls use one exact, transaction-scoped takeover; absent
  lease tuples retain the ordinary controlled CAS and partial tuples fail
  closed. Human provider work drains through the runtime blocking-work supervisor
  before storage closes.
- Ordinary shutdown atomically claims its exact store guard before the final
  backend release. Async close is off-loop and cancellation-drained; after the
  release point, warnings remain on idempotent readback and a leader's local
  control-flow exception is not replayed to followers. An already-released
  backend still permits graph teardown and stale exact-guard cleanup, with the
  missing durable shutdown evidence reported as an in-memory warning.
- A recovery-required fence is monotonic for its Runtime instance. Ordinary
  `close()`/`shutdown()` calls remain fail closed and retain the diagnostic store;
  they do not emit shutdown evidence, run finalizers, or release the backend
  lease. The explicit `Runtime.release_recovery_diagnostics()` (or awaited
  `arelease_recovery_diagnostics()`) handoff is admitted only for a genuine fence
  with no active admission or shutdown attempt. It writes no durable evidence,
  skips ordinary finalizers, runs only explicitly registered no-write recovery
  cleanup, stops transient workers, and atomically releases the exact store
  guard and backend lease. A failure or cancellation before backend ownership is
  released leaves the handoff retryable. Once ownership is irreversibly released,
  the old lifecycle becomes closed even if close reports a warning or the caller
  was cancelled; warnings remain available through idempotent release readback
  and control-flow interruption is propagated. Opening the same target then
  creates a new Runtime and performs authoritative startup recovery.
- Restore, fork, kill, and exec maintain durable revision and execution
  high-water identities. Snapshot concurrency metadata cannot recreate an old
  revision or execution token.
- MCP and JSON-RPC boundaries reject expired budgets before dispatch, settle
  unknown exchange costs conservatively, and expose only stable public error
  envelopes across primitive, JIT, Deno, ToolResult, and durable-result
  boundaries.
- A process exec operation covers the complete snapshot, publication, process,
  tool, boot, Skill, evidence, and compensation orchestration. Its terminal
  status is written only with the matching publication result. Host and worker
  admission use a complete concurrency CAS, and post-publication acknowledgement
  failures honor exact terminal receipts without replaying snapshots over a
  newer process epoch. Rollback restore also CASes the publication's exact
  admission tuple; a concurrent trusted pause or kill remains authoritative and
  moves unresolved compensation to fail-closed recovery instead of being
  overwritten.
- Checkpoint restore publishes durable, operation-bound reconciliation work for
  volatile Object payload delivery, image state, JIT sources, pruning, and
  Object finalizers. Hash-anchored payload replay precedes the general
  missing-payload sweep and remains retryable through fallible startup. Exact
  unchanged rows are rehydrated; newer rows with the same immutable creation
  identity, including ownership transfers, are never overwritten and converge
  through ordinary volatile-payload cleanup. Delivery uses a durable,
  owner-bound `preparing` attempt and hard-bounded `pending` -> `confirmed` ->
  `completed` pages. Operation truth is repaired independently before the exact
  attempt ACK and lifecycle OPEN share one outer database transaction. An
  ambiguous database completion error is resolved by typed exact-state readback:
  only `preparing` is compensable,
  while `acked` opens without replay and every mismatch fails closed. Phase and
  delivery receipts, claim leases, retry classification, attempt limits, exact
  legacy version-1 transcripts, and manual fail-closed state survive reopen.
- Process waits and outcomes are strict tagged values. Normal orchestration uses
  one transition service; the only exceptions are explicitly typed atomic
  execution/restore CAS primitives. Generic patch/update APIs cannot write
  semantic state, exec-epoch publication requires its exact publication-bound
  admission token and final pre-publication phase, and `status_message`
  is only a compatibility projection.
- Process/Resource/Publication, Snapshot/Checkpoint, and Module publication
  persistence use explicit typed repositories. Payload retention has its own
  typed scan/CAS repository; migrated runtime services are protected from raw
  SQL and generic-table regressions by an AST ratchet.
- SQLite and PostgreSQL use independent connection, dialect, and lease adapters
  over the same typed repository implementation. The supported product boundary
  is one writable Runtime per database or schema.
- Authority manifests distinguish `None` (unrestricted effect ceiling) from an
  explicit empty list (deny all). JSON-RPC and MCP approvals bind an immutable
  registry-spec digest and monotonic durable generation, revalidated inside the
  effect transaction before every provider phase, including first registration
  and reopen. MCP live tool-list refreshes use the same binding.
- Failed Runtime assembly has separate sync and async cleanup paths. Async hosts
  use `await Runtime.aopen()` so loop-affine teardown drains on the caller loop;
  sync open fails before opening storage when called from an active event loop.
  Async `aopen`, `afrom_store`, and `aassemble_existing` atomically install an
  identity-only store assembly reservation before worker handoff. Non-claimant
  lock, transaction, and query scopes fail fast until the exact worker claim
  exits, and scheduling, cancellation, decision-error, and failure paths
  compare-and-release the same token, eliminating the probe-to-worker RLock
  deadlock window.
  Incomplete teardown returns a public, retriable cleanup handle instead of
  closing storage underneath a live component. Both paths use the same
  `allocate_unassembled` subclass contract, validated before an owned store is
  opened, so a custom constructor cannot fail after starting the Runtime graph.
  Cached LLM clients are retired by exact object identity only after close
  succeeds; failed or interrupted closes remain owned by the cleanup handle for
  a later sync or async retry. Builder-owned failed opens atomically exchange
  the failed lifecycle guard for an exact close reservation: successors cannot
  assemble while it is pending, stale handles cannot close a successor, and
  async close is drained off-loop before cancellation is re-raised.
  If a cancelled async assembly has already reached `OPEN`, an exception group
  publishes the same extractable handle with
  `cleanup_kind=OPEN_RUNTIME_SHUTDOWN`; its sync or async release retries normal
  Runtime shutdown (not failed-assembly teardown), preserves ownership across
  exceptions, incomplete results, and renewed cancellation, and is idempotent
  after release.
- External-effect startup recovery is state-filtered, keyset-paged, hard-bounded,
  and index-backed. The payload-retention reduction/maintenance policy is
  disabled by default; when enabled it is explicit, monotonic, CAS-protected,
  transactionally audited, and recovery-safe. This is separate from the default
  `llm.persist_full_io: true` setting for newly recorded LLM calls.
- Provider-usage reservation recovery is startup-lease-only, status-first
  keyset-paged, and hard-bounded. Ambiguous reservations settle and charge
  atomically, overage convergence continues across the complete backlog, and
  Runtime diagnostics expose a bounded summary instead of every recovered ID.
- Every mutation-capable startup recovery entry is guarded by the opaque
  lifecycle recovery lease before its first read. Manual calls from an open
  runtime produce no claims, compensation, audit, events, or state changes.
  Prepared/provider-effect, capability/resource-reservation, stale-operation,
  and stale-execution diagnostics expose exact totals with bounded samples.
- Volatile Object payload cleanup and all three ObjectTask reopen scans are
  startup-lease-only, keyset-paged, index-backed, and expose bounded typed
  summaries. Object release precedes missing-result repair, eliminating dangling
  succeeded-task result references without constructor-time history scans.
- Stale finite-use capability reservations are abandoned only after prepared
  protected effects restore their exact linked reservations; cleanup is
  status-first keyset-paged and no longer parses all pending effect metadata.
- Stale process execution recovery is status/PID indexed and page-bounded;
  each transition commits its concurrency high-water and audit/event evidence
  in the same transaction.
- Runtime-publication startup recovery is exact-kind/state/marker filtered,
  keyset-paged, hard-bounded, and index-backed. Terminal launch/exec and
  committed checkpoint-restore operation repair skip durable completed markers;
  checkpoint plans are receipt-digest anchored before any recovery/finalizer
  replay, while failed/manual restores remain forward-recovery inputs. Orphan
  `CREATED` detection uses a bounded indexed anti-join rather than full-history
  materialization.

## Validation state

- Compilation, architecture/blocking-work checks, protected-operation coverage,
  release-contract checks, whitespace checks, and the invariant manifest pass.
  The checker resolves every declared invariant against the current pytest
  collection.
- The per-lane deterministic matrix passes all selected tests. PostgreSQL
  service coverage and the complete MCP SDK integration file run in dedicated
  gates; real remote MCP deployment and real-LLM coverage remain explicit
  environment gates. Deterministic mocked MCP coverage is part of the normal
  matrix. Platform-specific skips stay documented and real Deno runs by default
  when installed.
- The PostgreSQL CI job runs the complete `postgres` marker gate against a
  digest-pinned PostgreSQL 17.10 Bookworm image on Python 3.11 and permits no
  skips. This is a service-backed CI gate, not evidence that an arbitrary local
  PostgreSQL configuration has been validated.
- The isolated AgentDojo harness is a required CI matrix on Python 3.11 and
  3.12, using the subproject's own frozen environment; `release-artifacts`
  waits for both entries. This gate covers deterministic harness behavior only
  and makes no real-model AgentDojo utility or security claim.
- The GUI lane passes the complete checked-in Vitest suite, TypeScript type
  checking, and the production frontend build. Exact file and test counts are
  intentionally left to the CI receipt because they change as coverage grows.
- The runtime-safety release workflow gates all 32 checked-in deterministic
  tasks with complete audit evidence, no unauthorized effects, and no false
  denials. Its focused Git tasks cover managed-checkout containment, malicious
  repository config, remote misuse, and patch-label lineage.
- The practical-workflow evaluation passes three `native-live` scenarios and 80
  modeled scenarios while retaining their distinct evidence labels and using no
  modeled fallback for native scenarios.
- The 100k and one-million external-effect recovery profiles both use the
  matching composite index and perform work proportional to pending pages. Their
  elapsed times are diagnostic measurements, not release SLAs. Store
  read-only preflight, main initialization, and Runtime handler windows are all
  trace-observed with exact SELECT/DML ledgers; the benchmark executes the real
  prepared-effect handler, verifies every surviving seeded identity and final
  marker, and checks page-bounded Runtime diagnostics.
- The runtime-publication reconciliation `ci` profile executes the real startup
  handler over a 10k history with 1,001 incomplete records. Operation repair
  and the five-stage checkpoint payload handshake stay keyset/page bounded;
  page, attempt, ACK guard, control-state, operation-reconciliation, and
  invalid-domain plans use their exact indexes.
  The profile validates exact publication/operation convergence, attempt
  terminalization, and zero remaining `preparing` work without materializing
  the historical ID set.
- The `release-artifacts` CI job builds one canonical 1.0.0 wheel/source pair,
  rejects extra or non-regular output, and records an exact checksum manifest.
  Python 3.11 through 3.14 smoke jobs download and verify that same pair, install
  hash-checked dependencies exported from the root lock, and then install the
  artifact without dependency re-resolution. The source build uses the frozen
  release backend without build isolation. Each job checks dependency
  consistency and exercises all three installed console entrypoints plus the
  deterministic demo. The build waits for its declared pre-build gates; these
  smoke jobs run afterward, and the candidate is not release-validated until
  the full downstream matrix succeeds. No workflow publishes or pushes
  candidate distributions.

## Unarchived real-LLM observation

A local operator reported running a scoped GUI workflow against a custom LLM
endpoint to read a policy and CSV, compute a report, emit `human_output`, and
exit. No provenance-bearing report for that run is checked in with the source
revision, model/profile identity, redacted configuration, environment, and raw
test outcome needed to reproduce or compare it. It is therefore an unarchived
observation, not Agent libOS 1.0.0 release evidence, and supports no call-count,
token-count, approval-count, latency, or serial-versus-parallel claim. Promote a
future rerun only after using a documented opt-in real-model gate and preserving
its reproducible report outside this status summary.

## Supported release scope

- Python 3.11 through 3.14 is the declared package range. Per-change CI runs the
  Python lanes on Ubuntu 3.11 and 3.14, and the complete deterministic matrix
  in per-lane jobs on Windows 3.11, with the large runtime lane split into two
  deterministic file-weighted shards. This records checked-in CI coverage, not a
  separate local Windows run. The canonical release build job uses Python 3.11, while
  downstream artifact-smoke jobs cover Python 3.11 through 3.14; neither claim
  substitutes for evidence from an unrecorded local clean install.
- The GUI package declares Node `>=22.12.0` and npm `>=8`. Per-change CI checks
  Node 24 with its supplied npm version; it does not separately validate every
  lower declared Node or npm version.
- The release workflow configures a separate Ubuntu/macOS 14 Python 3.11 matrix
  for the manifest v2 host-filesystem-identity platform nodes. Each shard uses
  its platform marker with `--fail-on-skip`, and canonical release artifacts
  depend on that job. This is a configured CI gate, not a claim that a separate
  local macOS CI run was performed.
- SQLite and PostgreSQL implement the covered RuntimeStore contract. A 0.2 store
  or artifact is rejected before mutation and remains readable only with the
  archived 0.2 release.
- The Python wheel contains the core `agent_libos` package and its three console
  entrypoints: `agent-libos`, `agent-libos-gui-server`, and the explicit offline
  `agent-libos-migrate-tool-groups` migration command. Repository-level PTY
  module, example Skill and Image assets, benchmarks, tests, and documentation
  are source-distribution or checkout assets, as documented in the README.
- Git is a Python Runtime/model-tool surface only. It requires an existing
  non-bare workspace repository and system Git 2.26 or newer; unavailable Git
  fails individual calls without preventing Runtime startup. Host-configured
  remotes are the only first-class Git network exception. There is no Git CLI,
  GUI/HTTP surface, or real GitHub/GitLab API integration in 1.0.0.

## Remaining environment gates and non-blocking debt

- Native macOS process containment, filesystem locking, and PTY behavior outside
  the targeted configured host-filesystem-identity gate still require platform
  release-gate runs. The Windows 3.11 CI jobs exercise the
  implemented deterministic process, filesystem, Git, and `pywinpty`/ConPTY
  fallback paths, plus Deno's `KILL_ON_JOB_CLOSE` parent-death containment. It
  does not provide guarantees the ConPTY backend does not have: ConPTY has no
  Job Object parent-death containment or wall/CPU/RSS supervisor, and budgeted
  `SubprocessLimits` spawns fail closed.
- Native Electron desktop lifecycle and the production-build custom-protocol
  BrowserWindow smoke are separate environment gates; the source GUI and Python
  GUI server are covered. Installer packaging, signing, and notarization are
  not configured.
- Remote MCP server identity, real LLM, network proxy and TLS topology, and
  provider credentials remain explicit opt-in gates. The Ubuntu MCP SDK gate
  and deterministic loopback evidence are not presented as deployment-specific
  real-provider evidence.
- Real Git HTTPS/OpenSSH authentication and Host credential-manager variations
  remain environment gates on every platform. Deterministic local Git
  path/locking tests run in Windows CI, but temporary repositories and local
  bare remotes do not establish hosted-provider or real-credential
  interoperability.
- Payload retention is an operator-triggered maintenance policy, not an implicit
  startup behavior. Million-record benchmark timing remains informational rather
  than a latency guarantee.
