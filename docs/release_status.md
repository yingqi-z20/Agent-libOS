# Agent libOS 0.3.4 Status

Agent libOS 0.3.4 is release-ready for the core Python runtime scope exercised
by the checked-in CI workflow and defined in `docs/support_matrix.md`. This is
not a claim that every platform, desktop package, or real external-provider
configuration has been release-validated.

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
  and index-backed. LLM/effect payload retention is disabled by default and is
  explicit, monotonic, CAS-protected, transactionally audited, and recovery-safe.
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
  All 86 declared invariants resolve against 3,568 collected pytest nodes.
- The per-lane deterministic matrix passes all selected tests. PostgreSQL,
  MCP, and real-LLM coverage remains in dedicated or explicit gates, while
  platform-specific skips stay documented and real Deno runs by default when
  installed.
- The PostgreSQL CI job runs the complete `postgres` marker gate against
  PostgreSQL 17 on Python 3.11 and permits no skips; the current collection
  selects 259 tests. This is a service-backed CI gate, not evidence that an
  arbitrary local PostgreSQL configuration has been validated.
- The GUI lane passes all 19 Vitest files and 67 tests, TypeScript type checking,
  and the production frontend build.
- The runtime-safety release smoke passes all three selected tasks with complete
  audit evidence, no unauthorized effects, and no false denials. Four focused
  Git tasks additionally pass for managed-checkout containment, malicious
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
- The Python 3.11 `release-artifacts` CI job builds the 0.3.4 wheel and source
  distribution, checks content and metadata, clean-installs each artifact,
  checks dependency consistency, exercises both entrypoints plus the
  deterministic demo, and preserves those exact validated distributions for
  release use.

## Supported release scope

- Python 3.11 through 3.14 is the declared package range. Per-change CI runs the
  Python lanes on 3.11 and 3.14; local clean-install evidence above is Python
  3.11 and does not replace the CI version matrix.
- SQLite and PostgreSQL implement the covered RuntimeStore contract. A 0.2 store
  or artifact is rejected before mutation and remains readable only with the
  archived 0.2 release.
- The Python wheel contains the core `agent_libos` package and its two console
  entrypoints. Repository-level PTY module, example Skill and Image assets,
  benchmarks, tests, and documentation are source-distribution or checkout
  assets, as documented in the README.
- Git is a Python Runtime/model-tool surface only. It requires an existing
  non-bare workspace repository and system Git 2.26 or newer; unavailable Git
  fails individual calls without preventing Runtime startup. Host-configured
  remotes are the only first-class Git network exception. There is no Git CLI,
  GUI/HTTP surface, or real GitHub/GitLab API integration in 0.3.4.

## Remaining environment gates and non-blocking debt

- Native macOS and Windows process containment, filesystem locking, and PTY
  behavior require platform runs before those configurations are advertised as
  release-validated.
- Electron packaging, signing, notarization, and native desktop lifecycle are
  separate environment gates; the source GUI and Python GUI server are covered.
- Real MCP SDK/server, real LLM, network proxy and TLS topology, and provider
  credentials remain explicit opt-in gates. Deterministic or loopback evidence
  is not presented as real-provider evidence.
- Real Git HTTPS/OpenSSH authentication, Host credential-manager variations,
  and native Windows Git path/locking behavior are environment gates. Local
  bare-remote tests do not establish hosted-provider interoperability.
- Payload retention is an operator-triggered maintenance policy, not an implicit
  startup behavior. Million-record benchmark timing remains informational rather
  than a latency guarantee.
