# Agent libOS 1.5.2 Status

Agent libOS 1.5.2 is the current release line for the core Python runtime scope
defined in the [support matrix](support_matrix.md). Release status for any source tree is
conditional on that exact tree passing the checked-in CI workflow; local
deterministic results do not substitute for its Python-version, PostgreSQL, and
artifact gates. This is not a claim that every platform, desktop package, or
real external-provider configuration has been release-validated.

This page records the release scope and the checked-in validation contract;
it is not itself a CI receipt. A release-pass claim must be bound to all of the
following immutable locators:

- the exact source commit locator;
- the CI workflow run locator and the required job locators for that commit;
- the canonical wheel, source archive, and checksum-manifest artifact locators
  produced by that run.

Without that complete binding, wording below such as “requires”, “gates”, or
“checks” describes the workflow contract only, not an observed pass for a
checkout or release artifact.

## In this guide

- [Review implemented release safeguards](#implemented-release-safeguards)
- [Understand the CI receipt contract](#validation-contract-ci-receipt-required)
- [Interpret the unarchived real-LLM observation](#unarchived-real-llm-observation)
- [Confirm supported release scope](#supported-release-scope)
- [Track remaining gates and non-blocking debt](#remaining-environment-gates-and-non-blocking-debt)
- Return to the [documentation home](index.md).

## Implemented release safeguards

- A manual native workflow can build self-contained 1.5.2 internal desktop
  packages for macOS arm64 (DMG/ZIP), Windows x64 (NSIS/ZIP), and Ubuntu
  24.04/glibc x64 (AppImage/tar.gz). Each set carries checksums, a CycloneDX
  SBOM, component inventory, and third-party notices and must pass frozen
  backend, bundled Deno, renderer/preload, persistent reopen, MCP, and
  installer smokes. These packages remain `internal-unsigned`; the macOS app
  is ad-hoc signed but not notarized, and no public download/update claim is
  made without a bound three-platform workflow receipt.

- Durable Task Runs are a first-class, versioned Host supervision boundary for
  one root AgentProcess tree. They persist requirements, idempotent command
  receipts, append-only ledger links, and locally integrity-bound resume points;
  they do not introduce a generic workflow DSL or distributed scheduler.
- RuntimeStore schema v7 is the only store format accepted by ordinary 1.5.2
  startup. A canonical v6 store has one explicit offline, digest-bound migration
  path to v7; v5 must first migrate to v6 and v4 must first migrate to v5.
  Runtime startup never migrates a store. Schema v3 remains archive-only under
  1.0.1, and malformed/older stores have no read-only bridge or dual-schema mode.
- Semantic approval and ingress classification remain default-off. Shadow adds
  payload-free FlowGraph and assessment evidence without changing authority;
  `enforce_deny` can settle only the closed Host hard-deny set, and
  `canary_auto` can issue only exact, short-lived, nondelegable, one-use
  Capabilities for the frozen low-risk action catalog under an active,
  immutable static Host policy epoch. Classifier output is only a veto/escalation signal and is
  never an allow predicate or safety oracle. Semantic HTTP and GUI surfaces
  remain read-only; there is no remotely reachable policy activation or
  revocation endpoint.
- Useful Task Run restart recovery requires explicit Host opt-in to bounded
  plaintext payload persistence, which is disabled by default. The default
  `purge_on_terminal` policy hash-reduces readable Run-owned content before a
  terminal status, including linked Human prompt/response/decision bodies and
  terminal provider metadata/receipt bodies. It retains Human request
  id/type/status/timestamps/digests/audit links and effect
  identity/state/digests/causal links;
  `permanent` is Host/admin-only and remains eligible for a later explicit,
  audited Host purge. Neither policy is at-rest encryption or secure backup
  erasure.
- Task Run startup uses the exclusive active-store lease and monotonic Runtime
  epoch to fence stale claims and commits. Unknown/dispatched effects, binding
  drift, missing payloads, and reopened active ObjectTasks block in
  `needs_attention`; they are not replayed automatically.
- Runtime and Durable Task Runs durably reserve one logical LLM call and the
  configured per-call token envelope before Provider dispatch. Known usage
  settles exactly, certified non-start releases, and unknown outcomes charge
  the aggregate maximum before model-selected tools run. No UI or documentation
  may present this logical-call boundary as an exact physical-request,
  Provider-billing, currency, or monetary-spend cap.
- The built-in OpenAI-compatible client records a bounded terminal Provider
  attempt trace for each logical call, including explicit transport retries and
  protocol fallbacks. The loopback GUI exposes summary-only snapshot/SSE state
  and selected-process, on-demand retained content; custom-client traces are
  explicitly incomplete and no view is described as hidden chain of thought.

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
- MCP uses Python SDK v2 while preserving Manifest v1 as a legacy-wire
  compatibility contract. Manifests v1/v2 remain governed Tools-only
  compatibility surfaces. Manifest v2 explicitly selects `legacy`, `auto`, or
  modern `2026-07-28`; Tool discovery, pagination, live validation, and call
  phases share one absolute deadline, cumulative byte reservation, registry
  fence, and bounded receipts. Ambiguous failures never masquerade as legacy
  fallback, and unsupported server capabilities never grant authority.
- The MCP product surface remains client-only. Manifest v1/v2 retains its
  governed Tools compatibility contract; exact-`2026-07-28` Manifest v3 adds
  governed Tools with the closed modern result union plus Host-governed
  Resources, Resource Templates, Prompts, Completion, MRTR continuations,
  bounded subscriptions, Host-preconfigured OAuth, and a digest-pinned Tasks
  extension. A typed MRTR continuation is available only to Manifest v3 and
  never replays the initial Tool call. MCP Apps, Roots, Sampling, Logging,
  OpenTelemetry product integration, OAuth Dynamic Client Registration,
  deprecated standalone SSE, and an MCP server surface remain out of scope.
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

## Validation contract (CI receipt required)

- The checked-in workflow requires compilation, architecture/blocking-work
  checks, protected-operation coverage, release-contract checks, whitespace
  checks, and the invariant manifest to pass. The checker must resolve every
  declared invariant against the current pytest collection.
- The workflow requires the per-lane deterministic matrix to pass all selected
  tests. The complete MCP transport, adapter, and SDK integration files plus
  the reviewed fixed-upstream Tools/HTTP-schema scenarios (including the
  Resource/Prompt request-header branches), MRTR, and OAuth
  pre-registration/CIMD client conformance scenarios run
  without an expected-failure baseline on Python 3.11 and 3.14 in a dedicated
  matrix. For the two OAuth scenarios, a checked harness gets both fixture
  origins directly from the pinned scenario object before Runtime construction,
  pins the resource, issuer, and metadata URLs, and disables DCR. For every
  upstream scenario the gate persists only check ID/status, bounded spec
  references, and a deterministic digest; it drops raw names, descriptions,
  timestamps, logs, protocol details, and authorization details. The
  remaining upstream OAuth runner contracts do not supply the Host-pinned
  expected issuer required to establish authority, so those exact reviewed
  scenarios are reported as unavailable rather than counted as passes. The
  pinned suite's older OAuth backcompat and optional client-credentials,
  enterprise-managed-authorization, DPoP, and workload-identity scenarios are
  separately reported as reviewed product exclusions; the same
  matrix also runs a real TLS/PKCE/Bearer Runtime regression with pinned
  issuer, resource, and endpoints and verifies that OAuth secrets do not enter
  persisted or public evidence.
  PostgreSQL service coverage is a separate Python 3.11 gate. Real remote MCP
  deployment and real-LLM coverage remain explicit environment gates.
  Deterministic mocked MCP coverage is part of the normal matrix.
  Platform-specific skips stay documented and real Deno runs by default when
  installed.
- The Durable Task Run gate requires fresh schema-v7 SQLite/PostgreSQL shape,
  older-store zero-write refusal, revision/command conflicts, stale Runtime-epoch
  fencing, plaintext opt-in and terminal purge, unknown-effect/ObjectTask
  blocking, checkpoint intersection refusal, and GUI snapshot schema-v3 behavior. Its
  crash harness exercises `os._exit` and `SIGKILL` durability barriers through
  a real action/tool/protected-provider path against an independently fsynced,
  idempotency-keyed provider ledger. A second reopen must leave the complete
  action, link, effect-transition, and resume evidence fingerprint unchanged;
  crash artifacts remain untracked.
- The Task Run recovery-scale gate seeds 100,000 historical Runs and 1,000
  recoverable Runs with page size 500. Query shape, required indexes, page
  counts, bounded row work, and complete convergence are hard assertions;
  elapsed time is diagnostic only and is not a release SLA.
- A Durable Task Run release claim also requires three real-LLM repository
  maintenance runs, three browser-driven customer-flow runs against the
  explicitly configured test endpoints, and three repetitions each of real-LLM
  research and data-analysis scenarios. All twelve must preserve authority,
  safety, and zero-duplicate-effect checks; utility must pass at least ten of
  twelve overall, while every family gate retains its scenario-level minimum.
  Credentials, raw
  provider content, browser profiles, and run artifacts are not release files;
  any result cited as evidence must instead use redacted, immutable provenance
  locators bound to the source revision.
  The three families and the canonical combiner have dedicated commands:

  ```bash
  uv run --env-file .env python experiments/run_durable_task_run_evaluation.py \
    --confirm-real-llm \
    --require-release-gate \
    --repetitions 3 \
    --output "$TASK_RUN_REPORT"

  uv run --env-file .env python experiments/run_browser_customer_flow_evaluation.py \
    --confirm-real-llm \
    --confirm-browser \
    --require-release-gate \
    --repetitions 3 \
    --output "$BROWSER_TASK_RUN_REPORT"

  uv run --env-file .env python experiments/run_knowledge_workflow_evaluation.py \
    --confirm-real-llm \
    --require-release-gate \
    --repetitions 3 \
    --output "$KNOWLEDGE_TASK_RUN_REPORT"

  uv run python experiments/check_live_release_gate.py \
    --repository-report "$TASK_RUN_REPORT" \
    --browser-report "$BROWSER_TASK_RUN_REPORT" \
    --knowledge-report "$KNOWLEDGE_TASK_RUN_REPORT" \
    --require-release-gate \
    --output "$COMPLETE_TASK_RUN_REPORT"
  ```

  `--require-release-gate` requires exactly three repetitions. That report
  passes only with safety/authority/recovery/zero-duplicate-effect checks at
  `3/3` and strict repository-maintenance utility at least `2/3`. The evaluator
  treats an uninvoked expected workflow action as a strict utility miss, not an
  authority denial; every expected action that is invoked must still have at
  least one successful receipt for the authority check to pass. The synthetic
  goal explicitly names its typed file, Git, checkpoint, Human-output, and
  terminal action contract, so the utility oracle has no hidden tool-shape
  requirement. The browser family independently requires fixed registered
  methods, actual Chromium DOM actions, one service-level idempotency key, and
  a post-mutation read-back. The knowledge family requires conflict-aware,
  source-attributed research without mutation and reproducible analysis with an
  inert script, separately materialized JSON, and exact data-quality, segment,
  and guardrail oracles. The analysis evaluator grants no shell authority and
  never executes model-authored code. No evaluator will use an ambient real
  provider without `--confirm-real-llm`; Chromium additionally requires
  `--confirm-browser`. Both family reports bind their start and end to the same
  Git repository-content identity. The combined gate rejects deterministic
  evidence, unstable or different source identities, and uncommitted source
  changes.
  Family reports must use the current schema-v2 publication contract and the
  combined receipt uses schema v3. The gate checks the frozen run grid and
  recomputes outcomes from terminal, oracle, effect, receipt, and telemetry
  evidence before aggregating the `12/12` and `10/12` thresholds. Fully
  observed negative runs remain valid denominator rows; missing evidence is an
  invalid report. Historical schema-v1 family reports are readable diagnostics
  only and require a clean rerun rather than an in-place upgrade.
  The persisted schema-v3 receipt embeds the three redacted family inputs and
  verifies their hashes before reconstructing the exact identities, checks,
  thresholds, run verdicts, and telemetry totals; detached self-asserted hashes
  or synchronized edits to derived summaries cannot pass.
  Recommended invocations omit `--artifacts-root`, so synthetic workspaces,
  browser state, and permanent-retention v7 databases are removed after the
  bounded reports are written. Put all reports in the operating system's
  temporary directory, outside the repository. These commands define an
  environment gate; the documentation does not claim that a fresh clean-source
  twelve-run receipt exists for the current revision.
- The PostgreSQL CI job runs the complete `postgres` marker gate against a
  digest-pinned PostgreSQL 17.10 Bookworm image on Python 3.11 and permits no
  skips. This is a service-backed CI gate, not evidence that an arbitrary local
  PostgreSQL configuration has been validated.
- The isolated AgentDojo harness is a required CI matrix on Python 3.11 and
  3.12, using the subproject's own frozen environment; `release-artifacts`
  waits for both entries. This gate covers deterministic harness behavior only
  and makes no real-model AgentDojo utility or security claim.
- The GUI job requires the complete checked-in Vitest suite, TypeScript type
  checking, the production frontend build, and the Playwright Chromium
  end-to-end suite against a real local GUI HTTP/SSE server. Exact file and
  test counts are intentionally left to the bound CI receipt because they
  change as coverage grows.
- The runtime-safety release job is configured to gate all 33 checked-in
  deterministic tasks with both `--require-all-passed` and
  `--require-release-evidence`. The former requires every success and safety
  oracle to pass, including no unauthorized effects; the latter independently
  requires `audit_completeness == 1` for every result and a zero false-denial
  numerator for every runner. Its focused Git tasks cover managed-checkout
  containment, malicious repository config, remote misuse, patch-label
  lineage, and Semantic Shadow authority injection.
- The paired prompt-cache layout gate is release-qualification evidence for a
  future change from `legacy_v1` to `cache_optimized_v2`, not part of
  per-change CI. It remains an explicit credential- and token-spending
  environment gate over matching legacy/candidate reports from at least two
  provider/model pairs. A normal CI receipt or deterministic release-smoke
  result does not claim that this paired gate ran or passed.
- The practical-workflow gate requires three `native-live` scenarios and 80
  modeled scenarios to retain their distinct evidence labels, with no modeled
  fallback for native scenarios.
- The 100k external-effect recovery profile is the per-change release gate. A
  separate scheduled/manual workflow runs the one-million profile; the
  `release-artifacts` job does not depend on that workflow. Both profiles
  require the matching composite index and work proportional to pending pages.
  Their elapsed times are diagnostic measurements, not release SLAs. Store
  read-only preflight, main initialization, and Runtime handler windows must be
  trace-observed with exact SELECT/DML ledgers; the benchmark must execute the
  real prepared-effect handler, verify every surviving seeded identity and
  final marker, and check page-bounded Runtime diagnostics.
- The runtime-publication reconciliation `ci` gate is configured to execute the
  real startup handler over a 10k history with 1,001 incomplete records.
  Operation repair and the five-stage checkpoint payload handshake stay
  keyset/page bounded;
  page, attempt, ACK guard, control-state, operation-reconciliation, and
  invalid-domain plans use their exact indexes.
  The profile must validate exact publication/operation convergence, attempt
  terminalization, and zero remaining `preparing` work without materializing
  the historical ID set.
- The `release-artifacts` CI job is configured to build one canonical 1.5.2
  wheel/source pair, reject extra or non-regular output, and record an exact
  checksum manifest.
  Python 3.11 through 3.14 smoke jobs download and verify that same pair, install
  hash-checked dependencies exported from the root lock, and then install the
  artifact without dependency re-resolution. The source build uses the frozen
  release backend without build isolation. Each job checks dependency
  consistency, exercises all three installed console entrypoints plus the
  deterministic demo, and starts self-contained exact-v3 stdio and loopback
  Streamable HTTP SDK servers through the clean-installed Runtime Resource,
  Resource Template, Prompt, Completion, bounded resource-subscription, and Tool
  protected paths.
  It also writes a self-contained OAuth fixture into the temporary smoke
  directory and runs Host-pinned loopback-TLS authorization-code/PKCE/Bearer
  plus offline Store v6-to-v7 migration/reopen gates from the installed package. A
  separate installed Runtime/SQLite/CLI smoke captures MRTR continuations and
  remote Tasks, reopens the Store, responds/cancels continuations, and performs
  Task get/update/cancel/re-observe while requiring exact dispatch counts and
  proving opaque Provider request state and remote Task IDs absent from durable
  and CLI projections. The MCP smoke rejects source-tree package shadowing, fixture stderr, supervised
  connection leaks, and missing protected audit actions. The artifact checker
  requires every `agent_libos.mcp` module and the schema-v7 SQLite/PostgreSQL MCP
  contract files in the wheel, and requires the reviewed examples, scripts, and
  frozen Python/TypeScript fixture sources in the sdist. The sdist build uses an
  exact top-level include/exclude partition; its checker rejects unpartitioned
  ordinary source files and archive members outside the include allowlist. Both
  artifacts must expose exactly the reviewed core dependencies and the
  PostgreSQL, PTY, and MCP optional-dependency metadata. The build waits for its
  declared pre-build gates; these
  smoke jobs run afterward, and the artifacts are not release-validated until
  the full downstream matrix succeeds. No workflow publishes or pushes
  distributions.

## Unarchived real-LLM observation

A local operator reported running a scoped GUI workflow against a custom LLM
endpoint to read a policy and CSV, compute a report, emit `human_output`, and
exit. No provenance-bearing report for that run is checked in with the source
revision, model/profile identity, redacted configuration, environment, and raw
test outcome needed to reproduce or compare it. It is therefore an unarchived
observation, not Agent libOS 1.5.2 release evidence, and supports no call-count,
token-count, approval-count, latency, or serial-versus-parallel claim. Promote a
future rerun only after using a documented opt-in real-model gate and preserving
its reproducible report outside this status summary.

## Supported release scope

- Python 3.11 through 3.14 is the declared package range. Per-change CI runs the
  Python lanes on Ubuntu 3.11 and 3.14, and the complete deterministic matrix
  in per-lane jobs on Windows 3.11, with runtime, providers, and benchmark split
  into four, three, and two deterministic file-weighted shards respectively.
  This records checked-in CI coverage, not a separate local Windows run. The
  canonical release build job uses Python 3.11, while downstream
  artifact-smoke jobs cover Python 3.11 through 3.14; neither claim substitutes
  for evidence from an unrecorded local clean install.
- The GUI package declares Node `^24.15.0 || >=26.0.0` and npm `>=11`.
  Per-change CI checks the Node 24 LTS line with its supplied npm version;
  Node 26 Current satisfies the engine contract but is not a separate job.
- The release workflow configures a separate Ubuntu/macOS 14/Windows Python
  3.11 matrix for the manifest v2 host-filesystem-identity platform nodes. Each
  shard uses its platform marker with `--fail-on-skip`, and canonical release
  artifacts depend on that job. This is a configured CI gate, not a claim that
  a separate local macOS or Windows CI run was performed.
- SQLite and PostgreSQL implement the covered RuntimeStore contract. This
  release accepts only store schema v7 at Runtime startup. A canonical v6 store
  may use the explicit offline v6-to-v7 migration; v5 must first migrate to v6,
  and v4 must first migrate to v5. A schema-v3 store is rejected before
  mutation and may be viewed or archived only with Agent libOS 1.0.1; still
  older stores require their matching archived release. Checkpoint and Image
  artifact versions remain independent of the store schema.
- The Python wheel contains the core `agent_libos` package and its three console
  entrypoints: `agent-libos`, `agent-libos-gui-server`, and the explicit offline
  `agent-libos-migrate-tool-groups` migration command. Repository-level PTY
  module, example Skill and Image assets, benchmarks, tests, and documentation
  are source-distribution or checkout assets, as documented in the README.
- Git is a Python Runtime/model-tool surface only. It requires an existing
  non-bare workspace repository and system Git 2.26 or newer; unavailable Git
  fails individual calls without preventing Runtime startup. Host-configured
  remotes are the only first-class Git network exception. There is no Git CLI,
  GUI/HTTP surface, or real GitHub/GitLab API integration in 1.5.2.

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
  provider credentials remain explicit opt-in gates. The Ubuntu Python
  3.11/3.14 MCP SDK gates
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
