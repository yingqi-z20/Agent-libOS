# Runtime Storage

Agent libOS 1.0.1 stores durable runtime state through a `UnitOfWork` composed of
explicit domain boundaries, including `ProcessRepository`,
`ResourceRepository`, `RuntimePublicationRepository`,
`SnapshotCheckpointRepository`, `RuntimeModuleRepository`,
`PayloadRetentionRepository`,
`ObjectRepository`, `AuthorityRepository`, `EvidenceRepository`, and
`ExtensionRepository`. All repositories in one runtime share the same
transaction coordinator. Migrated runtime domains use those repositories. The
concrete SQL store remains a Host composition/lifecycle boundary and is still
passed to a small set of reviewed legacy services such as Explain; it is not yet
hidden from every runtime component. New domain persistence should use the
typed/repository boundary rather than extending that compatibility surface.

Process, resource-accounting, runtime-publication, operation/evidence,
module-publication, and Snapshot/Checkpoint persistence have explicit typed
Protocols and facade methods. Snapshot services exchange canonical `SnapshotRows` and
`ProcessSnapshot` aggregates with the repository; backend SQL, generic table
helpers, and Object-payload cache coordination stay behind that boundary. An
AST ratchet rejects raw SQL or generic table-helper regressions in the migrated
runtime services. Payload-retention scans and compare-and-swap reductions use a
separate typed repository so maintenance cannot regain the generic extension
facade or load full evidence history into Runtime services.

Those migrated repository surfaces, including payload retention, bind to
explicit backend Protocols. A `UnitOfWork` validates concrete method presence
and compatible positional and keyword call shapes before it constructs any
repository; dynamic `__getattr__` shims and transaction-only backends therefore
fail at assembly, not on their first request. The static no-reflection ratchet
is narrower than that backend-conformance check: it rejects `_delegate` and
`getattr` inside `ProcessRepository`, `ResourceRepository`,
`RuntimePublicationRepository`, `CheckpointRestorePublicationWriter`,
`SnapshotCheckpointRepository`, `EvidenceRepository`,
`RuntimeModuleRepository`, and `PayloadRetentionRepository`. The retention
repository calls its typed backend directly; it cannot route bounded scans or
CAS updates through the legacy facade's `_delegate`. Legacy event, audit,
human/LLM, ObjectTask, message/rating, context-materialization,
external-effect, registry, and provider facades still use reviewed allowlists.
They must be migrated explicitly before that compatibility mechanism can be
removed.

SQLite is the default local engine. SQLite and PostgreSQL are independent
connection/dialect/lease adapters over the same typed repository
implementation and canonical store schema v3. The shared implementation emits a
small SQLite-shaped SQL subset. The PostgreSQL dialect translates parameter
placeholders, `COLLATE BINARY`, `INSERT OR IGNORE`, the one reviewed
`INSERT OR REPLACE` upsert, `INDEXED BY`, the Skill-package description
`json_extract` expression, and the table/index metadata probes; a
syntax-surface test ratchets that set. PostgreSQL therefore does not inherit the
SQLite store or its connection/locking behavior, but it is also not a second,
copied repository implementation.

Library defaults are deliberately ephemeral. `Runtime.open()` and
`Runtime.aopen()` do not load the repository's `config.yaml`; with neither an
explicit target nor an explicit config they use `DEFAULT_CONFIG`, whose
`runtime.local_store_target` is `local`. The store factory maps both `local`
and `:memory:` to an in-memory SQLite database, and `SQLiteStore()` has the
same `:memory:` default. Each such connection is a separate store and all of
its state disappears when it is closed. A bare `sqlite://` target also maps to
`:memory:`. Persistence therefore requires an explicit filesystem path or
`sqlite:///...` file target, a config whose local target is a file, or a
PostgreSQL DSN. Product CLI and GUI entrypoints may load project
configuration before calling `Runtime.open`; that Host behavior does not
change the library default.

This release supports one writable Runtime per database/schema on either
backend. PostgreSQL's session advisory lease enforces that product boundary;
the current contract does not claim concurrent multi-Runtime writers or a
connection-pooled repository. Supporting those modes would require an explicit
database-level isolation design (for example row locks/epochs and corresponding
multi-connection tests), rather than relying on the in-process repository
lock.

Within one RuntimeStore, one backend connection and one process-local
`threading.RLock` serialize queries and transactions across threads. Reentrant
transactions on the owning thread use savepoints; another thread cannot join an
active outer transaction. PostgreSQL database concurrency does not imply that
this repository has a connection pool or concurrent per-Runtime transactions.

## Strict store schema v3

Fresh databases created by Agent libOS 1.0.1 use store schema v3 and create a
`runtime_schema` table with one marker row; the canonical DDL constrains its
`singleton` value to `1`. Opening an existing store requires the row selected
by `singleton = 1` to contain schema version `3`. Product version and store
schema version are independent identifiers: `1.0.1` is the current product
release, while `3` remains the persisted schema contract. Both backends apply
the same acceptance rules, with one backend-specific initial probe order:

1. SQLite validates `PRAGMA encoding` before reading the version marker;
   PostgreSQL reads the marker first and validates server encoding with the
   remaining shape probes.
2. For a version-3 marker, require every manifest table and required column,
   reject the obsolete `storage_migrations` table, and validate the required
   keyset text-column collations. A required SQLite relation must have
   `sqlite_master.type = 'table'`, and a required PostgreSQL relation must have
   `pg_class.relkind = 'r'`; a same-column view cannot impersonate a manifest
   table and is rejected before the initializer can mutate the schema.
3. Reject a wrong marker, a store missing any of that probed surface, or an
   unversioned target containing a probed user object. SQLite enumerates all
   non-`sqlite_%` schema objects; PostgreSQL's freshness probe covers `pg_class`
   relation kinds `r`, `p`, `v`, `m`, `S`, and `f`, not function/type/domain-only
   schemas.
4. In one transaction, run the idempotent initializer. For an accepted existing
   version-3 store this can create missing named indexes and insert missing
   canonical seed rows such as counters and the system namespace. For an empty
   target it creates the complete schema and then writes the marker.

An interrupted bootstrap rolls back both schema and marker, so reopening the
same empty target retries initialization instead of misclassifying it as a 0.2
database.

The DDL emitted for a fresh schema-v3 store is the current release shape,
including the typed process wait/outcome columns. It replaced draft v3 shapes
before schema v3 was frozen for the legacy Agent libOS 0.3 product release;
those development databases were never a supported release format. A version
marker alone is insufficient: drafts missing a
required table, required column, or canonical keyset collation are rejected,
and the runtime does not present that rejection as a migration.

The open-time compatibility probe is intentionally not a byte-for-byte DDL
validator. Apart from the checks above, it does not compare column types,
`NOT NULL`/`CHECK`/foreign-key/primary-key constraints, arbitrary collations,
extra columns or objects, or the definitions and uniqueness of existing
indexes. Idempotent initialization does not replace a same-named index with a
different definition. Operators must create stores through this release's
backend rather than treating a hand-built schema that passes the compatibility
probe as canonical.

Text columns that form durable startup, recovery, or retention keysets have a
canonical bytewise collation: `BINARY` on SQLite and `"C"` on PostgreSQL. Both
the primary timestamp and the stable textual tie-breaker use that physical
shape, and their covering indexes use the same collation. Queries inherit the
validated column collation so SQLite retains composite row-value range seeks.
SQLite database files and PostgreSQL servers must both use UTF-8 encoding;
under UTF-8, `BINARY` and `"C"` ordering match Python's Unicode string ordering
for persisted cursor values. Opening an existing version-3 store fails closed
when its database encoding, any required keyset text column, or any required
keyset text-column collation is not canonical; those required column
collations are checked with one set-based catalog probe. A UTF-16 SQLite file
or locale-inheriting draft PostgreSQL schema is therefore rejected rather than
silently paginating with a different order or degrading to a sort. This check
does not validate the collation of text columns outside the keyset manifest.

There are no migrations, backfills, or reconciliation paths from 0.2. A 0.2
database is archive-only and must be opened with an archived 0.2 release. The
current runtime raises `UnsupportedStoreVersion` before it initializes tables or
changes rows. Checkpoints and checkpoint-derived Image artifacts are likewise
strictly versioned and rejected before operation evidence or process state is
written.

## Transaction model

Top-level `UnitOfWork.transaction()`/store transactions use `BEGIN`/`COMMIT`.
An explicitly nested call to `transaction()`—including one made by a repository
method—creates a savepoint. The low-level single-write helper
`_join_or_begin_transaction()` instead joins an already active transaction
without adding a savepoint or a second post-commit failure boundary; when no
outer transaction exists, it opens and commits an explicit transaction.
Neither path commits independently while an outer transaction is active, so
lifecycle changes can publish authority, process, object, extension, audit,
event, operation, and protected-effect rows as one unit.

The PostgreSQL connection keeps connection-level autocommit enabled so a plain
read does not leave an implicit transaction open. Every repository mutation,
including a single-statement write, still enters an explicit outer transaction;
the lifecycle admission guard revalidates immediately before the real commit,
and rejection rolls that transaction back. SQLite uses the same mutation
helper and commit-guard contract.

When a Runtime is recovery-required, ordinary close/shutdown deliberately keeps
that exact admission guard and the SQLite/PostgreSQL active-runtime lease bound,
so no second writer can bypass the diagnostic fence. The explicit
`release_recovery_diagnostics()` handoff is the only no-write exit: after all
admissions and shutdown attempts are absent and transient workers have stopped,
it identity-matches the guard and closes the backend lease/store under the same
store lock. The store returns a structured ownership outcome: a failure while
the exact SQLite lease or PostgreSQL session is still owned restores the guard
and is retryable; a diagnostic raised after ownership is irreversibly released
never restores the stale guard, permanently disables the old store instance,
and completes the lifecycle handoff with warnings. It emits no
audit/event/terminal evidence and invokes no ordinary finalizers; only explicitly
registered no-write recovery cleanup may run. Only after an ownership-released
outcome may a newly opened Runtime take the same target and perform startup
recovery.

PostgreSQL handoff closes the owning session as its single release point; it
does not issue a separate `pg_advisory_unlock` first, because an ambiguous
unlock acknowledgement would make partial-close ownership unknowable. SQLite
closes the database connection before releasing its file lease, uses descriptor
close as the lease's single release point without a preceding `LOCK_UN`, and
probes the real driver handle after a close diagnostic. Builder-owned
failed-open cleanup atomically replaces the partial lifecycle guard with a
unique callable close reservation before graph teardown, so a stale cleanup
handle cannot close or yield the store to a successor Runtime.

Async close paths use two nonblocking store checks on the caller/event-loop
thread. `probe_admission_guard_close()` detects an active transaction, any
current-thread store lock scope, a lock held by another thread, or a stale
guard without changing state; lifecycle code uses it before teardown.
`claim_admission_guard_close()` repeats those exact checks and atomically marks
the guard close-pending immediately before worker offload. While claimed, new
transactions, `locked()` scopes, and dynamic identifier probes fail fast. A
pre-release backend failure that successfully restores the exact guard clears
the claim so diagnostics remain readable; retries must claim again. If guard
restoration itself is interrupted, the exact claim remains as the sole retry
token, blocks successor binding, and permits only that identity to probe,
re-claim, or finish releasing the backend. An
`OWNERSHIP_RELEASED` readiness result is terminal rather than retryable; the
exact release outcome clears the claim permanently with the guard, while a
stale caller may observe the terminal ownership fact without changing a live
guard.
Failed-open handles that legitimately need to repair an unbound owned guard use
`try_replace_admission_commit_guard()`, which applies the same nonblocking lock
and caller-scope checks and replaces only the exact expected owner; it cannot
take a guard installed by a successor Runtime.

Probe and claim return a structured `ownership_released` terminal outcome when
the backend lease/session is already gone. This is not treated as a retryable
lock failure: the exact lifecycle owner may finish graph teardown and call the
structured release operation to clear only its own stale guard. Ordinary async
close drains the off-loop release before reporting caller cancellation;
post-release warnings are retained by the lifecycle for idempotent readback.

Object payloads are runtime memory rather than ordinary SQL data. SQL Object
rows retain metadata and a live-payload marker. A transaction that changes
Object rows and payloads captures the in-memory payload state and restores both
layers on rollback. Checkpoints and Image artifacts explicitly serialize only
their bounded payload set.

There is one narrow startup-recovery projection outside the Object row. A
committed root `ProcessManager.spawn` publication carries the initial GOAL's
size-bounded JSON payload only when that goal is immutable and
`llm.persist_full_io=true`. Before the
general missing-payload sweep, startup may rehydrate only a matching live,
nonterminal root goal after validating the process id and creation time, current
goal id, Object identity and version, and payload digest. Exec may change Image
and preserve that original goal; an exec replacement goal is not recoverable by
the launch envelope. Generic
publication get/list paths redact the reversible value to a hash-only
projection. With `persist_full_io=false`, for non-root launches and other
Objects, and after terminal root cleanup, there is no reversible recovery value.
This exception is not a general durable-Object contract and direct database
administrators remain inside the Host trust boundary.

Launch compensation treats reversible content as part of the publication
state transition. Online rollback changes `planning`/`applying` to
`rollback_pending` only in the same outer transaction that reduces a full
initial-goal envelope to hashes; startup recovery likewise couples its claim
with that redaction. A redaction failure rolls back the transition/claim, keeps
the earlier recoverable state plus full envelope for a fenced retry, and cannot
leave `rollback_pending`, `rolled_back`, `failed`, or `manual` with reversible
goal content.

The SQL decoder retains a compatibility path for pre-marker rows that contain
a legacy full JSON payload: while such a row remains in an accepted store, an
explicit payload lookup can decode it into the volatile cache. New writes use
only runtime-memory markers, and startup's missing-payload sweep targets those
markers; it does not prove or rewrite arbitrary legacy full-payload rows. Hosts
handling a store produced by an older development build should therefore treat
those rows as durable sensitive data and migrate or recreate the store before
claiming marker-only historical retention.

If an ordinary commit, savepoint release, rollback, or rollback cleanup leaves
transaction state uncertain, the store is poisoned and its data plane is
closed. The sole narrower exception is the internal startup checkpoint-payload
ACK confirmation scope: after an ambiguous outer commit, the same thread may
perform one typed exact-state readback without opening a transaction. An exact
`acked` result explicitly accepts the commit and re-enables the store; an
unconfirmed or mismatched result closes the poisoned connection on scope exit.
No general caller receives this confirmation privilege. Otherwise later access
fails closed and callers must discard the Runtime and reopen a healthy database.
Exact ownership controls remain available only when a separate backend lease is still held, as
with a file-backed SQLite store whose SQL connection was poisoned before its
file lease could be released. If the connection/session itself was the final
ownership point and is already gone, those controls report the structured
terminal result instead of publishing an impossible cleanup retry.

Optional `expected_states` arguments on repository compare-and-swap mutations
have one uniform meaning: `None` disables the state predicate, while an
explicitly supplied empty iterable matches no state and returns `False` without
changing durable state. Non-empty iterables retain the exact state fence.

## Durable authority and evidence

The shared schema durably stores capability state and reservations, Task
Authority manifests, process/resource state, Human and process-message waits,
LLM pending actions and context label history, registry state, explainable
operations, audits, events, and protected external-effect intents.

External effects use a `pending`/`prepared` intent before provider dispatch.
The durable effect-state domain is `pending` or `finalized`; the durable
transaction-state domain is `prepared`, `authorized`, `approved`, `dispatched`,
`committed`, `failed`, `unknown`, or `compensated`. A provider certificate that
the first effectful phase never started restores reservations and deletes the
prepared intent, so there is no durable `not-started` transaction state. If a
later phase is certified not started after an earlier effectful phase, the row
is finalized as `committed` and records an outcome such as
`partial_not_started_after_prior_provider_effect` in provider metadata;
`partial` is not a transaction state. Capability-use reservations and effect
settlement share the enclosing transaction and preserve revoke-wins and
one-shot semantics.

An operation's runtime-publication binding is stored in a normalized nullable
column with a unique partial index, as well as in its versioned explanatory
metadata. The typed evidence repository performs exact indexed reverse lookup;
row decoding rejects a disagreement between the normalized column and metadata.
Publication planning and operation binding remain in one transaction, so the
index is an integrity constraint rather than a heuristic backfill.

Startup publication recovery uses typed, hard-bounded keyset pages over exact
kind, state, reconciliation marker, `created_at`, and publication id. Launch,
exec, and committed checkpoint-restore terminal-operation repair scans only
marker-false rows and exact-CAS marks completion; failed/manual checkpoint
restores remain forward-recovery inputs. RuntimeStore changes to a bound
operation atomically clear the marker, making the changed row eligible for
revalidation; direct database writes remain outside the application-integrity
boundary.

All mutation-capable startup recovery facades receive the lifecycle's bound
`require_recovery_lease` verifier, never its private token. Each facade and raw
backend invokes that verifier before its first durable read or transaction.
Consequently an `OPEN` runtime cannot manually scan, claim, reconcile, or
compensate startup work, and a same-shaped arbitrary ContextVar value cannot
impersonate the recovery lease. Recovery diagnostics are typed summaries with
exact totals and page-bounded samples rather than full-backlog lists.

Prepared protected-effect recovery runs before stale capability-use cleanup.
Once every valid prepared intent has restored its exact linked reservations,
the authority repository abandons remaining `reserved` rows through the
`(status, created_at, reservation_id)` keyset index. This ordering eliminates
the former startup-wide external-effect JSON scan and its unbounded protected
reservation set. Pending external effects are then reconciled without replay.

Active resource-usage reservations are recovered next through bounded
`(status, created_at, reservation_id)` keyset pages. A reservation whose linked
effect is absent or still `prepared` is released as not started; a reservation
whose surviving effect is in any other transaction state is conservatively
treated as possibly dispatched and charged at its maximum. Recovery permits
that charge to exceed the budget and commits any resulting resource-limit
process termination with it. Runtime retains exact totals and only a
page-bounded reservation-id sample.

After publication, Object-payload, and JIT recovery, running explainable
operations left by the prior Runtime are terminalized through bounded keyset
pages. An operation whose causal tree has a pending or `unknown` external effect
becomes `unknown`; the other stale running operations become `interrupted`.
The temporary membership index and store lock keep the cross-page view stable,
and Runtime retains exact totals plus a page-bounded operation-id sample.
Stale process executions then use a status/PID keyset index; each page commits
the `PAUSED` transition, concurrency high-water, audit, and event evidence
together while Runtime retains only a bounded PID sample.

Volatile Object payload cleanup is no longer a store-constructor side effect.
Runtime assembly invokes it under the opaque recovery lease, traverses the
partial `(created_at, oid)` recovery index in configured keyset pages, and uses
per-Object CAS writes to release metadata, links, and Object capabilities. It
runs before ObjectTask reconciliation, so succeeded tasks can deterministically
replace missing result references with `result_unavailable_after_reopen`.
Active tasks, missing results, and retryable terminal notifications each use
their own normalized, indexed `(created_at, task_id)` keyset scan. Runtime keeps
exact totals and at most one configured page of Object/ObjectTask identifiers.

Checkpoint-restore plans are complete at publication insert and carry an
immutable SHA-256 anchor in the publication receipt. Generic Host-visible
RuntimeStore methods reject checkpoint-restore insert, advance, recovery claim,
artifact append, plan update, and operation-reconciliation marking. An opaque
storage-owned writer is injected only into the restore reconciler; its state
machine enforces the main-commit marker, ordered phase/finalizer receipts,
recovery lease, failure classification, terminal receipt, and plan anchor.
Process-exec plans permit only effective no-op writes. The sole mutable
publication-plan slice is the exact
`boot_kind`/`materialized_workspace_root` pair for a planning or applying
process launch. Recovery verifies the checkpoint plan anchor before replaying
any reconciliation or durable-finalizer work, and terminal operation repair
also requires the complete causal transcript.

Orphaned `CREATED` process detection uses an indexed `NOT EXISTS`
launch-publication query. Neither that query nor the publication-recovery paths
above materialize complete publication or process history.

Process tool tables also have a transactionally maintained normalized reverse
projection in `process_tool_bindings`. Publication compensation checks an exact
tool identity through `(tool_id, pid)` and reads durable tool existence through
the `tools` primary key; it never decodes every process or loads the complete
tools table. Candidate receipts bind the exact Object Memory descriptor OID,
so cleanup and convergence checks use candidate/descriptor primary keys. The
capability effect and its exact receipt share the publication UnitOfWork, so
recovery has no metadata-scan fallback for unreceipted capabilities.
`process_tool_bindings` is part of the complete fresh version-3 release shape,
not a lazy projection or startup backfill. A draft version-3 database that
lacks it is rejected by the strict shape probe; supported stores therefore
retain the projection across reopen without scanning or rewriting processes.
The projection also stores transactionally derived JIT eligibility. A
binary-collated partial covering index keyset-pages only eligible bindings, so
database work follows the JIT backlog rather than unrelated callable aliases;
the exact Tool/candidate bulk lookup remains the authority check for every
returned page.
Exec rollback and checkpoint pruning use the same projection for both callable
and model-only bindings. A single-process exclusion performs its reference
check and exact tool deletion in one transaction; a multi-process restore scope
streams only the indexed matching PIDs instead of materializing process rows.

Data-flow evidence stores labels, source references, hashes, Sink/trust
generation, and decisions—not payload copies. LLM pending actions and context
generations retain canonical metadata-only `DataFlowContext` values. The
label/source JSON and pending-action context columns required by those records
are non-null in the fresh schema-v3 DDL, and row decoders require their canonical
object shapes and complete security labels. Malformed persisted values fail
closed instead of being repaired heuristically; other schema fields may still
be nullable where their domain permits it.

Process control state is persisted structurally. `wait_state_json` is a tagged
child/message/human/tool/pause/Host-resume wait, `outcome_json` is a tagged
exited/failed/killed outcome, and `state_generation` advances on every semantic
state transition. Normal runtime orchestration makes those transitions through
one `ProcessTransitionService`. The only explicit exceptions are typed
repository CAS primitives whose state update must be atomic with an execution
lease or snapshot restore; an exec-epoch commit additionally requires the exact
non-null admission token recorded by the matching applying `process_exec`
publication at its final pre-commit phase, then CASes RUNNING status,
generation, owner, and lease.
For ordinary waits and outcomes, `status_message` is only a compatibility
projection for older clients and is never parsed as the control protocol. The
narrow internal exception is checkpoint-fork publication: newly inserted
non-terminal rows carry the reserved `checkpoint_fork_pending_payload` sentinel
as a quarantine CAS marker until captured payloads are rehydrated and exact
target states are published. Wake tokens include the generation, so an
observer of an earlier wait cannot wake a later, textually identical wait after
an ABA cycle. Restore allocates a generation above the durable high-water mark,
while checkpoint fork resets the new process identity to generation zero and
remaps typed PID/Object references. Message-wait filters are strict JSON trees
with string object keys and finite numbers; no storage serialization may coerce
their identity. Checkpoint process rows require the physical JSON text values
(including the literal text `null`) rather than a SQL null for either tagged
column.
Public checkpoint-inspect projections strictly decode those tagged columns and
publish the canonical mappings with the snapshot `state_generation`; they never
derive control state from `status_message` or substitute current live state.
Generic process patch/update APIs reject `status`, `wait_state`, `outcome`, and
`state_generation` before writing. The transition repository primitive
revalidates the complete product type and CAS fence and computes
`state_generation + 1`; explicitly typed execution/restore CAS primitives own
the same generation increment at their atomic commit point. Callers cannot
supply or rewind the committed generation.

Runtime APIs apply domain-specific mutation rules rather than one universal
append-only/versioned policy. Audit records, events, operation-evidence links,
and external-effect transition rows are appended. Mutable projections and
lifecycle records—including processes, capabilities and their reservations,
Human/message/LLM state, operations, Objects, and external-effect intents—are
updated or deleted in place under their transaction, CAS, generation, or
state-machine fences. None of these application-level rules is
cryptographically tamper-proof against the Host or a database administrator.
Independent integrity requires externally signed or independently append-only
evidence.

The event row contract and the limits of event ordering and causal evidence are
specified in [Runtime Events](events.md).

## Backup and restore runbook

This runbook covers the supported recovery unit: one quiesced schema-v3 SQL
store restored into a new target and then opened by the same Agent libOS
release. It is an operational database backup, not a checkpoint restore and not
a snapshot of the whole environment.

Before either backend is backed up:

1. Stop admitting new work while the one owning Runtime is still active, and
   complete or explicitly diagnose in-flight provider operations. A database
   snapshot cannot make an external effect atomic with the backup time.
2. If volatile Object payloads must survive, capture the intended bounded
   payload set now in a checkpoint or checkpoint-derived Image artifact.
   Ordinary Object rows contain live-payload markers, not the in-memory payload
   bytes. This must happen before shutdown; a closed Runtime rejects checkpoint
   creation and a reopened Runtime cannot reconstruct those payloads. The sole
   automatic exception is the integrity-bound initial goal of a committed,
   nonterminal root spawn created with `llm.persist_full_io=true`; do not treat
   that narrow startup aid as artifact backup.
3. Stop every GUI, CLI, worker, and embedding that can open the target. For an
   embedded Runtime, call `shutdown()`/`ashutdown()`, require `ok: true`, and do
   not proceed from a recovery-required or incomplete shutdown result.
4. Record the Agent libOS product version, backend configuration, and the value
   of `runtime_schema.schema_version`. For this release the expected pair is
   product `1.0.1`, store schema `3`.
5. Prepare an owner-only backup directory and run the dump-producing command
   under `umask 077`. Before accepting either backend's archive, verify it is a
   regular, current-user-owned, single-link file with mode `0600`.

### SQLite

Use this procedure only for a file-backed SQLite target; `local` and
`:memory:` databases have no persistent file to back up.

1. After the Runtime has released its active-store lease, create an owner-only
   backup directory and use a restrictive umask with SQLite's own backup command:

   ```bash
   mkdir -p /srv/backups
   chmod 700 /srv/backups
   (
     umask 077
     sqlite3 /srv/agent-libos/runtime.sqlite ".backup '/srv/backups/runtime-2026-07-30.sqlite'"
   )
   chmod 600 /srv/backups/runtime-2026-07-30.sqlite
   ```

   Do not copy a live database file and do not copy Agent libOS lease files,
   `-wal`, or `-shm` sidecars into the backup. The supported procedure is the
   quiesced `.backup` output, which is a self-contained SQLite database. Before
   accepting it, verify that it is a regular, current-user-owned, single-link
   file with mode `0600`; a permissive default umask can otherwise make SQLite's
   new backup file group/world-readable.
2. Verify the backup before accepting it:

   ```bash
   sqlite3 /srv/backups/runtime-2026-07-30.sqlite \
     "PRAGMA quick_check; SELECT schema_version FROM runtime_schema WHERE singleton = 1;"
   ```

   The expected output includes `ok` and `3`.
3. To restore, keep the source database untouched and materialize the verified
   backup at a new, owner-only path. Do not restore over a path held by a live
   Runtime and do not restore old lease/sidecar files. Point a stopped Host at
   the new path and let `Runtime.open()` perform the complete schema-v3 shape,
   encoding, collation, and startup-recovery checks. Keep the old target until
   that open and a clean shutdown succeed.

### PostgreSQL

Use normal libpq credential mechanisms such as a service definition and
password file; avoid placing credentials in shell history. The PostgreSQL
Runtime identity and lease are scoped to `current_database()` plus the exact
`current_schema()`, so the backup unit is that one schema rather than the whole
database. Configure the service's `search_path` to select the Runtime schema;
the commands below use `agent_libos_runtime` as an example. With the Runtime
stopped and its advisory-lock session closed:

1. Confirm the selected schema and version, then create and inspect a
   custom-format logical dump of that exact schema. Replace
   `agent_libos_runtime` with the `current_schema()` result; do not use a schema
   wildcard.

   ```bash
   psql 'service=agent_libos' -At \
     -c 'SELECT current_schema(); SELECT schema_version FROM runtime_schema WHERE singleton = 1;'
   mkdir -p /srv/backups
   chmod 700 /srv/backups
   (
     umask 077
     pg_dump --format=custom --no-owner --no-privileges \
       --schema=agent_libos_runtime \
       --file=/srv/backups/agent-libos-2026-07-30.dump service=agent_libos
   )
   chmod 600 /srv/backups/agent-libos-2026-07-30.dump
   pg_restore --list /srv/backups/agent-libos-2026-07-30.dump
   ```

   A whole-database dump is acceptable only when that database is dedicated to
   exactly this Runtime schema and contains no unrelated application data.
   The `--schema=agent_libos_runtime` flag enforces that unit, while
   `--no-privileges` prevents source ACL/role dependencies from entering the
   archive; keep the matching restore flag below.
2. Provision a new empty UTF-8 database whose restore service selects the same
   application schema in `search_path`. A named schema selected by `--schema`
   is defined by the archive, so do not pre-create it and cause a
   `CREATE SCHEMA` conflict; a Runtime that deliberately uses the database's
   existing `public` schema keeps that normal empty-database bootstrap. Restore
   into the new target in one transaction; do not clean or overwrite the active
   source target:

   ```bash
   pg_restore --exit-on-error --single-transaction --no-owner --no-privileges \
     --dbname='service=agent_libos_restore' \
     /srv/backups/agent-libos-2026-07-30.dump
   psql 'service=agent_libos_restore' -At \
     -c 'SELECT current_schema(); SELECT schema_version FROM runtime_schema WHERE singleton = 1;'
   ```

   Require the restored `current_schema()` to equal the dumped schema and the
   schema version to equal `3` before opening the Runtime.

3. Point a stopped Host at the restored target. `Runtime.open()` must acquire
   the new target's advisory lease and pass the schema-v3 table, column,
   encoding, and keyset-collation probes before the target is promoted. Keep
   the original database until the restored Runtime also shuts down cleanly.

### Coverage and online-backup boundary

The SQL backup includes durable process, authority, registry, checkpoint/Image
artifact, operation, audit, event, and external-effect rows at the database
snapshot. It also includes any still-full internal root-spawn initial-goal
recovery envelope, so the backup must be protected as payload-bearing evidence.
It does not include other volatile Object payloads that were not explicitly
serialized into an artifact, process workspaces or arbitrary filesystem/Git
state, Host configuration and secrets, live provider sessions, or remote side
effects. Checkpoint restore has the same external-state limitation and is not a
substitute for this database runbook.

This release has no application-level online-backup barrier or cross-provider
recovery-point protocol. SQLite's online backup API and a PostgreSQL
transaction-consistent `pg_dump` may produce a valid SQL snapshot while a
Runtime is live, but Agent libOS does not certify that snapshot as a complete
recoverable runtime point: volatile payloads and external effects can straddle
it. The supported full runbook therefore requires quiescence. PostgreSQL PITR,
physical replication, SQLite filesystem snapshots, cross-version restore, and
in-place downgrade/upgrade are operator-managed and outside the tested product
contract.

Backups can contain Human requests, model/provider metadata, audit trails, and
retained payloads. Protect them at least as strongly as the live store and test
restore into an isolated new target rather than treating an unverified dump as
a recovery plan.

## Active-runtime leases

A persistent target is owned by one writable Runtime at a time.

- File-backed SQLite canonicalizes the database path on every platform. On the
  tested POSIX path, when `O_NOFOLLOW` and `fchmod` are available, it rejects
  unsafe file types and no-follow/path-identity violations for the canonical
  database and existing SQLite sidecars, and tightens their mode to `0600`;
  where `getuid` is available it also requires current-user ownership. An
  existing database symlink is first resolved to its canonical target, so an
  ordinary alias shares the same lease identity. When both `fcntl.flock` and
  `O_NOFOLLOW` are available, the Runtime holds both a non-blocking lock over a
  separately hardened path sidecar and an owner-only private identity lease
  keyed by the validated database `(st_dev, st_ino)`. The database, path lease,
  identity lease, and SQLite sidecars must be regular, current-user-owned,
  single-link files, and the database path is checked before and after opening
  against the selected identity. In addition, the live connection holds
  SQLite's exclusive database lock on the database it actually opened. That
  connection-level lock preserves the one-writer boundary even if a same-UID
  filesystem administrator races a pathname retarget during open. Python's
  standard SQLite driver does not expose the database file descriptor, however,
  so Agent libOS cannot prove that a path replaced at exactly the connect
  boundary still names the originally selected inode; a same-UID actor with
  write authority over the database directory can cause denial or misdirection
  and is inside the Host trust boundary. Keep the database and its parent
  owner-only, and never rename, replace, or relocate the database path or parent
  while a Runtime is alive because SQLite journal/WAL naming remains
  path-dependent. Where the POSIX sidecar/identity mechanism is unavailable,
  the exclusive database lock still supplies the single-writer lease but does
  not claim the unavailable path/inode, ownership, mode, no-follow, or
  single-link guarantees.
- PostgreSQL uses a session advisory lock derived from the database/schema
  identity. Session close is the single ownership-release point; close never
  attempts a separate explicit advisory unlock. Cleanup failures are reported
  without replacing the primary failure, and connection loss releases the lock.
- In-memory SQLite has no cross-process lease because every connection is an
  independent store.

Do not open a GUI server and another writable CLI Runtime against the same
persistent target. Do not edit capability, trust, label, decision, or evidence
rows directly; supported mutations must publish their coupled generation and
audit/event evidence.

See [Configuration Reference](configuration.md) for library and product
configuration precedence, [CLI Reference](cli.md) for backend selection, and
[Architecture](architecture.md) for the authority, evidence, primitive, and
Runtime dependency boundaries.
