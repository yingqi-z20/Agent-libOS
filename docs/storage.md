# Runtime Storage

Agent libOS 1.5.2 stores durable runtime state through a `UnitOfWork` composed of
explicit domain boundaries, including `ProcessRepository`,
`ResourceRepository`, `RuntimePublicationRepository`,
`SnapshotCheckpointRepository`, `RuntimeModuleRepository`,
`PayloadRetentionRepository`, `SemanticAssessmentRepository`,
`McpContinuationRepository`, `McpRemoteTaskRepository`,
`McpSubscriptionRepository`, `McpAuthMetadataRepository`,
`ObjectRepository`, `AuthorityRepository`, `EvidenceRepository`, and
`ExtensionRepository`. All repositories in one runtime share the same
transaction coordinator. Migrated runtime domains use those repositories. The
concrete SQL store remains a Host composition/lifecycle boundary and is still
passed to a small set of reviewed legacy services such as Explain; it is not yet
hidden from every runtime component. New domain persistence should use the
typed/repository boundary rather than extending that compatibility surface.

## In this guide

- [Strict store schema v7](#strict-store-schema-v7)
- [Offline migrations](#offline-v6-to-v7-migration)
- [Transaction model](#transaction-model)
- [Durable authority and evidence](#durable-authority-and-evidence)
- [Backup and restore runbook](#backup-and-restore-runbook)
- [Active-runtime leases](#active-runtime-leases)
- Return to the [documentation home](index.md).

Process, resource-accounting, runtime-publication, operation/evidence,
module-publication, and Snapshot/Checkpoint persistence have explicit typed
Protocols and facade methods. Snapshot services exchange canonical `SnapshotRows` and
`ProcessSnapshot` aggregates with the repository; backend SQL, generic table
helpers, and Object-payload cache coordination stay behind that boundary. An
AST ratchet rejects raw SQL or generic table-helper regressions in the migrated
runtime services. Payload-retention scans and compare-and-swap reductions use a
separate typed repository so maintenance cannot regain the generic extension
facade or load full evidence history into Runtime services.

Those migrated repository surfaces, including payload retention and semantic
assessment jobs/evidence, bind to explicit backend Protocols. A `UnitOfWork`
validates concrete method presence
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
implementation and canonical store schema v7. The shared implementation emits a
small SQLite-shaped SQL subset. The PostgreSQL dialect translates parameter
placeholders, `COLLATE BINARY`, `INSERT OR IGNORE`, the one reviewed
`INSERT OR REPLACE` upsert, `INDEXED BY`, the Skill-package description
`json_extract` expression, and the table/index metadata probes; a
syntax-surface test ratchets that set. PostgreSQL therefore does not inherit the
SQLite store or its connection/locking behavior, but it is also not a second,
copied repository implementation.

PostgreSQL support requires the optional `postgres` dependency extra and a
server whose major version is `17`. The committed canonical catalog records
PostgreSQL major 17, so startup fails closed on another major even if the
application relations otherwise appear compatible. PostgreSQL 17.10 is the
tested, manifest-generation baseline. Another 17.x release is not rejected by
the major-version field alone, but remains outside the validated release matrix
and must still match the complete canonical storage catalog.

The library default is the reserved SQLite target `user`.
`Runtime.open()` and `Runtime.aopen()` do not load the repository's
`config.yaml`; with neither an explicit target nor an explicit config they use
`DEFAULT_CONFIG`, whose `runtime.local_store_target` is `user`. That target
resolves to `~/.agent-libos/runtime/agent-libos.sqlite`, so state survives a
close and reopen without placing the RuntimeStore in the model-visible
workspace. On POSIX, the `~/.agent-libos/runtime` directory must be owned by the
current user and is fixed to mode `0700`; symlink or reparse-point components
are rejected on every supported platform.

Explicit `local` and `:memory:` targets remain independent in-memory SQLite
stores whose state disappears when closed; a bare `sqlite://` target has the
same in-memory meaning. An explicit filesystem path or `sqlite:///...` target
selects persistent SQLite, while a PostgreSQL DSN selects PostgreSQL. Before a
persistent SQLite database, runtime lease, journal, WAL, or SHM sidecar can be
created, Runtime startup Host-canonicalizes both the target and the effective
local workspace and rejects a target equal to or below that workspace. This
applies to implicit and explicit local substrates and
to caller-owned stores before model primitives are installed. In-memory
SQLite, PostgreSQL, and non-local substrates do not have this filesystem
overlap check.

There is no automatic migration from a legacy project-root
`.agent_libos.sqlite`. Stop every writer and move or restore that database to a
location outside the effective workspace before selecting its explicit path;
do not copy a live SQLite database or omit its required sidecars.

On Windows, startup also treats persisted filesystem identity as an authority
boundary. Before recovery, and again after recovery while its lease is held, it
uses bounded keyset pages to validate every active filesystem capability, live
file-label binding, and checkpoint capability row that restore could reactivate
against the current canonical resolver. A non-canonical record stops startup
before `STARTING`/`OPEN` with only its opaque capability, binding, or checkpoint
identifier in the error; no path is disclosed and no record is rewritten.
Inactive or tombstoned history that cannot become effective is excluded. There
is no automatic migration for legacy Windows aliases: an operator must inspect
and repair or replace the store offline before reopening it.

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

## Strict store schema v7

Fresh databases created by Agent libOS 1.5.2 use store schema v7 and create a
`runtime_schema` table with one marker row; the canonical DDL constrains its
`singleton` value to `1`. Opening an existing store requires the row selected
by `singleton = 1` to contain schema version `7`. Product version and store
schema version are independent identifiers: `1.5.2` is the current product
release, while `7` is the persisted schema contract. Both backends apply
the same acceptance rules, with one backend-specific initial probe order:

1. SQLite validates `PRAGMA encoding` before reading the version marker;
   PostgreSQL reads the marker first and validates server encoding with the
   remaining shape probes.
2. For a version-7 marker, run the focused relation, table/column, counter-seed,
   encoding, keyset-collation, and recovery-index probes, then require the
   complete backend canonical storage catalog. The v7 Task
   Run/recovery/semantic/FlowGraph index contract fixes each required index's
   table, ordered columns, uniqueness, full/partial shape, predicate, direction,
   and keyset collation. A PostgreSQL required index must additionally be valid,
   ready, and live. A required SQLite relation must have
   `sqlite_master.type = 'table'`, and a required PostgreSQL relation must have
   `pg_class.relkind = 'r'`; a same-column view cannot impersonate a manifest
   table and is rejected before the initializer can mutate the schema.
3. Reject a wrong marker, a store with a missing/extra manifest table or
   column, a store missing any other probed surface, or an unversioned target
   containing a probed user object. SQLite enumerates all
   non-`sqlite_%` schema objects; PostgreSQL's freshness probe covers `pg_class`
   relation kinds `r`, `p`, `v`, `m`, `S`, and `f`, not function/type/domain-only
   schemas.
4. In one transaction, run the idempotent initializer. For an accepted existing
   version-7 store this can insert allowed non-manifest seed rows such as the
   system namespace; required counter seeds are validated during preflight and
   cannot be repaired here. Its `CREATE INDEX IF NOT EXISTS` statements cannot
   repair a missing or malformed v7-manifest index because preflight has already
   rejected that store without writing. For an empty target it creates the
   complete schema and then writes the marker.

An interrupted bootstrap rolls back both schema and marker, so reopening the
same empty target retries initialization instead of misclassifying it as an
unsupported older database.

The DDL emitted for a fresh schema-v7 store is the current release shape,
including typed process wait/outcome fields and the Durable Task Run ledger,
payload, resume-point, command, requirement, and link tables, plus Human
request revisions, semantic job/evidence tables, FlowGraph, policy/control,
machine-settlement, review, and health evidence tables. Schema v7 additionally
adds five payload-free MCP tables: `mcp_continuations`, `mcp_remote_tasks`,
`mcp_subscriptions`, `mcp_auth_metadata`, and
`mcp_side_effect_preparations`. They contain local identities,
server/owner/auth-generation fences, SHA-256 commitments, lifecycle revisions,
expiry/counter state, closed diagnostic metadata, and opaque credential-broker
references only. OAuth tokens and client secrets, authorization codes,
PKCE/state, bearer-like remote task ids, raw MRTR request state, notification
bodies, and raw provider results have no Store column.
The preparation table is a crash-recovery ownership sidecar. The Runtime first
persists a preallocated Human request id and exact reserved credential-broker
slot references, then creates the Human request and writes those exact broker
slots. The same preparation precommits the bounded opaque refs and Human
preview digest that an update will supersede. One SQL transaction inserts,
revision-CASes, or retention-deletes the continuation/Task row and changes the
preparation from `prepared/abort` to `cleaning/retire`. Normal completion and
restart recovery then cancel the superseded Human request, delete only the old
broker slots, and revision-delete the sidecar. A precommit crash instead claims
the `prepared` row, cancels only the newly allocated Human request, deletes only
the newly reserved slots, and removes the sidecar; it never replays Provider
I/O. Retirement-only preparations may own no new slot: their strict typed
metadata is the durable ownership record before a terminal projection is
physically removed.
When a protected result publishes more than one durable MCP projection, each
prepared-row commit joins the same outer RuntimeStore transaction as the
originating effect settlement. Broker and Human cleanup is deliberately
deferred until that outer transaction commits. Rolling it back therefore
publishes none of the projections and restores every sidecar to `prepared`, so
startup can abort their preallocated external state without replaying the
Provider.
For an initial protected result, that same commit writes only a closed
`{kind, continuation_id|task_ref}` projection into the finalized external
effect's provider receipt. It contains no remote id, request state, Human
payload, or broker value. This receipt is the durable Host handoff if the
process exits after commit but before the local ref reaches its caller; the
Host accepts it only when the referenced main row carries the same effect
fence.
Every later continuation result uses the same transaction. A next-round
receipt points to the unchanged local continuation id while its new request
state remains broker-only; a completed result terminalizes the continuation
without persisting the Provider content; and a Task handoff binds the Task to
the response effect while the Continuation's broker receipt carries only the
safe Task ref and response-effect id needed to verify that recovery chain.
The sidecar contains only opaque local references, SHA-256 commitments, fixed
authority fences, and closed diagnostic codes, never the Human preview body or
credential-broker value.
Terminal continuation rows (`complete`, `cancelled`, `expired`) and terminal
remote-task rows (`completed`, `failed`, `cancelled`) have bounded oldest-first
retention queries and revision-fenced deletion. `needs_attention` rows use the
same retention path once the manager has preserved their unknown-dispatch
evidence; they do not consume the active-record cap forever. The manager first
writes a retirement-only sidecar, then atomically deletes only an unchanged
retained row while advancing that sidecar to `cleaning`; it never deletes a
broker slot or cancels a Human request ahead of the SQL fence. Cleanup is
idempotent and leaves the Human identity, effect, event, and audit evidence.
The narrow legacy terminal-delete repository method is safe only for a row with
no Human, broker, or result ownership and rejects every other row; callers must
use atomic terminal retirement for those rows.
An `input_required` remote-task row must bind a real durable
`human_requests.request_id`. A Human request cannot be shared by any MCP
continuation or remote-task row. A physical `NO ACTION` foreign key proves the
referenced Human row exists and prevents lifecycle cleanup from cascading away
MCP evidence. Both backends serialize cross-table binding writes through a
common durable fence so concurrent continuation/task inserts or CAS transitions
cannot pass independent absence checks. A task binding can be introduced only
with an `input_required` state, and another round atomically replaces it under
revision CAS. Opaque broker slots are also single-owner across continuation
state and both remote-task broker fields; a task's remote-id and state slots
must be distinct. Other task states may clear or retain the local audit reference,
but no Human preview or response is copied into the task row. Metadata keys and
values both use a closed Host-owned code vocabulary; an arbitrary provider
string is rejected even when it looks like a short lowercase diagnostic code.
A version marker
alone is insufficient: stores missing a required table, required column,
canonical keyset collation, required v7 index/unique shape, or required
recovery predicate are rejected, and the runtime does not present that
rejection as a migration.

The open-time validator does not compare raw DDL bytes: it normalizes the
backend catalog before comparison. It nevertheless requires the complete
canonical storage catalog captured for that backend, not merely the focused
manifest probes above. SQLite compares every user table, index, trigger, and
view; normalized `sqlite_master` definitions; table options; column type,
nullability, default, primary-key position, generated/hidden shape; foreign
keys; and complete index keys, direction, collation, uniqueness, origin, and
partial shape. PostgreSQL compares the pinned server major; relations and their
persistence/access method/options/replica identity/partition/RLS shape;
columns, types, order, nullability, defaults, collations, identity and generated
shape; constraints; indexes; triggers; policies; rewrite rules; and
inheritance. Missing, extra, or changed captured entries fail closed. The
PostgreSQL catalog does not claim to inventory a schema containing only
standalone functions, enum/base types, or domains that create none of the
captured relation kinds; those objects are outside the RuntimeStore DDL.
Operators must create stores through this release's backend rather than
treating a hand-built approximation as canonical.

Text columns that form durable startup, recovery, or retention keysets have a
canonical bytewise collation: `BINARY` on SQLite and `"C"` on PostgreSQL. Both
the primary timestamp and the stable textual tie-breaker use that physical
shape, and their covering indexes use the same collation. Queries inherit the
validated column collation so SQLite retains composite row-value range seeks.
SQLite database files and PostgreSQL servers must both use UTF-8 encoding;
under UTF-8, `BINARY` and `"C"` ordering match Python's Unicode string ordering
for persisted cursor values. Opening an existing version-7 store fails closed
when its database encoding, any required keyset text column, or any required
keyset text-column collation is not canonical; those required column
collations are checked with one set-based catalog probe. A UTF-16 SQLite file
or locale-inheriting draft PostgreSQL schema is therefore rejected rather than
silently paginating with a different order or degrading to a sort. That focused
set-based keyset probe does not itself inspect other text-column collations;
the complete canonical catalog comparison above still rejects drift in them.

The only supported migrations are the explicit, offline, operator-invoked
canonical v4-to-v5, v5-to-v6, and v6-to-v7 procedures below. They must be run
in order.
There are no automatic migrations, backfills, read-only compatibility modes,
or dual runtime schema paths. A v3
database remains archive-only and must be opened with Agent libOS 1.0.1.
Agent libOS 1.5.2 raises `UnsupportedStoreVersion` during ordinary Runtime
preflight, before initialization, index creation, seed insertion, recovery,
audit, or any other write. The same zero-write rule applies to v4/v5 before the
matching offline migrator runs, older/unversioned stores, and malformed v7 stores.
SQLite performs this probe against a private snapshot so rejection leaves the
original database, WAL/SHM sidecars, and any existing lease bytes and modes
unchanged.
Checkpoints and checkpoint-derived Image artifacts remain independently
versioned and are rejected before operation evidence or process state is
written when their own format is unsupported.

### Semantic assessment records

Schema v5 adds `human_requests.revision`, mutable
`semantic_assessment_jobs`, and append-only `semantic_assessments`.
Human-request terminal and delivery transitions use revision/status CAS so a
concurrent answer, cancel, or delivery update has one winner and stale ABA
attempts fail.

Jobs persist only typed references/digests and a bounded safe projection.
Approval and provider-ingress projections are metadata-only. A nonterminal
root-goal job may temporarily carry deterministic redacted intent only when its
`public`/`normal`, non-mixed-identity text is within 2,000 characters and passes
local secret/path detection; otherwise it is metadata-only. A job claim
increments its revision, records one lease owner/id/expiry, and raises its
external attempt count from zero to one. Terminalization requires the expected
revision/status/lease, clears the lease, writes a terminal status, appends the
assessment in the same transaction, and reduces the projection to `{}` with
`hash_only` retention. Expiry and off-mode cleanup use bounded pages without
claiming or dispatching the provider. An attempted external request whose
outcome is not known is terminal `provider_outcome_unknown`; lease recovery
does not turn it into another provider call.

Local Host DLP may place at most four closed
`category`/`code`/`evidence_sha256` objects in a nonterminal projection. It
never stores matched text. A hit forces metadata-only projection; terminal
production merges corresponding Host-source findings and monotonic data
findings before the same append-and-scrub transaction, including on classifier
failure paths.

The assessment repository exposes append, get, and bounded keyset query only.
Appending the exact same record is idempotent; the same ID with different
content fails and there is no generic update/delete path. Records carry closed
findings, Shadow result, action id, normalized Human observation,
timestamps/latency, calibration, versioned nullable token/cost fields, an
optional Host-supplied tenant bucket digest, and complete classifier/input/
feature/policy plus
applicable manifest/action/resource/arguments/state/source/Sink/tool/provider
and projection digests.
They never contain the job projection, prompt, goal/provider text, raw Human or
classifier response, or reasoning. Unicode keyset ordering uses the same
validated bytewise collation as other durable pages. Queries may additionally
apply exact `action_id` and `tenant_bucket_sha256` filters before the bounded
keyset page; the nullable tenant key is an opaque Host-derived digest rather
than stored tenant plaintext, and is not by itself an anonymization guarantee.
For external classifier dispatches, the three nullable token/cost fields may
contain only selected exact non-negative integers from an exact Host
`LLMCompletion` and an exact built-in usage dictionary limited to 64 exact
string keys. Values above the cross-layer JSON/TypeScript safe-integer maximum
`2^53 - 1` are rejected before persistence. The schema-v1 assessment record
uses that same ceiling for latency and nullable token/cost counters, so every
valid durable value remains exactly representable by the HTTP/CLI/GUI read
surfaces. The canonical
`input_tokens`/`output_tokens` keys accept `prompt_tokens`/`completion_tokens`
aliases only when both forms, if present, are valid and equal;
`cost_microunits` has no alias. Conflict or invalidity nulls only the affected
counter. Unknown keys, raw usage, provider response content, and arbitrary
mapping/completion subclasses are never serialized. Deterministic/scripted or
missing/untrusted telemetry remains `null`. These counters can accompany an
`invalid_schema` assessment because extraction follows a completed transport,
but they remain provider telemetry rather than billing proof or authority.
Semantic job, assessment, and cursor timestamps are normalized to canonical
UTC with six fractional digits and an explicit `+00:00` offset before
persistence and keyset comparison.

Checkpoint/image snapshot codecs do not own these Host evidence tables.
Restore and fork therefore neither copy nor delete semantic assessments.

### Phase 2–4 semantic records

Schema v6 adds four append-only FlowGraph relations:
`semantic_flow_entities`, `semantic_flow_activities`,
`semantic_flow_edges`, and `semantic_flow_label_assertions`. They retain
bounded identifiers, versions, labels, coverage, and provenance digests only;
raw bodies, paths, argv, prompts, responses, credentials, and reasoning are
forbidden. Cross-tenant edges fail atomically, Host edges are never removed,
and model assertions can only tighten sensitivity/integrity/trust.

`semantic_policy_epochs` is immutable and uniquely indexes both generation and
policy digest. Each row also stores a bounded canonical rollout scope made only
from tenant, auto-rule, hard-deny, classifier-artifact, and resource SHA-256
commitments plus action/right and bounded allow-parameter metadata; no raw rule
identifier, profile identifier, or resource is retained. This scope lets
startup prove subset narrowing and detect same-cardinality swaps, weakened
denies, relaxed confidence/TTL/rate limits, classifier drift, and budget-shard
rule replacement without treating aggregate metrics as authority evidence.
`semantic_control_state` is the one
revision/generation-CAS pointer for `off`, `shadow`, `enforce_deny`, or
`canary_auto`, including trip
state; every successful pointer change also appends
`semantic_control_transitions`. `semantic_rate_budgets` is the mutable
revisioned per-tenant/logical-rule counter. Its `epoch_id` is an immutable
origin/provenance anchor, not a partition key: rotating an epoch, reopening the
Store, or tightening a limit never resets minute, day, or inflight usage.
Static Host configuration is the only policy source; these tables do not create
a remote control API.

`semantic_machine_settlements`, `semantic_machine_outcomes`,
`semantic_review_labels`, and `semantic_health_events` are append-only. They
bind request/effect/epoch/tenant/action/binding/decision digests and closed
outcomes without source payloads. Reviews contain only `safe`, `unsafe`, or
`inconclusive` plus reviewer/evidence digests. A Host review append cannot
settle a request or mutate policy/control. Checkpoint restore and fork neither
copy, rewind, nor delete these Host-global records.

## Offline v6 to v7 migration

The v6-to-v7 command is an offline administrative surface. Ordinary Runtime
startup never imports or invokes it:

```bash
uv run agent-libos --db <target> store migrate --to 7 --dry-run \
  --sqlite-backup <verified-v6-backup>
uv run agent-libos --db <target> store migrate --to 7 --apply \
  --expected-plan-sha256 <digest> \
  --sqlite-backup <verified-v6-backup>
```

For PostgreSQL, replace the SQLite backup option with
`--postgres-snapshot-confirmed` on apply. Planning validates an exact canonical
v6 source and emits a deterministic contract and plan digest without writing.
SQLite planning additionally validates that an independent, current-user-owned,
single-link backup is a logical match for the source. Apply obtains the same
exclusive SQLite lease or PostgreSQL advisory lock as Runtime startup,
revalidates the source and recovery evidence, creates the five MCP v7 tables,
compare-and-swaps the singleton marker `6 -> 7`, and validates the complete
canonical v7 catalog before commit. Any DDL, marker, catalog, lease, backup, or
digest failure rolls back the whole transaction, including its migration
receipt.

Migration plan schema v2 binds the digest to the hashed database/schema and
cluster identity, exact source catalog, source logical or locked PostgreSQL
relation-state digest, source-observation receipt, migration-receipt contract,
migration implementation, and Agent libOS version. PostgreSQL planning and
apply hold the Runtime advisory lease, use a repeatable-read transaction, and
take schema-qualified `ACCESS EXCLUSIVE NOWAIT` locks on every frozen source
relation before capturing relation OIDs and visible `ctid`/`xmin` identities;
row payloads are neither persisted nor included in the plan. Apply appends an
exact plan-bound record to `audit_records` in the same transaction as DDL and
the marker update. If commit acknowledgement or the post-commit readback is
lost, retain the reviewed plan and recovery point and repeat `--apply` with the
same digest. Only a target with that exact receipt, reconstructed source state,
canonical catalog, and migration postconditions is reconciled, returning
`applied=false` and `already_applied=true`; a different database, modified
target, missing/tampered receipt, or generic same-version store is rejected.

The migration does not copy provider content or create auth credentials. New
tables begin empty. OAuth tokens, client secrets, authorization codes,
PKCE/state, remote task ids, continuation request state, resource/prompt
content, and subscription events remain outside RuntimeStore. Only later MCP
operations may write their typed, payload-free projections through the v7
repositories.

## Offline v5 to v6 migration

The v5-to-v6 command is an offline administrative surface and is never imported
or invoked by ordinary Runtime startup:

```bash
uv run agent-libos --db <target> store migrate --to 6 --dry-run \
  --sqlite-backup <verified-v5-backup>
uv run agent-libos --db <target> store migrate --to 6 --apply \
  --expected-plan-sha256 <digest> \
  --sqlite-backup <verified-v5-backup>
```

For PostgreSQL, omit the SQLite backup option and pass
`--postgres-snapshot-confirmed` on apply.

It uses the same safety protocol as the legacy migration below: stop all
writers, establish an independent recovery point, validate the complete
canonical v5 source storage catalog, obtain the SQLite exclusive lease or
PostgreSQL advisory lock, repeat source/backup validation, and run DDL plus
singleton marker CAS `5 -> 6` in one transaction. It then validates the
complete canonical v6 storage catalog, including every captured relation,
column, constraint, index, and backend hook, before commit. Failure at any
point rolls back DDL, the PostgreSQL migration receipt, and the marker.

SQLite dry-run is zero-write and requires `--sqlite-backup` to validate an
independent, current-user-owned, single-link `0600` regular file without live
journal/WAL/SHM sidecars. Apply revalidates that the backup still matches the
locked source. PostgreSQL apply requires `--postgres-snapshot-confirmed`; the
operator, not Agent libOS, owns and verifies that snapshot. The plan digest is
deterministic for the locked logical source and must be supplied exactly to
apply. A v4 store is not accepted by `--to 6`; it must independently complete
the v4-to-v5 workflow first.

The PostgreSQL migration role must be able to execute
`pg_catalog.pg_control_system()`. Migration identity fails closed if the
cluster system identifier is unavailable; only hashes of cluster/endpoint
identity are emitted, never the DSN or raw host values. The role must also own,
hold `MAINTAIN` on, or hold the applicable write privilege for every frozen
required table so both dry-run and apply can take `ACCESS EXCLUSIVE` locks.

## Offline v4 to v5 migration

The migration command is an offline administrative surface and is deliberately
not imported or invoked by normal Runtime startup:

```bash
uv run agent-libos --db <target> store migrate --to 5 --dry-run
uv run agent-libos --db <target> store migrate --to 5 --apply \
  --expected-plan-sha256 <digest> <backup-confirmation-option>
```

The supported procedure is:

1. Stop every Runtime that can write the SQLite file or PostgreSQL
   database/schema, and verify the Runtime has released its active-store lease.
2. Create an operator recovery point. For SQLite, use SQLite's own backup
   command to produce an independent, quiesced, current-user-owned,
   single-link `0600` regular file with no journal/WAL/SHM sidecars. For
   PostgreSQL, create and verify an operator-managed snapshot of the exact
   `current_database()` and `current_schema()`.
3. Run `--dry-run` (with `--sqlite-backup` when validating SQLite). It inspects
   a private SQLite snapshot, or holds the PostgreSQL advisory lease and
   schema-qualified relation locks in a read-only repeatable-read transaction,
   validates the complete canonical v4 storage catalog and source digest,
   performs zero source/lease/sidecar writes (the PostgreSQL locks are
   session-only), and prints a versioned plan plus deterministic `ddl_sha256`
   and `plan_sha256`.
4. Review the plan. Run `--apply` with that exact
   `--expected-plan-sha256`. SQLite additionally requires the same verified
   `--sqlite-backup`; PostgreSQL requires
   `--postgres-snapshot-confirmed`.
5. Apply obtains SQLite's exclusive offline lease or PostgreSQL's Runtime
   advisory lock, repeats canonical v4 and backup/source validation, and opens
   one transaction. It adds `human_requests.revision`, creates the two semantic
   tables/indexes, appends the exact plan-bound PostgreSQL migration receipt,
   compare-and-swaps the singleton marker from 4 to 5, runs the complete
   canonical v5 storage catalog validator, and only then commits.
6. Open the migrated target with this release and archive the plan/result with
   the operator recovery record. Keep the backup until application validation
   is complete.

A missing/mismatched plan digest, stale or non-self-contained SQLite backup,
absent PostgreSQL confirmation, lock conflict, noncanonical v4 input, failed
DDL, marker CAS miss, or failed v5 readback aborts. The transaction rolls back
schema changes, the PostgreSQL migration receipt, and the marker; it never
silently repairs a malformed source.
Planning an already-v5/newer or pre-v4 store is not an idempotent “success”:
planning accepts only an exact canonical v4 source. The narrower exception is
an apply retry with the same schema-v2 plan and recovery evidence after an
uncertain commit; an exact v5 migration result is reported as
`already_applied=true`. See [the CLI guide](cli.md#offline-store-migration)
for concrete commands.

On POSIX, SQLite apply additionally requires both source and independent backup
to be current-user-owned, single-link regular files with exact mode `0600`.
Dry-run does not chmod either file; an insecure source must be corrected by the
operator before apply.

### Durable Task Run records

Schema v4 adds first-class Task Run state. `task_runs` is the mutable current
projection and is updated only by revision compare-and-swap under the active
Runtime epoch. `task_run_requirements`, `task_run_payloads`,
`task_run_resume_points`, and `task_run_commands` hold the versioned inputs,
safe local continuation points, and idempotent Host command receipts.
`task_run_ledger` is append-only; `task_run_links` associates a ledger item
with existing Operation, external-effect, Human, LLM, checkpoint, ObjectTask,
or other evidence without copying that evidence into the Run. Process rows
carry an optional Run identity, epoch, and role so child spawn and fork can
inherit the supervising Run transactionally.

For a split-phase Host mutation, a `task_run_commands` row fixes the command
kind and canonical request hash. Every existing-Run mutation includes its
`expected_revision` in that canonical envelope; changing only the revision
under an existing command id is a command conflict and writes nothing. Its
bounded `result_json` and `result_revision` may then advance only by
compare-and-swap from a provisional
`settlement_state=pending` summary to the exact completed summary. This update
is intentionally separate from already committed Run/process state: a crash,
lost response, or failure to write the final result leaves the pending row as a
no-redispatch fence rather than evidence that the command never ran. Exact
replay of `run`, cancel/deadline, resume, or interrupt receipts may settle only
local generation-fenced state. It does not run another scheduler quantum,
Provider, Tool, or external effect.

Command-result decoding is strict and bounded. The Store rejects an oversized
raw UTF-8 `result_json` before parsing, and writes recheck the canonical encoded
size against `task_runs.command_result_max_bytes`. The version-1 envelope must
contain the complete canonical public TaskRun summary bound to `run_id` and
`result_revision`, and the Runtime accepts only the exact key set for the
command/request-selected variant. Missing/extra keys, pending-only fields on a
completed receipt, Boolean integers, and values outside signed BIGINT bounds
fail closed.

Split local-control pending and completed variants retain
`admission_ledger_seq`, `admission_ledger_item_id`, and
`admission_evidence_sha256`. Those fields identify the same-transaction
append-only `STATUS_TRANSITION` item whose Run, command id/kind, request hash,
from/to status, and canonical semantic evidence must all match before replay,
including a terminal or superseded early return. Interrupt pending receipts
also carry the raw admission Runtime epoch and canonical per-PID fences plus
`interrupt_provenance_sha256`; completion drops the raw epoch/fences but retains
that digest and the ledger reference. The digest still binds the removed
pending fields through the immutable admission item.

Authoritative effect-receipt recovery commits the provisional command row and
the verifier-normalized ExternalEffect settlement in one transaction. If that
commit survives but completion of the command result does not, replay checks
the stored Run/effect/cancellation generation/admission epoch together with the
exact `settlement_transition_seq` and `settlement_audit_record_id`. The former
must name the append-only finalized effect transition and the latter the
matching `external_effect.recovery_settled` audit decision sourced from
`host_verified_receipt`; replay does not depend on purgeable provider metadata
or receipt bodies and completes locally without calling the verifier again.
For a recoverable nonterminal Run,
startup settles already-staged complete provider results before it interprets a
pending interrupt. Its eligible interrupt command scan is hard bounded; an
oversized set, duplicate current-generation pending interrupt, or malformed
interrupt receipt is not truncated into a recovery decision and instead blocks
the Run in `needs_attention`. Cancellation/deadline state may outrank an older
interrupt, and a terminal Run is not scanned merely to complete a command row;
that historical receipt is completed only by exact client replay. See the
complete [crash-safe command contract](durable_task_runs.md#crash-safe-command-settlement).

Linked recovery also has a bounded command-only repair path for the crash gap
between its committed deterministic nested rerun and its outer recover receipt.
The nested request hash includes a versioned parent command id and outer-request
hash. Exact outer replay may insert the missing receipt only after matching that
nested command, its immutable source/target summaries, the target's create
receipt and current Run identity, and exactly one append-only `rerun_of` link.
It copies the nested result and result revision; it does not create another Run
or select the source's current revision. A changed request or malformed,
missing, duplicate, or over-bound evidence rolls back the repair.

The Task Run ledger is durable metadata, not a tamper-proof event log. A
database administrator can modify it. A Task Run's readable goal, follow-ups,
resume bundle, and result material are a separate plaintext payload boundary.
They may be written only when the Host has explicitly enabled Durable Task
payload persistence. This release does not encrypt those values at rest; SQL
database files, server administrators, and backups can read payloads retained
as `plaintext`.

The default Run retention policy is `purge_on_terminal`. Terminalization first
enters `finalizing` and, in one transaction, replaces Run-owned readable
payloads and linked LLM/tool-output content with canonical hash-only
projections, deletes pending LLM continuation rows, and removes resume points.
It deletes durable messages whose Run binding was derived from their Run-member
recipient; an ordinary caller cannot suppress, override, or forge that binding.
It replaces Run-linked Human request prompt, response, and decision bodies with
hash projections while retaining request id, type, status, timestamps, audit
linkage, and content digests. Linked terminal external-effect provider metadata
and provider receipt bodies also move monotonically to canonical hash-only
envelopes. Effect identity, state, classification, canonical-argument hash,
original payload digest, receipt digest, and causal links remain; readable
receipt content does not. The Run does not report `succeeded`, `failed`, or
`cancelled` if that cleanup cannot commit. `permanent` retention is a Host/admin
choice made at Run creation and skips automatic Run-terminal cleanup; ordinary
evidence-retention policy remains independently applicable. A Host/admin may
later apply the same audited cleanup explicitly to a terminal permanent Run.
Neither policy makes ordinary Object Memory payloads durable.

The backend's active-store lease still enforces one writable Runtime per
database/schema. Schema v4 additionally records a monotonic Runtime epoch for
Task Run execution. Claims, child publication, resume-point publication, and
terminal settlement reject stale epochs. Every Task Run command insert and
result update also performs a conditional no-op update of the global
`task_run_runtime_epoch` counter row, using it as the SQLite/PostgreSQL row lock,
then requires that same current epoch in the command mutation. This includes
the linked-recovery missing-parent receipt path, so a superseded manager cannot
insert or complete a command after a successor Runtime owns the store. This fencing protects cooperating
Runtime instances; it does not protect against direct database writes.

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

While holding the lifecycle recovery lease, the builder validates recoverable
TaskRun plaintext and integrity bindings without dispatch. It then drains
durable recovery work in this exact dependency order:

1. reconcile crash-interrupted MCP continuations;
2. reconcile crash-interrupted MCP remote Tasks;
3. reconcile crash-interrupted MCP subscriptions;
4. recover prepared protected operations;
5. reconcile pending external effects;
6. recover semantic authority;
7. abandon stale capability-use reservations;
8. recover resource-usage reservations;
9. recover incomplete process-exec publications;
10. recover incomplete process-launch publications;
11. recover incomplete checkpoint-restore publications;
12. recover root-spawn initial-goal payloads;
13. recover missing volatile Object payloads;
14. rehydrate registered JIT tools;
15. interrupt stale Explainable Operations;
16. recover stale process execution leases;
17. recover Object Tasks;
18. recover incomplete process-terminal cleanup intents; and
19. perform TaskRun startup recovery.

The three MCP restart reconciliations use durable compare-and-set transitions
and are never provider replay. Prepared protected-effect recovery restores exact
linked reservations only when its intent proves that no provider phase began.
Pending-effect reconciliation follows and must complete before the authority
repository abandons remaining `reserved` rows: a provider receipt can prove
that an effect never started and atomically restore its bound reservation. The
subsequent stale cleanup uses the
`(status, created_at, reservation_id)` keyset index. This ordering eliminates
the former startup-wide external-effect JSON scan and its unbounded protected
reservation set.

Active resource-usage reservations are then recovered through bounded
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
`process_tool_bindings` is part of the complete fresh version-7 release shape,
not a lazy projection or startup backfill. A draft version-7 database that
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
are non-null in the fresh schema-v7 DDL, and row decoders require their canonical
object shapes and complete security labels. Malformed persisted values fail
closed instead of being repaired heuristically; other schema fields may still
be nullable where their domain permits it.

Process control state is persisted structurally. `wait_state_json` is a tagged
child/message/human/tool/pause/Host-resume/stale-execution wait, `outcome_json` is a tagged
exited/failed/killed outcome, and `state_generation` advances on every semantic
state transition. Normal runtime orchestration makes those transitions through
one `ProcessTransitionService`. The only explicit exceptions are typed
repository CAS primitives whose state update must be atomic with an execution
lease or snapshot restore; an exec-epoch commit additionally requires the exact
non-null admission token recorded by the matching applying `process_exec`
publication at its final pre-commit phase, then CASes RUNNING status,
generation, owner, and lease.
Startup stale-execution recovery is the other Store-owned semantic boundary.
Its exact owner/lease/state/execution-generation CAS atomically pauses the row,
clears the raw execution token, and writes an owner/lease-identity-hash-only
`StaleExecutionProcessWait`; an owner or generation that changes after the
recovery SELECT makes that row a CAS loser and it is not paused. Only the PID,
canonical recovering-Runtime owner-id/prior-owner/prior-lease SHA-256 values,
and generation receipt are projected. The recovering-owner digest is an
identity hash rather than a cryptographic signature; this provenance relies on
the RuntimeStore and database administrator being inside the trusted computing
base. A later exclusive Runtime may accept the historical recovering-owner hash
across another reopen, but TaskRun admission/current epochs, per-process
generation fences, safe-point integrity, and current bindings remain
authoritative in their existing rows and are not copied into the generic
process wait. Normal transition and execution-completion APIs reject callers
that attempt to create this reserved receipt.
For every current-v7 typed wait and outcome, including stale execution,
`status_message` is only a compatibility projection for older clients and is
never parsed as the control protocol. The
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

This runbook covers the supported recovery unit: one quiesced schema-v7 SQL
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
   product `1.5.2`, store schema `7`.
5. Prepare an owner-only backup directory and run the dump-producing command
   under `umask 077`. Before accepting either backend's archive, verify it is a
   regular, current-user-owned, single-link file with mode `0600`.

### SQLite

Use this procedure only for a file-backed SQLite target, including the resolved
`user` path `~/.agent-libos/runtime/agent-libos.sqlite`; `local`, `:memory:`,
and bare `sqlite://` databases have no persistent file to back up.

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

   The expected output includes `ok` and `7`.
3. To restore, keep the source database untouched and materialize the verified
   backup at a new, owner-only path. Do not restore over a path held by a live
   Runtime and do not restore old lease/sidecar files. Point a stopped Host at
   the new path and let `Runtime.open()` perform the complete schema-v7
   canonical storage catalog, encoding, and startup-recovery checks. Keep the
   old target until that open and a clean shutdown succeed.

### PostgreSQL

Use normal libpq credential mechanisms such as a service definition and
password file; avoid placing credentials in shell history. The PostgreSQL
server must be major version 17; 17.10 is the tested baseline used to generate
the committed catalog manifest. Check this before the dump with
`SHOW server_version_num`; a restored target on another major is rejected by
`Runtime.open()`. The PostgreSQL Runtime identity and lease are scoped to
`current_database()` plus the exact `current_schema()`, so the backup unit is
that one schema rather than the whole database. Configure the service's
`search_path` to select the Runtime schema;
the commands below use `agent_libos_runtime` as an example. With the Runtime
stopped and its advisory-lock session closed:

1. Confirm the PostgreSQL server version, selected schema, and store version,
   then create and inspect a custom-format logical dump of that exact schema.
   Replace `agent_libos_runtime` with the `current_schema()` result; do not use
   a schema wildcard.

   ```bash
   psql 'service=agent_libos' -At \
     -c 'SHOW server_version_num; SELECT current_schema(); SELECT schema_version FROM runtime_schema WHERE singleton = 1;'
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
     -c 'SHOW server_version_num; SELECT current_schema(); SELECT schema_version FROM runtime_schema WHERE singleton = 1;'
   ```

   Require the restored `server_version_num` to have major version 17,
   `current_schema()` to equal the dumped schema, and the store schema version
   to equal `7` before opening the Runtime.

3. Point a stopped Host at the restored target. `Runtime.open()` must acquire
   the new target's advisory lease and pass the complete schema-v7 canonical
   storage catalog and encoding probes before the target is promoted. Keep the
   original database until the restored Runtime also shuts down cleanly.

### Coverage and online-backup boundary

The SQL backup includes durable process, authority, registry, checkpoint/Image
artifact, operation, audit, event, Task Run, and external-effect rows at the
database snapshot. It also includes any still-full internal root-spawn
initial-goal recovery envelope and any plaintext Task Run payload or full linked
Human request/provider receipt captured before terminal purge, so the backup
must be protected as payload-bearing evidence.
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

A persistent target is owned by one writable Runtime at a time. The reserved
`user` target follows the same file-backed SQLite lease rules after it resolves
to `~/.agent-libos/runtime/agent-libos.sqlite`.

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

See [the configuration guide](configuration.md) for library and product
configuration precedence, [the CLI guide](cli.md) for backend selection, and
[Architecture](architecture.md) for the authority, evidence, primitive, and
Runtime dependency boundaries.
