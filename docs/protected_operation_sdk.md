# Protected Operation SDK

`agent_libos.sdk` is the stable extension boundary for provider-backed host
operations. It combines capability reservation, Task Authority Manifest effect
ceilings, canonical argument binding, provider dispatch, effect
classification, event/audit evidence, resource settlement, and Explainable
Operations linkage in one fail-closed state machine. A tool or extension must
not call the runtime-internal effect lifecycle helpers directly.

The SDK does not replace domain authorization or provider interfaces. The
primitive still validates arguments, chooses the exact capability decisions,
and supplies safe evidence. The SDK controls when those decisions are reserved
and committed and how provider ambiguity is represented.

## In this guide

- [Define and invoke a protected contract](#contract-and-invocation)
- [Implement a synchronous operation](#synchronous-operation)
- [Compose async providers](#async-and-composite-providers)
- [Handle failure and at-most-once behavior](#failure-and-at-most-once-behavior)
- [Use prepare, settle, and compensation hooks](#prepare-settle-and-compensation-hooks)
- [Understand enforcement](#enforcement)
- Return to the [documentation home](index.md).

## Contract and invocation

Register contracts during trusted Runtime composition:

```python
from agent_libos.models import DataFlowDirection, DataIntegrity
from agent_libos.sdk import ProtectedOperationContract, ResourcePolicy

runtime.protected_operations.register_contract(
    ProtectedOperationContract(
        name="primitive.example.fetch",
        provider="example",
        operation="fetch",
        evidence_roles=("audit", "event", "effect"),
        resource_policy=ResourcePolicy.REQUIRED,
        information_flow=True,
        data_flow_direction=DataFlowDirection.BIDIRECTIONAL,
        minimum_egress_integrity=DataIntegrity.UNKNOWN,
    )
)
```

`AuthorityMode.CAPABILITY` is the default and requires one or more allowed
`CapabilityDecision` values for the acting pid. `AuthorityMode.RUNTIME_INTERNAL`
requires a non-empty `internal_reason`; it is for a Runtime-owned continuation,
not a shortcut for extension code. `state_mutation` and `information_flow` are
conservative upper bounds used when the provider outcome or classifier is
unknown. `ResourcePolicy` has these exact semantics:

- `none` forbids preflight, reservation, success, and failure accounting on the
  invocation;
- `optional` permits an invocation with no accounting, or one that explicitly
  supplies the relevant accounting fields; and
- `required` requires either `preflight_usage` or `reservation_usage` before
  dispatch and a `ResourceSettlement` on successful completion.

`preflight_usage` is only a budget check. It does not reserve or charge quota.
On a dispatched failure, `failure_resource` supplies measured partial usage; a
required invocation without that measurement conservatively charges its
preflight usage, but only after the failure effect is durably settled. If that
effect settlement fails, the intent remains pending and a preflight-only path
has not yet charged usage.

`reservation_usage` instead creates a durable maximum-usage reservation in the
same transaction as the prepared effect. A successful `ResourceSettlement`
settles actual usage within that envelope; `charge_reserved_maximum=True`
settles the full envelope. Certified not-started paths release the reservation.
An unknown dispatched outcome without measured usage charges the maximum.
After a crash, startup recovery releases an active reservation whose effect is
absent or still `prepared`, and charges the maximum for every other active
reservation. A successful invocation that supplied `reservation_usage` must
therefore also pass a settlement to `complete()` to settle it online; otherwise
the active reservation is left for startup recovery.

Effect/evidence finalization commits before the SDK performs the success-path
resource charge or reservation settlement. A charge or overage failure is
reported to the caller but cannot roll back the provider effect. With a durable
reservation, a failed post-effect settlement stays active and is handled by
startup recovery; without a reservation, there is no deferred charge record to
recover.

`data_flow_direction` is independently `none`, `ingress`, `egress`, or
`bidirectional`. Do not infer egress from `information_flow`: filesystem reads,
DNS, and clocks observe information but do not send the caller's payload.
Every egress/bidirectional invocation must provide a concrete primary
`DataSink`, trusted `DataFlowContext`, canonical payload descriptor, and
non-empty data-flow operation. When the same provider operation has more than
one real recipient, it must also provide every other recipient in
`additional_data_sinks`; modeling only the primary recipient is not sufficient.
The value must be a tuple of concrete Sinks and all Sink identities must be
unique. Additional Sink identities must remain stable for the invocation; the
single `data_sink_revalidator` hook described below applies only to the primary
Sink. Every ingress/bidirectional invocation must additionally provide a trusted
`data_flow_ingress_context`; `none` and egress-only invocations must omit it. A
contract with a data-flow direction must also declare `information_flow=True`.

`minimum_egress_integrity` defaults to `untrusted` and may be raised only for
an egress or bidirectional contract. The SDK checks it before provider dispatch
and on every dispatch revalidation; Sink trust and one-shot sensitivity release
do not override it. Prepared and final effect evidence retain the selected
floor without retaining the payload.
Trusted deployment configuration may tighten core descriptors by exact name
through `data_flow.operation_minimum_integrity`. It cannot weaken a
code-declared floor or configure an unknown/non-egress descriptor.

`ProtectedOperationInvocation.data_flow_request_release` defaults to `true` for
backward compatibility. A trusted runtime-internal caller may set it to
`false` when recursively prompting for release would be unsafe. The SDK still
runs the complete DataFlow checks, but a conditional result is returned as a
denial rather than creating a `data_release_approval`. Semantic classifier
egress uses this flag and records `egress_blocked` without dispatching the
classifier.

The Host may bind one default post-commit result observer to the SDK. Binding is
one-shot: a second default registration is rejected. An invocation may add its
own observer, but it cannot replace or suppress the Host observer. After a
successful provider effect/evidence commit, both callbacks receive the original
result plus stable effect, process, provider, operation, target, DataFlow, and
result-identity metadata. They run independently; one exception neither skips
the other nor changes the already committed result, and failures are reduced to
a bounded payload-free envelope.

Result identity never invokes arbitrary provider conversion hooks. The general
safe canonical projection accepts only exact `None`, `bool`, `int`, finite
`float`, `str`, `bytes`, and `bytearray` values; `StrEnum` values; exact
`list`/`tuple` containers; exact `dict` containers with exact string keys; and
explicitly allowlisted Host result dataclasses, subject to 4,096 nodes and 256
KiB. It does not accept an arbitrary `Enum`. When that small projection is
insufficient, Filesystem, Shell, Git, JSON-RPC, MCP, and LLM contracts plus
explicitly Host-bound result dataclasses may use a streaming digest capped at
500,000 nodes and 64 MiB. Streaming accepts the same exact built-in scalars and
containers, but an enum must be a Host-owned, module-bound `Enum` whose stored
value is an exact string; dataclasses must likewise be Host-bound and
allowlisted. It neither constructs a second aggregate plaintext value nor
persists source text.

For the allowlisted `LLMCompletion`, only normalized fields consumed by the
Runtime participate in result identity and local text traversal. Raw provider
objects, hidden reasoning, provider request options, compatibility-removal
metadata, provider trace, and provider-attempt sequence state are explicitly
excluded; this path does not claim to classify or bind those opaque fields.

The observation descriptor is at most 4 KiB and contains only schema version,
bounded type identity, digest mode, and canonical byte count. Type identity is
named only for exact built-ins and module-bound Host-allowlisted classes;
everything else becomes the fixed `opaque` label. Unsupported, cyclic,
colliding-key, non-finite, opaque, or over-budget values yield a null digest and
`digest_unavailable`, not a type-only substitute. Semantic
provider-ingress capture requires the real digest and therefore records only an
isolated capture failure in that case. Callers must prevent the classifier's
own runtime-internal operation from recursively becoming a new ingress
assessment.

If `prepare` changes durable domain state, declare a named
`prepared_recovery` policy on the contract and register its trusted recovery
handler during Runtime composition with
`register_prepared_recovery(name, handler)`. Policy names are non-empty strings
and are normalized by trimming surrounding whitespace. Starting an operation
whose declared policy has no registered handler fails before `prepare`,
authority reservation, intent creation, or provider code.

Startup Runtime Module entrypoints run, and their buffered handler
registrations are applied, before prepared-operation recovery. The registered
provider/startup hook callbacks themselves run only after recovery completes.
An entrypoint receives the buffered `ModuleContext`, not the hook-time Host
surface, so it cannot use `protected_operations` directly; registering a
prepared-recovery handler from a provider/startup hook is too late for that
startup. Such a handler must therefore be installed by core composition or
another trusted pre-recovery Host composition step. A module must not persist a
prepared policy that depends on its later hook execution after a crash; startup
will correctly fail with a missing handler.

The recovery-specific effect metadata stores the policy name, safe observation,
contract/actor identity, and reservation IDs; canonical arguments remain
represented only by their hash. The handler has the signature
`handler(effect: ExternalEffectRecord) -> None`, receives that payload-safe
record, and may repair only local state in the current RuntimeStore transaction.
It must not call a provider or perform another non-transactional side effect.
On startup, under the internal recovery lease, the SDK validates the stored
metadata and reservation bindings, runs the handler, restores the reservations,
and abandons the intent in one transaction. Every linked reservation ID must be
unique, present, still in `reserved` state, carry the exact JSON-integer count
`1`, and be bound to the stored actor and contract reason. Prepare also appends
an immutable `effect_reservation` operation-evidence link containing the exact
effect, capability, count, contract, and actor. Recovery requires one
unambiguous effect/operation link plus matching reservation and direct-effect
links; the reservation row's `cap_id` and count must agree with both. A missing
handler, missing or non-live reservation, duplicate link, missing/ambiguous or
mismatched evidence, failed restoration, or handler exception fails recovery;
the transaction rolls back, the prepared intent and any existing reservation
rows remain unchanged, and Runtime startup does not continue to general
provider reconciliation. A prepared row from an older development build that
lacks the direct binding fails closed and requires trusted store repair or
recreation; do not replay the provider call. A `prepared` SDK intent is never
sent to a provider reconciler because no provider phase was dispatched.

An invocation contains full canonical arguments and a separate safe
observation. The SDK hashes canonical arguments for approval/idempotency
binding but persists only the observation:

```python
ingress_context = runtime.data_flow.unclassified_ingress_context(
    flow_context,
    origin="external:example",
)
invocation = ProtectedOperationInvocation(
    pid=pid,
    actor=pid,
    target="example:item-7",
    decisions=(decision,),
    canonical_args={"item": "item-7", "credential": credential},
    observation={"item": "item-7", "credential_present": True},
    preflight_usage=ResourceUsage(external_read_bytes=max_bytes),
    resource_source="primitive.example.fetch",
    failure_resource=lambda error, phase: ResourceSettlement(
        usage=measured_partial_usage(),
        source="primitive.example.fetch",
        context={"failure_phase": phase},
    ),
    data_flow_ingress_context=ingress_context,
    data_sink=sink,
    additional_data_sinks=(repository_sink,),
    data_flow_context=flow_context,
    data_flow_payload=provider_request,
    data_flow_operation="example.fetch",
)
```

Omit `failure_resource` only when conservative preflight charging is the right
failure policy. Its factory runs after the provider effect has been classified
and settled, receives only the exception object and safe phase name, and must
not make a provider call or place exception text in persisted context.

Do not put Object payloads, credentials, Human content, raw LLM I/O, provider
payloads, stdout/stderr, or exception text in observations or evidence.

Before preparing the effect, the SDK authorizes the primary and every additional
Sink against one Sink-registry generation and the exact source Object
versions/content hashes. It checks the shared payload and release bindings, then
reserves ordinary authority and every required release capability with the
intent in one transaction. It repeats the complete all-Sink data-flow check
immediately before every provider phase; a release consumed by an earlier phase
is accepted only through that same protected-operation reservation. Prepare,
not-started restoration, success commit, and unknown settlement treat the
complete release set atomically. A failed prepare recheck creates no intent,
while a failed dispatch recheck calls no later provider phase. Effect metadata
stores the primary egress under the normal fields and additional recipients in
`additional_egresses`. Each entry contains the same metadata-only decision,
Sink identity/trust, labels, source references, hashes, registry generation, and
release-capability reference; it never contains the egress payload. Together,
the early primitive check, transactional prepare, and per-phase dispatch recheck
close provider-before-policy and mutable-source TOCTOU paths.

Preflight authorization visits the primary and additional Sinks in tuple order.
The first successful authorization captures the registry generation used by the
rest. If a conditional Sink needs Human release, that attempt suspends before
later Sinks are visited; resumed attempts can therefore surface independent
per-Sink release requests sequentially. The complete release set becomes atomic
at protected-operation prepare, after all Sinks have authorized successfully.

Use the invocation revalidators for mutable identities that the generic checks
cannot reconstruct:

- `authority_revalidator` re-derives the exact ordered capability-decision set
  inside the prepare transaction, after the optional `prepare` hook and before
  reservations are created. If omitted, the SDK calls
  `CapabilityManager.reauthorize_decision()` for each original decision. Before
  each provider phase, the SDK then reauthorizes reusable decisions and checks
  the exact finite-use reservations; it does not call the custom revalidator a
  second time.
- `data_sink_revalidator` resolves the live primary `DataSink` identity
  immediately before every provider phase. It must return the same trusted
  identity captured by `data_sink`; a changed or unresolvable Sink is denied
  before provider code runs and the denial is retained as data-flow evidence.
  There is no per-additional-Sink resolver, so extensions must not place a
  mutable identity in `additional_data_sinks`.
- `data_flow_target_state_version` binds the preflight decision to the captured
  target-state version. When the target can change, pair it with
  `data_flow_target_state_version_resolver`; the resolver supplies the live
  version for every per-phase authorization, which rejects a stale binding.

These callbacks are trusted, synchronous Host code. They must not call the
provider, must return the declared typed value, and must not place payloads or
secrets in exceptions or persisted evidence.

Registry-backed provider invocations may additionally supply an immutable
`ProviderRegistryBinding`, its typed live resolver, and a synchronous phase
guard. These three fields are all-or-none. For every `call()`, the SDK enters the
guard and, while it remains held, reauthorizes, compares the captured spec
SHA-256 and generation with the resolver result, persists dispatch, and invokes
the provider callable. The guard must be the same lock/context used by every
supported registry mutator; this is a trusted extension obligation that the SDK
cannot infer from an arbitrary context manager. A mismatch before the first
phase abandons the prepared intent and restores finite-use reservations without
calling provider code; a mismatch after an earlier effectful phase blocks later
phases and follows conservative partial/unknown settlement.

Registry-guarded operations must use synchronous `call()`. `acall()` rejects
them before provider dispatch; the surrounding operation exit restores and
abandons a first-phase preparation. Async facades must offload the complete
synchronous `call()` interval rather than holding a thread lock across an
`await` or guarding only the resolver.

## Synchronous operation

Every real provider call is a named phase:

```python
with runtime.protected_operations.start(
    "primitive.example.fetch", invocation, provider=provider
) as operation:
    response = operation.call(
        ProviderPhase("transport", information_flow=True),
        provider.fetch,
        item_id,
    )
    return operation.complete(
        response,
        ProtectedOperationEvidence(
            event_type=EventType.EXTERNAL_READ,
            event_source=pid,
            event_target="example:item-7",
            event_payload={"status": response.status},
            audit_action="primitive.example.fetch",
            audit_actor=pid,
            audit_target="example:item-7",
            audit_decision={"status": response.status},
            effect_metadata={"status": response.status},
            provider_receipt={"receipt_id": response.receipt_id},
        ),
        classification_result={"status": response.status},
        resource=ResourceSettlement(
            ResourceUsage(external_read_bytes=response.bytes_read),
            source="primitive.example.fetch",
        ),
    )
```

`complete()` atomically runs the local `settle_success` hook, emits event/audit,
and finalizes the prepared effect id. Resource charge or usage-reservation
settlement runs afterward. A charge or overage error is reported to the caller
and may terminate the process, but cannot hide or roll back an already committed
provider effect.

## Async and composite providers

Use `acall()` for an async provider. Composite operations use sibling phases on
the same handle:

```python
addresses = operation.call(
    ProviderPhase("dns", information_flow=True),
    resolve_registered_host,
    endpoint,
)
reply = await operation.acall(
    ProviderPhase("transport", state_mutation=True, information_flow=True),
    provider.acall,
    endpoint,
    addresses,
)
```

`ProviderPhase.commits_authority` defaults to `True`. A successful phase with
that default commits finite authority even if both `state_mutation` and
`information_flow` are false. A purely local coordination phase such as a lock
or validation fence must explicitly use `commits_authority=False`; it must also
remain free of provider-visible mutation and information flow. If dispatch is
rejected after only completed phases whose three effect flags are all false,
the SDK can abandon the not-started intent and restore all ordinary and release
reservations. Once any completed phase mutates state, observes information, or
commits authority, a later `ProviderEffectNotStarted` or revalidation failure
finalizes the confirmed partial/unknown effect and cannot restore authority.
The active-phase record, completed-phase transcript, and final
`provider_phases` evidence persist the phase name and all three flags, so this
authority floor remains explainable after settlement or reopen.

For an ingress/bidirectional contract, the SDK observes the invocation's
ingress context automatically and at most once after the first actually
started phase whose `information_flow` flag is true. It observes before
returning a successful result and before propagating an ordinary exception,
cancellation, or otherwise uncertain failure. A current phase certified by
`ProviderEffectNotStarted`, whether raised or returned as the structured marker
below, certifies only that current phase and does not propagate that phase's
ingress context. It is not a certificate that the whole operation did not
start. Any context already observed by an earlier completed information-flow
phase remains in force.

When a primitive must return a structured domain error instead of propagating
`ProviderEffectNotStarted`, its provider callable may return
`ProviderEffectNotStartedResult(error, result, outcome=...)`. The SDK settles
the certificate without adding the current phase to the completed-phase floor;
the primitive returns `result` only after observing that marker. This prevents
a not-started mutating phase from becoming a false committed mutation.

## Failure and at-most-once behavior

- Failure before dispatch atomically restores revoke-safe reservations, runs
  `restore_not_started`, and abandons the prepared intent.
- `ProviderEffectNotStarted` in the first provider phase has the same result.
- An ordinary exception or cancellation consumes authority and best-effort
  finalizes `unknown`; if settlement itself fails, the dispatched pending intent
  remains durable for reconciliation.
- Success, provider error, and finalized-unknown metadata all retain the same
  payload-free data-flow evidence (decision, Sink/trust generation, source
  refs and label hashes). Finalizing an intent never replaces that evidence
  with an error-only envelope.
- A dispatched required-resource operation settles measured `failure_resource`
  usage. Without a measurement it charges the reserved maximum when
  `reservation_usage` was used, or conservatively charges `preflight_usage`.
  Charging happens only after effect settlement, so an overage cannot erase the
  provider outcome. If effect settlement itself fails, the intent remains
  pending and only a durable usage reservation supplies a recoverable deferred
  accounting record.
- A classifier failure uses the contract's conservative effect ceiling and
  records only the classifier error type.
- Exiting without a provider phase or without `complete()`, completing twice,
  or charging resources contrary to the contract raises
  `ProtectedOperationProtocolError`.

The idempotency contract is deliberately narrower than exactly-once execution:

- `ProtectedOperationInvocation.idempotency_key` is a Host-selected, non-secret
  protocol token. The store claims it under the exact `(pid, idempotency_key)`
  pair; another process may use the same token. The token is persisted in clear
  and retained with summarized evidence, so it must not contain payloads or
  credentials.
- While an effect row is retained, the key is claimed in every transaction
  state, including `prepared`, `dispatched`, `committed`, `failed`, and
  `unknown`. A duplicate invocation raises `ValidationError` before provider
  dispatch; it does not return the prior result, coalesce callers, or replay the
  effect. Transactional `prepare` work and reservations from the duplicate
  attempt roll back.
- A pre-dispatch abort, a PENS certificate with no effectful or
  authority-committing completed phase, or prepared startup recovery deletes the
  not-started intent and therefore releases its key. Reusing it is then safe
  because the lifecycle proved that no effectful provider phase began. An
  ambiguous or reconciled row keeps its key. A new, intentionally distinct
  attempt after a confirmed provider failure needs a new key.
- If no explicit key is supplied, the SDK hashes the current operation/effect
  identity together with the provider, operation, target, and canonical argument
  hash. This blocks a duplicate lifecycle for that identity, including an
  approval-bound effect, but a normal independent top-level operation receives a
  fresh identity. The default is therefore not cross-request or cross-restart
  semantic deduplication. Supply a stable explicit key only when the provider
  protocol and recovery logic define that retry identity.
- An LLM-selected Python tool receives Host-captured
  `llm_transcript_output_key`, `llm_tool_call_id`, and `llm_tool_name` values in
  `ToolContext.metadata`. They survive supported Human/child/message waits and
  can be inputs to a non-secret explicit key when the provider protocol defines
  that native tool call as the retry identity. They do not authorize the
  operation, and the runtime never deduplicates calls merely because their
  function and arguments match.

Startup reconciliation may query by key or provider receipt, but never invokes
the original provider operation. An unresolved outcome remains `unknown`; do
not bypass the retained key and blindly retry it.

The default is `PostProviderFailureMode.PROPAGATE`: a local completion-settlement
failure is raised even though the provider may already have returned, so the
pending/unknown intent must not be treated as safe to retry. Select
`PRESERVE_RESULT` only when returning the accepted provider result is necessary
to prevent replay and the caller can continue with conservative pending/unknown
evidence. This mode does not suppress a failure of the conservative fallback
settlement itself; that secondary error still propagates.

Both `primitive.human.read` and `primitive.human.write` use
`PostProviderFailureMode.PRESERVE_RESULT`. Once the Human provider accepts an
answer or output and fallback settlement succeeds, a later local completion
failure returns that accepted result, keeps pending/unknown evidence, and never
invokes the Human provider again.

## Prepare, settle, and compensation hooks

`prepare`, `restore_not_started`, `settle_success`, and `failure_settlement`
are synchronous hooks that may mutate only local transactional state. They
must not call a provider. Returning an awaitable, generator, or async generator
is rejected and the containing transaction fails closed rather than silently
discarding deferred work; ordinary eager synchronous return values have no
meaning.
`failure_settlement(error, phase)` runs at most once, in its own transaction,
only after dispatch may have crossed a provider boundary; it is not run for a
first-phase `ProviderEffectNotStarted`. It lets a primitive preserve local
conservative state such as a file-path label when the provider outcome is
unknown. A compensating host action is another phase on the same handle:

```python
handle = None
try:
    handle = operation.call(ProviderPhase("create", state_mutation=True), provider.create)
    publish_local_handle(handle)
except Exception:
    if handle is not None and not operation.terminal:
        operation.call(ProviderPhase("cleanup", state_mutation=True), handle.close)
    raise
```

Never blindly retry an `unknown` provider outcome. Startup reconciliation may
query an idempotency key or receipt when the provider explicitly supports it;
otherwise the effect stays `unknown`.

## Enforcement

`scripts/check_protected_operations.py` is an AST policy check over Python files
under `agent_libos/` and the repository-level `modules/` directory. It rejects:

- direct imports or calls of the internal prepare/dispatch/finalize/abandon
  helpers outside the four explicitly allowed lifecycle implementation files,
  including imports through the public evidence re-export, `as` aliases,
  wildcard lifecycle imports, and simple or chained assignment aliases;
- use of the former private reservation-restoration API;
- recognized direct `self.provider.*()` calls and recognized provider-handle
  methods outside a callable passed to an SDK `call()`/`acall()` phase; and
- missing ingress/egress descriptor keywords for its explicit, hard-coded core
  contract inventory when the local invocation construction can be resolved.

The intentional exceptions are narrow and visible in the checker: filesystem
path normalization; Human delivery-buffer `read`/`write`; Git's local,
non-dispatching `preflight_remote_fingerprint`, `preflight_path_kind`,
`repository_lock`, and `validate_read_only_operation`; the exact Host-only
`GitPrimitive._semantic_read_flow_snapshot` call graph described below; the
lifecycle implementation files; and recovery-time `handle.close()` in a
function whose first executable statement is the exact recovery-cleanup lease
guard. The last exception is evidence-free cleanup of an already-published
transient handle; none of these entries is a general provider-call allowance.

`GitPrimitive._semantic_read_flow_snapshot` is the one semantic pre-intent
provider exception. While revalidating a canary grant, before authority exists
to enter the ordinary Protected Operation, it may observe only local repository
layout/state and run checker-reviewed read-only Git commands. It cannot make a
remote or mutating provider call, returns only bounded digests plus DataFlow
label/source-reference metadata, and has no durable external-effect intent of
its own. The checker pins the owning class, private root method, reachable
provider methods, `read_only=True`, and the absence of dynamic runner keyword
arguments. If a grant is later used, the ordinary Git read creates its durable
intent and repeats the flow/state observation inside the protected provider
phase before returning payload. This exception is a trusted local security
preflight, not evidence that a protected provider dispatch occurred.

The checker is not a sound whole-program Python call-graph analysis. It does not
scan tests, scripts, examples, installed third-party packages, or module source
outside the two roots; it does not prove reflective/dynamic dispatch, arbitrary
provider aliases, cross-file helper reachability, or dynamically registered
data-flow contracts. New provider shapes and contract names require updating
the checker inventory and adding runtime denial-path tests. Separate runtime
tests assert that the registered SDK contracts equal the Explainable Operations
external-primitive boundary set and that every core data-flow contract declares
an exact direction. LLM is included in that runtime inventory.

The SDK is Host-only infrastructure: it adds no model tool, syscall, CLI, or
HTTP endpoint.
