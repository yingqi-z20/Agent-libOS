---
name: agent-libos-capability-delegation
description: Delegate one attenuated Capability to a known direct child or revoke one exact Capability. Do not create children, grant peers or grandchildren, amplify authority, or treat revocation as rollback of completed effects.
allowed-tools: delegate_capability revoke_capability
---
# Delegate and revoke capabilities

Use `agent-libos-child-processes` first to identify a direct child and message
it. Delegation does not reveal tools, load Skills, or expand Task Authority;
Skill activation grants no authority. Standard base/coding/review images do not
normally contain these mutation tools, and permission cannot manufacture an
unavailable tool.

Use `agent-libos-authority-basics` to inspect parent/target records. Its lists
are count- and byte-bounded; follow `has_more`/`next_cursor` to exhaustion, and
do not treat an omitted constraints projection as an empty policy.

## Tool guide

### `delegate_capability`

Call:

`delegate_capability(child_pid, resource, rights, effect="allow", expires_at=null, uses_remaining=null, delegable=false, constraints={}, metadata={})`

This is a local, transactional, side-effectful, non-idempotent mutation.

- `child_pid` must be this process's current direct child; peers,
  grandchildren, unrelated/missing/stale processes are rejected.
- `resource` must be canonical and covered by the selected parent; prefer the
  narrow exact child scope. `rights` must be nonempty, legal, and a subset of
  one parent. There is no `delegate` right.
- `effect` is `allow`, `ask`, or `deny`. Default `allow` for work. `ask`/`deny`
  can block the child's other/future allows, and the child cannot self-revoke
  them. Use a restrictive effect only as deliberate policy; `ask` matters only
  where the owning primitive supports per-use approval. Restrictive effects
  cannot have `uses_remaining`: `ask`/`deny` decisions do not consume a use, so
  such a lease would never expire by use count. Bound temporary restrictions
  with a future `expires_at` and an issuer/admin revocation plan instead.
- `expires_at` must be valid ISO, strictly in the future at delegation, and no
  later than the parent; omission inherits parent expiry. `uses_remaining`
  applies only to `allow` and must be an integer >=1; omission makes the child
  lease unlimited. A finite parent can never be delegated.
- Keep `delegable=false` unless the child must delegate again and inherited
  depth has room. There is no model `max_delegation_depth` argument.
- `constraints` authorize operations. Copy every selected-parent constraint
  exactly; only recognized additional narrowing keys are accepted, and a Shell
  policy constraint cannot be newly added. Parent constraints plus `{}` fail.
- `metadata` is bookkeeping, not policy. It cannot narrow resource, rights,
  argv, state, or effects; never put secrets in it.

There are no `parent_cap_id`, `max_depth`, or `revocable` arguments; the record
uses the runtime's revocable default. Normal model-visible success returns
`{capability:{...}}` with the generated `cap_id` and action-oriented fields that
fit the projection. `subject`, selected `parent_cap_id`, issuer/parent lineage,
revocability, metadata, and `metadata_projection`/`constraints_projection`
omission receipts exist only in durable ToolResult/Host evidence and are not
model-visible. Consequently absent model policy/provenance fields cannot verify
that the stored value is empty or identify the selected parent. Do not repeat
this non-idempotent mutation to try to obtain hidden fields; use the child
baseline and trusted Host/child readback.

If even the authority identity cannot fit the presentation budget, committed
success instead returns
`{capability:{cap_id,status,presentation_omitted:true}}`. At a still smaller
global result limit the side-effect framework may return top-level
`result_omitted=true`. Both are positive settlement receipts, not validation
failures: never repeat the mutation; use the child baseline and trusted
readback to recover missing fields.

#### Automatic parent selection

The runtime finds active, unexpired, parent-chain-valid, delegable `allow`
records covering resource and every right. Any active intersecting `deny` or
`ask` blocks delegation. It chooses the longest resource string, then newest
`issued_at`, and only afterward validates finite use, expiry, depth, and
constraints.

Thus a more-specific/newer parent can be selected and fail even when an older
broad parent seems usable. The model cannot choose another parent ID. Inspect
visible candidates to narrow the possibilities, but do not claim which record
was selected: model projection omits `issued_at` and selected-parent provenance.
When exact selected constraints or lease/depth facts matter, require the owning
primitive's or Host's explain evidence. Never broaden or distort child scope to
force another parent; report/correct the conflict through an authorized actor.

### `revoke_capability`

`revoke_capability(cap_id, reason=null)` revokes one exact known ID. Use a short
non-secret reason. Repetition can duplicate evidence even if status is already
revoked.

The target must be `revocable`. The caller must be its original issuer, the
subject relinquishing its own `allow`, or hold applicable `revoke`/`admin` over
the resource. A subject cannot self-revoke its `ask`/`deny`; knowing a child ID
does not grant revocation authority.

Normal success returns `{capability:{...}}`; require the requested ID and
`status="revoked"`. Apply the same projection and minimal-success receipt rules
as delegation. Parent revocation immediately invalidates descendants
through chain checks although their stored rows may remain active. It affects
future authorization only and never undoes committed files, messages, remote
changes, subprocesses, or other effects.

## Recommended workflow

### Delegate

1. Confirm a live direct child. Ask it to report pre-existing IDs for the exact
   resource/right; this baseline supports lost-result recovery.
2. Choose least resource, rights, expiry, use count, and depth. Default to
   `allow`, nondelegable, and empty metadata.
3. Apply the documented automatic-selection rule only as a candidate analysis.
   Because the model cannot see `issued_at` or selected-parent provenance,
   confirm the selected record's use, depth, expiry, and constraints through
   owning-primitive or Host explain evidence when selection is not unambiguous.
   Any intersecting active `deny`/`ask` is a stop.
4. Call once. Validate the returned `cap_id`, resource, rights/effect, active
   status, and any visible lease, constraints/rules, or delegation summary.
   `subject`, selected parent, lineage, revocability, and durable omission
   receipts are not model-visible. Do not repeat when those fields are absent:
   obtain trusted Host/child readback against the pre-call baseline or report
   verification as inconclusive.
5. Have the child report that exact ID and use it only through the owning
   primitive. Delegation does not prove its tool, Skill, Task Authority,
   data-flow, or operation-context gates are satisfied.

### Revoke

1. Obtain one exact ID, not a vague resource match. Inspect self-owned records
   or use trusted prior output/subject reporting.
2. Confirm revocability and issuer, subject-allow, or revoke/admin basis.
3. Revoke once and validate exact ID/status. Message affected direct children
   when their plan must stop; a revoke event does not coordinate them.

Revoke the smallest obsolete record rather than broad parent authority still
needed by unrelated descendants.

## Failure and recovery

- Direct-child, schema, coverage, effect, past/invalid timestamp, finite-use,
  restrictive-effect-with-use-count, constraint,
  depth, restrictive-boundary, revocability, or caller-authority failures are
  definite. Transactional mutation publishes no partial result; fix the exact
  input or report the boundary, never widen scope or retry unchanged.
- Delegation row, child attachment, event, and audit commit together. A returned
  evidence-sink failure rolls them all back.
- Lost result/runtime crash after dispatch can make settlement unknown. Never
  repeat either non-idempotent mutation blindly.
- `presentation_omitted=true` or top-level `result_omitted=true` on a successful
  call means the mutation settled but evidence was reduced; it is not unknown
  settlement and must not be retried. Recover details through the same baseline
  comparison used for a lost result.
- For unknown delegation, have the child compare exact resource/right and new
  IDs to its pre-call baseline, and use Host evidence when lineage is required.
  Traverse every capability page; missing model policy/provenance fields or
  otherwise inconclusive readback mean unresolved settlement, not permission to
  duplicate.
- For unknown self-record revocation, use `inspect_capability(cap_id)`; for
  another subject, obtain its report. Default listing may hide a parent-invalid
  descendant; inactive listing may show stored active without effectiveness.
- A finite, depth-exhausted, short-lived, or differently constrained selected
  parent will be selected again for the same args. Do not loop.

Stop after one verified mutation or a precise unknown-settlement report.

## Completion evidence

Delegation needs a successful model receipt naming the new ID and all available
action-oriented fields, plus child-side confirmation of that exact ID. When
completion requires subject, parent/lineage, revocability, durable omission
receipts, or another hidden field, obtain trusted Host/child readback; absence
from the model result is not failure and must not trigger another mutation.
Actual child use must separately succeed or identify its remaining primitive
gate.

Revocation needs the successful exact-ID result with `status="revoked"`; a
top-level omitted success additionally needs trusted readback of that known ID.
For a parent, record descendants expected to become invalid; stored status need
not change. Never claim rollback of prior effects. If readback after a lost
result is inconclusive, record “unknown settlement” with pre/post observations
rather than mutating again.
