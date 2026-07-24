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
are silently bounded (default 100), so absence is not conclusive.

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
  where the owning primitive supports per-use approval.
- `expires_at` must be valid ISO and no later than the parent; omission inherits
  parent expiry. `uses_remaining` must be an integer >=1; omission makes the
  child lease unlimited. A finite parent can never be delegated.
- Keep `delegable=false` unless the child must delegate again and inherited
  depth has room. There is no model `max_delegation_depth` argument.
- `constraints` authorize operations. Copy every selected-parent constraint
  exactly; only recognized additional narrowing keys are accepted, and a Shell
  policy constraint cannot be newly added. Parent constraints plus `{}` fail.
- `metadata` is bookkeeping, not policy. It cannot narrow resource, rights,
  argv, state, or effects; never put secrets in it.

There are no `parent_cap_id`, `max_depth`, or `revocable` arguments; the record
uses the runtime's revocable default. Success returns `{capability:{...}}` with
the generated `cap_id` and selected `parent_cap_id`.

#### Automatic parent selection

The runtime finds active, unexpired, parent-chain-valid, delegable `allow`
records covering resource and every right. Any active intersecting `deny` or
`ask` blocks delegation. It chooses the longest resource string, then newest
`issued_at`, and only afterward validates finite use, expiry, depth, and
constraints.

Thus a more-specific/newer parent can be selected and fail even when an older
broad parent seems usable. The model cannot choose another parent ID. Inspect
the record this algorithm selects and copy its constraints. Never broaden or
distort child scope to force another parent; report/correct the conflict through
an authorized actor.

### `revoke_capability`

`revoke_capability(cap_id, reason=null)` revokes one exact known ID. Use a short
non-secret reason. Repetition can duplicate evidence even if status is already
revoked.

The target must be `revocable`. The caller must be its original issuer, the
subject relinquishing its own `allow`, or hold applicable `revoke`/`admin` over
the resource. A subject cannot self-revoke its `ask`/`deny`; knowing a child ID
does not grant revocation authority.

Success returns `{capability:{...}}`; require the requested ID and
`status="revoked"`. Parent revocation immediately invalidates descendants
through chain checks although their stored rows may remain active. It affects
future authorization only and never undoes committed files, messages, remote
changes, subprocesses, or other effects.

## Recommended workflow

### Delegate

1. Confirm a live direct child. Ask it to report pre-existing IDs for the exact
   resource/right; this baseline supports lost-result recovery.
2. Choose least resource, rights, expiry, use count, and depth. Default to
   `allow`, nondelegable, and empty metadata.
3. Apply automatic selection to parent candidates. Confirm the selected record
   is unlimited-use, has depth and expiry room, and has constraints you can
   preserve. Any intersecting active `deny`/`ask` is a stop.
4. Call once. Validate returned `cap_id`, child `subject`, resource,
   rights/effect, constraints, lease, delegation/depth, `parent_cap_id`,
   revocability, and active status.
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

- Direct-child, schema, coverage, effect, timestamp, finite-use, constraint,
  depth, restrictive-boundary, revocability, or caller-authority failures are
  definite. Transactional mutation publishes no partial result; fix the exact
  input or report the boundary, never widen scope or retry unchanged.
- Delegation row, child attachment, event, and audit commit together. A returned
  evidence-sink failure rolls them all back.
- Lost result/runtime crash after dispatch can make settlement unknown. Never
  repeat either non-idempotent mutation blindly.
- For unknown delegation, have the child compare exact resource/right/lineage
  to its pre-call ID baseline. Respect the silent list window; inconclusive
  readback means unresolved settlement, not permission to duplicate.
- For unknown self-record revocation, use `inspect_capability(cap_id)`; for
  another subject, obtain its report. Default listing may hide a parent-invalid
  descendant; inactive listing may show stored active without effectiveness.
- A finite, depth-exhausted, short-lived, or differently constrained selected
  parent will be selected again for the same args. Do not loop.

Stop after one verified mutation or a precise unknown-settlement report.

## Completion evidence

Delegation needs the successful result naming the new ID and intended subject,
resource, rights, effect, lease, constraints, parent/delegation fields,
revocability, and active status, plus child-side confirmation of that exact ID.
Actual child use must separately succeed or identify its remaining primitive
gate.

Revocation needs the successful exact-ID result with `status="revoked"`. For a
parent, record descendants expected to become invalid; stored status need not
change. Never claim rollback of prior effects. If readback after a lost result
is inconclusive, record “unknown settlement” with pre/post observations rather
than mutating again.
