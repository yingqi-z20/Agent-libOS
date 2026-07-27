---
name: agent-libos-authority-basics
description: Diagnose this process's Capability authority and request one exact Human policy decision when a protected operation lacks requestable authority. Do not use it for intent questions, validation errors, provider failures, or speculative privilege expansion.
allowed-tools: list_capabilities inspect_capability request_permission
---
# Inspect and request authority

Use this Skill for a concrete `permission_denied` or a Host-declared exact
Capability requirement. Never run a risky operation merely to provoke denial.
A visible tool, loaded Skill, identifier, Human prose, and permission-request
ceiling are not grants.

Keep four independent gates separate:

1. **Tool visibility/schema**: the tool must be projected and arguments valid.
2. **Capability**: an applicable resource/right decision must authorize the pid.
3. **Task Authority**: manifest authority, effect, and resource budgets still
   bound the task.
4. **Primitive/Sink policy**: constraints, approvals, containment, data-flow,
   provider policy, and target state still apply.

`request_permission` changes only Capability policy. It cannot reveal a tool,
expand Task Authority, approve an exact external effect, declassify data, fix
arguments, or restore a provider.

## Tool guide

### `list_capabilities`

`list_capabilities(include_inactive=false, limit=<current-schema-default>,
after_cap_id=null)` returns `{capabilities:[...], has_more, next_cursor}` for
this process only and never grants or consumes authority.

- `limit` must be positive and is capped by the Host
  `capability.list_limit` (default 100). When `has_more=true`, pass the non-null
  `next_cursor` as the next page's `after_cap_id`. Only infer absence after all
  pages have been traversed without a matching record. `limit` is a maximum,
  not a promised page length: the model-result byte budget can end a page
  earlier and still return a continuation cursor.
- The default keeps stored-active, unexpired rows with a currently valid parent
  chain. It does not evaluate constraints against the proposed operation.
- `include_inactive=true` is diagnostic and subject to the same paging bound. It may
  show `revoked`, `disabled`, `exec_revoked`, expired, exhausted, or
  parent-invalid rows. Stored `status="active"` can still be ineffective.
- Important fields are `cap_id`, `subject`, canonical `resource`, `rights`, base
  `effect`, `constraints`/`rules`, lease, status, delegation, revocability, and
  lineage.
- Every row is a bounded presentation, not the durable authority record.
  If `metadata_projection` or `constraints_projection` has `omitted=true`, its
  sibling `metadata` or `constraints` is deliberately `{}` and the latter also
  makes `rules=[]`. The receipt gives the canonical byte count and SHA-256 of
  the omitted value. Never interpret those empty containers as real empty
  policy or metadata; use the hash only for identity/change detection, not to
  reconstruct or authorize the hidden value.

Do not infer a final decision from one row. An `allow` may be a scoped miss,
become `ask`/`deny` under an Authority Rule, lose to an applicable deny, expire,
or fail a primitive-specific constraint when exact operation context is known.
When a policy-bearing projection is omitted or exact operation context is not
available, list/inspect cannot determine the final policy. Stop and require the
owning primitive's or Host's explain evidence; do not guess or run a risky probe.

### `inspect_capability`

`inspect_capability(cap_id)` returns `{capability:{...}}` for an exact known ID
only when this process is its subject. Unknown IDs are validation failures;
cross-subject inspection is denied. Use it to check lease, constraints,
lineage, `delegable`, and `revocable`. `metadata` is descriptive, not authority.
Inspection uses the same bounded projection and omission receipts as listing,
so it may not expose full constraints, rules, or metadata. It does not evaluate
a proposed operation and cannot prove success.

### `request_permission`

Call `request_permission(resource, rights, reason,
human=<current-schema-default>)` once:

- Take the canonical typed `resource` from the owning primitive's contract or
  denial, an inspected row, or a visible request ceiling. Do not derive it from
  an unnormalized path.
- Use the smallest nonempty `rights` list on that one resource. Legal rights are
  `read`, `write`, `execute`, `link`, `diff`, `materialize`, `delete`, `grant`,
  `revoke`, `approve`, and `admin`; neither `*` nor `delegate` is a right.
- Give a task-specific, non-secret `reason`. Omit `human` to use the default
  shown by the current tool schema; an explicit recipient needs `write` on
  `human:<human>`.

The request must fit the manifest's authorized or requestable Capability specs.
That ceiling is not authority. The runtime checks the ceiling, Human write,
model-request restrictions, and Human Sink/data-flow before a decision.

Once the request is durably queued, that successful publication consumes a
finite `human:<human>` `write` use. A later rejection or cancellation does not
refund it; only a failure that rolls back request publication preserves the use.

Model requests fail before reaching the Human for privileged Capability/meta
authority, root/global filesystem write/delete, workspace-wide delete, and
broad Shell execute. More generally, every root-prefix `<kind>:*` request that
contains `admin`, `grant`, `revoke`, `write`, `execute`, or `delete` is rejected
before the Human, regardless of resource kind; Host manifests reject the same
impossible requestable ceilings at bind time. Shell is stricter: every non-Git
`shell:<command>/execute` request is rejected. Only `shell:git` is requestable,
and it is constrained to six exact read-only argv forms: `status`,
`status --short`, `branch --show-current`, `rev-parse --show-toplevel`, `diff`,
and `diff --stat`. For other commands use `agent-libos-command-execution` and
its exact per-use approval path, or report that Host Shell policy is required.

The first call may suspend; runtime resumes the exact call. Do not poll, change
arguments, or duplicate it. Completion returns only
`{request_id, resource, rights, status}`. `resource` echoes the submitted value;
the result omits canonical policy details and Capability ID. Status is not the
installed policy: `approved` may mean `always_allow` or `ask_each_time`, while
`rejected` may mean `always_deny` or `ask_each_time`. This tool cannot request
`allow_once`.

## Recommended workflow

1. Classify the failure. Route validation, stale-state, budget, provider,
   data-flow, and unavailable-tool errors to their owning Skills.
2. Identify the exact resource/right without a risky probe. Resources are
   typed; bare `*` is invalid, and wildcards are terminal only: `kind:*` is a
   prefix and `kind:body/*` a subtree. Prefer an exact resource.
3. Page through current rows to exhaustion and inspect a known candidate ID.
   A partial page, `has_more=true`, or an omission receipt is not proof of
   absence or of an empty policy. If projection or operation context is
   insufficient to establish the final policy, stop and require owning-
   primitive or Host explain evidence instead of guessing or probing.
4. Follow a final policy only when complete operation-specific evidence or the
   owning primitive/Host has established it:
   - **Allow**: request nothing; invoke the original operation once with exact
     intended arguments. Its primitive is the final verifier.
   - **Ask**: invoke the original operation once only when its owning primitive
     explicitly provides a per-use approval bridge for this operation, then
     accept that exact Human wait. Without such a bridge the operation returns
     `permission_denied`; stop instead of treating Ask as a pending approval.
     Do not request standing policy to evade approval.
   - **Deny**: stop. A new allow cannot override an applicable deny. Its issuer
     or a covering `revoke`/`admin` holder must revoke it through
     `agent-libos-capability-delegation`, or the Host must change policy.
   - **Missing/scoped miss**: request only when the exact resource/right is
     inside a Host ceiling and supported by this model-request path.
5. After one completed policy decision, list/inspect if possible. Never replay
   when readback confirms an applicable `always_deny`. Replay the original
   operation once with unchanged arguments only when authority may have changed,
   no applicable deny is established, and any required Ask bridge exists.

For an operation needing several resources, resolve only the exact gate reported
at each step. One permission call cannot combine different resources; do not
create speculative standing authority.

## Failure and recovery

- `validation_error`: fix malformed resource syntax or empty/unknown rights;
  never ask a Human to approve invalid input.
- Ceiling denial, missing Human write, applicable deny, and unsupported Shell
  request are stops; unchanged repetition cannot help.
- Pending request: issue no new action. Runtime holds the process and resumes
  the same durable request, including after reopen.
- Completed `approved`: treat it as a status receipt, not authority proof;
  inspect if possible and replay only under workflow step 5.
- Completed `rejected`: never replay when readback confirms `always_deny` or any
  applicable deny. Retry once only when authority may nevertheless have changed,
  no applicable deny is established, and any required Ask bridge exists;
  otherwise stop.
- Cancelled/malformed Human decision, data-release rejection, provider error,
  or resume mismatch: do not create a replacement until Human/Host explicitly
  reopens the decision.
- If the original operation remains denied after one completed request, stop.
  Do not widen resource scope, add `admin`/`grant`/`revoke`, or cycle requests.

## Completion evidence

Diagnosis should name the exact operation, canonical resource/right, visible
Capability IDs and relevant fields, plus uncertainty from a bounded list.

A permission workflow needs: (1) the single request receipt and request ID;
(2) any resulting list/inspect evidence without inventing the omitted policy;
and (3) one fresh result from the original operation showing success, per-use
rejection, or the remaining blocker. Never claim completion from `approved`, an
`allow` row, Skill activation, or Human prose alone. Use `ask_human` for intent
or facts, never Capability approval.
