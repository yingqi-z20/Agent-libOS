# Task Authority Manifests

`TaskAuthorityManifest` is the Host-authored launch contract for an
`AgentProcess`. It closes the gap between an image declaring what it may need
and a real user, workflow controller, or enterprise policy deciding what this
particular task may receive.

`runtime.launch_authority_mode` is fixed to `manifest_required`. Image
`required_capabilities` are copied into the durable manifest as requirements
for comparison and Explain output, but they are never granted. Only
`authorized_capabilities` compile into root capabilities.

A manifest records:

- the process, image, goal reference, issuer, parent manifest, expiry, and a
  deterministic SHA-256 hash;
- authorized and image-required capability specs;
- permitted provider effect classes;
- resource budget, approval policy, and process receive-domain data-flow policy;
- opaque operator metadata supplied at the Host/admin boundary.

Root launch compiles only the authorized capability specs. Child launch uses
the intersection of the parent manifest, current capability policy, and the
child manifest ceiling. `CapabilityManager.derive_authority()` and
`transition_allowed_rights()` are the public transition boundaries used by
process and checkpoint flows; finite-use authority is never duplicated.
Manifest `max_delegation_depth` values are compiled into the durable
capability and may only stay equal or decrease across a transition.

Checkpoint fork creates a new hashed manifest for every remapped process in
the same transaction as its process and capability rows. Explicit source
manifests retain their effect, approval, expiry, budget, and data-flow
ceilings. A fork derived from an implicit source records an empty model-request
contract, so copied Host authority does not silently become requestable
authority.

Every launch receives a durable manifest record. When the Host does not supply
an explicit template, the implicit record contains any capability specs the
Host supplied through the launch `capabilities` argument; it is empty only when
that argument is empty. Those recorded specs compile into root capabilities and
bound model requests, so requests outside them are denied. An implicit root
manifest is not a transition ceiling for authority the Host deliberately grants
after launch. Supplying an empty object `{}` is different from omission: it
marks the resulting manifest as explicit and therefore as a transition ceiling;
its authorized list is empty only when the separate launch `capabilities`
argument is also empty. Only an explicit Host manifest (and manifests derived
beneath it) constrains later child/fork authority and manifest budgets. Derived
manifests inherit the parent approval, effect, expiry, and data-flow ceilings
unless the Host supplies a narrower child template. Omitted child fields inherit
those ceilings. An explicit child template cannot add or replace parent
data-flow or approval policy keys, request authority outside the parent's
authorized/requestable space, extend the parent expiry, or widen a concrete or
deny-all effect ceiling to unrestricted `null` (or to patterns outside the
parent's ceiling).

Model permission requests are checked against the manifest before a Human
request is created. `approval_policy.requestable_capabilities` declares
authority that may be requested but is not granted at launch; a Human decision
still creates the narrower one-time or policy capability.

`permitted_effects` is an additional provider-boundary ceiling with exact
entries or terminal wildcards such as `jsonrpc.*`. Omitting the field or using
JSON `null` preserves capability-only effect gating for compatibility. An
explicit empty list is deny-all: no provider effect may cross the Task
Authority boundary even when an ordinary capability would otherwise allow it.
That deny-all ceiling also covers runtime-mediated LLM and Human provider
effects owned by the process. The explicit `"*"` entry permits every current
and future effect class and should be used only when that deliberately broad
ceiling is intended.

Typed Git adds five effect families that must be present when a concrete
ceiling is used: `git.read`, `git.mutate`, `git.fetch`, `git.push`, and
`git.pull_request`. The corresponding protected-operation descriptors use the
`primitive.git.*` namespace. Exact Runtime boundary names such as
`runtime.git.status`, `runtime.git.commit`, and `runtime.git.pull` are linked in
Explain evidence, while the protected provider descriptor supplies the effect
class. An existing `shell:git` authorized capability or old image requirement
does not imply any Git effect or capability; a new manifest must explicitly
authorize the configured `git.repository_resource` (default `git:workspace`),
any filesystem paths, and the selected remote/PR resources. Mandatory
destructive-operation approval remains necessary even when the effect family is
within the manifest ceiling.

The durable effect-policy envelope is schema version 2 so unrestricted `null`
and deny-all `[]` cannot collapse during persistence. Legacy version 1 rows are
upcast on read: their empty list retains its historical unrestricted meaning,
while every newly written manifest uses the tagged version 2 representation.
Derived manifests inherit an omitted parent value. An unrestricted parent may
be narrowed to any list, including deny-all; a concrete or deny-all parent
cannot be widened back to unrestricted.

Per-use external-operation approval preallocates the eventual external
`effect_id` and binds it with the canonical argument hash and target state
version in the one-shot capability. The capability reservation carries that
same id into the provider intent, so a same-argument call cannot create a
different approved effect ledger entry.

## Closed input schema

Manifest input is closed at the top level. The supported fields are
`authorized_capabilities`, `permitted_effects`, `resource_budget`,
`approval_policy`, `data_flow_policy`, `expires_at`, `issued_by`, and
`metadata`. Process identity, image requirements, parent linkage, hashes, and
timestamps are runtime-owned and cannot be supplied as input. Any unknown
top-level field fails validation; for example, misspelling `permitted_effects`
or `expires_at` cannot silently remove an effect or expiry ceiling.

Each entry in `authorized_capabilities` and
`approval_policy.requestable_capabilities` is also closed. It accepts only:

- required `resource` and `rights` fields;
- optional `constraints`, `delegable`, `revocable`, `expires_at`,
  `uses_remaining`, and `max_delegation_depth` fields.

Unknown capability-entry fields fail validation instead of being discarded.
When present, `delegable` and `revocable` must be JSON booleans (or Python
`bool` values through the mapping API). Strings such as `"false"`, numbers,
and `null` are rejected rather than coerced. `permitted_effects` entries must
be non-empty strings; a wildcard must be exactly `"*"` or a single terminal
`.*`, such as `jsonrpc.*`. `resource_budget` is validated against
`ResourceBudget`, while
`data_flow_policy` has the separate closed schema below. Capability
`constraints`, other `approval_policy` values, and `metadata` are deliberately
policy-defined mappings rather than self-authorizing fields.

The entire manifest is trusted Host/admin-plane input, not model-authored task
content. `metadata` is opaque: it is persisted and covered by the manifest
hash, but the runtime does not scan or redact its values. The Host is therefore
responsible for excluding task payloads, credentials, and unnecessary personal
data. Metadata cannot grant authority, and runtime-owned keys such as
`launch_authority_mode`, `explicit`, `transition_ceiling`, and effect-policy
provenance are set by the runtime rather than trusted from the supplied mapping.

CLI example:

```bash
uv run agent-libos spawn \
  --goal "review one report" \
  --authority-manifest-json '{
    "authorized_capabilities": [
      {"resource":"filesystem:workspace:reports/*","rights":["read"]}
    ],
    "permitted_effects": ["filesystem.read_bytes"],
    "resource_budget": {"max_tool_calls": 20},
    "metadata": {"policy":"review-v1"}
  }'
```

Effect classes use protected-operation provider/operation ids, not model-facing
tool or syscall names. For example, the `read_text` convenience operation is
classified as `filesystem.read_bytes`. The current built-in inventory is
aggregated by `agent_libos/runtime/descriptor_catalog.py` from primitive, LLM,
Human, and module descriptor sets. The manifest parser deliberately does not
require catalog membership because a trusted Host extension may register an
additional protected-operation contract. An unknown exact entry, or an unknown
namespace wildcard, authorizes no currently emitted effect; it takes effect
only if a registered operation emits a matching `provider.operation` class.
It never bypasses the capability, approval, data-flow, or resource checks at
that operation boundary.

`POST /api/processes` and `POST /api/workflows/run` accept the same object as
`authority_manifest`. The Host-only
`AuthorityManifestManager.summary_for_process()` projection contains the
manifest id/hash, declared authority, unmet image requirements, effect ceiling,
budget, and policies, and omits opaque operator `metadata`. Explain does not
embed that mapping verbatim: `summary.authority_manifest` is a sanitized,
bounded observation envelope containing a preview, SHA-256 digest, byte count,
and `truncated`/`redacted` flags. A large manifest therefore need not expose
every summary field in its Explain preview.

The manifest is not a substitute for primitive capability checks. Tool
projection, Skills, image metadata, and requirement declarations remain
visibility or planning inputs only.

## Data identity domain

`data_flow_policy` has one strict v1 shape:

```json
{
  "schema_version": 1,
  "allowed_tenants": ["tenant-a"],
  "allowed_principals": ["analyst-a"]
}
```

Unknown keys, unsupported versions, wildcard/`mixed` entries, malformed
identities, and non-list fields fail closed. JSON inputs require arrays and the
Python mapping API requires `list` values; Python tuples are rejected. A
child/fork manifest inherits the parent sets when omitted and may only keep or
remove entries. Empty sets allow only data without a tenant/principal.

A fresh root goal created from text, a payload mapping, or the configured
default goal has the ordinary Object-label defaults: `sensitivity=normal`,
`trust_level=unknown`, `integrity=unknown`, `origin=local`, and no tenant or
principal. An existing goal `ObjectHandle` retains its Host-managed labels.
The manifest does not assign or reclassify those labels; after the manifest is
bound, launch validates the goal's tenant/principal against this receive-domain
policy before compiling the root capabilities.

This policy controls what identity domain a process may receive through a
goal, message, result, Object Task notification, memory merge, fork, or exec.
Those boundaries preserve the source labels; observing a labeled message also
creates a metadata-only process carrier so later text goals or replies inherit
the same label.

It is deliberately not an external Sink allowlist or trust root. A manifest
cannot mark `llm:*`, `jsonrpc:*`, a file, executable, or Human recipient as
trusted, cannot lower a Host Sink clearance, and does not need to repeat a
trusted Sink pattern. External transmission is governed by the independent
Host registry described in [Data Flow](data_flow.md), followed by ordinary
capability, `permitted_effects`, approval, and budget checks.
