---
name: agent-libos-skill-navigation
description: Discover an applicable Agent Skill, activate its exact snapshot, read declared resources from that loaded snapshot, and unload it when its guidance or tools are no longer needed. Use when the required Skill is not already loaded or when a registered Skill must be found; never use Skill lifecycle operations to obtain Capability authority or bypass a denied tool.
allowed-tools: discover_skills activate_skill read_skill_resource unload_skill
---
# Navigate Skills

First inspect `Loaded skills` and visible schemas. If the exact Skill is already usable, do not rediscover or reactivate it: activation and unload are durable, audited, non-idempotent lifecycle mutations. Select the smallest direct match; a similar name is not an interchangeable implementation.

## Tool guide

### `discover_skills`

Use `{"text":"git pull request","limit":8}` only when the exact ID is uncertain or a registered Skill may apply.

- The result contains `skills`, `catalog_scope`, and `has_more`. Applicable built-ins are a complete prefix and do not count against `limit`; `text` and `limit` bound only registered entries.
- Model-facing discovery merges applicable built-ins with registered entries; it does not scan directories for unregistered packages. `catalog_scope="builtin_only"` means registry visibility was absent, not that no other Skill exists.
- Registered discovery needs `read` on the configured `skill:*` registry resource. A limited grant can be consumed, so use a useful first-call limit. Missing authority silently narrows scope to built-ins.
- There is no cursor. If `has_more=true`, refine `text` or raise `limit` within the Host maximum. If a later call changes to `builtin_only`, report the authority change instead of concluding that prior registered entries disappeared.
- Built-in summaries include `active` and `available_tools`; registered summaries and the metadata-only prompt catalog omit `active`. Use `Loaded skills` as the primary state check.
- Discovery neither loads instructions nor grants tool, Capability, provider, or approval authority.

### `activate_skill`

Pass exactly one returned or prompt-provided ID, for example `{"skill_id":"agent-libos-jsonrpc"}`.

A built-in activates only when the image already owns every declared static tool under the exact IDs. It atomically projects all of them into model visibility; it does not change the full tool table, add JIT, grant Capability, or need current-process `skill:<id>` authority.

A registered activation requires `execute skill:<id>` and can add static/JIT bindings. Resume an ASK decision rather than duplicating it. Reactivation atomically replaces its prior snapshot/bindings and retires superseded JIT.

Both persist an immutable snapshot/provenance plus audit/event evidence. `authority_changed=false` means Capability did not expand, not that no state changed.

Built-in success includes `activation_kind="builtin_projection"`, `authority_changed=false`, maps/hashes, and empty `jit_tool_ids`. Registered success includes identity, maps, and hashes but omits those two fields. New guidance/schemas appear next turn.

Direct projection exposes each registered JIT schema. Multiplexed images expose only `run_jit_tool`; use the loaded contract and `{"tool_name":"exact-name","arguments":{...}}`. Never guess a hidden schema.

### `read_skill_resource`

Read a package-relative resource after activation and before unload, for example `{"skill_id":"example-skill","path":"references/protocol.md","max_bytes":8192}`.

- Use an exact path from loaded instructions or `resources` metadata. Absolute paths, `..`, and inventions fail; this reads neither workspace files nor the live registry package.
- Reads use the immutable loaded snapshot, unaffected by later registry replacement, and need no second `skill:<id>` read grant.
- `max_bytes` rejects rather than truncates. Omit it for the Host default or choose at least disclosed `size_bytes`; do not probe repeatedly.
- Inspect `kind`, `size_bytes`, and `sha256`. For `kind="text"`, use `content`; for `kind="base64"`, use `content_base64`. Exactly one payload form is useful.
- Built-in tool-Skill packages contain only `SKILL.md`, so bundled resources normally belong to registered Skills.

### `unload_skill`

Call `{"skill_id":"example-skill"}` only after its guidance/resources are no longer needed.

Built-in unload needs no current-process `skill:<id>` right. Registered unload separately requires `execute skill:<id>`. It removes that snapshot/bindings, retires JIT, and restores overlapping/base bindings. It never revokes Capability or reverses external effects.

The result includes `activation_kind`, `removed_tools`, and `authority_changed=false`. Empty `removed_tools` is normal for a built-in because image-owned full-table bindings remain.

## Recommended workflow

1. Check `Loaded skills` and visible schemas. If the exact Skill is already usable, stop navigation and perform the task.
2. Activate an exact catalog built-in directly; otherwise discover once with focused text and a sufficient limit.
3. Select by exact ID, description, tools, source type, and package hash—not name resemblance.
4. Activate once. Match returned `skill_id` and any discovered `package_sha256`.
5. Validate the tool maps. For a built-in, require `jit_tool_ids == {}` and `set(tool_names) == keys(tool_ids)`, aligned with `available_tools`. For a registered Skill, require `set(tool_names) == keys(tool_ids) union keys(jit_tool_ids)`.
6. Next turn, confirm its `Loaded skills` instructions and read each schema before use. Read only required declared resources.
7. Use the Skill to finish the user task. Keep it loaded across related steps; unload only when reduced prompt/tool exposure is useful and no later step depends on it.
8. Recheck after transitions. Same-image fork/restore can preserve snapshots; fresh-child spawn or exec uses target-image defaults and may drop projections.

## Failure and recovery

- Already loaded: do not reactivate; it rewrites lifecycle evidence and can replace registered JIT.
- Unknown ID or absent registered entry: refine discovery only when `has_more=true`; otherwise report that no visible registered match exists. Do not scan paths or invent an ID through these tools.
- Unsupported built-in membership, tool-ID mismatch, invalid provenance, or hash mismatch: stop. Do not request partial activation or substitute an unverified global tool.
- Discovery/activation denial: request only exact catalog `read` or `skill:<id>` `execute`; never duplicate pending ASK.
- Unknown activation settlement: inspect the next turn's `Loaded skills`; for built-ins, a later discovery `active` value is also usable. Do not blindly repeat a non-idempotent activation.
- Missing/oversized resource: compare loaded metadata and correct the path or ceiling once; no filesystem bypass.
- Multiplexed JIT without a documented name/argument contract: stop and report the incomplete Skill package rather than guessing.
- Unknown unload settlement: verify loaded prompt state on the next turn. Schema presence alone is insufficient because the base image or another Skill may own the same binding.
- Process transition: reassess current loaded state and exact schemas; never rely only on what was loaded in the parent or pre-exec process.

## Completion evidence

Navigation is complete when one exact Skill is loaded, identity/maps are consistent, guidance/schemas are visible, and required resources came from that snapshot with hashes recorded. Domain completion still follows that Skill.

After unload, require the exact Skill to be absent from the next `Loaded skills` section. For a built-in, discovery may additionally show `active=false`; registered summaries have no such field. Report `removed_tools` as full-table deletions only, not as a claim that every model schema or prior external effect disappeared.
