---
name: agent-libos-skill-navigation
description: Discover an applicable Agent Skill on demand, activate its exact snapshot, read declared resources from that loaded snapshot, and unload it when its guidance or tools are no longer needed. Use whenever required guidance or a domain tool is not already visible; never use Skill lifecycle operations to obtain Capability authority or bypass a denied tool.
allowed-tools: discover_skills activate_skill read_skill_resource unload_skill
---
# Navigate Skills

First inspect `Loaded skills` and visible schemas. If the exact Skill is already usable, do not rediscover or reactivate it: activation and unload are durable, audited, non-idempotent lifecycle mutations. Otherwise discover with two to four concrete domain/action terms, select the smallest direct match, and load only that Skill. A similar name is not an interchangeable implementation.

## Tool guide

### `discover_skills`

Use a focused query such as `{"text":"git pull request","limit":8}` whenever the exact ID is uncertain or the required domain schema is not visible. Query terms are matched independently against visible identity and description metadata, then results are relevance-ranked. A multi-capability query can return multiple narrowly owned Skills when each has sufficient term coverage; activate each relevant exact ID instead of trying to force one Skill to own the whole query.

- `text` and `limit` apply uniformly to every visible Skill. Start with concrete nouns and actions, omit `limit` or use at least 5 as an unquoted JSON integer, and avoid pasting the whole goal. A broad empty query is allowed but is usually wasteful.
- The result contains one `skills` page plus `has_more`, `visibility_limited`, and `next_step`. Every summary has the same fields regardless of package source: identity, description, declared tool/action/JIT names, required capabilities, package hash, and `active`.
- `active=true` means the trusted loaded snapshot has the same `package_sha256` as this catalog entry. A loaded older snapshot remains immutable and usable, but discovery reports it as `active=false` after package replacement or a catalog upgrade.
- `next_step=activate_skill` means at least one returned match is not the current loaded content. `next_step=use_loaded_skill` means every returned match is already loaded at the discovered hash. `next_step=refine_search` means no match was returned, so shorten or replace the terms; never repeat an unchanged query.
- There is no cursor. `has_more=true` means more matches exist for this exact query, so refine `text` or raise `limit` within the Host maximum. It does not control whether a zero-result query may be refined.
- `visibility_limited=true` means catalog authority prevented searching every configured source. It says nothing about the origin of returned entries, does not invalidate a plausible returned match, and does not by itself justify requesting broader authority.
- Model discovery does not scan directories for packages the Host has not made visible. Do not invent IDs or inspect paths as a fallback.
- Discovery returns metadata only. It does not load instructions, expose declared domain schemas, grant Capability, contact providers, or approve effects.

### `activate_skill`

Pass exactly one ID and its hash from the same discovery result, for example `{"skill_id":"example-skill","expected_package_sha256":"<64 lowercase hex characters>"}`. Do not activate from a guessed name or omit, normalize, or substitute the discovered hash.

Activation always has one model-visible result contract: the outer envelope is `{"result": {...}}`, whose nested result contains `pid`, `skill_id`, `name`, `version`, `tool_names`, `tool_ids`, `jit_tool_ids`, `instructions_hash`, and `package_sha256`. Package origin, Host trust mechanism, and internal activation provenance are not model routing inputs.

The runtime compare-and-swaps against `expected_package_sha256`, then validates the exact package snapshot and every binding atomically under the registry lifecycle lock. If registration or catalog content changed after discovery, activation fails before process, tool, or JIT publication. A static binding may reveal an image-owned tool; a package-declared JIT binding may add a process-local tool. Neither case grants the primitive Capability or Human approval needed for an effect. Activation authority is determined by Host policy and package contents; if an ASK is pending, resume it rather than submitting a duplicate activation.

New instructions and schemas appear in the next turn. Reactivation is not a harmless read: it replaces the previous snapshot and bindings and can retire superseded JIT, so avoid it when `active=true`. If the same ID is loaded but discovery reports `active=false`, keep the old immutable snapshot unless the newly discovered metadata and hash justify an explicit replacement.

Direct JIT exposure shows each activated JIT schema. Multiplexed images expose only `run_jit_tool`; use the loaded contract and `{"tool_name":"exact-name","arguments":{...}}`. The prompt omits the JIT catalog/source resource entries, but an exact loaded-snapshot path already supplied by trusted instructions or earlier authorized evidence remains readable. Never guess a hidden schema or probe filenames.

### `read_skill_resource`

Read a package-relative resource after activation and before unload, for example `{"skill_id":"example-skill","path":"references/protocol.md","max_bytes":8192}`.

- Use an exact path from loaded instructions, visible `resources` metadata, or prior authorized evidence. Absolute paths, `..`, and inventions fail; this reads neither workspace files nor the live registry package. In multiplexed mode an omitted JIT-contract path is still readable if its exact path is already known, but the tool does not list or discover it.
- Reads use the immutable loaded snapshot, unaffected by later registry replacement, and need no second `skill:<id>` read grant.
- `max_bytes` rejects rather than truncates. Omit it for the Host default or choose at least disclosed `size_bytes`; do not probe repeatedly.
- The outer result is `{"resource": {...}}`; inspect the nested `kind`, `size_bytes`, and `sha256`. Both payload keys are present. For `kind="text"`, use non-null `content` and expect `content_base64=null`; for `kind="base64"`, use non-null `content_base64` and expect `content=null`.
- A Skill with no declared resources needs no resource read. Do not infer package origin from that absence.

### `unload_skill`

Call `{"skill_id":"example-skill"}` only after its guidance/resources are no longer needed.

Unload removes that snapshot and its contributed visibility, retires unshared JIT, and restores overlapping or base bindings. It never revokes Capability or reverses external effects. Host policy may require authority for the lifecycle mutation.

The outer envelope is `{"result": {...}}`; the nested result contains `pid`, `skill_id`, and `removed_tools`. Empty `removed_tools` is normal when the full process table or another loaded Skill still owns every binding; it does not mean the prompt body remained loaded.

## Recommended workflow

1. Check `Loaded skills` and visible schemas. If the exact Skill is already usable, stop navigation and perform the task.
2. Discover once with two to four task terms and a sufficient but bounded limit.
3. Select by exact ID, description, declared behavior, and package hash—not name resemblance or presumed source.
4. If `next_step=use_loaded_skill`, do not reactivate. Otherwise activate once with the selected discovery row's exact ID and `package_sha256`; match both against the activation receipt.
5. Validate `set(tool_names) == keys(tool_ids) union keys(jit_tool_ids)`. Treat any mismatch as incomplete settlement.
6. On the next turn, confirm the exact body under `Loaded skills` and inspect each newly visible schema before use. Read only required declared resources.
7. Use the Skill to finish the user task. Keep it loaded across related steps; unload only when reduced prompt/tool exposure is useful and no later step depends on it.
8. Recheck after transitions. Fork or restore can preserve snapshots; fresh spawn or exec may start with no loaded Skills or with target-image defaults.

## Failure and recovery

- Already loaded: do not reactivate; it rewrites lifecycle evidence and can replace package-declared JIT.
- No matching entry: follow `next_step=refine_search` and change or shorten the terms once; `has_more` only describes additional matches to the same query. If visibility is limited and no refined query matches, report that authority bounds the conclusion. Do not scan paths or invent an ID through these tools.
- Tool-ID mismatch or invalid provenance: stop. Do not request partial activation or substitute an unverified similarly named tool.
- `SkillPackageChanged` activation error: the stale activation made no lifecycle mutation. Rediscover, re-evaluate the changed metadata and hash, and activate only if that exact new content is still appropriate; never retry the stale hash.
- Discovery or activation denial: request only the exact catalog or Skill right made available by Host policy; never duplicate a pending ASK.
- Unknown activation settlement: inspect the next turn's `Loaded skills` or repeat focused discovery and check `active`. Do not blindly repeat a non-idempotent activation.
- Missing/oversized resource: compare loaded metadata and correct the path or ceiling once; no filesystem bypass.
- Multiplexed JIT without a retained contract or an exact trusted path to its loaded-snapshot contract resource: stop and report the incomplete Skill package rather than guessing names, arguments, or paths.
- Unknown unload settlement: verify loaded prompt state on the next turn. Schema presence alone is insufficient because the base image or another Skill may own the same binding.
- Process transition: reassess current loaded state and exact schemas; never rely only on what was loaded in the parent or pre-exec process.

## Completion evidence

Navigation is complete when one exact Skill is loaded, identity and common tool maps are consistent, guidance and schemas are visible, and required resources came from that snapshot with hashes recorded. Domain completion still follows that Skill.

After unload, require the exact Skill to be absent from the next `Loaded skills` section; focused discovery may additionally show `active=false`. Report `removed_tools` as full-table deletions only, not as a claim that every model schema or prior external effect disappeared.
