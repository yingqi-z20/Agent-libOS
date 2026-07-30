# Agent libOS Historical Paper Roadmap

> **HISTORICAL ROADMAP — NOT A CURRENT CONTRACT.** This dated planning record is
> retained to explain past paper milestones. It is not the implementation reference
> and intentionally contains incomplete or superseded command names,
> feature baselines, priorities, and schedules. Do not use it for current
> runtime behavior, security guarantees, release status, or CLI syntax; use
> [README.md](README.md) and the current `docs/` references instead.

Paper title:

> Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents

Date: 2026-06-13
Roadmap horizon: 2026-06-13 to 2026-09-24 AoE
Target venue: EuroSys 2027 Fall full paper submission

Note: the 2026-09-24 AoE date is the working deadline used by this roadmap.
Before artifact freeze, re-check the official CFP and submission system rather
than treating this file as the source of truth for conference dates.
This roadmap remains historical even when an individual milestone happens to
match current code. Re-check current documentation and tests rather than
updating operational guidance from this file.

## 1. Submission Goal

The paper goal is to make Agent libOS a systems prototype for self-evolving LLM
agents: agents that can persist state, fork work, ask humans, call tools,
write or activate new Skills, register process-local Deno/TypeScript JIT tools,
register or execute new AgentImage definitions, use remote resources, and
change their working context over long executions.

The central claim is:

> Self-evolving LLM agents need a runtime substrate that separates evolution of
> the model-visible action surface from authority over external resources.
> Agent libOS provides that substrate with process identity, capabilities,
> primitive-mediated effects, and append-only audit.

The runtime should allow adaptation without implicit privilege escalation. A
Skill, JIT tool, child process, checkpoint fork, newly registered/executed or
checkpoint-committed image, JSON-RPC endpoint, or future provider may expand what the model can ask
to do, but it must not expand what the process is authorized to affect unless a
capability, policy decision, or human approval explicitly permits it.

## 2. Paper Thesis

Agent libOS is a runtime substrate for capability-controlled self-evolving LLM
agents.

The paper is not about a larger tool catalog, a new prompt pattern, or a
general-purpose sandbox. Its contribution is the boundary:

```text
process identity + capability + primitive + audit
```

This boundary makes self-evolution auditable and safe to evaluate:

- model-visible affordances can change dynamically,
- resource authority remains checked at primitive use,
- human approval is part of the runtime operation, not a retry protocol,
- side effects are provider-classified and append-only in audit,
- checkpoints restore reconstructable agent state without claiming to rewind
  external reality,
- LLM calls and tool/JIT decisions are persistently recorded for cost and
  reproducibility analysis.

## 3. Current System Baseline

The current repository already implements the substrate needed for the paper
prototype:

- `AgentProcess` lifecycle: spawn, fork, exec, wait, signal, pause, resume,
  exit, process-local cwd, and message queues.
- Capability: typed resource matching, deny/ask/allow effects, issue,
  delegate, revoke, one-shot consumption, attenuation, and audit lineage.
- Primitive boundary: filesystem, typed Git, shell, clock, human, process,
  image, checkpoint, Object Memory, Skill, JIT syscall, JSON-RPC, and MCP calls
  go through runtime primitives instead of direct model-facing tool authority.
- Object Memory: process-private namespaces by default, explicit shared
  namespaces through capabilities, and file/object bridges.
- Human-as-device: questions, output, approval, ordinary messages, and
  interrupts are runtime objects and queues.
- Deno/TypeScript JIT: agent-authored tools run with no ambient host
  permissions and can access libOS only through `libos.syscall`.
- AgentImage registry, exec, and checkpoint commit: images can change prompt,
  default tool table, default Skills, baked internal Object Memory/JIT state,
  and lifecycle behavior, while `exec` and checkpoint-derived image boot never
  grant target-image required capabilities automatically.
- Standard Agent Skills: `SKILL.md` packages can add prompt instructions,
  tool visibility, bundled resources, and JIT candidates without granting
  filesystem/shell/object/remote authority.
- Runtime Modules: trusted pre-start Python extensions can register host-side
  runtime components but are TCB, not process authority.
- JSON-RPC over HTTP: client-only, pre-registered endpoints/methods, no
  model-supplied URLs or secrets, capability-checked method invocation.
- Typed Git: the system-Git provider is pinned to the workspace repository and
  covers local changes, managed worktrees, immutable patch Objects, existing
  Host-configured remotes, and repository-local simulated pull requests with
  capability/state-token/approval enforcement. It is not a GitHub/GitLab API.
- Scoped checkpointing: reconstructable process-subtree state can be restored
  or forked without rewinding audit/events, LLM-call identity, or external
  provider evidence. Audit and effect-transition history remain append-only;
  guarded effect state and retained LLM/effect payload projections can advance
  in place.
- SQLite persistence: process metadata, capabilities, messages, human
  requests, LLM calls, audit, events, tool candidates, Skills, JSON-RPC
  endpoints, modules, checkpoints, and external-effect records.
- M1 benchmark harness: 32 deterministic runtime-safety tasks, baselines,
  ablations, self-evolution subset, side-effect oracle, and metrics output.

The remaining gap is evaluation depth and paper-grade explanation, not another
round of unrelated feature growth.

## 4. Submission Contributions

The paper should present four contributions.

1. Runtime model:
   Agent libOS models a self-evolving agent as an `AgentProcess` with dynamic
   tool/Skill/JIT/image visibility, child processes, object namespaces,
   messages, human I/O, checkpoints, and capability-controlled primitives.

2. Implementation:
   A Python runtime substrate with Resource Provider Substrate, Capability,
   ToolBroker, standard Skills, Deno/TypeScript syscall-only JIT tools,
   JSON-RPC remote resources, scoped checkpoints, persistent LLM accounting,
   and audit/event persistence.

3. Benchmark:
   Runtime-safety workloads that exercise not just ordinary tool calls, but
   self-evolution paths: Skill activation, JIT registration, image
   registration/exec/checkpoint commit, child process delegation, checkpoint fork, remote endpoint
   use, shell policy, Object Memory access, and human approval.

4. Evaluation:
   Baseline comparison, ablations, overhead, approval burden, task success,
   unauthorized side-effect rate, and audit completeness.

Regression tests support artifact credibility. They should not replace the
paper evaluation.

## 5. Milestones

| Milestone | Dates | Main Outcome |
| --- | --- | --- |
| M0 | 2026-06-05 to 2026-06-12 | Hygiene, invariant map, thesis freeze |
| M1 | 2026-06-13 to 2026-06-30 | Runtime-safety benchmark and baseline harness |
| M2 | 2026-07-01 to 2026-07-20 | Self-evolution workload: Skills, JIT, images, child processes, remote resources |
| M3 | 2026-07-21 to 2026-08-05 | Audit explain and context materialization metadata |
| M4 | 2026-08-06 to 2026-08-22 | Security and systems experiments, ablations, first figures |
| M5 | 2026-08-23 to 2026-09-07 | Paper rewrite and anonymous artifact freeze |
| M6 | 2026-09-08 to 2026-09-17 | Internal review, artifact dry run, title/abstract lock |
| M7 | 2026-09-18 to 2026-09-24 | Final submission checks |

## 6. M0: Hygiene And Story Freeze

Status: complete or nearly complete.

Exit criteria:

- README and docs agree with current implementation.
- `docs/invariants.md` maps core invariants to tests or explicit gaps.
- `docs/paper_thesis.md` states the fixed Agent libOS title and thesis.
- Artifact checklist exists and no longer uses the old temporary PAR name.
- The historical M0 benchmark-schema milestone is recorded; the current
  shipped contract uses task schema v1 and run-output schema v2.
- No current docs claim Python JIT, direct external framework adapters,
  unsupported rollback, or Skill-as-permission behavior.

## 7. M1: Runtime-Safety Benchmark And Baselines

Goal: make safety claims measurable before adding more system surface.

Already implemented:

- 20+ deterministic runtime-safety tasks.
- `experiments/run_benchmark.py`.
- `experiments/collect_metrics.py`.
- Side-effect oracle.
- Baselines and ablations:
  - `direct_tool_wrapper`
  - `confirmation_wrapper`
  - `sandbox_only`
  - `agent_libos_full`
  - `no_primitive_approval`
  - `no_audit_linkage`
  - `no_namespace_isolation`
  - `no_fork_attenuation`

Completed M1 hardening:

- Added self-evolution tasks for Skill activation, Skill-loaded JIT syscall
  denial, image registration/exec without target-image capability grants, child
  process delegation, checkpoint fork after revocation, checkpoint-to-image
  commit, and JSON-RPC visibility without method authority.
- `collect_metrics.py` reports self-evolution-specific counters:
  `skill_activations`, `jit_registrations`, `image_commits`,
  `image_registrations`, `image_execs`, `child_processes`,
  `checkpoint_forks`, and `remote_calls`.
- The default benchmark path remains deterministic and no-token.

Exit criteria:

- One command runs the deterministic subset.
- One command emits machine-readable metrics.
- The benchmark contains ordinary tool-call and self-evolution attack classes.
- Every modeled side effect is classified as allowed, forbidden, or unknown.

## 8. M2: Self-Evolution Workload

Goal: demonstrate that Agent libOS controls agents that modify their own
operational surface.

Deliverables:

Status as of 1.0.1: the local Git/worktree, patch artifact, and simulated-PR
deliverables below are implemented through `Runtime.git`; hosted Git provider
integration remains intentionally out of scope.

- A workload where agents use or attempt to use:
  - standard `SKILL.md` packages,
  - Deno/TypeScript JIT tools,
  - new AgentImage registration and `exec`,
  - child processes,
  - checkpoint fork/restore,
  - Object Memory namespace sharing,
  - JSON-RPC remote resources.
- A coding-agent workload with local git/worktree fixtures.
- Patch artifacts connected to process, tool/syscall, primitive, capability,
  human approval, and source evidence.
- Mock merge/PR primitives as high-risk operations; no real GitHub dependency.

Key invariant:

Self-evolution may increase expressiveness but must not increase authority.

Tests and checks:

- Skill activation changes prompt/tool visibility, not capabilities.
- JIT tools cannot bypass Deno sandbox, syscall broker, or Capability.
- Image registration and `exec` can change prompts/default tools/default
  Skills, but cannot grant target-image required capabilities.
- Child processes receive only attenuated delegated capabilities.
- Checkpoint fork remaps process/object/capability identity and does not
  resurrect revoked authority.
- JSON-RPC method calls require method capability and redact secrets in audit.
- Managed worktree paths remain workspace-contained and separately authorized;
  every selected worktree still shares the repository's refs/common metadata
  under the typed provider lock and state-token checks.

Exit criteria:

- Baseline and Agent libOS runners execute the same self-evolution tasks.
- At least one coding task produces an auditable patch artifact.
- Merge/PR operations are policy-controlled and auditable.
- No benchmark-critical path depends on real external services.

## 9. M3: Audit Explain And Context Policy

Goal: make the runtime boundary explainable enough for paper evaluation.

Deliverables:

- `audit explain` CLI/API.
- `audit why-allowed` and `audit why-denied`.
- Capability chain explanation.
- Object and patch lineage explanation.
- LLM context materialization metadata:
  - included object ids,
  - omitted object ids,
  - summaries/truncation reason,
  - materialization policy id,
  - token budget,
  - tool-result compaction provenance.

Minimum explain shape:

```json
{
  "process": "...",
  "tool": "...",
  "syscall": "...",
  "primitive": "...",
  "resource": "...",
  "decision": "allowed",
  "capability": "...",
  "issuer_chain": ["..."],
  "human_approval": "...",
  "provider_effect": "...",
  "output_object": "..."
}
```

Exit criteria:

- Every side-effecting benchmark event can be explained or explicitly marked
  as a known gap.
- Unauthorized Object Memory payloads cannot leak into prompts through summary,
  retrieval, or resource reads.
- Benchmark metrics include audit completeness.

## 10. M4: Security And Systems Experiments

Goal: produce the quantitative results that carry the paper.

Required task families:

- Prompt-injected file read/write/delete.
- Secret exfiltration attempts.
- Shell policy bypass and nested interpreter attempts.
- Skill/JIT/image capability escalation attempts.
- Child-process delegation leakage.
- Checkpoint restore/fork authority resurrection attempts.
- Object Memory namespace leakage.
- Human approval spoofing.
- JSON-RPC remote endpoint misuse.
- Context materialization leakage.
- Coding-agent patch/merge misuse.

Required figures or tables:

1. Unauthorized side effects blocked.
2. Task success vs safety mode.
3. Approval burden.
4. Audit completeness.
5. Primitive/audit/context/JIT overhead.
6. Ablation matrix.
7. Self-evolution safety table: Skill, JIT, image, fork, checkpoint, JSON-RPC.

Exit criteria:

- 50+ adversarial/runtime tasks.
- 20+ coding-agent-style tasks.
- 3+ baselines.
- 4+ ablations.
- Each main claim has a figure, table, or quantified result.

## 11. M5: Paper Rewrite And Artifact Freeze

Goal: turn implementation and results into an anonymous, reproducible systems
submission.

Paper structure:

1. Introduction: why self-evolving agents need a runtime substrate.
2. Motivation: tool visibility is not authority.
3. Runtime model.
4. Implementation.
5. Benchmark and baselines.
6. Evaluation.
7. Limitations.
8. Related work.
9. Conclusion.

Artifact requirements:

- deterministic subset runs without real LLM credentials,
- real-model subset is optional and documented,
- no private endpoints, keys, or account metadata,
- no local absolute paths or identity leaks,
- figure/table regeneration is scripted,
- paper title and artifact docs use the fixed Agent libOS title consistently.

Exit criteria:

- Fresh clone dry run passes.
- CI and deterministic benchmark subset pass.
- Paper text and artifact commands agree.
- Anonymous paper and artifact scans are clean.

## 12. M6: Internal Review And Submission Lock

Goal: surface paper and artifact problems while there is still time to fix
them.

Review roles:

- systems novelty and significance,
- threat model and security claims,
- self-evolution framing,
- benchmark validity,
- coding-agent realism,
- artifact reproducibility,
- writing clarity under page limits.

Exit criteria:

- Title and abstract are frozen.
- Main claims are frozen.
- No new experiments are planned except emergency fixes.
- Limitations are explicit:
  - no kernel-grade sandbox claim,
  - no full prompt-injection solution,
  - no rollback of irreversible external effects,
  - no production multi-tenant guarantee,
  - no claim that external frameworks are trusted security boundaries.

## 13. M7: Final Submission

Goal: submit a clean, anonymous, internally consistent paper and artifact.

Final checklist:

- Official CFP deadline and double-blind rules re-checked.
- Title: `Agent libOS: A Runtime Substrate for Capability-Controlled
  Self-Evolving LLM Agents`.
- AI-use disclosure included if required.
- Figures readable in grayscale.
- Page limit respected.
- Artifact link anonymous.
- Reproduction instructions work from a clean checkout.
- No author identity leaks in paper, artifact, scripts, paths, or metadata.
- No claims depend on missing experiments.

## 14. Deprioritized Before Submission

- Full OpenAI Agents SDK compatibility.
- Full MCP ecosystem compatibility.
- Real GitHub PR creation and OAuth.
- Slack, calendar, browser, or enterprise provider integrations.
- Distributed multi-worker runtime.
- Full time-travel UI.
- Complex semantic RAG or vector database integration.
- LLM-based risk classifier as a safety authority.
- Persistent signed public Skill marketplace.
- Multi-model routing.

Minimal acceptable versions:

- Agent adapter: benchmark adapter, not ecosystem-complete adapter.
- Hosted Git providers: repository-local simulated PR and adversarial/local
  remote workloads, not production GitHub/GitLab integrations. MCP Tools have a
  separate client implementation but broader ecosystem compatibility remains
  out of scope.
- Durable execution: scoped checkpoint restore/fork and replay diagnostics,
  with overhead and authority-resurrection evaluation.
- Risk-aware approval: deterministic rules, not intelligent classifier.

## 15. Minimum Bar For A Strong Submission

- 50+ adversarial/runtime tasks.
- 20+ coding-agent-style tasks.
- 10+ self-evolution-specific tasks.
- 3+ baselines.
- 4+ ablations.
- 5 metric families:
  - unauthorized side-effect rate,
  - task success,
  - approval burden,
  - audit completeness,
  - overhead/cost.
- One command for deterministic tests.
- One command for a benchmark subset.
- One command for figure/table regeneration.
- Anonymous artifact branch.
- Roughly 4 pages of evaluation, not only a regression-test table.

Without these, Agent libOS reads as a promising architecture prototype. With
them, it can argue a systems contribution: self-evolving LLM agents become
safer and more explainable when action-surface evolution is decoupled from
resource authority by a capability-controlled runtime substrate.
