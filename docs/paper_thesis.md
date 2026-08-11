# Research Thesis

This document explains the research framing of the implemented runtime. It is
not an API, security, support, or release-status contract; those contracts live
in the references linked from [README.md](../README.md).

Paper title:

> Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents

## Thesis

Self-evolving LLM agents need a runtime substrate that lets their model-visible
action surface change without letting resource authority grow implicitly.

Modern agent systems increasingly persist memory, fork work, call shells,
activate Skills, register self-authored tools, create, execute, or commit new images,
use remote resources, ask humans, and resume across long executions. Those
behaviors are useful, but they make prompt-only control, wrapper-level tool
lists, and host isolation insufficient as the primary authority boundary.

Agent libOS argues for two coupled, Host-enforced admission boundaries and a
separate evidence plane. Applicable policy, approval, and budget terms are
conjunctive checks; none is implied by tool visibility:

```text
operation admission: process identity + capability + Task Authority ceiling
                     + operation policy/Human approval + ResourceBudget + primitive
information flow:   trusted data labels + Host Sink trust + exact release
evidence (not authority): durable intent + classified outcome + event/audit
                          + explicit causal links
```

A process may see a model-facing tool, Skill, JIT tool, image definition,
child-process handle, checkpoint, or remote endpoint. Resource access is still
decided only when a libOS primitive runs under that process id. Capability and
Task Authority checks, policy, human approval, resource accounting, provider
containment, external-effect classification, events, and audit all happen at
that primitive boundary. Audit and other evidence record what the boundary did;
they do not grant authority or make an otherwise denied operation admissible.
Trusted source labels propagate through runtime Objects, messages, Tool/JIT
results, Human answers, and provider ingress. Before runtime-mediated external
egress, the Host resolves a canonical Sink identity and clearance from its
durable registry; a model cannot declare a Sink trusted. Conditional
high-sensitivity egress requires an exact one-shot Human release bound to the
Sink, payload, source versions, labels, manifest, and registry generation.
Internal process handoff is not cleared as an external Sink: it preserves
labels and must fit the receiving process's Task Authority receive-domain.

The contribution is not a larger tool catalog. The contribution is a runtime
substrate for capability-controlled self-evolution.

## Contributions

1. Runtime model.
   Agent libOS models an agent as an `AgentProcess` with process-local Object
   Memory namespaces, process-local working directories, message queues, child
   lifecycle, AgentImage registration, exec, and checkpoint commit, standard Skill activation,
   process-local Deno/TypeScript JIT tools, checkpoints, human I/O, and
   Task Authority ceilings, Human approval and policy, hierarchical resource
   budgets, and capability-controlled primitives.

2. Implementation.
   The current implementation realizes the model in Python with Capability,
   Resource Provider Substrate, runtime store persistence, audit/events, scoped
   checkpoint restore/fork/replay diagnostics, persistent LLM call accounting,
   image registry/exec/commit primitives, standard `SKILL.md` packages, a typed
   system-Git provider for the configured, runtime-pinned workspace repository,
   JSON-RPC over HTTP
   client endpoints, MCP client tools over registered servers, and
   Deno/TypeScript JIT tools that can reach libOS only through syscall RPC.
   The same implementation carries trusted data-flow labels, enforces
   Host-owned Sink clearance at filesystem-write, LLM, Human, JSON-RPC, MCP,
   Shell, PTY, and GUI presentation boundaries, checks internal process handoff
   against the receiving Task Authority Manifest's data-flow policy, and routes
   provider ingress through the shared Protected Operation SDK. The protected
   prepare transaction reauthorizes and reserves finite capability/release
   authority together with its durable external-effect intent; provider side
   effects remain outside that transaction, so ambiguous dispatched outcomes
   remain explicit recovery inputs. Launch, exec, fork, and checkpoint restore
   use separate, recovery-bound runtime publications. See the exact
   [data-flow contract](data_flow.md),
   [protected-operation lifecycle](protected_operation_sdk.md), and
   [publication reconciliation model](explainable_operations.md).

3. Benchmark suite.
   The current implementation includes a deterministic runtime-safety
   harness with 33 schema-v1 adversarial tasks, wrapper baselines, ablations,
   declared allowed/forbidden side effects, evidence-backed outcome records,
   explicit exact/prefix/glob oracle matching, and fail-closed stable metrics
   output. The suite includes a first self-evolution subset covering Skill
   activation, JIT registration, image registration/exec/checkpoint commit,
   child-process delegation, checkpoint fork, JSON-RPC remote-resource
   visibility, Git worktree/config/remote/patch-lineage boundaries, and a
   Semantic Shadow authority-injection fixture.

4. Evaluation.
   The benchmark supports comparison against the deterministic
   `direct_tool_wrapper`, `confirmation_wrapper`, and static-category
   `sandbox_only` interventions. These runners operate on copied fixtures;
   `sandbox_only` is not Host, container, VM, or other operating-system
   isolation, and none of these deterministic baselines invokes a real LLM.
   The implemented stable columns discussed here are `tasks`,
   `task_success_rate`, `safety_pass_rate`, `unauthorized_side_effect_rate`,
   `false_denial_rate`, `approval_count`, `tool_calls`, `primitive_calls`,
   `llm_tokens`, `wall_time_s`, and `audit_completeness`. The benchmark also
   emits named self-evolution attempt counters and explicit rate numerators,
   denominators, and validity diagnostics; it does not emit cost or overhead
   metrics. Unauthorized side-effect rate uses definitely performed effects as
   its denominator; false-denial rate uses only allowed attempts with definite
   performed/denied outcomes. Unknown or missing evidence invalidates the rate
   row rather than being inferred from tool success/failure.
   Practical workflow results must additionally label their evidence level.
   Only `native-live` rows with real ToolBroker calls, provider state oracles,
   external-effect records, and explicit operation links are runtime evidence.
   `modeled` rows are design-coverage experiments and use a separate
   denominator; trace completeness never upgrades them to native execution.

## Non-Goals

- Agent libOS does not claim kernel-grade sandboxing. Host isolation layers
  such as containers, Deno, WASM, or VMs are useful provider backends, not
  replacements for agent-level authority.
- Agent libOS does not solve all prompt injection. It constrains side effects
  and authority even when prompt content is adversarial.
- Agent libOS does not roll back irreversible external side effects.
  Checkpoint restore reconstructs scoped runtime state; provider-classified
  external effects remain in durable history and are report-only unless a
  future provider compensation API implements explicit repair. Their prepared
  intent rows are finalized or retention-reduced in place under guarded state
  transitions rather than forming a cryptographically append-only ledger.
- Agent libOS does not rely on MCP, GitHub, OpenAI Agents SDK, LangGraph, or
  any external framework as a trusted security boundary. Those systems are
  workload inspiration or adapter targets; the authority boundary is inside
  the libOS primitive layer.
- Agent libOS does not treat Skills, JIT tools, Runtime Modules, image
  definitions, process exec, or JSON-RPC endpoint visibility as permission
  grants. They can change model-visible affordances; resource authority still
  comes from process capabilities, primitive checks, policy, approval, and
  audit.
- Host Sink trust constrains only runtime-mediated delivery. Trusting a Shell,
  PTY, MCP stdio executable, remote endpoint, or other provider authorizes that
  delivery; it does not control the recipient's later direct I/O or forwarding.
  Trusted provider/module code and a direct RuntimeStore administrator remain
  inside the Host TCB, and local evidence is not cryptographically tamper-proof
  against that administrator.
- Data labels and Sink clearance are not at-rest encryption or automatic
  deletion. Optional [payload-retention maintenance](evidence_payload_retention.md)
  is Host-triggered and disabled by default; it monotonically reduces eligible
  terminal LLM/external-effect provider payloads while retaining causal
  identity, classification, links, and digests needed for evidence.

Research evaluation results must be published as source-bound artifacts under
the [benchmark contract](benchmark.md); this narrative intentionally contains
no copied pass counts or internal submission milestones.
