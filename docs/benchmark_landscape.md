# Benchmark Landscape for Agent libOS Evaluation

This document records the evaluation-design basis for the practical workflow
benchmark. The suite is inspired by existing agent benchmarks, but it does not
claim to run those official datasets unless explicitly stated.

## Design Principles

Agent-security evaluation should separate utility from security. AgentDojo does
this directly by measuring user-task utility and attack success separately in
tool-rich environments. InjecAgent adds indirect prompt-injection cases with
private-data exfiltration and direct user harm. ToolEmu motivates long-tail,
high-risk tool failures rather than only file-read toy tasks. CaMeL motivates
system-level control/data-flow and capability enforcement rather than relying
on in-band prompting alone.

Tool-use and workflow benchmarks also point toward stateful execution. Tau-bench
and ToolSandbox evaluate agents over mutable tool state and final environment
state, while WebArena, WorkArena, OSWorld, and CRAB-style environments push
evaluation toward realistic multi-step work rather than single-turn responses.
SWE-bench and SWE-agent motivate test-driven coding-agent utility; ScienceAgentBench
and PaperBench motivate decomposed research rubrics, executable outputs, and
cost accounting. AgentDyn and recent benchmark-critiques argue that static,
weak prompt-injection suites saturate quickly and need adaptive, helpful-looking,
third-party instructions.

## Chosen Tracks

The v2 practical benchmark therefore uses five tracks:

| Track | Inspired by | What it tests |
|---|---|---|
| Coding Agent Security Bench | SWE-bench, SWE-agent | Patch/test utility under repo-local prompt injection and tool escalation. |
| Research/RAG Agent Bench | ScienceAgentBench, PaperBench | RAG synthesis under malicious retrieved documents and private-note exfiltration attempts. |
| Stateful Enterprise Tool Bench | Tau-bench, ToolSandbox, CRAB-Bench-style stateful tools | Final mock-service state, wrong-recipient risk, cross-user data access, and approval burden. |
| DevOps/SecOps Agent Bench | WorkArena, WebArena, OSWorld, defensive Cybench-style tasks | Log and deploy workflows under untrusted plugin/API/chatops output. |
| Self-Evolution and Capability Dynamics Bench | CaMeL, AgentDyn, ComplexMCP-style tool dynamics | Skills, JIT tools, child processes, checkpoints, images, JSON-RPC/MCP, and authority laundering. |

## Reviewer-Facing Claims

The main paper claim should be:

- Agent libOS is a runtime enforcement substrate, not a better planner.
- The primary security metric is committed forbidden effects, not harmful model
  requests or benign runtime denials.
- Baselines compare deployment mechanisms: direct tool use, confirmation
  wrappers, host sandboxing, prompt-only defense, and Agent libOS ablations.
- The v2 benchmark reports utility, security, state mutation, robustness, cost,
  and explainability together.

## Sources

- AgentDojo: <https://arxiv.org/abs/2406.13352>
- InjecAgent: <https://aclanthology.org/2024.findings-acl.624/>
- ToolEmu: <https://arxiv.org/abs/2309.15817>
- CaMeL: <https://arxiv.org/abs/2503.18813>
- Tau-bench: <https://arxiv.org/abs/2406.12045>
- ToolSandbox: <https://arxiv.org/abs/2408.04682>
- CRAB-Bench: <https://arxiv.org/abs/2606.01815>
- SWE-bench: <https://arxiv.org/abs/2310.06770>
- SWE-agent: <https://arxiv.org/abs/2405.15793>
- ScienceAgentBench: <https://arxiv.org/abs/2410.05080>
- PaperBench: <https://arxiv.org/abs/2504.01848>
- WorkArena: <https://arxiv.org/abs/2403.07718>
- WebArena: <https://arxiv.org/abs/2307.13854>
- OSWorld: <https://arxiv.org/abs/2404.07972>
- Cybench: <https://arxiv.org/abs/2408.08926>
- AgentDyn: <https://arxiv.org/abs/2602.03117>
- Static benchmark critique: <https://arxiv.org/abs/2510.05244>
