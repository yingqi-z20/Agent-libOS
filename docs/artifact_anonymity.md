# Artifact Anonymity Checklist

This checklist is the M0 artifact hygiene baseline for an anonymous systems
submission. It is not a legal review or a release checklist; it is the minimum
set of checks required before sharing a paper artifact or benchmark bundle with
reviewers.

## Paper Title And System Name

The paper title is fixed as:

> Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents

Use `Agent libOS` consistently in paper drafts and artifact documentation. Do
not use the older temporary anonymous name `Primitive Agent Runtime` (`PAR`).
For double-blind review, anonymize author, institution, repository, and
deployment metadata rather than renaming the runtime in source code.

## License Consistency

- `LICENSE` must contain Apache License 2.0.
- `pyproject.toml` must use `license = { text = "Apache-2.0" }`.
- README and artifact docs must not claim MIT, proprietary, or dual licensing.
- Generated distributions should include `LICENSE` and should not add a
  conflicting classifier.

## Double-Blind Content Scan

Before making an anonymous artifact branch or archive, scan all tracked files
and generated paper/artifact files for:

- author names, lab names, school names, school emails, and personal emails,
- personal GitHub, GitLab, homepage, cloud bucket, or institutional URLs,
- absolute local paths such as `C:\Users\...`, `/Users/...`, or `/home/...`,
- private API endpoints, dashboard URLs, project ids, tenant ids, and account ids,
- `.env` contents, API keys, access tokens, SSH keys, cookies, and credentials,
- LLM provider account metadata in logs, traces, screenshots, notebooks, or
  benchmark results,
- non-anonymous git remotes, branch names, tags, commit messages, and issue links,
- PDF, DOCX, PPTX, image, and archive metadata that may contain author identity.

Reviewers must not need real credentials to run the deterministic artifact
subset. Real-model experiments may be optional, but their instructions must
make the credential requirement explicit and must not embed secrets.

## Runtime Artifact Commands

These commands are the M0 baseline checks for the code artifact:

```bash
uv sync --frozen --all-groups
uv run python -m compileall agent_libos tests scripts experiments benchmarks
uv run python scripts/test_matrix.py --lane all
uv run python scripts/check_test_invariants.py
```

For an anonymous artifact branch, add a fresh-clone dry run before submission:

```bash
uv sync --frozen --all-groups
uv run python scripts/test_matrix.py --lane all
```

Deno-backed tests run by default when `deno` is installed. Tests that require a
real Deno installation skip with a clear message when `deno` is missing; use
`--skip-real-deno` only for runs that intentionally exclude them.

## Documentation Consistency

- README is the current project entrypoint and documentation index.
- `agent_libos_design_doc.md` is a historical design archive.
- `docs/invariants.md` is the invariant-to-test map.
- `docs/paper_thesis.md` carries the fixed paper title, thesis, contributions,
  and non-goals.
- `docs/architecture.md`, `docs/runtime_model.md`, `docs/capabilities.md`,
  `docs/object_memory.md`, `docs/tools_and_jit.md`, `docs/skills.md`,
  `docs/checkpoints.md`, `docs/jsonrpc.md`, `docs/modules.md`, `docs/gui.md`,
  `docs/cli.md`, `docs/development.md`, and `docs/benchmark.md` are the core
  implementation guides.
- `docs/mini_swe_agent_image.md` documents the package-only mini-swe-agent
  compatibility image.
- `benchmarks/runtime_safety/schema.md` defines benchmark task shape for the M1
  runtime-safety harness.
- Documentation must not describe Python JIT, direct external framework
  adapters, real GitHub/MCP providers, or unsupported rollback semantics as
  current behavior.

## Submission Exit Gate

M0 is complete when:

- the license metadata is internally consistent,
- the CI workflow runs compile and unit tests on supported Python versions,
- every core invariant has test coverage or an explicit gap,
- benchmark task schema v0 exists,
- benchmark harness documentation exists,
- a one-page paper thesis with the fixed Agent libOS title exists,
- this anonymity checklist exists and is linked from README.
