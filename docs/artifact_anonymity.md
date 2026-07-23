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
- absolute local paths such as `C:\Users\...`, `/Users/...`, `/home/...`,
  `/private/...`, or `/tmp/...`,
- private API endpoints, dashboard URLs, project ids, tenant ids, and account ids,
- `.env` contents, API keys, access tokens, SSH keys, cookies, and credentials,
- LLM provider account metadata in logs, traces, screenshots, notebooks, or
  benchmark results,
- non-anonymous git remotes, branch names, tags, commit messages, and issue links,
- PDF, DOCX, PPTX, image, and archive metadata that may contain author identity.

Reviewers must not need real credentials to run the deterministic artifact
subset. Real-model experiments may be optional, but their instructions must
make the credential requirement explicit and must not embed secrets.

## Required Scan Procedure And Evidence

The scan applies to the exact commit and exact generated files that will be
shared. Scanning only the working tree is insufficient because ignored paper
builds, screenshots, binary documents, and the final archive can carry identity
that is absent from tracked source. Start from a clean anonymous worktree and
record these inventories in a private submission log:

```bash
ANON_OUTPUT_DIR=/absolute/path/to/generated-submission
test -d "$ANON_OUTPUT_DIR"
git rev-parse HEAD
git status --short
git ls-files
git ls-files -s | rg '^120000 '
find "$ANON_OUTPUT_DIR" -print | LC_ALL=C sort
find "$ANON_OUTPUT_DIR" -type l -exec ls -ld '{}' \;
```

`git status --short` must be empty unless every listed change is intentionally
included in the artifact. Set `ANON_OUTPUT_DIR` to the directory containing all
generated paper, supplement, benchmark, log, image, document, and archive
outputs. No output from the `120000` query means there are no tracked symlinks;
otherwise inspect each tracked and generated symlink target for identity,
absolute paths, and files outside the intended artifact root. Do not put the
private scan log in the shared artifact.

Before running the text scan, the submission owner must build and record an
`ANON_IDENTITY_PATTERN` containing every author name and username, email and
institutional domain, lab/institution name, personal repository/homepage, issue
tracker, cloud bucket, and deployment identifier known to the authors. The
example values below are placeholders and must be replaced:

```bash
ANON_IDENTITY_PATTERN='author-one|author@example\.edu|lab-name|institution-domain\.edu'
git grep -nEI "$ANON_IDENTITY_PATTERN"
rg -ni --hidden --no-ignore -e "$ANON_IDENTITY_PATTERN" "$ANON_OUTPUT_DIR"

git grep -nEI '(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)'
rg -ni --hidden --no-ignore -e '(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)' "$ANON_OUTPUT_DIR"

git grep -nEI '(api[_-]?key|access[_-]?token|client[_-]?secret|password|authorization)[[:space:]]*[:=]'
rg -n --hidden --no-ignore -e '(?i)(api[_-]?key|access[_-]?token|client[_-]?secret|password|authorization)[[:space:]]*[:=]' "$ANON_OUTPUT_DIR"
git grep -nE -- '-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----'
rg -n --hidden --no-ignore -e '-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----' "$ANON_OUTPUT_DIR"

git ls-files | rg '(^|/)\.env($|\.)'
find "$ANON_OUTPUT_DIR" -type f -name '.env*' -print
```

`git grep` exit status 1 and `rg` exit status 1 mean no matches. These are
candidate scans, not automatic proofs: test fixtures and documentation can
produce intentional matches, while account ids or project names may not match a
generic credential pattern. Record every hit, the reviewer, and one of
`removed`, `replaced`, or `intentional non-identifying fixture`, then rerun the
same command after remediation. An organization-approved secret scanner may be
added, but its name, version, configuration, exact command, and findings must be
recorded; it does not replace the identity pattern or manual review.

Inspect repository identity and history separately:

```bash
git config --get-regexp '^(user|remote)\.'
git remote -v
git branch --all --no-color
git tag --list
git log --all --format='%H%x09%an%x09%ae%x09%cn%x09%ce%x09%s'
```

If the reviewers receive Git metadata, names, emails, remotes, branch/tag names,
commit messages, and issue links in all reachable history must be anonymous. If
they receive an archive without Git metadata, inspect its member list and prove
that `.git/`, repository credentials, and local Git configuration are absent;
an anonymous current checkout does not sanitize embedded history by itself.

Binary formats require format-aware inspection. First inventory every relevant
file, then record the tool versions and output for each file:

```bash
find "$ANON_OUTPUT_DIR" -type f \( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.pptx' -o -iname '*.xlsx' -o -iname '*.ipynb' -o -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.gif' -o -iname '*.tiff' -o -iname '*.webp' -o -iname '*.svg' -o -iname '*.zip' -o -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' \) -print
exiftool -ver
pdfinfo -v
```

For every listed file (replace angle-bracket placeholders below with real paths,
without the brackets):

- run `exiftool -a -G1 -s <file>` and remove or replace author, creator,
  company, host, software-user, GPS, source-path, comment, and custom fields;
- run `pdfinfo <file.pdf>` and `pdftotext <file.pdf> -` for each PDF, apply the
  identity/path scan to the extracted text, render every page, and visually
  inspect title pages, acknowledgements, headers, footers, annotations, and
  links;
- inspect `docProps/core.xml`, `docProps/app.xml`, comments, revisions, notes,
  hidden slides/sheets, custom properties, and relationships in each OOXML
  document (for example, `unzip -p <file.docx> docProps/core.xml`), then render
  and visually inspect the document;
- inspect image metadata and visually inspect every plot and screenshot for
  account names, avatars, browser tabs, terminal prompts, absolute paths,
  dashboard ids, and hidden/cropped identity;
- render every notebook and inspect its kernel metadata, cell outputs, embedded
  images, widget state, execution errors, and stored paths; and
- inspect archive comments and the complete member list with `zipinfo -v`,
  `unzip -l`, or `tar -tvf` as appropriate. Check member paths, ownership names,
  symlink targets, `.git`, `.env`, logs, and secret-like filenames. Extract only
  with a path-traversal-safe tool into a fresh directory, then repeat the text
  and metadata scans recursively over the extracted contents. For the project
  wheel and sdist, also run
  `uv run python scripts/check_release_artifacts.py <artifact-directory>`.

Finally, hash the exact files that passed review and record those hashes beside
the commit, scan patterns, tool versions, commands, outputs, hit dispositions,
manual reviewer, and review date. If an artifact is regenerated or repackaged,
its hash changes and the binary/archive inspection must be repeated.

```bash
find "$ANON_OUTPUT_DIR" -type f -exec shasum -a 256 '{}' \; | LC_ALL=C sort
```

On systems without `shasum`, use `sha256sum` and record that substitution.

## Runtime Artifact Commands

These commands are the M0 baseline checks for the code artifact:

```bash
uv sync --frozen --all-groups
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
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
- `docs/release_status.md` describes only the current version's readiness,
  validation outcomes, and remaining environment boundaries;
  `docs/prelaunch_hardening_report.md` is historical evidence only.
- `agent_libos_design_doc.md` is a historical design archive.
- `docs/invariants.md` is the invariant-to-test map.
- `docs/paper_thesis.md` carries the fixed paper title, thesis, contributions,
  and non-goals.
- `docs/architecture.md`, `docs/threat_model.md`, `docs/runtime_model.md`,
  `docs/python_api.md`, `docs/capabilities.md`,
  `docs/task_authority_manifest.md`, `docs/data_flow.md`,
  `docs/object_memory.md`, `docs/tools_and_jit.md`, `docs/skills.md`,
  `docs/checkpoints.md`, `docs/git.md`, `docs/jsonrpc.md`, `docs/mcp.md`,
  `docs/modules.md`,
  `docs/storage.md`, `docs/evidence_payload_retention.md`,
  `docs/protected_operation_sdk.md`, `docs/explainable_operations.md`,
  `docs/gui.md`, `docs/cli.md`, `docs/configuration.md`, `docs/providers.md`,
  `docs/support_matrix.md`, `docs/development.md`, and `docs/benchmark.md` are
  the core implementation guides.
- `docs/gui_api_schema.json` is the versioned machine-readable subset for GUI
  snapshots, errors, and confirmed high-risk mutations; it is not a complete
  public OpenAPI contract.
- `docs/mini_swe_agent_image.md` documents the package-only mini-swe-agent
  compatibility image.
- `benchmarks/runtime_safety/schema.md` defines benchmark task shape for the M1
  runtime-safety harness.
- Documentation must not describe Python JIT, direct external framework
  adapters, real GitHub providers, MCP Resources/Prompts, or unsupported
  rollback semantics as current behavior.

## Submission Exit Gate

M0 is complete when:

- the license metadata is internally consistent,
- the CI workflow runs static compilation on Python 3.11 and deterministic
  Python/security lanes on the endpoint versions listed in
  `docs/support_matrix.md`; Python 3.12/3.13 remain declared but are not claimed
  as per-change CI jobs,
- every core invariant has test coverage or an explicit gap,
- benchmark task/output schema v1 existed at the M0 milestone; the current
  contract keeps task schema v1 and uses run-output schema v2,
- benchmark harness documentation exists,
- a one-page paper thesis with the fixed Agent libOS title exists,
- this anonymity checklist exists and is linked from README,
- the tracked-file and generated-output inventories cover the exact commit and
  exact artifacts to be shared,
- the recorded identity, absolute-path, credential, and secret scans have been
  run against both inventories and every hit has a reviewed disposition,
- Git remotes, refs, and reachable history are anonymous when Git metadata is
  shipped, or the final archive member list proves Git metadata is absent,
- every Office, PDF, image, and archive file has completed the format-aware
  metadata/content inspection above with no unresolved identity or secret,
- the final archive has been safely extracted and rescanned recursively, and
- a second human has reviewed the scan record and the recorded SHA-256 hashes
  match the exact files being submitted.

The checklist's existence is not evidence that these checks ran. Any unresolved
hit, uninspected generated file, changed post-scan hash, or missing human review
keeps the M0 anonymity gate open.
