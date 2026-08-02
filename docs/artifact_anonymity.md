# Artifact Publication and Anonymity Checklist

This contributor-only checklist covers identity and secret scanning for a
source archive, benchmark bundle, or anonymous research artifact. It is not a
legal review, an end-user setup guide, or evidence that any particular release
artifact has passed these checks.

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
ANON_COMMIT=$(git rev-parse --verify 'HEAD^{commit}') || exit 1
ANON_STATUS=$(git status --porcelain=v1 --untracked-files=all) || exit 1
test -z "$ANON_STATUS"
printf '%s\n' "$ANON_COMMIT"
printf '%s' "$ANON_STATUS"
git ls-files
git ls-files -s | rg '^(120000|160000) '
git submodule status --recursive
git submodule foreach --recursive \
  'git rev-parse HEAD; git status --porcelain=v1 --untracked-files=all'
git ls-files -z | git check-attr --cached --stdin -z -a
find "$ANON_OUTPUT_DIR" -print | LC_ALL=C sort
find "$ANON_OUTPUT_DIR" -type l -exec ls -ld '{}' \;
```

The status capture must succeed and its `test` must report a clean index and
working tree, including ordinary untracked files. If the intended submission
includes a working-tree change, first commit it on the anonymous branch and
restart the scan against that exact commit; an untracked or merely staged file
is otherwise not covered by history-based scans. Ignored files are not part of
that commit: they are covered only if they appear under the exact
`ANON_OUTPUT_DIR` that will be shared. Do not point that variable at a broader
staging parent or a narrower subset. Prefer an output directory outside the
worktree; if it is inside an ignored path, the explicit output inventory and
recursive scans remain mandatory.

Mode `120000` identifies tracked symlinks and mode `160000` identifies
gitlinks/submodules. Inspect every tracked/generated symlink target and every
submodule commit independently. Reject absolute symlink targets and targets
that resolve outside the intended artifact root unless they are an explicitly
documented part of the submission and their target bytes are separately
inventoried and scanned. Any leading `-` in `git submodule status` means that
content is not initialized and therefore has not been reviewed; initialize it
from a trusted source or exclude the gitlink from the artifact. Neither the
superproject text scan nor an archive of the superproject proves a submodule's
contents anonymous.

The NUL-delimited attribute inventory identifies content filters without
misparsing unusual filenames. If it reports Git LFS, run the additional checks
below with a recorded Git LFS version; apply an equivalent object inventory for
any other filter:

```bash
git lfs version
git config --show-origin --get-regexp '^lfs\.(fetchinclude|fetchexclude)$'
git lfs ls-files --all --long
git lfs fsck --objects
rg -l --hidden --no-ignore -F \
  'version https://git-lfs.github.com/spec/v1' "$ANON_OUTPUT_DIR"
```

Remove fetch exclusions in the disposable review clone before treating the LFS
checks as complete. `git lfs fsck --objects` verifies the current HEAD/index
objects, may move a corrupt local object into `.git/lfs/bad`, and does not verify
the historical inventory produced by `git lfs ls-files --all --long`; run it in
that disposable clone. Retrieve, hash, and review every distinct LFS OID
referenced by the shipped history separately, and review every pointer reported
in the exact output. A pointer is acceptable only when the submission
intentionally requires a later fetch, that fact is disclosed, and the fetched
object bytes were independently scanned; otherwise export and scan the reviewed
object bytes. Do not put the private scan log in the shared artifact.

Before running the text scan, the submission owner must build and record an
`ANON_IDENTITY_PATTERN` containing every author name and username, email and
institutional domain, lab/institution name, personal repository/homepage, issue
tracker, cloud bucket, and deployment identifier known to the authors. The
example values below are placeholders and must be replaced:

```bash
ANON_IDENTITY_PATTERN='author-one|author@example\.edu|lab-name|institution-domain\.edu'
ANON_PATH_PATTERN='(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)'
ANON_SECRET_PATTERN='(api[_-]?key|access[_-]?token|client[_-]?secret|password|authorization)[[:space:]]*[:=]'
ANON_PRIVATE_KEY_PATTERN='-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----'

git grep -lEI -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN"
rg -li --hidden --no-ignore \
  -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" "$ANON_OUTPUT_DIR"
git ls-files | rg -i -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN"
(cd "$ANON_OUTPUT_DIR" && find . -print) | \
  rg -i -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN"

git grep -lEI -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN"
rg -li --hidden --no-ignore \
  -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" "$ANON_OUTPUT_DIR"

git ls-files | rg '(^|/)\.env($|\.)'
find "$ANON_OUTPUT_DIR" -type f -name '.env*' -print
```

`git grep` exit status 1 and `rg` exit status 1 mean no matches. These are
candidate scans, not automatic proofs: test fixtures and documentation can
produce intentional matches, while account ids or project names may not match a
generic credential pattern. The commands intentionally print only filenames:
inspect each candidate in a restricted local view, and record only a redacted
location, detector/category, and disposition. Never copy a live secret or an
unredacted matching line into the scan log. Add an organization-approved,
redacting secret/entropy scanner when available and cover provider-specific key
formats, bearer/JWT material, cloud credentials, cookies, connection strings,
and encoded secrets; the generic grep is only a first-pass locator. Record every
hit, the reviewer, and one of
`removed`, `replaced`, or `intentional non-identifying fixture`, then rerun the
same command after remediation. Record the scanner's name, version,
configuration, exact redacted command, and findings; it does not replace the
identity pattern or manual review.

Inspect repository identity and history separately:

```bash
set -o pipefail
ANON_GIT_SCAN_DIR="$(mktemp -d)" || exit 1
test -n "$ANON_GIT_SCAN_DIR" || exit 1
trap 'test -n "$ANON_GIT_SCAN_DIR" && rm -rf -- "$ANON_GIT_SCAN_DIR"' EXIT

git config --get-regexp '^(user|remote)\.'
git remote -v
git branch --all --no-color
git tag --list
git for-each-ref \
  --format='%(refname)%09%(objectname)%09%(objecttype)%09%(authorname)%09%(authoremail)%09%(committername)%09%(committeremail)%09%(taggername)%09%(taggeremail)' \
  refs/heads refs/remotes refs/tags
git log --all --format='%H%x09%an%x09%ae%x09%cn%x09%ce'
git --no-replace-objects rev-list --objects --all \
  > "$ANON_GIT_SCAN_DIR/reachable-objects" || exit 1
if rg -i -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" \
  "$ANON_GIT_SCAN_DIR/reachable-objects"; then
  : # review the matching object paths in the restricted local session
elif test "$?" -ne 1; then
  exit 1
fi

git --no-replace-objects cat-file \
  --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' \
  < "$ANON_GIT_SCAN_DIR/reachable-objects" \
  > "$ANON_GIT_SCAN_DIR/reachable-metadata" || exit 1
while read -r object_oid object_type object_size object_path; do
  case "$object_type" in
    commit|tag|tree|blob)
      if ! git --no-replace-objects cat-file "$object_type" "$object_oid" \
        > "$ANON_GIT_SCAN_DIR/object-bytes"; then
        printf 'git cat-file failed for %s\n' "$object_oid" >&2
        exit 1
      fi
      if rg -a -qi \
          -e "$ANON_IDENTITY_PATTERN" \
          -e "$ANON_PATH_PATTERN" \
          -e "$ANON_SECRET_PATTERN" \
          -e "$ANON_PRIVATE_KEY_PATTERN" \
          "$ANON_GIT_SCAN_DIR/object-bytes"; then
        printf '%s\t%s\t%s\t%s\n' \
          "$object_oid" "$object_type" "$object_size" "$object_path"
      elif test "$?" -ne 1; then
        exit 1
      fi
      ;;
  esac
done < "$ANON_GIT_SCAN_DIR/reachable-metadata"
```

If the reviewers receive Git metadata, names, emails, remotes, branch/tag names,
annotated-tag contents, full commit bodies and signatures, issue links, paths,
and file contents in all reachable history must be anonymous. The object loop
reads the complete raw bytes of every reachable commit, annotated tag, and
distinct tree and blob while printing only object identity/type/size/path for a
pattern hit. Raw tree bytes are included because filenames can themselves carry
identity even when unusual names make a line-oriented path inventory awkward.
Redact the path too when it is identifying. Run the approved secret scanner over
Git history as well; the four regular expressions are not sufficient secret
detection.

Pattern matching is not a substitute for semantic review. View full commit
bodies with `git --no-replace-objects log --all --format=fuller --show-signature`
and each annotated tag with
`git --no-replace-objects cat-file tag <tag-object-id>` in a restricted local
session. Disabling replace-object substitution is essential: a copied
`refs/replace/*` entry must not hide the original object's bytes from review. Do
the same for the value-bearing Git config and remote commands above because a
remote URL can embed a credential. Do not capture any of their raw output in the
submission log; record only redacted findings and dispositions.

Prefer an export that excludes `.git`; a clean current tree is much easier to
audit than a working repository. If a local `.git` directory itself will be
copied, scanning refs and reflogs is not sufficient because loose or packed
dangling objects are copied too. Inventory **every object in the object
database** into a private temporary file and feed it through the same raw-byte
loop (the output has no path for an unreachable object). Run this continuation
in the same shell before the earlier `EXIT` trap removes the private directory;
its first guard aborts rather than redirecting through an unset path:

```bash
test -n "${ANON_GIT_SCAN_DIR:-}" && test -d "$ANON_GIT_SCAN_DIR" || exit 1
git --no-replace-objects cat-file --batch-all-objects \
  --batch-check='%(objectname) %(objecttype) %(objectsize)' \
  > "$ANON_GIT_SCAN_DIR/all-object-metadata" || exit 1
while read -r object_oid object_type object_size; do
  case "$object_type" in
    commit|tag|tree|blob)
      if ! git --no-replace-objects cat-file "$object_type" "$object_oid" \
        > "$ANON_GIT_SCAN_DIR/object-bytes"; then
        printf 'git cat-file failed for %s\n' "$object_oid" >&2
        exit 1
      fi
      if rg -a -qi \
          -e "$ANON_IDENTITY_PATTERN" \
          -e "$ANON_PATH_PATTERN" \
          -e "$ANON_SECRET_PATTERN" \
          -e "$ANON_PRIVATE_KEY_PATTERN" \
          "$ANON_GIT_SCAN_DIR/object-bytes"; then
        printf '%s\t%s\t%s\n' "$object_oid" "$object_type" "$object_size"
      elif test "$?" -ne 1; then
        exit 1
      fi
      ;;
  esac
done < "$ANON_GIT_SCAN_DIR/all-object-metadata"
```

Every producer above writes a file and must exit successfully before its
consumer runs. The temporary directory must also be created successfully and
validated as non-empty before the cleanup trap is installed. Each object read
is checked independently; a failed `git cat-file` or `rg` exits the checklist
instead of looking like a no-match result. `set -o pipefail` remains mandatory
for any locally added pipelines.
Scan `.git` metadata files without logging raw credentials. Historical binary
blobs need the same format-aware inspection as current binaries or must be
removed by an audited history rewrite. After a rewrite, regenerate or clone the
deliverable and rerun the all-object scan; pruning alone is not evidence that a
copied object database is anonymous.

`git log` alone does not inspect blobs, and inspecting only the current tree
cannot prove reachable history anonymous. If reviewers receive an archive
without Git metadata, inspect its member list and prove that `.git/`, repository
credentials, and local Git configuration are absent; an anonymous current
checkout does not sanitize an archive that embeds history by itself.

Binary formats require format-aware inspection. Raw `git grep`, `rg -a`, and
blob scans do not parse compressed OOXML members, PDF metadata, nested archives,
or rendered content. Materialize the regular-file blobs from the exact commit
into the private review directory before running the format-aware pass; do not
substitute the current working tree or treat `git archive` as an identity-
preserving materialization. Archive generation obeys committed
`.gitattributes`, so `export-ignore` can omit tracked paths and `export-subst`
can rewrite bytes.

```bash
test -n "${ANON_GIT_SCAN_DIR:-}" && test -d "$ANON_GIT_SCAN_DIR" || exit 1
ANON_COMMIT_TREE="$ANON_GIT_SCAN_DIR/exact-commit-tree"
test ! -e "$ANON_COMMIT_TREE" || exit 1
mkdir "$ANON_COMMIT_TREE" || exit 1
python3 - "$ANON_COMMIT" "$ANON_COMMIT_TREE" <<'PY' || exit 1
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

commit, destination = sys.argv[1:]
root = Path(destination).resolve(strict=True)
listing = subprocess.run(
    ["git", "--no-replace-objects", "ls-tree", "-rz", "--full-tree", commit],
    check=True,
    stdout=subprocess.PIPE,
).stdout
for record in listing.split(b"\0"):
    if not record:
        continue
    header, raw_path = record.split(b"\t", 1)
    mode, object_type, oid = header.decode("ascii").split()
    if object_type != "blob" or mode == "120000":
        # Symlink target blobs and gitlinks are reviewed separately by the
        # mandatory inventory above; never create a live symlink here.
        continue
    if mode not in {"100644", "100755"}:
        raise SystemExit(f"unsupported tree mode {mode}")
    path_text = os.fsdecode(raw_path)
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise SystemExit("unsafe tree path")
    target = root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "blob", oid],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with target.open("xb") as stream:
        stream.write(payload)
PY

# Separately inspect the projection that git archive would deliver. This view
# may intentionally differ from the raw tree because of committed attributes.
ANON_COMMIT_EXPORT="$ANON_GIT_SCAN_DIR/archive-projection"
test ! -e "$ANON_COMMIT_EXPORT" || exit 1
mkdir "$ANON_COMMIT_EXPORT" || exit 1
git --no-replace-objects archive --format=tar "$ANON_COMMIT" \
  > "$ANON_GIT_SCAN_DIR/archive-projection.tar" || exit 1
tar -tf "$ANON_GIT_SCAN_DIR/archive-projection.tar" \
  > "$ANON_GIT_SCAN_DIR/archive-projection-members" || exit 1
tar -xf "$ANON_GIT_SCAN_DIR/archive-projection.tar" \
  -C "$ANON_COMMIT_EXPORT" || exit 1
```

The blob materializer deliberately skips live symlink creation; review every
mode `120000` target from the earlier inventory and scan the referenced bytes
under the stated symlink policy. Review gitlinks recursively. Keep the tar
member inventory private, inspect unusual names and symlink entries, and
confirm extraction stayed inside the fresh empty directory. Compare the
archive projection with the cached attribute inventory and disposition every
path affected by `export-ignore` or `export-subst`. Both views are tied to
`ANON_COMMIT`; regenerating either from another revision requires restarting
the review.

Apply every MIME, metadata, extraction, rendering, and manual visual check below
to the raw `$ANON_COMMIT_TREE`, the `$ANON_COMMIT_EXPORT` archive projection,
and `$ANON_OUTPUT_DIR`. First inventory every relevant file, then record the
tool versions and output for each file:

```bash
file --version
for ANON_BINARY_ROOT in \
  "$ANON_COMMIT_TREE" "$ANON_COMMIT_EXPORT" "$ANON_OUTPUT_DIR"; do
  test -d "$ANON_BINARY_ROOT" || exit 1
  find "$ANON_BINARY_ROOT" -type f -exec file --mime-type -- '{}' \; |
    LC_ALL=C sort || exit 1
  find "$ANON_BINARY_ROOT" -type f \( \
    -iname '*.pdf' -o -iname '*.doc' -o -iname '*.docx' -o \
    -iname '*.ppt' -o -iname '*.pptx' -o -iname '*.xls' -o \
    -iname '*.xlsx' -o -iname '*.odt' -o -iname '*.ods' -o \
    -iname '*.ipynb' -o -iname '*.png' -o -iname '*.jpg' -o \
    -iname '*.jpeg' -o -iname '*.gif' -o -iname '*.tiff' -o \
    -iname '*.webp' -o -iname '*.svg' -o -iname '*.heic' -o \
    -iname '*.mp4' -o -iname '*.mov' -o -iname '*.webm' -o \
    -iname '*.whl' -o -iname '*.zip' -o -iname '*.7z' -o \
    -iname '*.tar' -o -iname '*.tar.gz' -o -iname '*.tgz' -o \
    -iname '*.gz' -o -iname '*.bz2' -o -iname '*.xz' -o \
    -iname '*.parquet' -o -iname '*.pkl' -o -iname '*.pickle' -o \
    -iname '*.npy' -o -iname '*.npz' -o -iname '*.onnx' -o \
    -iname '*.bin' -o -iname '*.exe' -o -iname '*.dll' -o \
    -iname '*.so' -o -iname '*.dylib' \
  \) -print || exit 1
done
exiftool -ver
pdfinfo -v
```

The MIME inventory, not the extension list, is authoritative: classify every
file, including extensionless files, executables, fonts, media, databases, model
weights, and unfamiliar data/container formats. Record the `file` version (or
the equivalent classifier and version on the review platform), then select a
format-aware inspection tool for every non-plain-text result. For every listed
file (replace angle-bracket placeholders below with real paths, without the
brackets):

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
  and metadata scans recursively over the extracted contents. Treat wheels as
  ZIP archives and inspect their `METADATA`, `WHEEL`, `entry_points.txt`, license
  files, and every packaged source/resource; a structural package check is not
  an anonymity scan. For the project wheel and sdist, also run
  `uv run python scripts/check_release_artifacts.py <artifact-directory>`, which
  validates release structure but does not replace the anonymity review.

Finally, hash the exact files that passed review and record those hashes beside
the commit, scan patterns, tool versions, commands, outputs, hit dispositions,
manual reviewer, and review date. If an artifact is regenerated or repackaged,
its hash changes and the binary/archive inspection must be repeated.

```bash
ANON_FINAL_COMMIT=$(git rev-parse --verify 'HEAD^{commit}') || exit 1
ANON_FINAL_STATUS=$(git status --porcelain=v1 --untracked-files=all) || exit 1
test "$ANON_FINAL_COMMIT" = "$ANON_COMMIT"
test -z "$ANON_FINAL_STATUS"
find "$ANON_OUTPUT_DIR" -type f -exec shasum -a 256 '{}' \; | LC_ALL=C sort
```

On systems without `shasum`, use `sha256sum` and record that substitution.

## Runtime Artifact Commands

These commands are baseline checks for a source artifact:

```bash
uv sync --frozen
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/test_matrix.py --lane all
uv run python scripts/check_test_invariants.py
```

For an anonymous artifact branch, add a fresh-clone dry run before submission:

```bash
uv sync --frozen
uv run python scripts/test_matrix.py --lane all
```

Deno-backed tests run by default when `deno` is installed. Tests that require a
real Deno installation skip with a clear message when `deno` is missing; use
`--skip-real-deno` only for runs that intentionally exclude them.

## Documentation Consistency

Use [README.md](../README.md) as the documentation index,
[release_status.md](release_status.md) for the release contract, and
[support_matrix.md](support_matrix.md) for environment coverage. Archived
design and prelaunch notices are not current evidence. Documentation must not
present Python JIT, direct external framework adapters, real GitHub providers,
MCP Resources/Prompts, or unsupported rollback semantics as implemented.

## Publication Exit Gate

The artifact is ready to share only when:

- the license metadata is internally consistent,
- the CI workflow runs static compilation on Python 3.11 and deterministic
  Python/security lanes on the endpoint versions listed in
  `docs/support_matrix.md`; Python 3.12/3.13 remain declared but are not claimed
  as per-change CI jobs,
- every core invariant has test coverage or an explicit gap,
- the benchmark contract uses task schema v1 and complete run-output schema v2,
- benchmark harness documentation exists,
- the raw tracked-blob tree, archive projection, and generated-output
  inventories cover the exact commit and exact artifacts to be shared,
- the recorded identity, absolute-path, credential, and secret scans have been
  run against all three inventories and every hit has a reviewed disposition,
- Git remotes, refs, and reachable history are anonymous when Git metadata is
  shipped, or the final archive member list proves Git metadata is absent,
- every Office, PDF, image, and archive file in the raw exact-commit tree,
  deliverable archive projection, and generated-output inventory has completed
  the format-aware metadata/content inspection above with no unresolved
  identity or secret,
- the final archive has been safely extracted and rescanned recursively, and
- a second human has reviewed the scan record and the recorded SHA-256 hashes
  match the exact files being submitted.

The checklist's existence is not evidence that these checks ran. Any unresolved
hit, uninspected generated file, changed post-scan hash, or missing human review
keeps the publication gate open.
