# Maintainer Release Runbook

This runbook covers the current core Python release-candidate process. The
checked-in GitHub Actions workflows validate and preserve artifacts but have
read-only repository permission and no PyPI, tag, or GitHub-release authority.
Every tag, upload, yank, or other external mutation below requires a separate,
explicit decision by a human repository/package owner.

Public publication is blocked while [the security policy](../SECURITY.md)
reports that no confidential vulnerability intake is enabled. Before a release
preview can be authorized, a repository owner must enable GitHub private
vulnerability reporting, verify that the repository Security page presents a
working private-report form to a signed-in reporter, and update `SECURITY.md` to
state the verified channel. Changing the repository setting is an external
mutation and is not performed by CI or by this runbook.

Run the command blocks in order in the same trusted Bash session. Each block
reasserts strict fail-fast mode: any failed command, unset variable, or failed
pipeline stops that block and invalidates the release attempt. Do not continue
with a later block after a failure; correct the cause and restart the affected
release gate from its documented beginning.

## 1. Select and align the version

Choose an unreused final-form numeric `X.Y.Z` version. The current
workflow uses the literal version in wheel/source-archive filenames and in the
private GUI package's SemVer field, so pre-release, development, local, epoch,
and other PEP 440 spellings that a build backend can normalize are not
supported by this runbook. The repository may describe an unpublished
`X.Y.Z` build as a release candidate without changing that artifact
version. Never reuse a version already uploaded to a package index.

Align all current version-bearing surfaces:

- `pyproject.toml` and `agent_libos/__init__.py`;
- the root `uv.lock` editable project entry;
- `gui/package.json`, `gui/package-lock.json`, and its root package entry;
- `experiments/agentdojo/uv.lock`'s editable Agent libOS metadata;
- release artifact names in `.github/workflows/test.yml`;
- current-version assertions in release-contract tests;
- [release status](release_status.md), relevant versioned compatibility
  documentation, and the [changelog](../CHANGELOG.md); and
- literal artifact filenames in maintained build/install examples.

Do not mechanically replace store, event, GUI-schema, MCP-protocol, artifact,
or report-schema versions: those identifiers evolve independently.

Regenerate and verify locks with the repository's pinned toolchains:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

uv lock
uv lock --check
(
  cd experiments/agentdojo
  uv lock
  uv lock --check
)
npm --prefix gui install --package-lock-only
```

Review every resulting diff. Search for both the old and new product version;
an unexplained old occurrence or unexpected protocol-version edit blocks the
release.

The local release checker proves only that checked-in version identifiers are
aligned and use the final ASCII `X.Y.Z` spelling without leading zeros. It does
not contact a package index or inspect remote Git references. For this release
line it additionally pins the exact target `1.5.0`; a future release must
intentionally update that target and its regression contract rather than
passing merely because stale identifiers agree with one another. Immediately
before publication authorization, the human owner must re-check the complete
PyPI project history and the intended remote's tags to confirm that the exact
version remains unused.

## 2. Run local deterministic and artifact checks

Start from a clean candidate branch. Install locked dependencies and run the
maintained checks:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

uv sync --frozen
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/check_architecture.py
uv run python scripts/check_protected_operations.py
uv run python scripts/check_test_invariants.py
uv run python scripts/test_matrix.py --lane all
npm --prefix gui ci
uv run python scripts/test_matrix.py --lane gui
git diff --check
```

Run optional PostgreSQL, MCP, real-Deno, real-LLM, browser, and native-platform
gates exactly as documented when the release claim includes those cells. Skips
are not positive evidence for an environment gate.

Build the local pair only as a packaging preflight. Derive filenames from the
project version rather than copying a stale example:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

RELEASE_VERSION="$(.venv/bin/python - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version'])
PY
)"
LOCAL_RELEASE_DIR="$(mktemp -d)"
LOCAL_RELEASE_WHEEL="$LOCAL_RELEASE_DIR/agent_libos-${RELEASE_VERSION}-py3-none-any.whl"
LOCAL_RELEASE_SDIST="$LOCAL_RELEASE_DIR/agent_libos-${RELEASE_VERSION}.tar.gz"

uv sync --frozen --no-dev --group release
uv build --no-build-isolation --out-dir "$LOCAL_RELEASE_DIR" \
  --python .venv/bin/python --no-create-gitignore
.venv/bin/python scripts/check_release_artifacts.py "$LOCAL_RELEASE_DIR" \
  --write-checksums
uv run --frozen --no-dev --group release twine check \
  "$LOCAL_RELEASE_WHEEL" "$LOCAL_RELEASE_SDIST"
uv run --frozen --no-dev --group release check-wheel-contents \
  "$LOCAL_RELEASE_WHEEL"
.venv/bin/python scripts/check_release_artifacts.py "$LOCAL_RELEASE_DIR" \
  --verify-checksums
```

This local build is not the artifact to publish. It proves that the candidate
can build; the canonical pair is built once by CI after all upstream gates.
Keep the fresh local directory only as long as needed to review the preflight
artifacts. This procedure never clears or writes the repository's existing
`dist/` directory.

## 3. Obtain the bound CI receipt

Merge or otherwise select the exact release commit through the normal reviewed
repository process. The checked-in `tests` workflow must complete for that
commit. Its release build depends on the static, AgentDojo, Python, security,
host-filesystem-identity, MCP SDK, deterministic-release, PostgreSQL, GUI, and
Windows jobs. The downstream artifact-smoke matrix must then pass on Python
3.11, 3.12, 3.13, and 3.14.

Record all of these immutable locators:

- exact source commit;
- workflow run and required job URLs;
- canonical wheel, source archive, and `SHA256SUMS` artifact locator; and
- any separately required native-provider or live-evaluation receipt.

Download the canonical CI artifact before its configured 14-day retention
expires. Set `RELEASE_DOWNLOAD_DIR` to the absolute path of a fresh directory
containing only the three downloaded files. The following validation rejects a
relative path, symlinked entry, extra entry, or filename mismatch and then
binds every later upload to that resolved directory:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

RELEASE_DOWNLOAD_DIR=/absolute/path/to/download
case "$RELEASE_DOWNLOAD_DIR" in
  /*) ;;
  *) printf 'RELEASE_DOWNLOAD_DIR must be absolute\n' >&2; exit 1 ;;
esac

RELEASE_DOWNLOAD_DIR="$(.venv/bin/python - "$RELEASE_DOWNLOAD_DIR" "$RELEASE_VERSION" <<'PY'
from pathlib import Path
import sys

requested = Path(sys.argv[1])
version = sys.argv[2]
root = requested.resolve(strict=True)
if not root.is_dir():
    raise SystemExit("release download path is not a directory")
expected = {
    f"agent_libos-{version}-py3-none-any.whl",
    f"agent_libos-{version}.tar.gz",
    "SHA256SUMS",
}
entries = list(root.iterdir())
actual = {entry.name for entry in entries}
if actual != expected or len(entries) != len(expected):
    raise SystemExit(
        f"canonical download entries differ: expected={sorted(expected)!r} "
        f"actual={sorted(actual)!r}"
    )
for entry in entries:
    if entry.is_symlink() or not entry.is_file():
        raise SystemExit(f"canonical download entry is not a regular file: {entry}")
    if entry.resolve(strict=True).parent != root:
        raise SystemExit(f"canonical download entry escapes directory: {entry}")
print(root)
PY
)"
RELEASE_WHEEL="$RELEASE_DOWNLOAD_DIR/agent_libos-${RELEASE_VERSION}-py3-none-any.whl"
RELEASE_SDIST="$RELEASE_DOWNLOAD_DIR/agent_libos-${RELEASE_VERSION}.tar.gz"
RELEASE_CHECKSUMS="$RELEASE_DOWNLOAD_DIR/SHA256SUMS"

.venv/bin/python scripts/check_release_artifacts.py "$RELEASE_DOWNLOAD_DIR" \
  --verify-checksums
(
  cd "$RELEASE_DOWNLOAD_DIR"
  sha256sum --check --strict SHA256SUMS
)
```

On a platform without `sha256sum`, use an equivalent SHA-256 verifier and record
the substitution. Do not rebuild or modify the canonical files after this
point. Preserve the checksum manifest and receipt locators in the private
release record.

The opt-in twelve-run Durable Task Run live gate, native desktop packaging, and
other cells marked as environment gates are not made true by deterministic CI.
If a release claim includes one, obtain and bind its documented clean-source
receipt before authorization.

## 4. Explicit publication authorization

Before any external mutation, a human repository/package owner reviews one
preview containing:

- package name and target index;
- version, source commit, and proposed tag;
- workflow/job locators;
- exact artifact filenames and SHA-256 values;
- the verified confidential vulnerability-reporting channel;
- environment-gate receipts included in the release claim; and
- the [changelog](../CHANGELOG.md) plus known limitations from the
  [release status](release_status.md) and [support matrix](support_matrix.md).

The owner then explicitly authorizes or rejects that exact preview. Silence,
prior releases, a model decision, CI success, or access to a credential is not
authorization. Credentials stay in the trusted maintainer environment and must
not be written to the repository, command log, benchmark output, or issue.

## 5. Tag and publish outside CI

Tags and uploads are manual, separately authorized operations. Create the tag
only at the recorded commit, verify it, then push that exact tag:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

: "${RELEASE_VERSION:?set the exact authorized version}"
: "${RELEASE_COMMIT:?set the exact authorized commit}"
RELEASE_COMMIT="$(git rev-parse --verify "${RELEASE_COMMIT}^{commit}")"
git tag -a "v${RELEASE_VERSION}" "$RELEASE_COMMIT" \
  -m "Agent libOS ${RELEASE_VERSION}"
git show --no-patch --decorate "v${RELEASE_VERSION}"
TAG_COMMIT="$(git rev-list -n 1 "v${RELEASE_VERSION}")"
test "$TAG_COMMIT" = "$RELEASE_COMMIT"
git push origin "v${RELEASE_VERSION}"
```

For a first publication or packaging change, the owner may upload the same
canonical files to TestPyPI and perform a clean `--no-deps` import/entrypoint
smoke. TestPyPI is an external mutation and needs the same explicit preview;
never rebuild between TestPyPI and production.

```bash
set -Eeuo pipefail
IFS=$'\n\t'

uv run --frozen --no-dev --group release twine upload \
  --repository testpypi \
  "$RELEASE_WHEEL" "$RELEASE_SDIST"
```

Publish only the validated wheel and source archive from the downloaded
canonical directory. For a maintainer environment configured with the intended
index and token, the release-group tool is:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

uv run --frozen --no-dev --group release twine upload \
  --repository pypi \
  "$RELEASE_WHEEL" "$RELEASE_SDIST"
```

Preview the resolved repository configuration without printing credentials.
Immediately read back the index project/version page and install the exact
version into a fresh environment. Verify package metadata, import, all three
console entrypoints, and the deterministic in-memory demo. Record URLs and
results; do not record tokens.

Creating a GitHub release or attaching `SHA256SUMS` is another optional external
mutation, not an existing CI step. If done, it must reference the same tag,
commit, and digests and receive a separate explicit preview/authorization.

## 6. Failure, yank, and recovery

Package-index files are immutable. Never overwrite or reuse a published
version, and never move a tag associated with a published release.

If validation fails before upload, stop, leave the candidate unpublished, fix
forward on a new commit, and rerun the entire bound CI/artifact process. An
unpublished erroneous remote tag may be removed only after a human owner
reviews that exact destructive action; record the deletion and use a fresh tag
after validation.

If a bad version was uploaded:

1. stop further promotion and preserve the release receipt and incident facts;
2. use the package index's owner interface to yank the exact version with a
   concise reason; yanking is a human-authorized external mutation;
3. do not delete/re-upload files or reuse the version;
4. publish a fixed, incremented version through this runbook; and
5. for a vulnerability, follow the [security policy](../SECURITY.md),
   coordinate an advisory and disclosure, and rotate any exposed credential
   outside the repository.

A yank reduces accidental selection; it does not remove existing downloads.
Document migration or rollback guidance that respects the store and artifact
compatibility boundaries. Runtime checkpoint restore is not a package rollback
and does not reverse external provider state.

## 7. Close the release record

The release record is complete only when it binds the authorization decision,
source/tag, CI jobs, canonical artifact digests, publication/readback results,
environment gates, changelog, and any incident/yank action. Update the changelog
for the next development cycle without rewriting the facts of the released
entry.
