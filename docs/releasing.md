# Maintainer Release Runbook

This runbook covers the current core Python release-candidate process. The
checked-in GitHub Actions workflows validate and preserve artifacts but have
read-only repository permission and no PyPI, tag, or GitHub-release authority.
Every tag, upload, yank, or other external mutation below requires a separate,
explicit decision by a human repository/package owner.

[The security policy](../SECURITY.md) records that GitHub private vulnerability
reporting is enabled and identifies the signed-in private-report form. Before a
release preview can be authorized, a repository owner must re-verify that the
repository Security page still presents that working confidential channel and
that `SECURITY.md` still states the verified status. Public publication is
blocked if either check fails. Enabling or repairing the repository setting is
an external mutation and is not performed by CI or by this runbook.

Run the command blocks in order in the same trusted Bash session. Each block
reasserts strict fail-fast mode: any failed command, unset variable, or failed
pipeline stops that block and invalidates the release attempt. Do not continue
with a later block after a failure; correct the cause and restart the affected
release gate from its documented beginning.

## In this guide

- [Select and align the version](#1-select-and-align-the-version)
- [Run deterministic and artifact checks](#2-run-local-deterministic-and-artifact-checks)
- [Obtain the bound CI receipt](#3-obtain-the-bound-ci-receipt)
- [Record explicit publication authorization](#4-explicit-publication-authorization)
- [Tag and publish outside CI](#5-tag-and-publish-outside-ci)
- [Handle failure, yank, and recovery](#6-failure-yank-and-recovery)
- [Close the release record](#7-close-the-release-record)
- Return to the [documentation home](index.md).

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
- the modern MCP `clientInfo`, `desktop/runtime-manifest.json`, desktop checker
  constants, and every versioned name in `.github/workflows/desktop-internal.yml`;
- current-version assertions in release-contract tests;
- generated `docs/cli_reference.md`, `docs/configuration_reference.md`, and
  `README.pypi.md` (the package long description);
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
uv run --frozen python scripts/generate_cli_reference.py
uv run --frozen python scripts/generate_config_reference.py
uv run --frozen python scripts/generate_pypi_readme.py
uv run --frozen python scripts/generate_cli_reference.py --check
uv run --frozen python scripts/generate_config_reference.py --check
uv run --frozen python scripts/generate_pypi_readme.py --check
```

Review every resulting diff. Search for both the old and new product version;
an unexplained old occurrence, unexpected protocol-version edit, generated-file
diff after `--check`, `/blob/main/` link in `README.pypi.md`, or missing
`/blob/vX.Y.Z/` release-tag link blocks the release. Generated references are
derived artifacts: update their source/parser/configuration first and rerun the
generator rather than editing them directly.

The local release checker proves only that checked-in version identifiers are
aligned and use the final ASCII `X.Y.Z` spelling without leading zeros. It does
not contact a package index or inspect remote Git references. It also verifies
that every tracked ordinary source file belongs to the exact sdist
include/exclude partition; an unclassified top-level file blocks the build
until a maintainer makes an explicit inclusion decision. For this release
line it additionally pins the exact target `1.5.1`; a future release must
intentionally update that target and its regression contract rather than
passing merely because stale identifiers agree with one another. Immediately
before publication authorization, the human owner must re-check the complete
PyPI project history and the intended remote's tags to confirm that the exact
version remains unused.

## 2. Run local deterministic and artifact checks

Start from a clean candidate branch. Before entering the block, set
`RELEASE_BASE_SHA` to the exact reviewed base commit used for the candidate
diff, and set `RELEASE_DEFAULT_BRANCH` to the intended remote default-branch
name. The base must be an available ancestor distinct from the release
candidate. The full base object id and branch name belong in the release receipt;
checking only an already-clean working tree would miss whitespace errors in
committed candidate lines. Install locked dependencies and run the maintained
checks:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

: "${RELEASE_BASE_SHA:?set the exact reviewed base commit SHA}"
: "${RELEASE_DEFAULT_BRANCH:?set the intended default-branch name}"
RELEASE_BASE_SHA="$(git rev-parse --verify "${RELEASE_BASE_SHA}^{commit}")"
RELEASE_HEAD_SHA="$(git rev-parse --verify 'HEAD^{commit}')"
test "$RELEASE_BASE_SHA" != "$RELEASE_HEAD_SHA"
git merge-base --is-ancestor "$RELEASE_BASE_SHA" "$RELEASE_HEAD_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"

uv sync --frozen
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/check_architecture.py
uv run python scripts/check_protected_operations.py
uv run python scripts/check_test_invariants.py
uv run python scripts/generate_cli_reference.py --check
uv run python scripts/generate_config_reference.py --check
uv run python scripts/generate_pypi_readme.py --check
uv run python scripts/test_matrix.py --lane all
npm --prefix gui ci
uv run python scripts/test_matrix.py --lane gui
uv run python scripts/check_changed_whitespace.py \
  --base-sha "$RELEASE_BASE_SHA" \
  --default-branch "$RELEASE_DEFAULT_BRANCH"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
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

For the 1.5.1 internal desktop candidate, native packaging is a separate
manual workflow (`.github/workflows/desktop-internal.yml`). It builds only on
the three locked native runners, uploads Actions artifacts for 14 days, and has
read-only repository permission. Its macOS artifact is ad-hoc signed and not
notarized; Windows and Linux artifacts are unsigned. Record all three workflow
job locators, every package/SBOM/component/notice checksum, and the aggregate
upload-verification job if the internal desktop claim is included. Do not turn
that receipt into a public download, tag, Release, signing, notarization, or
auto-update action without a new explicit authorization.

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
canonical files to TestPyPI and perform the index readback below with
`RELEASE_INDEX_JSON_BASE=https://test.pypi.org/pypi`. TestPyPI is an external
mutation and needs the same explicit preview; never rebuild between TestPyPI
and production. A successful TestPyPI readback is not authorization to upload
to production.

```bash
set -Eeuo pipefail
IFS=$'\n\t'

RELEASE_INDEX_JSON_BASE=https://test.pypi.org/pypi
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

RELEASE_INDEX_JSON_BASE=https://pypi.org/pypi
uv run --frozen --no-dev --group release twine upload \
  --repository pypi \
  "$RELEASE_WHEEL" "$RELEASE_SDIST"
```

Immediately read the uploaded files back from the selected index. For the
production upload, set the JSON base exactly as below. For a TestPyPI smoke,
run the same block immediately after that upload with the TestPyPI JSON base
shown above. The block requires exactly the canonical wheel and sdist, compares
the index-declared and downloaded SHA-256 values and sizes with the canonical
files, installs the downloaded wheel into a fresh environment, and exercises
metadata, import, all three console entrypoints, and the deterministic
in-memory demo:

```bash
set -Eeuo pipefail
IFS=$'\n\t'

: "${RELEASE_INDEX_JSON_BASE:?set the authorized PyPI or TestPyPI JSON base}"
case "$RELEASE_INDEX_JSON_BASE" in
  https://pypi.org/pypi|https://test.pypi.org/pypi) ;;
  *) printf 'unexpected release index JSON base\n' >&2; exit 1 ;;
esac
RELEASE_READBACK_DIR="$(mktemp -d)"

.venv/bin/python - \
  "$RELEASE_INDEX_JSON_BASE" \
  "$RELEASE_VERSION" \
  "$RELEASE_WHEEL" \
  "$RELEASE_SDIST" \
  "$RELEASE_READBACK_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote, urlsplit
from urllib.request import urlopen

json_base, version, wheel_text, sdist_text, output_text = sys.argv[1:]
canonical = {
    path.name: path
    for path in (Path(wheel_text).resolve(strict=True), Path(sdist_text).resolve(strict=True))
}
output = Path(output_text).resolve(strict=True)
api_url = f"{json_base.rstrip('/')}/agent-libos/{quote(version, safe='')}/json"
with urlopen(api_url, timeout=30) as response:
    payload = json.load(response)
info = payload.get("info")
if not isinstance(info, dict) or info.get("version") != version:
    raise SystemExit("index metadata version does not match the authorized version")
long_description = info.get("description")
if not isinstance(long_description, str):
    raise SystemExit("index metadata has no long description")
if f"/blob/v{version}/" not in long_description:
    raise SystemExit("index long description lacks release-tag-pinned links")
if "/blob/main/" in long_description:
    raise SystemExit("index long description contains a mutable main-branch link")
print(f"long-description-links-ok /blob/v{version}/")

published: dict[str, dict[str, object]] = {}
for item in payload.get("urls", []):
    if not isinstance(item, dict) or not isinstance(item.get("filename"), str):
        raise SystemExit("index returned malformed file metadata")
    filename = str(item["filename"])
    if filename in published:
        raise SystemExit(f"index returned duplicate filename: {filename}")
    published[filename] = item
if set(published) != set(canonical):
    raise SystemExit(
        f"index file set differs: expected={sorted(canonical)!r} "
        f"actual={sorted(published)!r}"
    )

for filename, source in canonical.items():
    item = published[filename]
    digest = item.get("digests")
    declared_sha = digest.get("sha256") if isinstance(digest, dict) else None
    with source.open("rb") as stream:
        local_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    expected_size = source.stat().st_size
    if declared_sha != local_sha or item.get("size") != expected_size:
        raise SystemExit(f"index digest/size differs from canonical file: {filename}")
    file_url = item.get("url")
    if not isinstance(file_url, str) or urlsplit(file_url).scheme != "https":
        raise SystemExit(f"index returned a non-HTTPS file URL: {filename}")
    target = output / filename
    downloaded_size = 0
    downloaded_hash = hashlib.sha256()
    with urlopen(file_url, timeout=60) as response, target.open("xb") as stream:
        if urlsplit(response.geturl()).scheme != "https":
            raise SystemExit(f"artifact redirect is not HTTPS: {filename}")
        while chunk := response.read(1024 * 1024):
            downloaded_size += len(chunk)
            if downloaded_size > expected_size:
                raise SystemExit(f"download is larger than canonical file: {filename}")
            downloaded_hash.update(chunk)
            stream.write(chunk)
    if downloaded_size != expected_size or downloaded_hash.hexdigest() != local_sha:
        raise SystemExit(f"downloaded file differs from canonical file: {filename}")
    print(f"index-readback {api_url} {filename} sha256={local_sha}")
PY

RELEASE_READBACK_WHEEL="$RELEASE_READBACK_DIR/$(basename "$RELEASE_WHEEL")"
uv venv --python 3.11 "$RELEASE_READBACK_DIR/venv"
RELEASE_READBACK_PYTHON="$RELEASE_READBACK_DIR/venv/bin/python"
RELEASE_READBACK_BIN="$RELEASE_READBACK_DIR/venv/bin"
uv pip install \
  --python "$RELEASE_READBACK_PYTHON" \
  --index-url https://pypi.org/simple \
  "$RELEASE_READBACK_WHEEL"

"$RELEASE_READBACK_PYTHON" - "$RELEASE_VERSION" <<'PY'
from importlib import metadata
import sys

import agent_libos

expected_version = sys.argv[1]
distribution = metadata.distribution("agent-libos")
if distribution.version != expected_version or agent_libos.__version__ != expected_version:
    raise SystemExit("installed metadata/import version mismatch")
expected_scripts = {
    "agent-libos",
    "agent-libos-gui-server",
    "agent-libos-migrate-tool-groups",
}
actual_scripts = {
    entry.name
    for entry in distribution.entry_points
    if entry.group == "console_scripts"
}
if actual_scripts != expected_scripts:
    raise SystemExit(
        f"console entrypoint set differs: expected={sorted(expected_scripts)!r} "
        f"actual={sorted(actual_scripts)!r}"
    )
print(f"metadata-and-entrypoints-ok {expected_version}")
PY

"$RELEASE_READBACK_BIN/agent-libos" --help >/dev/null
"$RELEASE_READBACK_BIN/agent-libos-gui-server" --help >/dev/null
"$RELEASE_READBACK_BIN/agent-libos-migrate-tool-groups" --help >/dev/null
(
  cd "$RELEASE_READBACK_DIR"
  "$RELEASE_READBACK_BIN/agent-libos" --db local demo > demo.json
)
"$RELEASE_READBACK_PYTHON" - "$RELEASE_READBACK_DIR/demo.json" <<'PY'
import json
from pathlib import Path
import sys

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(report, dict) or report.get("target_file", {}).get("content_matches") is not True:
    raise SystemExit("deterministic demo did not return its successful JSON contract")
print("entrypoint-help-and-demo-ok")
PY
```

Success prints `long-description-links-ok /blob/vX.Y.Z/`, one `index-readback`
line per canonical file with its SHA-256, then
`metadata-and-entrypoints-ok` and `entrypoint-help-and-demo-ok`. Any mutable or
missing long-description link, missing/extra index file, hash or size mismatch,
non-HTTPS artifact URL, install failure, missing entrypoint, or demo contract
failure stops the release. Index propagation may delay this block; a
404 is a readback failure to retry after waiting, not a reason to upload the
same files again. Preserve the printed URLs, hashes, and success lines in the
private release record without credentials.

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
