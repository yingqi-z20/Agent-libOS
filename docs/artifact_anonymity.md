# Artifact Publication and Anonymity Checklist

This contributor-only runbook covers identity and secret scanning for a source
archive, benchmark bundle, or anonymous research artifact. It is not legal
advice, an end-user setup guide, or evidence that any artifact passed review.
The commands below are intentionally fail closed: an unreadable input, failed
detector, unreviewed match, changed byte, unsafe archive member, or incomplete
final rescan stops publication.

## Paper title, system name, and license

The paper title is fixed as:

> Agent libOS: A Runtime Substrate for Capability-Controlled Self-Evolving LLM Agents

Use `Agent libOS` consistently. Do not revive the temporary anonymous name
`Primitive Agent Runtime` (`PAR`). For double-blind review, anonymize authors,
institutions, repository ownership, and deployment metadata instead of renaming
the runtime in source code.

The license gate is:

- `LICENSE` contains Apache License 2.0;
- `pyproject.toml` declares `Apache-2.0` and no conflicting classifier;
- README and artifact documentation do not claim another license; and
- the wheel and source distribution contain the expected license material.

## What must be scanned

Scan the exact commit, the exact `git archive` projection, every generated file,
and the final packed deliverable. The review covers:

- author, username, institution, lab, email, repository, issue-tracker, cloud,
  tenant, project, account, and deployment identifiers;
- absolute local paths such as `C:\Users\...`, `/Users/...`, `/home/...`,
  `/private/...`, and `/tmp/...`;
- `.env` files, API keys, access tokens, private keys, cookies, connection
  strings, credentials, private endpoints, and provider account metadata;
- Git remotes, every ref name, commit/tag identity, signatures, messages, and
  reachable history when Git metadata will be delivered; and
- metadata and visible content in PDF, Office, notebook, image, media, model,
  data, executable, and nested archive formats.

Deterministic reviewers must not need real credentials. Optional real-provider
instructions must say that credentials and paid tokens are required, without
embedding either.

## One strict review session

Run all shell blocks in the same Bash process. If the shell is restarted, rerun
this prologue. Do not translate it to a shell without equivalent `errexit`,
`nounset`, and `pipefail` behavior.

```bash
#!/usr/bin/env bash
set -Eeuo pipefail
set -o pipefail
IFS=$'\n\t'
umask 077

: "${ANON_OUTPUT_DIR:?set this to the exact generated-output directory}"
: "${ANON_FINAL_ARCHIVE:?set this to the new absolute .tar deliverable path}"
: "${ANON_REVIEW_DIR:?set this to a persistent private review directory}"

case "$ANON_OUTPUT_DIR" in /*) ;; *) printf 'ANON_OUTPUT_DIR must be absolute\n' >&2; exit 1;; esac
case "$ANON_FINAL_ARCHIVE" in /*) ;; *) printf 'ANON_FINAL_ARCHIVE must be absolute\n' >&2; exit 1;; esac
case "$ANON_REVIEW_DIR" in /*) ;; *) printf 'ANON_REVIEW_DIR must be absolute\n' >&2; exit 1;; esac
case "$ANON_FINAL_ARCHIVE" in *.tar) ;; *) printf 'ANON_FINAL_ARCHIVE must end in .tar\n' >&2; exit 1;; esac

test -d "$ANON_OUTPUT_DIR"
test -d "$ANON_REVIEW_DIR"
test -d "$(dirname "$ANON_FINAL_ARCHIVE")"
test ! -e "$ANON_FINAL_ARCHIVE"

for command_name in git rg python3; do
  command -v "$command_name" >/dev/null
done

ANON_COMMIT="$(git rev-parse --verify 'HEAD^{commit}')"
ANON_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test -z "$ANON_STATUS"
ANON_DISPOSITIONS="$ANON_REVIEW_DIR/dispositions-$ANON_COMMIT.json"
ANON_RECORDED_CANDIDATES="$ANON_REVIEW_DIR/candidates-$ANON_COMMIT.json"

python3 - \
  "$(git rev-parse --show-toplevel)" \
  "$ANON_OUTPUT_DIR" \
  "$ANON_REVIEW_DIR" \
  "$ANON_FINAL_ARCHIVE" \
  "$ANON_DISPOSITIONS" \
  "$ANON_RECORDED_CANDIDATES" <<'PY'
from pathlib import Path
import sys

worktree = Path(sys.argv[1]).resolve(strict=True)
output = Path(sys.argv[2]).resolve(strict=True)
review = Path(sys.argv[3]).resolve(strict=True)
archive = Path(sys.argv[4]).resolve(strict=False)
dispositions = Path(sys.argv[5]).resolve(strict=False)
recorded_argument = Path(sys.argv[6])
if recorded_argument.is_symlink():
    raise SystemExit("recorded candidate path must not be a symbolic link")
recorded_candidates = recorded_argument.resolve(strict=False)


def within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


for label, path in (("output", output), ("review", review)):
    if within(path, worktree) or within(worktree, path):
        raise SystemExit(
            f"{label} and worktree must not contain one another: {path}"
        )
if within(archive, worktree):
    raise SystemExit(f"archive must be outside the worktree: {archive}")
if within(output, review) or within(review, output):
    raise SystemExit("output and private review directories must not contain one another")
if within(archive, output) or within(archive, review):
    raise SystemExit("final archive must be outside output and private review directories")
for label, path in (
    ("dispositions", dispositions),
    ("recorded candidates", recorded_candidates),
):
    if path.parent != review:
        raise SystemExit(
            f"{label} must remain directly inside the private review directory"
        )
PY

ANON_SCAN_DIR="$(mktemp -d "$ANON_REVIEW_DIR/scan.XXXXXX")"
test -n "$ANON_SCAN_DIR"
test -d "$ANON_SCAN_DIR"
trap 'test -n "${ANON_SCAN_DIR:-}" && rm -rf -- "$ANON_SCAN_DIR"' EXIT

ANON_COMMIT_TREE="$ANON_SCAN_DIR/exact-commit-tree"
ANON_COMMIT_EXPORT="$ANON_SCAN_DIR/archive-projection"
ANON_FINAL_EXTRACT="$ANON_SCAN_DIR/final-extract"
mkdir "$ANON_COMMIT_TREE" "$ANON_COMMIT_EXPORT" "$ANON_FINAL_EXTRACT"

printf '%s\n' "$ANON_COMMIT"
```

The commands resolve and enforce that the output and private review directories
are outside the worktree and do not contain one another, and that the final
archive is outside all three. The private disposition and recorded-candidate
paths are derived inside the review directory; the latter is rejected if it is
a symbolic link. Neither can be packed accidentally. The final archive must
not already exist: publication never overwrites an earlier reviewed deliverable.
Commit an intended source change first, then restart against the new commit.
Ignored or merely staged files are not commit evidence.

### Match-command status handling

Both `rg` and `git grep` return `0` for a match, `1` for no match, and a value
greater than `1` for an error. Under strict shell mode, use this helper for every
candidate-producing invocation. It accepts only `0` and `1`, verifies that a
no-match command emitted no candidate bytes, and propagates every real error.

```bash
capture_matches() {
  local destination="$1"
  shift
  local status

  test ! -e "$destination"
  set +e
  "$@" >"$destination"
  status=$?
  set -e

  case "$status" in
    0) test -s "$destination" ;;
    1) test ! -s "$destination" ;;
    *)
      printf 'candidate detector failed (%s):' "$status" >&2
      printf ' %q' "$@" >&2
      printf '\n' >&2
      return "$status"
      ;;
  esac
}
```

Never append `|| true` to a detector. A missing optional tool is an explicit
environment gate, not a clean result.

## Commit, attribute, submodule, and all-ref inventory

Record these private inventories. NUL-delimited files remain private because
they may contain identifying paths or URLs.

```bash
git ls-files -z >"$ANON_SCAN_DIR/tracked-files.nul"
git ls-files -s >"$ANON_SCAN_DIR/tracked-modes"
git check-attr --cached --stdin -z -a \
  <"$ANON_SCAN_DIR/tracked-files.nul" \
  >"$ANON_SCAN_DIR/tracked-attributes.nul"

capture_matches "$ANON_SCAN_DIR/symlink-or-gitlink-modes" \
  rg '^(120000|160000) ' "$ANON_SCAN_DIR/tracked-modes"

git submodule status --recursive >"$ANON_SCAN_DIR/submodules"
git submodule foreach --quiet --recursive \
  'git rev-parse HEAD && git status --porcelain=v1 --untracked-files=all' \
  >"$ANON_SCAN_DIR/submodule-status"

git for-each-ref \
  --format='%(refname)%09%(objectname)%09%(objecttype)%09%(authorname)%09%(authoremail)%09%(committername)%09%(committeremail)%09%(taggername)%09%(taggeremail)' \
  >"$ANON_SCAN_DIR/all-refs"
git log --all --format='%H%x09%an%x09%ae%x09%cn%x09%ce%x09%s' \
  >"$ANON_SCAN_DIR/all-log-identities"
git config --show-origin --list >"$ANON_SCAN_DIR/git-config"
git remote -v >"$ANON_SCAN_DIR/git-remotes"
```

`git for-each-ref` deliberately has no `refs/heads`, `refs/remotes`, or
`refs/tags` restriction. Custom refs, notes, stash, pull refs, and replace refs
can identify an author just as easily as branches and tags.

Mode `120000` is a symlink and `160000` a gitlink. This runbook rejects both
from the packed artifact rather than following or silently omitting them. If an
artifact genuinely needs either, flatten reviewed bytes into ordinary files on
the anonymous branch and restart. A leading `-` in submodule status means the
submodule bytes were never reviewed and keeps the gate open.

Parse the NUL-delimited attribute triples exactly. LFS is the only recognized
content-filter value, but this runbook does not support publishing an LFS
pointer or its separately stored object. An unknown or valueless filter fails;
an exact LFS value takes a distinct blocking path instead of being silently
skipped:

```bash
python3 - \
  "$ANON_SCAN_DIR/tracked-attributes.nul" \
  "$ANON_SCAN_DIR/lfs-required.json" <<'PY'
import json
import os
from pathlib import Path
import sys

raw = Path(sys.argv[1]).read_bytes().split(b"\0")
if raw and raw[-1] == b"":
    raw.pop()
if len(raw) % 3:
    raise SystemExit("git check-attr emitted an incomplete NUL-delimited triple")

lfs_paths: list[str] = []
for index in range(0, len(raw), 3):
    path, attribute, value = raw[index:index + 3]
    if attribute != b"filter":
        continue
    if value != b"lfs":
        raise SystemExit(
            f"unsupported Git content filter {os.fsdecode(value)!r} "
            f"on {os.fsdecode(path)!r}"
        )
    lfs_paths.append(os.fsdecode(path))

if lfs_paths:
    Path(sys.argv[2]).write_text(
        json.dumps(sorted(lfs_paths), indent=2) + "\n",
        encoding="utf-8",
    )
PY

if test -f "$ANON_SCAN_DIR/lfs-required.json"; then
  printf 'Git LFS paths are unsupported by this runbook:\n' >&2
  sed 's/^/  /' "$ANON_SCAN_DIR/lfs-required.json" >&2
  printf 'flatten reviewed LFS bytes, remove the filter, commit, and restart\n' >&2
  exit 4
fi
```

Fetch each required LFS object only in a private review environment, inspect
its actual bytes, then replace the pointer with those reviewed ordinary bytes
and remove the filter attribute on the anonymous branch. Commit that flattened
tree and restart this runbook. The pointer alone is never evidence that its
object is anonymous, and fetch include/exclude rules make an automated partial
LFS scan unsafe for this checklist.

## Patterns and tracked-tree cross-check

The submission owner must replace the identity placeholders with a complete,
private pattern covering every known author identity and deployment identifier.
Keep the pattern and scan log outside the deliverable.

```bash
ANON_IDENTITY_PATTERN='author-one|author@example\.edu|lab-name|institution-domain\.edu'
ANON_PATH_PATTERN='(/Users/|/home/|/private/|/tmp/|[A-Za-z]:\\Users\\)'
ANON_SECRET_PATTERN='(api[_-]?key|access[_-]?token|client[_-]?secret|password|authorization)[[:space:]]*[:=]'
ANON_PRIVATE_KEY_PATTERN='-----BEGIN ([A-Z ]+ )?PRIVATE KEY-----'

test "$ANON_IDENTITY_PATTERN" != \
  'author-one|author@example\.edu|lab-name|institution-domain\.edu'

capture_matches "$ANON_SCAN_DIR/all-refs-sensitive" \
  rg -ali \
    -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" \
    -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" \
    "$ANON_SCAN_DIR/all-refs"
capture_matches "$ANON_SCAN_DIR/git-metadata-sensitive" \
  rg -ali \
    -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" \
    -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" \
    "$ANON_SCAN_DIR/all-log-identities" \
    "$ANON_SCAN_DIR/git-config" \
    "$ANON_SCAN_DIR/git-remotes"

capture_matches "$ANON_SCAN_DIR/git-grep-identity.nul" \
  git grep -lzEI \
    -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" \
    "$ANON_COMMIT" --
capture_matches "$ANON_SCAN_DIR/git-grep-secret.nul" \
  git grep -lzEI \
    -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" \
    "$ANON_COMMIT" --
```

These two results are a cross-check, not the final decision. The byte-exact
materialized tree below is scanned with the same detectors as the archive
projection and generated output. A non-empty candidate list requires an exact
reviewed disposition; a detector error is never interpreted as no match.

## Safe materialization and archive extraction

Create a helper that extracts only ordinary files and directories. It rejects
absolute paths, `.`/`..`, control characters, duplicate canonical paths,
symlinks, hard links, devices, FIFOs, and every other special member. It never
calls `extractall()`.

```bash
cat >"$ANON_SCAN_DIR/safe_tar.py" <<'PY'
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts:
        raise ValueError(f"unsafe archive path: {name!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise ValueError(f"control character in archive path: {name!r}")
    return tuple(path.parts)


parser = argparse.ArgumentParser()
parser.add_argument("archive")
parser.add_argument("destination")
args = parser.parse_args()
destination = Path(args.destination).resolve(strict=True)
seen: set[tuple[str, ...]] = set()

with tarfile.open(args.archive, "r:*") as archive:
    for member in archive:
        parts = safe_parts(member.name)
        if parts in seen:
            raise ValueError(f"duplicate archive path: {member.name!r}")
        seen.add(parts)
        target = destination.joinpath(*parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=False)
            target.chmod(0o755)
            continue
        if not member.isfile():
            raise ValueError(f"non-regular archive member: {member.name!r}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"unreadable archive member: {member.name!r}")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
        target.chmod(0o755 if member.mode & 0o111 else 0o644)
PY

python3 - "$ANON_COMMIT" "$ANON_COMMIT_TREE" <<'PY'
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

commit, destination_text = sys.argv[1:]
destination = Path(destination_text).resolve(strict=True)
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
    if object_type != "blob" or mode not in {"100644", "100755"}:
        raise SystemExit(f"anonymous artifacts require regular tracked files: {raw_path!r}")
    path_text = os.fsdecode(raw_path)
    if os.fsencode(path_text) != raw_path:
        raise SystemExit("tracked path is not reversibly decodable")
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise SystemExit("unsafe tracked path")
    if any(ord(character) < 32 or ord(character) == 127 for character in path_text):
        raise SystemExit("control character in tracked path")
    target = destination.joinpath(*relative.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = subprocess.run(
        ["git", "--no-replace-objects", "cat-file", "blob", oid],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with target.open("xb") as output:
        output.write(payload)
    target.chmod(0o755 if mode == "100755" else 0o644)
PY

git --no-replace-objects archive --format=tar "$ANON_COMMIT" \
  >"$ANON_SCAN_DIR/archive-projection.tar"
python3 "$ANON_SCAN_DIR/safe_tar.py" \
  "$ANON_SCAN_DIR/archive-projection.tar" "$ANON_COMMIT_EXPORT"
```

The raw tree ignores export transformations by design. The archive projection
applies committed `export-ignore` and `export-subst` attributes. Review both;
neither is a substitute for the other.

## Equal scans for all three inventories

The following helper rejects symlinks, special files, undecodable/control
characters in paths, and detector failures. It applies the same content
detectors to the raw commit tree, archive projection, and generated output.
Path-name candidates use the same identity/path patterns, while secret-like
filenames receive a separate detector.

```bash
cat >"$ANON_SCAN_DIR/path_candidates.py" <<'PY'
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

parser = argparse.ArgumentParser()
parser.add_argument("root")
parser.add_argument("pattern")
args = parser.parse_args()
root = Path(args.root).resolve(strict=True)
pattern = re.compile(args.pattern, re.IGNORECASE)

for current, directories, files in os.walk(root, followlinks=False):
    current_path = Path(current)
    for name in sorted([*directories, *files]):
        path = current_path / name
        if path.is_symlink():
            raise SystemExit(f"symlink is not allowed: {path}")
        relative = path.relative_to(root).as_posix()
        if ".git" in Path(relative).parts:
            raise SystemExit(f"Git metadata is not allowed in packed roots: {relative!r}")
        if any(ord(character) < 32 or ord(character) == 127 for character in relative):
            raise SystemExit(f"control character in path: {relative!r}")
        if path.is_dir() or path.is_file():
            if pattern.search(relative):
                print(os.fspath(path), end="\0")
        else:
            raise SystemExit(f"special file is not allowed: {path}")
PY

scan_root() {
  local label="$1"
  local root="$2"
  local destination="$3"
  mkdir -p "$destination"
  test -d "$root"

  capture_matches "$destination/$label.identity-content.nul" \
    rg -ali0 --hidden --no-ignore -e "$ANON_IDENTITY_PATTERN" -- "$root"
  capture_matches "$destination/$label.path-content.nul" \
    rg -ali0 --hidden --no-ignore -e "$ANON_PATH_PATTERN" -- "$root"
  capture_matches "$destination/$label.secret-content.nul" \
    rg -ali0 --hidden --no-ignore \
      -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" -- "$root"

  python3 "$ANON_SCAN_DIR/path_candidates.py" \
    "$root" "$ANON_IDENTITY_PATTERN" \
    >"$destination/$label.identity-name.nul"
  python3 "$ANON_SCAN_DIR/path_candidates.py" \
    "$root" "$ANON_PATH_PATTERN" \
    >"$destination/$label.path-name.nul"
  python3 "$ANON_SCAN_DIR/path_candidates.py" \
    "$root" '(^|/)(\.env($|\.)|[^/]*(key|token|credential|cookie|secret)[^/]*)' \
    >"$destination/$label.secret-name.nul"
}

ANON_PRIMARY_SCAN="$ANON_SCAN_DIR/primary-scan"
mkdir "$ANON_PRIMARY_SCAN"
scan_root commit-tree "$ANON_COMMIT_TREE" "$ANON_PRIMARY_SCAN"
scan_root archive-projection "$ANON_COMMIT_EXPORT" "$ANON_PRIMARY_SCAN"
scan_root generated-output "$ANON_OUTPUT_DIR" "$ANON_PRIMARY_SCAN"
```

An approved entropy/secret scanner must also run over the same three roots. Its
redacted candidate paths are added to the disposition process below. Generic
patterns do not cover provider-specific tokens, JWTs, encoded credentials, or
high-entropy secrets.

### Exact candidate dispositions

Intentional fixtures and documentation often contain words such as `api_key`.
They may remain only after two humans review the exact bytes. Build one private
candidate manifest whose key is detector, root label, relative path, kind, and
SHA-256 (directories use `null`). Filenames are recorded; matching contents are
not copied into the log.

```bash
cat >"$ANON_SCAN_DIR/candidate_manifest.py" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re


def digest(path: Path) -> str | None:
    if path.is_dir():
        return None
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in ("detector", "root", "path", "kind", "sha256"))


parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="command", required=True)
build = subparsers.add_parser("build")
build.add_argument("--scan-dir", required=True)
build.add_argument("--root", action="append", required=True)
build.add_argument("--output", required=True)
verify = subparsers.add_parser("verify")
verify.add_argument("--candidates", required=True)
verify.add_argument("--dispositions", required=True)
verify.add_argument("--root", action="append")
args = parser.parse_args()

if args.command == "build":
    roots: dict[str, Path] = {}
    for item in args.root:
        label, separator, raw_root = item.partition("=")
        if not separator or not label or label in roots:
            raise SystemExit("each --root must be a unique label=/absolute/path")
        roots[label] = Path(raw_root).resolve(strict=True)
    rows: dict[tuple[object, ...], dict[str, object]] = {}
    for match_file in sorted(Path(args.scan_dir).glob("*.nul")):
        label, separator, detector = match_file.name.removesuffix(".nul").partition(".")
        if not separator or label not in roots:
            raise SystemExit(f"unexpected match file: {match_file.name}")
        for raw_path in match_file.read_bytes().split(b"\0"):
            if not raw_path:
                continue
            path_text = os.fsdecode(raw_path)
            if os.fsencode(path_text) != raw_path:
                raise SystemExit("candidate path is not reversibly decodable")
            path = Path(path_text)
            try:
                relative = path.relative_to(roots[label]).as_posix()
            except ValueError as error:
                raise SystemExit(f"candidate escaped root: {path}") from error
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise SystemExit(f"candidate is not a regular file/directory: {path}")
            row = {
                "detector": detector,
                "root": label,
                "path": relative,
                "kind": "file" if path.is_file() else "directory",
                "sha256": digest(path),
            }
            rows[key(row)] = row
    output = Path(args.output)
    output.write_text(
        json.dumps(sorted(rows.values(), key=key), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
else:
    candidates = json.loads(Path(args.candidates).read_text(encoding="utf-8"))
    dispositions = json.loads(Path(args.dispositions).read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not isinstance(dispositions, list):
        raise SystemExit("candidate and disposition documents must be JSON arrays")
    selected_roots = set(args.root or [])
    candidate_by_key = {
        key(row): row for row in candidates
        if not selected_roots or row.get("root") in selected_roots
    }
    disposition_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    for row in dispositions:
        if selected_roots and row.get("root") not in selected_roots:
            continue
        row_key = key(row)
        if row_key in disposition_by_key:
            raise SystemExit(f"duplicate disposition: {row_key}")
        if row.get("disposition") != "intentional non-identifying fixture":
            raise SystemExit(f"candidate must be removed/replaced or explicitly retained: {row_key}")
        reviewer = str(row.get("reviewer") or "").strip()
        second = str(row.get("second_reviewer") or "").strip()
        reviewed_at = str(row.get("reviewed_at") or "")
        if not reviewer or not second or reviewer == second:
            raise SystemExit(f"two distinct reviewers are required: {row_key}")
        if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", reviewed_at) is None:
            raise SystemExit(f"invalid reviewed_at date: {row_key}")
        disposition_by_key[row_key] = row
    missing = sorted(set(candidate_by_key) - set(disposition_by_key))
    stale = sorted(set(disposition_by_key) - set(candidate_by_key))
    if missing or stale:
        raise SystemExit(f"disposition mismatch: missing={missing}, stale={stale}")
PY

ANON_PRIMARY_CANDIDATES="$ANON_SCAN_DIR/primary-candidates.json"
python3 "$ANON_SCAN_DIR/candidate_manifest.py" build \
  --scan-dir "$ANON_PRIMARY_SCAN" \
  --root "commit-tree=$ANON_COMMIT_TREE" \
  --root "archive-projection=$ANON_COMMIT_EXPORT" \
  --root "generated-output=$ANON_OUTPUT_DIR" \
  --output "$ANON_PRIMARY_CANDIDATES"

python3 - "$ANON_PRIMARY_CANDIDATES" "$ANON_RECORDED_CANDIDATES" <<'PY'
import os
from pathlib import Path
import sys

current = Path(sys.argv[1]).read_bytes()
recorded = Path(sys.argv[2])
if recorded.is_symlink():
    raise SystemExit("recorded candidate path must not be a symbolic link")
if recorded.exists():
    if not recorded.is_file() or recorded.read_bytes() != current:
        raise SystemExit("candidate inventory changed; discard the old review and restart")
else:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(recorded, flags, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        output.write(current)
PY

if ! test -f "$ANON_DISPOSITIONS"; then
  printf 'review is required; candidates persisted at %s\n' \
    "$ANON_RECORDED_CANDIDATES" >&2
  printf 'create the reviewed disposition array at %s, then rerun\n' \
    "$ANON_DISPOSITIONS" >&2
  exit 3
fi

python3 "$ANON_SCAN_DIR/candidate_manifest.py" verify \
  --candidates "$ANON_PRIMARY_CANDIDATES" \
  --dispositions "$ANON_DISPOSITIONS"
```

The first scan always persists its exact candidate manifest in
`ANON_REVIEW_DIR` and exits with status `3` when the matching disposition file
does not exist. The temporary-directory cleanup therefore cannot erase the only
copy needed to bootstrap review. Review candidate metadata—not matching
content—and create the disposition JSON. Each retained entry repeats the five
candidate key fields and adds:

```json
{
  "disposition": "intentional non-identifying fixture",
  "reviewer": "private-reviewer-id",
  "second_reviewer": "different-private-reviewer-id",
  "reviewed_at": "2099-01-31"
}
```

The values above are format examples, not real review evidence. Removed or
replaced matches disappear on the next scan; they must not be approved as still
present. Any byte change changes SHA-256 and invalidates its disposition. Stale
dispositions also fail so the record cannot silently describe another artifact.

## Git history when Git metadata is delivered

Prefer the final source archive below, which excludes `.git`. If Git metadata,
a bundle, or repository clone is itself the deliverable, all-ref inventory is
not enough. Scan every reachable object without replace-object substitution and
then every object in the object database, including unreachable objects:

```bash
git --no-replace-objects rev-list --objects --all \
  >"$ANON_SCAN_DIR/reachable-objects"
git --no-replace-objects cat-file \
  --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' \
  <"$ANON_SCAN_DIR/reachable-objects" \
  >"$ANON_SCAN_DIR/reachable-metadata"
git --no-replace-objects cat-file --batch-all-objects \
  --batch-check='%(objectname) %(objecttype) %(objectsize)' \
  >"$ANON_SCAN_DIR/all-object-metadata"

scan_object_metadata() {
  local metadata="$1"
  local findings="$2"
  : >"$findings"
  while IFS=' ' read -r object_oid object_type object_size object_path; do
    case "$object_type" in
      commit|tag|tree|blob)
        git --no-replace-objects cat-file "$object_type" "$object_oid" \
          >"$ANON_SCAN_DIR/object-bytes"
        local status
        set +e
        rg -a -qi \
          -e "$ANON_IDENTITY_PATTERN" -e "$ANON_PATH_PATTERN" \
          -e "$ANON_SECRET_PATTERN" -e "$ANON_PRIVATE_KEY_PATTERN" \
          "$ANON_SCAN_DIR/object-bytes"
        status=$?
        set -e
        case "$status" in
          0) printf '%s\t%s\t%s\t%s\n' \
               "$object_oid" "$object_type" "$object_size" "${object_path:-}" \
               >>"$findings" ;;
          1) ;;
          *) return "$status" ;;
        esac
        ;;
      *) printf 'unexpected object type: %s\n' "$object_type" >&2; return 1 ;;
    esac
  done <"$metadata"
}

scan_object_metadata "$ANON_SCAN_DIR/reachable-metadata" \
  "$ANON_SCAN_DIR/reachable-findings"
scan_object_metadata "$ANON_SCAN_DIR/all-object-metadata" \
  "$ANON_SCAN_DIR/all-object-findings"
```

Review ref names, config, remotes, reflogs, signatures, commit/tag bodies,
object paths, object bytes, hooks, and every copied `.git` metadata file. Do not
put raw URLs, credentials, or matched bytes in the shared log. A history rewrite
requires a fresh clone/export and a complete rerun; pruning does not prove a
copied object database anonymous. Any unresolved history finding keeps the gate
open.

## Format-aware and visual review

Raw regexes do not parse compressed OOXML, PDF metadata, notebook outputs,
nested archives, or screenshots. Apply the same format-aware review to
`$ANON_COMMIT_TREE`, `$ANON_COMMIT_EXPORT`, and `$ANON_OUTPUT_DIR`:

```bash
for ANON_BINARY_ROOT in \
  "$ANON_COMMIT_TREE" "$ANON_COMMIT_EXPORT" "$ANON_OUTPUT_DIR"; do
  test -d "$ANON_BINARY_ROOT"
  find "$ANON_BINARY_ROOT" -type f -exec file --mime-type -- '{}' \; \
    | LC_ALL=C sort \
    >"$ANON_SCAN_DIR/$(basename "$ANON_BINARY_ROOT").mime-inventory"
done
```

Record tool versions. For each non-plain-text MIME type:

- inspect metadata with a format-aware tool such as `exiftool`;
- use `pdfinfo`, `pdftotext`, and a full-page render for every PDF;
- inspect OOXML core/app/custom properties, comments, revisions, relationships,
  notes, hidden slides/sheets, and then render the document;
- inspect notebook kernels, outputs, widgets, errors, embedded media, and paths;
- inspect and visually review every image, plot, screenshot, audio, and video;
- inspect executables, databases, model/data files, fonts, and unfamiliar
  containers with a recorded suitable tool; and
- list every nested archive member, reject unsafe/link/special members, extract
  with a path-safe tool into a fresh directory, and recursively repeat the same
  text, secret, MIME, metadata, and visual checks.

For the project wheel and source distribution, also run
`uv run python scripts/check_release_artifacts.py <artifact-directory>`.
Structural validation is not an anonymity scan.

## Build, hash, safely unpack, and rescan the final archive

Only the reviewed archive projection and reviewed generated output are packed.
The raw commit tree is evidence for transformations, not a second copy in the
deliverable. The builder rejects links and special files, emits deterministic
ownership/time metadata, and never includes `.git` implicitly.

```bash
cat >"$ANON_SCAN_DIR/tree_inventory.py" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("root")
args = parser.parse_args()
root = Path(args.root).resolve(strict=True)
rows: list[dict[str, object]] = []

for path in [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]:
    if path.is_symlink() or not (path.is_dir() or path.is_file()):
        raise SystemExit(f"only ordinary files/directories may be inventoried: {path}")
    relative = path.relative_to(root).as_posix()
    if ".git" in Path(relative).parts:
        raise SystemExit(f"Git metadata is not allowed in packed roots: {relative!r}")
    if any(ord(character) < 32 or ord(character) == 127 for character in relative):
        raise SystemExit(f"control character in inventoried path: {relative!r}")
    row: dict[str, object] = {
        "path": relative,
        "kind": "directory" if path.is_dir() else "file",
        # Compare the normalized mode emitted by build_final_tar.py, not the
        # review host's umask-dependent staging mode.
        "mode": "0755" if path.is_dir() or path.stat().st_mode & 0o111 else "0644",
        "size": 0 if path.is_dir() else path.stat().st_size,
        "sha256": None,
    }
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        row["sha256"] = digest.hexdigest()
    rows.append(row)

print(json.dumps(rows, indent=2, sort_keys=True))
PY

python3 "$ANON_SCAN_DIR/tree_inventory.py" "$ANON_COMMIT_TREE" \
  >"$ANON_SCAN_DIR/commit-tree.inventory.json"
python3 "$ANON_SCAN_DIR/tree_inventory.py" "$ANON_COMMIT_EXPORT" \
  >"$ANON_SCAN_DIR/source.expected.inventory.json"
python3 "$ANON_SCAN_DIR/tree_inventory.py" "$ANON_OUTPUT_DIR" \
  >"$ANON_SCAN_DIR/generated.expected.inventory.json"

cat >"$ANON_SCAN_DIR/build_final_tar.py" <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("archive")
parser.add_argument("--source", required=True)
parser.add_argument("--generated", required=True)
args = parser.parse_args()
archive_path = Path(args.archive)
roots = {
    "artifact/source": Path(args.source).resolve(strict=True),
    "artifact/generated": Path(args.generated).resolve(strict=True),
}

with tarfile.open(archive_path, "x", format=tarfile.PAX_FORMAT) as archive:
    for prefix, root in roots.items():
        entries = [root, *sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())]
        for path in entries:
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError(f"only ordinary files/directories may be packed: {path}")
            relative = path.relative_to(root).as_posix()
            name = prefix if relative == "." else f"{prefix}/{relative}"
            if any(ord(character) < 32 or ord(character) == 127 for character in name):
                raise ValueError(f"control character in packed path: {name!r}")
            info = archive.gettarinfo(path, arcname=name)
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if path.is_dir() or path.stat().st_mode & 0o111 else 0o644
            if path.is_file():
                with path.open("rb") as source:
                    archive.addfile(info, source)
            else:
                archive.addfile(info)
PY

python3 "$ANON_SCAN_DIR/build_final_tar.py" "$ANON_FINAL_ARCHIVE" \
  --source "$ANON_COMMIT_EXPORT" \
  --generated "$ANON_OUTPUT_DIR"

sha256_file() {
  python3 - "$1" <<'PY'
import hashlib
from pathlib import Path
import sys

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
}

ANON_FINAL_SHA256="$(sha256_file "$ANON_FINAL_ARCHIVE")"
printf '%s  %s\n' "$ANON_FINAL_SHA256" "$ANON_FINAL_ARCHIVE"

python3 "$ANON_SCAN_DIR/safe_tar.py" \
  "$ANON_FINAL_ARCHIVE" "$ANON_FINAL_EXTRACT"

ANON_FINAL_SCAN="$ANON_SCAN_DIR/final-scan"
mkdir "$ANON_FINAL_SCAN"
scan_root archive-projection \
  "$ANON_FINAL_EXTRACT/artifact/source" "$ANON_FINAL_SCAN"
scan_root generated-output \
  "$ANON_FINAL_EXTRACT/artifact/generated" "$ANON_FINAL_SCAN"

ANON_FINAL_CANDIDATES="$ANON_SCAN_DIR/final-candidates.json"
python3 "$ANON_SCAN_DIR/candidate_manifest.py" build \
  --scan-dir "$ANON_FINAL_SCAN" \
  --root "archive-projection=$ANON_FINAL_EXTRACT/artifact/source" \
  --root "generated-output=$ANON_FINAL_EXTRACT/artifact/generated" \
  --output "$ANON_FINAL_CANDIDATES"
python3 "$ANON_SCAN_DIR/candidate_manifest.py" verify \
  --candidates "$ANON_FINAL_CANDIDATES" \
  --dispositions "$ANON_DISPOSITIONS" \
  --root archive-projection \
  --root generated-output

python3 "$ANON_SCAN_DIR/tree_inventory.py" "$ANON_COMMIT_EXPORT" \
  >"$ANON_SCAN_DIR/source.refreshed.inventory.json"
python3 "$ANON_SCAN_DIR/tree_inventory.py" "$ANON_OUTPUT_DIR" \
  >"$ANON_SCAN_DIR/generated.refreshed.inventory.json"
python3 "$ANON_SCAN_DIR/tree_inventory.py" \
  "$ANON_FINAL_EXTRACT/artifact/source" \
  >"$ANON_SCAN_DIR/source.extracted.inventory.json"
python3 "$ANON_SCAN_DIR/tree_inventory.py" \
  "$ANON_FINAL_EXTRACT/artifact/generated" \
  >"$ANON_SCAN_DIR/generated.extracted.inventory.json"

python3 - "$ANON_SCAN_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for name in ("source", "generated"):
    expected = (root / f"{name}.expected.inventory.json").read_bytes()
    refreshed = (root / f"{name}.refreshed.inventory.json").read_bytes()
    extracted = (root / f"{name}.extracted.inventory.json").read_bytes()
    if expected != refreshed:
        raise SystemExit(f"{name} input changed while the archive was built")
    if expected != extracted:
        raise SystemExit(f"{name} extracted inventory differs from reviewed input")
PY

find "$ANON_FINAL_EXTRACT" -type f -exec file --mime-type -- '{}' \; \
  | LC_ALL=C sort \
  >"$ANON_SCAN_DIR/final-extract.mime-inventory"
```

Repeat the MIME/metadata/nested-archive review on the safely extracted final
tree. The commands compare deterministic path, kind, normalized mode, size, and
SHA-256 inventories both against refreshed inputs and the extracted tree. Any
difference is a failure. The initial archive hash is recorded only after
construction and is recomputed after all extraction and review commands; the
extracted bytes—not the staging directories—are the final evidence surface.

Finally prove that the source did not move during review:

```bash
ANON_FINAL_COMMIT="$(git rev-parse --verify 'HEAD^{commit}')"
ANON_FINAL_STATUS="$(git status --porcelain=v1 --untracked-files=all)"
test "$ANON_FINAL_COMMIT" = "$ANON_COMMIT"
test -z "$ANON_FINAL_STATUS"
test -f "$ANON_FINAL_ARCHIVE"
test -n "$ANON_FINAL_SHA256"
ANON_VERIFIED_FINAL_SHA256="$(sha256_file "$ANON_FINAL_ARCHIVE")"
test "$ANON_VERIFIED_FINAL_SHA256" = "$ANON_FINAL_SHA256"
```

If the archive, source commit, generated output, disposition file, or any
reviewed binary changes, discard the result and restart.

## Runtime validation

Run the deterministic artifact baseline in a fresh clone before submission:

```bash
uv sync --frozen
uv run python -m compileall agent_libos tests scripts experiments benchmarks modules
uv run python scripts/test_matrix.py --lane all
uv run python scripts/check_test_invariants.py
```

Deno-backed tests run when Deno is installed. Use `--skip-real-deno` only when
the submitted validation record explicitly says real Deno was excluded.

## Publication exit gate

The artifact is ready only when all of these statements are backed by the same
private review record:

- license and package metadata agree;
- the exact source commit is clean and unchanged;
- the raw commit tree, archive projection, generated output, and safely
  re-extracted final archive have complete inventories;
- every detector completed successfully on all required roots;
- every retained candidate has an exact current-byte disposition and two
  distinct reviewers, with no stale or missing disposition;
- all refs and, when shipped, all reachable/unreachable Git objects and Git
  metadata have been reviewed;
- every binary/container has completed format-aware, recursive, and visual
  review with no unresolved identity or secret;
- the final archive contains only regular files/directories, was extracted by
  the rejecting path-safe extractor, and the extracted inventories match their
  reviewed inputs;
- fresh-clone runtime checks completed with environment gates stated; and
- the recorded final SHA-256 matches the exact file being submitted.

This checklist is not a receipt. Any missing command result, detector error,
unresolved match, changed hash, unsafe member, incomplete environment gate, or
missing second review keeps publication blocked.
