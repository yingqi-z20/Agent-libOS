# Changelog

This changelog starts with the currently maintained release line. It does not
reconstruct older release notes or infer publication dates from Git history.
Git history remains the record for earlier development snapshots.

## Unreleased

Changes intended for the next published version must be summarized here before
release. Do not treat an entry in this section as shipped behavior.

## 1.5.2

`1.5.2` is the current stabilization release version aligned across the Python
project, package lockfiles, GUI package, MCP client identity, desktop metadata,
and release workflows. It builds on the 1.5.1 Runtime authority and data-flow
semantics with further filesystem-authorization hardening, storage and recovery
safety, generated-reference tooling, and a comprehensive documentation accuracy
and completeness pass. As with every version entry here, publication still
requires the separately authorized, receipt-bound process in
[docs/releasing.md](docs/releasing.md); this entry alone does not claim a tag,
package-index upload, signed desktop distribution, or completed external-provider
gate.

- Hardened filesystem authorization against path rebinding between
  authorization and use.
- Implemented runtime safety and recovery enhancements across the storage
  factory and backend boundaries, checkpoint restore and snapshot remapping,
  startup recovery, Git command policy, and Windows store identity handling.
- Introduced the generated CLI reference, the generated configuration field
  reference, and the version-pinned PyPI README generator, extracting the CLI
  parser build for generated-reference use.
- Fixed an ObjectTask wait race that could return a stale snapshot from
  before the worker published its waiting notification, terminal result
  collection in the ask-file-then-show example script, and a CI readiness
  race in the GUI end-to-end suite.
- Fixed documentation accuracy and completeness across the guide set. The
  generated CLI reference now reports the effective destination default for
  store-constant flags, and `task-run list --status` help and rejection
  messages enumerate the closed status set.
- Completed a comprehensive documentation review and remediation across the
  guide set: corrected drifted inventories (MCP Resource surface, image
  bindings, checkpoint images), documented the CLI exit-code contract and exec
  image-package authority grant, fixed the storage migration backup example,
  replaced stale version tokens, pinned the PyPI README quick-start clone to
  the release tag, added help text for required CLI options, and added a
  glossary version-map tripwire regression test.

## 1.5.1

`1.5.1` is the current stabilization release version aligned across the Python
project, package lockfiles, GUI package, MCP client identity, desktop metadata,
and release workflows. It preserves the Runtime authority and data-flow
semantics while hardening first-run behavior, offline migration reconciliation,
terminal-owner cleanup, live-evaluation evidence recomputation, and release
artifact validation.

The source distribution now uses an explicit include/exclude partition. Its
checker rejects ordinary source files outside that partition and validates the
exact core, PostgreSQL, PTY, and MCP `Requires-Dist`/`Provides-Extra` metadata
in both the wheel and source archive. See
[docs/release_status.md](docs/release_status.md) for the implemented scope and
remaining environment gates.

As with every version entry here, publication still requires the separately
authorized, receipt-bound process in [docs/releasing.md](docs/releasing.md).
This entry alone does not claim a tag, package-index upload, signed desktop
distribution, or completed external-provider gate.

## 1.5.0

`1.5.0` was the preceding aligned release version across the Python project,
package lockfiles, GUI package, and release-artifact workflow.

This release preserves Manifest v1/v2 governed Tools compatibility and adds
the exact-`2026-07-28` Manifest v3 client for governed Tools, Resources,
Resource Templates, Prompts, Completion, Host-owned OAuth, MRTR, pinned remote
Tasks, and bounded subscriptions. It also introduces the explicit RuntimeStore
schema-v6-to-v7 migration and tightens Tool argument, deadline, redaction,
retry-classification, registry-search, artifact, and cross-SDK conformance
contracts.

The tag, GitHub Release, and downloadable Python artifacts must remain bound to
the exact source commit, CI receipt, and checksums recorded by the separately
authorized process in [docs/releasing.md](docs/releasing.md). This entry does
not claim a PyPI upload, signed desktop distribution, or completion of the
environment gates listed in the release status.

## 1.4.2 — prior release candidate

`1.4.2` was the previously aligned release-candidate version. This historical
entry does not claim that a tag, package-index upload, or GitHub release exists.
