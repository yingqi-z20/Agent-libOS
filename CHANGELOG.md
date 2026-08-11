# Changelog

This changelog starts with the currently maintained release-candidate line. It
does not reconstruct older release notes or infer publication dates from Git
history. Git history remains the record for earlier development snapshots.

## Unreleased

Changes intended for the next published version must be summarized here before
release. Do not treat an entry in this section as shipped behavior.

## 1.5.0 — release candidate

`1.5.0` is the version currently aligned across the Python project, package,
lockfiles, GUI package, and release-artifact workflow. Its implemented scope,
validation contract, and remaining environment gates are maintained in
[docs/release_status.md](docs/release_status.md).

This candidate preserves Manifest v1/v2 governed Tools compatibility and adds
the exact-`2026-07-28` Manifest v3 client for governed Tools, Resources,
Resource Templates, Prompts, Completion, Host-owned OAuth, MRTR, pinned remote
Tasks, and bounded subscriptions. It also introduces the explicit RuntimeStore
schema-v6-to-v7 migration and tightens Tool argument, deadline, redaction,
retry-classification, registry-search, artifact, and cross-SDK conformance
contracts.

This heading does not claim that a tag, PyPI upload, GitHub release, or current
CI receipt exists. Publication requires the separately authorized process in
[docs/releasing.md](docs/releasing.md).

## 1.4.2 — prior release candidate

`1.4.2` was the previously aligned release-candidate version. This historical
entry does not claim that a tag, package-index upload, or GitHub release exists.
