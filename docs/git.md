# Git Provider and Primitive

Agent libOS exposes Git as a first-class `Runtime.git` primitive backed by the
Host system Git. The boundary is deliberately typed: model tools select a
documented operation and structured parameters, never an arbitrary Git argv,
URL, credential, refspec, hook, or configuration value.

This release covers the Python Runtime and model tool surfaces only. It does
not add Git commands to the CLI, GUI, or HTTP API, and it does not integrate
with GitHub, GitLab, or another hosting service.

## Repository scope and availability

The provider is pinned to the Runtime workspace root. That path must already
be the root of a non-bare Git repository; the provider does not search parent
directories, initialize a repository, or clone one. `worktree_id="main"`
selects the root checkout. Other ids map deterministically to
`<managed-worktree-root>/<id>` and are accepted only after structural checks:
the candidate and Git-file path must not traverse a link/reparse indirection,
Git must report the expected worktree identity, and the metadata root must be
the repository common directory or an explicitly configured
`git.trusted_metadata_roots` entry. This is structural containment and Host
trust, not a persistent proof that this Runtime instance originally created the
worktree. Linked worktrees outside those roots are omitted from process-visible
lists and cannot be selected.

Git is optional at Runtime startup. `git.enabled: false`, a missing executable,
or a version older than `git.minimum_version` leaves the rest of the Runtime
usable. Git calls then fail with a stable `git_unavailable` or
`unsupported_git_version` error. The default minimum is Git 2.26.0 because safe
effective-configuration inspection requires `git config --show-scope`. Both
SHA-1 and SHA-256 repositories are supported.

At every call the provider verifies the workspace root, worktree, Git directory,
common directory, object format, and filesystem identity. It rejects:

- parent-repository discovery and bare workspace roots;
- a symlinked `.git`, repository config, attributes source, or metadata path;
- a forged or untrusted external gitfile, object alternate, or linked-worktree
  metadata root;
- worktrees outside the configured workspace/metadata trust boundary;
- repository identity changes before, during, or after an operation.

Managed worktrees are created below `git.worktree_root`, which must resolve to
a proper subdirectory of the workspace and outside `.git`. The Runtime creates
the path and opaque id; the model cannot supply an arbitrary destination. The
provider adds that root to the repository-local `info/exclude` without editing
the tracked `.gitignore`. Removal is explicit, and unknown or dirty worktrees
are never automatically deleted. Listing uses the configured managed-root
identity rather than following a later symlink replacement, so an external
checkout cannot become visible by redirecting that root.

Preparing a create may materialize the managed-root directories and add the
`info/exclude` entry before `git worktree add` succeeds. Those repository-local
metadata effects are therefore part of failure reconciliation even when no
managed worktree is ultimately listed; callers must not remove them through
ordinary filesystem tools.

## Public Runtime and tool surface

Every method below has a synchronous form on `Runtime.git` and an asynchronous
form prefixed with `a`, such as `status`/`astatus` and `push`/`apush`. All model
arguments use strict Pydantic schemas with unknown fields rejected.

For `git_branch`, `git_tag`, `git_stash`, and `git_worktree`, the model-facing
JSON schema names the discriminator `operation`; the accepted values are
`create/delete/rename`, `create/delete`, `push/apply/pop/drop/clear`, and
`create/remove`, respectively. The legacy input spelling `action` is accepted
for compatibility and is also the keyword used by the lower-level Python
Runtime methods, but it is not emitted in the model-facing schema.

| Category | Model tools |
| --- | --- |
| Inspect | `git_repository_info`, `git_status`, `git_diff`, `git_log`, `git_show`, `git_blame`, `git_list_refs`, `git_list_remotes`, `git_list_worktrees` |
| Local change | `git_stage`, `git_unstage`, `git_commit`, `git_restore`, `git_branch`, `git_switch`, `git_tag`, `git_integrate`, `git_stash`, `git_reset`, `git_clean`, `git_worktree` |
| Patch | `git_create_patch`, `git_apply_patch` |
| Remote | `git_fetch`, `git_pull`, `git_push` |
| Simulated pull request | `git_create_pull_request`, `git_list_pull_requests`, `git_inspect_pull_request`, `git_review_pull_request`, `git_merge_pull_request`, `git_close_pull_request` |

`git_integrate` accepts only `merge`, `rebase`, `cherry_pick`, `revert`, and
`abort`. Pull defaults to `ff_only` and also supports `merge` and `rebase`.
Push requires explicit remote and local refs (remote deletion omits the local
ref). A forced update is available only as force-with-lease with the exact
expected remote OID; naked force is not an interface option.

There is no typed operation for init, clone, remote/config mutation, arbitrary
argv, submodule update, LFS, signing, bisect, GC/maintenance, plumbing, custom
upload/receive pack, or model-selected credentials.

## Results, paths, state, and bounds

Public dataclasses include `GitRepositoryInfo`, `GitStateToken`, `GitPath`,
`GitStatusEntry`, `GitStatusResult`, `GitDiffResult`, `GitRef`, `GitCommit`,
`GitRemoteInfo`, `GitWorktreeInfo`, `GitOperationResult`, `GitPatchArtifact`,
and `GitPullRequest`.

Git output is captured and parsed as bytes. A `GitPath` contains a display
string plus a base64 token for the exact repository-relative bytes. When a path
is not valid UTF-8, `lossy` is true and callers must return `path_b64` instead
of reconstructing the path from `display`. Literal pathspecs and `--` are used
for path-bearing commands, so a filename beginning with `-`, containing a
newline, or using non-UTF-8 bytes cannot become an option.

List and content responses declare `truncated`, byte count, and SHA-256. The
normal result limit may produce an explicitly truncated result where the
operation supports it; crossing the configured hard limit raises
`output_too_large`. A patch artifact must fit in full—no partial Object is
created.

Every mutation accepts an opaque 64-hex `expected_state_token` obtained from a
prior repository read. The token commits to repository/worktree identity,
HEAD/ref state, index, effective configuration, refs, worktree registry,
simulated-PR metadata, and bounded worktree content state. Mutation acquires a
cross-process repository lock and compares the token again immediately before
dispatch. Drift returns `stale_state`; success returns the post-operation
observation token in `GitOperationResult.after`. It is a deterministic digest,
not a freshly generated nonce, so a verified no-op or a remote-only mutation
whose local inputs are unchanged may return the same hex value.

Only an initial state-token comparison that fails before the first
state-changing provider effect certifies that the invocation did not mutate
Git. A later `stale_state` can follow an earlier phase or a successful Host
command and is a possibly dispatched outcome: inspect and reconcile before any
retry rather than classifying it from the error code alone.

Repository reads as well as mutations acquire the same bounded cross-process
repository lock, so either kind of operation can fail with `repository_busy`.
Worktree access then acquires canonical filesystem-label locks in repository →
file-label order and retains them through final label refresh and settlement.
Reads that observe checkout bytes lock their normalized paths, or the selected
worktree root for a whole-tree read. Any mutation that may write or delete
checkout bytes conservatively locks that whole worktree root even when its
paths are enumerable; staged/ref-only operations may need no file-label lock.
The repository lock is reentrant within one thread, but extensions must not
invert this ordering.

Refs are restricted to validated full refs, strict branch/tag names, or exact
object ids resolved by the provider. Model-supplied revision expressions,
path/ref ambiguity, option injection, and arbitrary refspecs are not accepted.

## Capability and approval model

Git authority is independent of tool visibility and legacy Shell grants:

| Resource | Rights and use |
| --- | --- |
| value of `git.repository_resource` (default `git:workspace`) | `read`, `diff`, `write`, `delete`, and `admin` for the fixed repository |
| `git_remote:workspace:<remote>` | `read` for fetch/pull input, `write` for push, and `delete`/`admin` for deletion or force-with-lease |
| `git_pr:workspace:<pr-id>` | `read`, `write`, `approve`, and `delete` for one simulated PR; wildcard read is used for listing |
| `object_namespace:<process-namespace>` | `write` when `git_create_patch` stores a new immutable patch Object in the caller's process namespace |
| `object:<patch-oid>` | `read` when `git_apply_patch` resolves and verifies an existing patch Object |

Remote capability constraints may bind `git_remote`, `git_url_fingerprint`,
`git_allowed_refs`, `git_expected_state_token`, and `git_old_oid`. These are
matched against primitive-built operation context, not raw model assertions.
The primitive validates and canonicalizes the remote, exact ref, expected state
token, and old OID inputs before placing them in that context;
`git_url_fingerprint` comes from the Host-observed registered remote
configuration.

Read-only Git inspection does not require filesystem capability, even when the
provider observes checkout bytes; repository Git authority, Task Authority, and
data-flow policy govern those reads. Filesystem capability is an additional
boundary for Git mutations that read, write, or delete checkout files. Exact
path mutations check each path. When a safe preflight cannot enumerate the
affected set, the mutation requires the applicable read/write/delete authority
for the selected worktree subtree. Git metadata is never authorized through
filesystem capabilities.

Some writes contain a separately protected Git read phase and therefore need
repository `read` in addition to their mutation right: commit preflight, clean
candidate capture, pull when it must resolve the current branch, push lineage/
preflight reads, and simulated-PR create/review/merge/close verification. These
reads use the ordinary capability transaction; a finite grant can be consumed
for each distinct protected read phase (push can perform more than one). A
prior state token is CAS evidence, not a substitute for this read authority.

The following actions require `delete` and `admin` authority plus a mandatory
one-use Human approval bound to the exact parameters, old state token, and
relevant old OID:

- reset, clean, amend, every restore, rebase, and every integration abort;
- branch/tag/stash/worktree/ref deletion, branch rename, and forced branch create;
- stash pop, stash including untracked files, forced switch/tag, and fetch
  prune;
- remote-ref deletion and force-with-lease;
- simulated-PR merge;
- a patch application whose preview deletes files.

Restore has an additional exact filesystem matrix. Every form has the Git
`write/delete/admin` and approval requirements above. When `worktree=true`,
each selected path also requires filesystem `write` and `delete` subtree
authority so file/tree type changes remain authorized. A staged-only restore
(`worktree=false`) requires no filesystem authority. A source-specific,
worktree-only restore is unsupported; select `staged` as well.

Ordinary commit, `git_integrate` merge/cherry-pick/revert, non-rebase pull,
and non-forced push follow the selected capability effect (`allow`, `ask`, or
`deny`). Rebase and every `git_integrate` abort instead use the mandatory
destructive controls above. A mandatory approval cannot be satisfied by a broad
unbound allow. Capability decisions, finite-use reservations, approval binding,
the pending effect, event, audit, and operation evidence use the
protected-operation lifecycle.

Git provider effects must also pass the process Task Authority Manifest. The
relevant effect classes are `git.read`, `git.mutate`, `git.fetch`, `git.push`,
and `git.pull_request`; their protected-operation descriptors use the
`primitive.git.*` namespace. Exact method boundary names such as
`runtime.git.status` and `runtime.git.commit` are included in Explain evidence.
An old `shell:git` grant confers none of these capabilities.

Capability and Task Authority are necessary but not sufficient. Git also uses
the runtime data-flow boundary:

| Effect class | Data-flow direction | Additional behavior |
| --- | --- | --- |
| `git.read` | ingress | observed repository/file/ref labels are attached to returned data and carrier state |
| `git.mutate` | bidirectional | source/carrier labels must clear the configured repository Sink; affected paths/refs are bound in canonical payload and state evidence, not created as independent Sinks |
| `git.fetch` | bidirectional | remote input contributes ingress while the local mutation clears the configured repository Sink |
| `git.push` | egress | source/carrier labels must clear the exact configured `git_remote:workspace:<remote>` Sink; the destination ref is canonically bound to that operation |
| `git.pull_request` | bidirectional | the exact PR resource is the primary Sink; create and merge additionally clear the configured repository Sink |

The primitive aggregates source and carrier labels, uses stable repository, PR,
and remote resource identities as Sinks, and canonically binds worktree,
path/ref, and state details to the operation payload. Those payload bindings do
not create additional per-path or per-ref Sinks. It revalidates source and
target state immediately before dispatch. Host Sink trust or an exact
conditional release is therefore still required when labels exceed an actual
Sink's clearance; a matching Git capability and manifest effect ceiling alone
do not authorize the flow.

## External-effect classification

Every Git provider call uses the protected-operation intent lifecycle. Read-only
provider results are classified `no_rollback_required/not_required`, with
`state_mutation=false` and `information_flow=true`. Mutations are classified
`irreversible/not_supported` with `state_mutation=true`; the provider marks
network operations as information flow, while local mutations can still acquire
information-flow evidence from earlier primitive phases. Mutation descriptors
also use the irreversible/not-supported ceiling if post-dispatch classification
fails.

The pending intent is durable before the first Host effect and advances to a
final classification when settlement succeeds. An ordinary provider error or a
post-dispatch classification/settlement failure preserves `unknown` or pending
evidence rather than implying rollback. Provider receipts and repository state
tokens are evidence for reconciliation, not permission to replay.

## Repository configuration and command hardening

Every Git subprocess uses a Host-selected executable outside the workspace and
a fixed non-interactive environment. The provider disables or neutralizes:

- pagers, optional locks, fsmonitor, untracked cache, maintenance, hooks,
  editors, merge auto-edit, signing, replace refs, submodule recursion, external
  diff/textconv, and implicit lazy fetch;
- workspace-controlled global configuration and executable lookup;
- prompts and interactive credential acquisition.

Repository configuration is treated as data, not executable authority. An
operation fails `unsafe_repository_config` before dispatch when its active
configuration or attributes select an external clean/smudge/process filter,
LFS filter, diff command, merge driver, alternate-refs command, custom SSH
command, upload/receive pack, remote helper, promisor/partial clone, shell
credential helper, repository credential helper, repository-scoped HTTP
proxy/TLS/header/cookie/redirect override, or workspace-controlled include.
Commands that materialize a target tree inspect its `.gitattributes` blobs as
well as the current worktree and index, including nested and binary attribute
files. Rebase inspects every bounded replay candidate. A checkout, blame, or
rebase therefore cannot activate a previously dormant filter, diff command, or
`textconv` driver. Hooks are redirected to an empty Host-owned directory and
commands also use `--no-verify` where applicable.

Commit author/committer identity is read from effective repository or Host Git
configuration. The model can supply only the commit message; author overrides,
environment identity, signing, and editor invocation are unavailable.

Filesystem reads/writes/deletes reject `.git` path components, the worktree
`.git` file, and Git metadata aliases. `git.protect_git_metadata` is a
compatibility/configuration field fixed to `true`; configuration validation
rejects attempts to disable this security invariant.
Recursive deletion preflights and then removes entries without following
symlinks or reparse points; it fails closed if any descendant is Git metadata,
including metadata inserted during traversal. A partial deletion is recorded
as an unknown mutation rather than certified as not started.
The typed Git provider is the only Runtime primitive that mutates repository
metadata; filesystem capability does not authorize metadata access.

For compatibility, Shell, the optional PTY module, and benchmark provenance
share the provider's repository validation for exactly six directly invoked
inspection commands: `git status`, `git status --short`,
`git branch --show-current`, `git rev-parse --show-toplevel`, `git diff`, and
`git diff --stat`. Executable matching is case-insensitive and accepts the
platform `git.exe` spelling. The commands receive the same
no-pager/no-lock/no-fsmonitor/no-external-diff and no-lazy-fetch hardening. Every
other direct Git command, including supported transparent launcher wrappers, is
rejected before shell policy or Human approval even when Shell policy is
`always_allow`.

That check is argv mediation, not an OS sandbox. An authorized interpreter,
script, or native executable can invoke Git later or modify `.git` directly as
the host user. Such authority is outside the typed Git boundary and must be
isolated by a container/WASM/service provider when those downstream effects are
not acceptable.

## Remote operations

The model supplies only an existing remote name. Fetch/pull/push use the URL in
the repository config; the model cannot supply a URL, credential, refspec,
protocol helper, or transport command. The default protocol set is HTTPS and
controlled OpenSSH. URLs with HTTPS userinfo/passwords, query/fragment data,
non-`git` SSH users, `ext::`, custom protocols, or custom helpers are rejected.
Each remote must resolve to exactly one fetch URL and one push URL. Configured
fetch refspecs may map only a branch (or the branch wildcard) into that
remote's matching `refs/remotes/<remote>/` namespace. Typed fetch/pull also
override implicit prune and tag-fetch settings; typed push disables implicit
follow-tags, push certificates, and configured push options, so repository or
Host defaults cannot broaden the approved effect.
Typed pull supplies a Host-generated exact refspec and updates only the selected
branch's matching remote-tracking ref, including when repository configuration
contains a wildcard fetch refspec.
`git_list_remotes` exposes those validated fetch refspec strings alongside the
redacted URL placeholders and hashes, so callers can determine whether a
specific pushed branch maps to a locally observable remote-tracking ref. The
values are read-only configuration observations and cannot be supplied back as
transport input.
Local `file:` remotes are disabled by default and exist only as an explicit
Host configuration option for controlled deployments and deterministic tests.

HTTPS may use standard Host credential helpers only when the helper was loaded
from system/global config, resolves to an executable outside the workspace,
and its executable identity can be hashed. `!shell` helpers and repository
helpers are forbidden. Host system/global HTTP proxy and TLS policy remains
available, while repository-local transport overrides are forbidden. SSH uses
a Host-resolved OpenSSH executable, batch mode, the inherited SSH agent when
enabled, no user config, no forwarding, and no proxy/local commands.
Authentication material is never placed in model-visible argv, tool results,
audit, events, or provider error text.

Before approval the primitive captures hashes of the fetch/push URLs, effective
configuration, credential/SSH executable identities, remote-tracking refs, and
the expected old remote OID. The provider recomputes that fingerprint after
approval and immediately before dispatch. A change returns `stale_state`.
Timeout or ambiguous transport failure remains `unknown`. Startup
reconciliation is local and query-only: it returns current repository state and,
when the configured remote remains locally valid, freshly computed
configuration/helper/ref fingerprint components as bounded evidence. The
current reconciler does not compare them with the recorded fingerprint, prove a
remote dispatch outcome, or change the `unknown` state. It does not contact the
remote, query a network-side receipt, or replay the operation.

Host-configured remotes are the only first-class Git-provider network exception
to the general rule that remote targets must be separately registered. This is
not a real GitHub/GitLab API integration and does not create hosted pull
requests.

## Patch Objects and simulated pull requests

`git_create_patch` creates an immutable `ObjectType.CODE_PATCH`. Its payload
contains the complete patch bytes, base/head/index OIDs, source state token,
changed byte-safe paths, byte count, and SHA-256. Object provenance and data
labels conservatively include observed file bindings, repository/index carriers,
and returned commits. A range patch also includes the current index carrier even
when unrelated staged content did not contribute patch bytes, so this is safe
overtainting rather than an exact-minimal contributor set. Every Git read
inherits the stable repository carrier; diff, show, log, and blame additionally
recover their operation-specific lineage and revalidate repository and label
generations before returning. A monotonic,
repository-scoped content carrier also preserves the highest classification of
every Runtime-mediated mutation. It is bound after final policy revalidation
and before the first Host effect, so renamed or deleted content, ref-only
outputs, long histories, and post-effect settlement failures cannot lose their
lineage. If the complete patch exceeds `git.patch_max_bytes` or the Object hard
limit, the call fails without creating an Object. Creating it requires Git
`diff` plus `write` on the caller's process Object namespace; successful
creation grants the caller a handle for the new immutable Object under the
ordinary Object Memory contract.

`git_apply_patch` accepts only an existing patch Object created by this
primitive. It validates the artifact hash and schema, checks the expected state,
runs `git apply --check` as a preview, determines affected/deleted paths, then
applies through the typed mutation boundary. The source Object's labels flow to
the result and affected file bindings. Resolving the artifact requires
`ObjectRight.READ` on that exact `object:<patch-oid>` independently of the Git
and filesystem rights required for application.

`git_clean` builds a bounded, hashed reconciliation manifest from `git
ls-files` observations and also hashes Git's dry-run preview before approval,
then recomputes both before dispatch. The path manifest is intentionally a
conservative candidate set: for example, without `directories=true`, it may
list files nested below an untracked directory that Git does not remove. The
preview hash and drift check bind the decision, but neither `candidate_count`
nor `changed_paths` certifies that every listed path was actually deleted.
Post-operation status and literal-path inspection are required for that claim.

Simulated pull requests are repository-local workflow records. Creation accepts
only `refs/heads/*` refs (or shorthand local branch names), resolves them through
the main checkout's repository, and requires `write` on both the PR resource and
configured repository resource. Immutable base and head snapshots live below
`refs/agent-libos/pull-requests/`; versioned
metadata and review-body hashes are written atomically below the Git common
directory. To validate those hashes and preserve the local review record, the
same metadata file also retains the PR body and each review body as plaintext.
Inspection returns the PR body but projects review entries as hashes only; this
output redaction is not encryption or a retention policy, and a Host or
repository administrator with common-directory access can read the stored
text. Creation captures base/head OIDs and patch hash. Review supports
comment, approve, and request-changes. Close retains evidence. Merge supports
fast-forward, merge commit, and squash. The selected worktree must be clean and
its live HEAD ref and OID must exactly match the recorded base, while the live
head and metadata hashes must still match their recorded values. Merge requires
PR `approve`, repository `write/delete/admin`, selected-worktree subtree
filesystem `read/write/delete`, and mandatory one-use approval. PR metadata has
persistent per-record and
collection lineage, prebound before provider writes so an unknown settlement
cannot downgrade later inspection. Creation and merge must pass both the PR
sink and repository sink clearances in one protected operation. These records
do not contact a hosting platform.

## Stable errors and recovery

`GitError.code` is stable and includes: `git_unavailable`,
`unsupported_git_version`, `not_repository`, `unsafe_repository`,
`repository_busy`, `stale_state`, `invalid_path`, `invalid_ref`,
`dirty_worktree`, `conflict`, `identity_missing`,
`unsafe_repository_config`, `auth_required`, `non_fast_forward`,
`remote_rejected`, `timeout`, `output_too_large`, `unknown_effect`,
`command_failed`, `not_found`, `already_exists`, and `unsupported`.
Model tools map these to the normal Tool error envelope and expose only the
stable Git code and operation, never raw provider stderr that might contain a
secret.

Provider-certified failures before any protected Git effect starts abandon the
pending intent and may restore a finite capability reservation. After dispatch,
timeouts, cancellation, repository identity loss, failed post-validation, and
unclassifiable outcomes retain an `unknown` effect. Startup reconciliation is
query-only; it inspects local repository refs, worktree state, simulated-PR
metadata, and locally available remote fingerprints. It does not contact a
remote or inspect a network-side receipt, and never automatically retries a
mutation or network call.

When a caller supplies a `LocalResourceProviderSubstrate`, `Runtime.open` binds
the active Runtime Git configuration atomically to both the Local Git provider
and the Shell raw-Git guard. An explicitly conflicting provider configuration,
including a Local provider subclass whose effective configuration differs,
causes open to fail closed rather than leaving a partially bound substrate.

Checkpoint restore and image commit do not capture, package, rewind, or delete
Git metadata, checkout state, managed worktrees, remote state, or simulated-PR
metadata. They report the already-recorded Git external effects. `.git` remains
excluded from image packages.
