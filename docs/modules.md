# Startup Runtime Modules

Configured startup Runtime Modules are trusted Python extensions loaded during
runtime assembly before `Runtime.open()` returns. A trusted Host may also call
`runtime.modules.load_module_manifest(...)` to load an additional module into
an already-open Runtime. Both paths extend the Host composition root; neither
gives an AgentProcess extra authority or exposes module loading as
model-controlled installation.

## Boundary

Modules are part of the host trusted computing base:

- They run in the Python interpreter, not in the Deno/JIT sandbox.
- They are loaded only from explicit manifests.
- The entrypoint source file, or its inferred Python source package, must match
  the manifest `sha256`.
- The `(module_id, manifest_sha256, source_sha256)` trust key must be trusted
  by config or CLI for normal use. The weaker digest-pair trust list accepts
  any module id with that manifest/source hash pair and is intended for local
  development.
- Loading a module never grants filesystem, shell, Object Memory, human,
  process, checkpoint, Skill, image, or JSON-RPC capabilities to a process.
- Module load, hook, and rollback failures keep raw exception text in memory
  only. Durable module rows, audit decisions, and warning-policy results carry
  a correlated text-free error envelope plus byte-count/SHA-256 diagnostic
  fingerprints, so exception-authored secrets and paths are not published.
- Import-string entrypoints are resolved to a concrete source file under the
  manifest directory without importing package code, then trusted source bytes
  are loaded fresh under an isolated module name so a previous `sys.modules`
  entry cannot satisfy a newer trusted hash. Multi-file packages use a unique
  isolated namespace for each load, so repeated loads of the same trusted
  package hash do not reuse module globals, while the namespace remains
  available for registered tools and hooks that perform runtime relative
  imports.
- Manifest and source files are read to EOF through one identity-stable,
  non-reparse descriptor with a configured byte cap before parsing, hashing,
  or import. Unix opens the lexical path from the filesystem root one component
  at a time with no-follow directory descriptors, retains the root-to-parent
  chain, and revalidates every parent/name edge after the target access. The
  only POSIX alias expansion is Darwin's fixed root aliases (`/etc`, `/tmp`,
  and `/var`) to their `/private` targets; arbitrary ancestor symlinks fail;
  Windows uses replacement-blocking `CreateFileW` handles and verifies file
  identity with `GetFileInformationByHandle`; it holds a no-delete handle chain
  from the absolute root through the selected parent so name-based Win32
  enumeration cannot be redirected by renaming an ancestor. Multi-file package
  traversal keeps every parent directory guarded, is path-count bounded, and
  hashes only the bytes read from guarded file descriptors. Source files respect
  `AgentLibOSConfig.modules.source_max_bytes`; package snapshots also respect
  `package_max_bytes` and `max_package_files`.
- A module that performs direct Python/OS I/O is outside the runtime-mediated
  data-flow guarantee. Module syscalls and providers that handle process data
  must declare a protected-operation direction and use the common SDK gate;
  Host trust in the module is not a substitute for Sink clearance.

The manifest/source SHA-256 values are unkeyed content and identity bindings,
not signatures, MACs, publisher/source-authentication proofs, or protection
against a Host or database administrator who can replace trusted configuration
and bytes together. Module trust comes from how the Host obtains, reviews, and
pins the exact digest tuple; `modules verify` computes the tuple but does not
establish its provenance or safety.

Use Skills when the model needs workflow instructions, tool visibility, or JIT
tool candidates at runtime. Use Runtime Modules when the system owner wants to
install new host-side runtime surfaces before startup.

## Manifest V1

```yaml
schema_version: 1
module_id: example-module:v0
name: Example module
version: v0
entrypoint: ./example_module.py:register_module
provides:
  tools:
    - example_echo
  images:
    - example-agent:v0
  syscalls:
    - example.ping
  provider_hooks: []
  startup_hooks:
    - initialize_example
  durable_object_release_finalizers:
    - example.provider-resource-release:v1
sha256: "<sha256 of example_module.py bytes, or the inferred package digest>"
metadata:
  owner: local
```

The unwrapped mapping above is the canonical authoring form. For compatibility,
the loader also accepts the same mapping beneath a sole top-level `module` key:

```yaml
module:
  schema_version: 1
  module_id: example-module:v0
  name: Example module
  entrypoint: ./example_module.py:register_module
  provides: {}
  sha256: "<sha256 of example_module.py bytes>"
```

The wrapper must be the only top-level key and its value must be a mapping.
After removing that optional wrapper, Manifest V1 is closed: the only allowed
fields are `schema_version`, `module_id`, `name`, `version`, `entrypoint`,
`provides`, `sha256`, and `metadata`. Unknown fields are rejected rather than
ignored. `provides` is also closed; its only allowed fields are `tools`,
`images`, `syscalls`, `provider_hooks`, `startup_hooks`, and
`durable_object_release_finalizers`.

The field contract is:

| Field | Required | V1 contract |
| --- | --- | --- |
| `schema_version` | yes | Integer `1`; booleans are not integers for this field. |
| `module_id` | yes | Non-empty identifier matching `[A-Za-z0-9][A-Za-z0-9_.:/@+-]*`. |
| `name` | yes | Non-empty display string. |
| `version` | no | Defaults to `v0` when omitted. Authors should otherwise supply a non-empty string; the current compatibility parser also normalizes an explicit falsy decoded value to `v0`. |
| `entrypoint` | yes | Non-empty `<module-or-path>:<callable>` reference resolved relative to the manifest directory. |
| `provides` | yes | Mapping, including `{}` when the entrypoint buffers no declared registration. |
| `sha256` | yes | Exactly 64 hexadecimal characters; normalized to lowercase. |
| `metadata` | no | A mapping containing only finite JSON values. Omission and explicit `null` both normalize to `{}`. |

Every `provides` child is optional. An omitted child or an explicit `null`
normalizes to an empty list; a present child must be a list of unique,
non-empty strings. Syscall, provider-hook, startup-hook, and durable-finalizer
names must match `[A-Za-z][A-Za-z0-9_.:-]*`. Tool and image declarations use
the general non-empty identifier-string validation because their concrete
registries perform the surface-specific validation during preflight.

With the default `ModuleDefaults`, the manifest string and inventory limits
are:

| Value | Default limit |
| --- | ---: |
| `module_id` | 128 characters |
| `name` | 128 characters |
| `version` | 64 characters |
| `entrypoint` | 512 characters |
| each `provides` item | 128 characters |
| `provides.tools` | 128 items |
| `provides.images` | 128 items |
| `provides.syscalls` | 128 items |
| `provides.provider_hooks` | 64 items |
| `provides.startup_hooks` | 64 items |
| `provides.durable_object_release_finalizers` | 64 items, sharing `max_declared_startup_hooks` |

These are Host-configurable positive limits. Strings are trimmed before length
checking and control characters are rejected. Duplicate mapping keys are
rejected at every YAML or JSON mapping level, and YAML mapping keys must be
strings. `metadata` has no separate item-count allowance: it is bounded by the
whole-manifest and parser limits below and must remain JSON-serializable with
no NaN or infinity.

Default file and package limits are:

| Config/default | Enforced boundary |
| --- | --- |
| `manifest_max_bytes = 262144` | Maximum manifest file bytes read by normal `resolve`, `verify`, and load paths. |
| `manifest_hard_limit_bytes = 1048576` | Maximum UTF-8 bytes accepted by `parse_manifest`; configuration requires it to be at least `manifest_max_bytes`. |
| `source_max_bytes = 1048576` | Maximum bytes for a direct entrypoint source or any selected Python source file. |
| `package_max_bytes = 8388608` | Maximum aggregate bytes of Python files included in a package snapshot; configuration requires it to be at least `source_max_bytes`. |
| `max_package_files = 256` | Maximum directory entries visited while collecting a package, not merely the number of `.py` files finally hashed. |

Package traversal charges a visited path before deciding whether to descend,
prune it, or include its bytes. There is no separate Module package-depth
setting; every descended directory consumes the same `max_package_files`
budget, so the path-count ceiling also bounds recursive depth. Only the sorted
`.py` snapshot is executed and represented in the package digest.

Manifest decoding also has fixed structural ceilings independent of
`ModuleDefaults`. YAML allows at most 64 levels, 32,768 expanded nodes, 60,000
parse events, 64 aliases, 4,300 integer digits, and 1 MiB of expanded scalar
text. JSON allows at most 256 levels, 100,000 decoded nodes, and 4,300 integer
digits. Both formats reject non-finite numbers; the YAML loader additionally
has a fixed 1 MiB UTF-8 input ceiling.

The manifest may also use an import-string entrypoint such as
`example_package.module:register_module`. Import strings and file entrypoints
are resolved relative to the manifest directory; file entrypoints cannot escape
that directory with `../`. Package import strings require each package parent to
exist under the manifest directory with an `__init__.py`, which keeps manifest
resolution from drifting to arbitrary installed packages. The package parents
are not imported or executed by the module loader.

If `sha256` matches the entrypoint source file bytes, the module is treated as a
single-file module and loaded as before. If it does not match the entrypoint
file, the loader infers a Python source package from the entrypoint, includes
only `.py` files under that package root, sorts them by manifest-relative path,
and compares the manifest `sha256` to that package digest. Multi-file modules
are imported from the verified in-memory snapshot under a synthetic package
name, so bundled helpers should use package-relative imports such as
`from .helper import make_tool`. The manifest directory is not added to
`sys.path`, and local helpers are not imported through their original absolute
package name. If import or source verification is interrupted by any
`BaseException`, including cancellation or `KeyboardInterrupt`, the synthetic
namespace and its temporary importer are removed before the interruption is
propagated.

The package reader rejects symlinks, hard links, non-regular selected entries,
path escapes, directory replacement during traversal, and likely secret
material. During recursive multi-file package collection it prunes cache/VCS
directories rather than treating their mere presence as a package error; those
pruned files are not executed or covered by the digest. A direct single-file
entrypoint is verified as that file and does not recursively apply the
package-directory pruning rule. Run
`uv run agent-libos modules verify <module.yaml>` after authoring a module to
get the current `manifest_sha256`, `source_sha256`, `trust_key`, and the file
list covered by the source digest.

YAML and JSON manifests both reject duplicate mapping keys. Treat duplicate-key
errors as authoring bugs rather than relying on parser-specific overwrite
behavior.

## Entrypoint

The entrypoint is synchronous:

```python
def register_module(ctx):
    ctx.register_tool(MyTool())
    ctx.register_image(my_image)
    ctx.register_syscall("example.ping", ping_handler)
    ctx.bind_durable_object_release_finalizer(
        "example.provider-resource-release:v1",
        prepare_release_intent,
        finalize_release_intent,
    )
    ctx.add_startup_hook(initialize_example)
```

The entrypoint's buffered `ModuleContext` records registrations first. The
registry verifies that these buffered registrations use only resources declared
in `provides`, checks name collisions, then applies them before the runtime is
returned to the caller. Each module's buffered registration, module metadata,
audit/event rows, and in-memory tool/image/syscall/hook registries use one
lifecycle transaction and registration journal.

Hooks use the same journal discipline, but startup and post-startup failure
scope differ. During `Runtime.open()`, every configured module is applied before
the combined provider/startup-hook phase begins. If any hook in that phase
fails, the Runtime rolls back **all external startup modules** from that opening
attempt, in reverse order, and records their rows as failed; it does not preserve
the otherwise-successful configured modules from that batch. The internal core
module is excluded from this rollback. After startup, a trusted Host load runs
only the newly loaded module's hooks immediately, and a failure rolls back that
module without removing modules that were already loaded successfully.

A hook receives an explicit `ModuleHookServices`-backed surface, not the
concrete Runtime or another manager's private registry. Hook code may still
perform arbitrary external trusted host-side effects that the runtime cannot
compensate, so hooks should be kept small and idempotent and should not treat a
registration rollback as an external-effect rollback.

There are two deliberately different context lifecycles:

1. The entrypoint receives a buffered `ModuleContext`. Its `runtime` member is
   only a `ModuleRuntimeView` exposing `config`; attempts to reach other Runtime
   attributes fail before preflight. The entrypoint may call
   `register_tool`, `register_image`, `register_syscall`,
   `register_provider_hook`, `add_startup_hook`, and
   `bind_durable_object_release_finalizer`. Every registered name/id must appear
   in the corresponding manifest `provides` list. The registry verifies that
   each public registration buffer still contains exactly the objects admitted
   through those methods, so direct list/dict mutation cannot bypass
   `provides` before preflight or application. A tool's concrete publication
   name must also equal the name admitted when `register_tool` ran; mutating a
   tool implementation after admission fails and rolls back its publication.
2. After buffered registrations pass preflight and are applied, each declared
   provider/startup hook receives a `ModuleHookContext`. The context is backed
   by `ModuleHookServices`, and every supported registration or installed
   module-owned attribute records an inverse in the module's registration
   journal. Hook-time registrations are trusted TCB behavior and are not checked
   against the entrypoint-oriented `provides` inventory; the journal provides
   lifecycle rollback, not declarative least-authority enforcement. Modules that
   need a complete supply-chain inventory should still declare and review those
   dynamic surfaces explicitly.

`ModuleHookServices.from_host(...)` is the assembly/test boundary that captures
the explicit services used to build hook contexts. Its current public fields
are `config`, `workspace_root`, `audit`, `events`, `shell`, `human`,
`resources`, `data_flow`, `operations`, `memory`, `process`, `capability`,
`protected_operations`, `store`, `substrate`, `images`, `tools`,
`image_registry`, `syscalls`, `provider_hooks`, `lifecycle`, `state`, and
`add_handle_to_process_view`. Normal module code receives the narrower
`ModuleHookContext`, not this dataclass or the concrete Runtime.

The hook context exposes read/operational properties for `module_id`, `actor`,
`config`, `workspace_root`, `audit`, `events`, `shell`, `human`, `resources`,
`data_flow`, `operations`, `memory`, `process`, `capability`,
`protected_operations`, `store`, `substrate`, and a deep-copied read-only
`images` mapping. `substrate` rejects attribute writes; module-owned additions
must use `set_substrate_attribute`. `memory` blocks private/non-journaled bind
access while forwarding ordinary public operations and exposing journaled
Object finalizer binding.

Hook-only mutation and lifecycle methods are:

- `register_tool(..., scope=None, ephemeral=False)`, `register_image(...,
  source=None)`, `register_syscall(...)`, and `register_provider_hook(...)`;
- `bind_shutdown_finalizer(...)`, `bind_recovery_cleanup(...)`,
  `bind_object_release_finalizer(...)`, and
  `bind_durable_object_release_finalizer(...)`;
- `set_runtime_attribute(...)` for new names beginning `_agent_libos_`, and
  `set_substrate_attribute(...)` for new public substrate attributes;
- `get_runtime_attribute(...)`, `add_handle_to_process_view(...)`, and
  `require_recovery_cleanup_lease()` for installed adapters.

The registry deactivates the hook context in a `finally` block as soon as that
synchronous hook returns or raises. Registration, binding, and setter methods
then reject later calls. Installed adapters may retain the context for its
operational services; `get_runtime_attribute`, `add_handle_to_process_view`,
and the lifecycle-scoped `require_recovery_cleanup_lease` remain callable.
Hooks must not return awaitables. If a hook or its evidence sink fails, journal
rollback removes its owned registrations in reverse order; this rollback does
not compensate arbitrary external I/O performed directly by trusted Python.

The runtime serializes module resolution, import, buffered application, hook
execution, rollback, failed-record publication, and shutdown cleanup with the
same re-entrant registry lifecycle lock used by ToolBroker and ImageRegistry.
A failed module rollback therefore cannot erase a successful module or registry
publication that raced with it. Module list limits must be positive
integers no larger than `ModuleDefaults.discover_limit`.

`Runtime.open()` loads the internal core module plus manifests selected through
`AgentLibOSConfig.modules.manifest_paths` and its explicit `module_manifests`
argument, and runs all of their startup hooks before returning. This timing is a
guarantee for configured startup modules, not a ban on trusted Host
reconfiguration. If Host code calls
`runtime.modules.load_module_manifest(...)` after startup has completed, that
module's provider and startup hooks run immediately; failure rolls back the
newly loaded module.

`AgentLibOSConfig.modules.load_policy` controls ordinary per-manifest load
failure handling:

- `fail` (the default) records a bounded failed-module diagnostic and raises the
  error.
- `warn` records the same diagnostic and returns a `status: failed` result, so
  startup manifest iteration can continue. It also changes a post-startup
  dynamic module/hook failure into that failed result after rolling back the new
  module.

`load_policy: warn` does **not** make the later combined startup-hook phase
optional. A provider or startup hook failure while `Runtime.open()` is running
still rolls back the external startup-module batch and makes `Runtime.open()`
fail. This preserves the contract that `Runtime.open()` never returns a Runtime
whose configured startup hooks only partially ran.

## Registration Surfaces

`ctx.register_tool(tool)` registers a static model-facing tool through
`ToolBroker`. Tool visibility remains separate from resource authority.

`ctx.register_image(image)` registers an `AgentImage` through the image
primitive validation path. Image `required_capabilities` are applied only by
normal process bootstrap rules and never by module loading itself. Image
`required_modules` can declare startup module prerequisites as
`{module_id, source_sha256}` pairs; process spawn and exec fail closed unless
the current runtime has already loaded those exact module ids and source
hashes. The image declaration does not load modules and does not grant any
process authority.

`ctx.register_syscall(name, handler)` adds a module syscall to the syscall
router. The handler receives the `LibOSSyscallSession` and syscall args. It
must call existing primitives for protected effects.

`ctx.register_provider_hook(kind, hook)` records one trusted provider hook for a
declared provider hook kind. Hooks are startup module code, not process
capabilities.

`ctx.add_startup_hook(fn)` runs a synchronous hook after all startup module
registrations have been applied.

Inside a trusted startup hook,
`ctx.bind_recovery_cleanup(fn)` registers an idempotent, process-local
teardown callback for recovery-diagnostics handoff. This is deliberately
separate from `bind_shutdown_finalizer`: ordinary shutdown callbacks are never
implicitly recovery-safe and are not called by handoff. A recovery cleanup may
stop workers and close transient host handles only; it must not read or mutate
Object Memory, the runtime store, audit, or events. It returns `False` until
cleanup has fully converged, which retains the registration and keeps the store
open for an exact retry. Async callbacks require
`await runtime.arelease_recovery_diagnostics()`.

`ctx.bind_durable_object_release_finalizer(id, prepare, finalize)` registers a
manifest-declared, restart-stable Object cleanup handler during buffered module
application. Unlike a startup-hook registration, it is available before
checkpoint publication recovery. `prepare` must be side-effect free and return
a bounded JSON mapping; `finalize` is at-least-once and receives the persisted
intent plus the same idempotency key on every retry. Handler ids are durable
protocol identifiers: changing their meaning requires a new id.

## PTY Module

`modules/pty/module.yaml` is the standard trusted module for interactive
terminal sessions. When loaded and trusted, it registers the tools
`pty_create`, `pty_read`, `pty_write`, `pty_resize`, `pty_close`, and
`pty_list`, plus the `pty-agent:v0` image. The adapter, local PTY provider,
reader thread, buffer limits, and timeout/window defaults live inside this
module, not in the core Runtime or default Resource Provider Substrate.
The module files are shipped in the repository and Python source distribution,
not in the core wheel. Installing the wheel's optional `pty` dependency extra
can provide `pywinpty` on Windows, but does not install `modules/pty`; a
wheel-only deployment must separately supply and trust the module package.
The adapter receives the explicit module host services it uses and shares
Shell launch authorization only through the public `ShellExecutionPolicy`
surface; it does not retain the concrete Runtime or call `ShellAdapter` private
methods.
Spawn, read, write, resize, close, automatic exit cleanup, and compensating
close phases use the same [Protected Operation SDK](protected_operation_sdk.md)
contracts as core providers; the module does not manage effect intents or
finite-use reservations through a private lifecycle.
The declared data-flow directions are:

| Operation | Direction | Flow behavior |
| --- | --- | --- |
| public `pty_create` / `primitive.pty.spawn` | bidirectional | `argv` plus `cwd` egress to the resolved executable Sink; provider/session output is external ingress |
| `pty_read` / `primitive.pty.read` | ingress | returned output/status observes the session's accumulated ingress context |
| internal `primitive.pty.ingest` | ingress | the continuous reader drains provider output under the session ingress context |
| `pty_write` / `primitive.pty.write` | egress | text goes to the pinned session Sink and raises the session label high-water on success or ambiguous dispatch |
| `pty_resize` / `primitive.pty.resize` | egress | dimensions are a control payload to the pinned session Sink |
| public `pty_close` / `primitive.pty.close` | egress | force/timeout controls go to the pinned session Sink; lifecycle-only internal close has runtime-internal authority rather than a process payload |
| `pty_list` | local ingress observation | no provider call/effect is created; labels from all returned readable sessions are aggregated into the caller context |

`pty_list` is provider-side-effect-free and idempotent with respect to PTY
state, but it is not authority-free: each returned session performs the normal
Object `read` check, so a finite read grant may be consumed and a later list may
return fewer sessions. It also records its audit observation and raises the
caller's ingress context as described above.

The session
pins the resolved executable identity and Sink trust snapshot established at
spawn; a later write cannot switch the session to another trusted executable.
Every successful write raises the session's data-flow high-water mark and the
published session Object labels. A write that crossed the provider boundary
but returned an ambiguous error raises the same high-water mark
conservatively. Later clean-looking PTY output/read results therefore retain
all sensitivity and source references ever written to that session.
The continuous reader is one runtime-internal `pty.ingest` child operation for
the lifetime of the session. It drains and probes the provider handle only
inside that SDK phase, records one information-flow effect when it stops, and
does not infer causality from thread timing.
During an explicit recovery-diagnostics handoff, the PTY module's separately
tagged recovery cleanup stops the reader and resource monitor, closes the live
provider handle directly, and waits for both workers without publishing close,
audit, event, Object, or operation state. The durable PTY Object remains for
the next Runtime's existing stale-session recovery. A close or join failure
keeps the in-memory session registered and the store open so handoff can retry;
the ordinary PTY shutdown finalizer is not used on this path.
The raw handle close is guarded by an opaque lifecycle ContextVar lease that is
present only while RuntimeLifecycle invokes the registered cleanup. Direct
adapter calls fail before changing session state. The protected-operation
static ratchet recognizes only a leading instance-host lease check followed by
`handle.close()`; reads, writes, resizes, provider methods, late checks, and
similarly named guards remain rejected.

`pty_create(argv, cwd=None, cols=None, rows=None, startup_timeout_s=None,
max_output_chars=None, name=None)` launches a local PTY through the shell
primitive's argv validation, workspace cwd checks, shell policy, human approval,
resource budget, provider classification, events, and audit path. `None`
values use `PtyModuleSettings` defaults such as `default_cols`,
`default_rows`, `startup_timeout_s`, and `startup_output_max_chars`. It returns
a mutable Object Memory `EXTERNAL_REF` object id as `session_oid`; the payload
is descriptive metadata only and is not an authority source.
If provider spawn succeeds but any later registration, reader startup,
event/audit, or effect-recording step fails, the adapter attempts to close the
host handle, remove the in-memory session, and release the object before
returning failure. Cleanup is best effort: an ambiguous close retains an
unresolved session/Object and its evidence instead of pretending that the host
resource was removed.

A finite-use shell-policy decision is reserved before the SDK's
`provider.spawn` phase.
The adapter then persists a structured pending spawn-effect intent before
`provider.spawn`. `ProviderEffectNotStarted` certifies only that this spawn
phase did not create a child. Reservation restoration and intent abandonment
are permitted only when no earlier effectful phase has already observed
information; for example, a preceding resolver observation still commits the
reservation and retains conservative evidence. Any other spawn exception
commits the use and conditionally finalizes the same effect id as `unknown` when the result
sinks succeed; a post-provider sink failure leaves it pending. Successful spawn
followed by setup/classification failure is contained by closing the handle and
removing the Object, but the already-crossed provider outcome remains
conservatively `unknown`; cleanup is not proof that the process had no external
effect. If session Object creation is the failed phase, the recorded metadata
names `session_object_creation` and includes whether cleanup was attempted and
succeeded, so operators can distinguish containment from non-execution.

The local PTY backend resolves bare executables on a safe host PATH that
excludes workspace entries, rejects workspace PATH hijacks, and gives child
processes a workspace-scoped `HOME`/`USERPROFILE`.

On POSIX, PTY resource accounting runs in a monitor worker independent of the
blocking output reader, so a provider read cannot suspend wall/CPU/RSS
enforcement. Each successful metrics sample covers the complete process tree.
CPU is accumulated by `(pid, create_time)` and retains each process's maximum
observed total, so an exited child cannot make aggregate charged CPU decrease;
wall time is charged cumulatively and RSS records the session peak. If
process-tree discovery or CPU/RSS inspection is denied, the adapter closes the
session and releases its Object handle instead of continuing without the
required accounting. There is one narrower exception: a
strictly wall-only budget has no CPU or memory value to enforce, so a denied
process-metrics sample does not stop monotonic wall charging; wall overage still
kills/closes the session. That mode provides no CPU/RSS accounting claim.
Cleanup signals the process group first; if that is denied it explicitly
signals the discovered descendant tree, and surfaces cleanup failure rather
than reporting an uncontained session as closed.

The current Windows backend is only the `pywinpty`/ConPTY session adapter. It
does not install a Job Object, process-tree wall/CPU/RSS supervisor, or
parent-death containment. `LocalPtyProvider.supports_subprocess_limits` is false
there, so any process budget that produces `SubprocessLimits` is rejected before
spawn rather than run without accounting. Unbudgeted PTY use still has the
module's output/startup bounds and explicit close lifecycle, but deployments
must not treat it as process-tree resource containment.

Follow-on tools use that `session_oid` as the public handle:

- `pty_read(session_oid, timeout_s=0, max_chars=32000)` requires object `read`.
- `pty_write(session_oid, text)` requires object `write` and the original
  session owner pid.
- `pty_resize(session_oid, cols, rows)` requires object `write`.
- `pty_close(session_oid, force=True, timeout_s=2)` requires object `delete`,
  closes the host PTY, releases the object, and revokes its object
  capabilities.
- `pty_list()` returns active sessions whose PTY object is readable by the
  caller.

Finite object write/delete decisions used by `pty_write`, `pty_resize`, and
`pty_close` are reserved before the provider call. If the first operation
boundary certifies `ProviderEffectNotStarted`, the reservation and pending
intent are restored/abandoned together. Ordinary exceptions, partial writes,
or any failure after provider information has been observed commit the use and
leave finalized-unknown or pending evidence rather than making a one-use PTY
handle reusable.

`pty_write`, `pty_resize`, and `pty_close` each persist a pending effect before
their provider operation and CAS the same id to a final event/audit-linked
record afterward. `ProviderEffectNotStarted` abandons the pending row and
restores an exact finite-use reservation only when the first boundary did not
start. An ordinary provider exception best-effort finalizes `unknown`; if
event/audit/effect settlement itself fails, the dispatched pending intent
remains durable. The local provider classifies write and close as
irreversible/not-supported and resize as rollbackable/not-applied; v1 records
that classification but does not perform compensation. If a provider does not
classify one of those operations, or its classifier raises after the operation,
the module finalizes an `unknown`/`unknown` fallback rather than losing the fact
that the host boundary was crossed. An ambiguous close keeps the Object/session
registered but marks its close outcome unresolved; automatic cleanup and a
second close do not blindly call the provider again.

PTY sessions are memory-resident host resources. They are closed by explicit
`pty_close`, object release, process-owned memory release on process exit,
runtime shutdown, or PTY child process exit. Reopening a runtime does not
reconnect old PTY objects; stale PTY `EXTERNAL_REF` rows are released during
startup.

Automatic child-exit cleanup is itself a close provider boundary. The monitor
persists a close intent before reading the exit code or calling `close()`. A
failure before either observation can abandon only when certified not-started;
after the exit-code read, a later close failure or not-started result retains
the pending information-flow intent. Event/audit failure after successful
automatic close likewise leaves that intent pending for reconciliation.

The internal close state machine is idempotent under races. The first closer
sets `closing`, later lifecycle/resource/shutdown closers wait on the same
completion event, and only the caller that removes the registered session emits
the close event/audit record. A timeout or provider close failure is surfaced;
the module does not report `closed` while a handle is still uncontained. Once
the public Object/session has been removed, a later `pty_close` has no authority
target and returns the normal not-found failure.

Tests and hosts can override module defaults by setting `substrate.pty_settings`
to a mapping before loading the module, and can inject a fake provider by
setting `substrate.pty`. Without that injection, the module constructs its own
local provider from the runtime workspace root. On Windows, the real backend
uses `pywinpty`; install it through the optional `pty` extra when real ConPTY
support is needed:

```bash
uv sync --frozen --extra pty
```

## CLI

Verify a manifest without loading it:

```bash
uv run agent-libos modules verify modules/pty/module.yaml
```

The checked-in project-root `config.yaml` already lists and trusts the PTY
module, so list it without repeating the manifest on the command line:

```bash
uv run agent-libos --db local modules list
```

On a Host whose configuration does not already list that manifest, pass
`--module-manifest modules/pty/module.yaml` together with the exact
`--trusted-module <trust_key>` returned by `modules verify`. Do not combine a
config entry and a CLI entry for the same manifest: duplicate module ids are
rejected during startup.

`modules verify` returns the exact `trust_key` value accepted by
`--trusted-module`. The weaker
`--trusted-module-sha256 <manifest_sha256>:<source_sha256>` trusts any module id
with that manifest/source digest pair and is intended for local development
only.

Every command that needs module-provided images, tools, or syscalls must pass
the same startup module configuration, or use an application-level config that
sets `AgentLibOSConfig.modules.manifest_paths` and trusted hashes. Relative
paths in `modules.manifest_paths` are resolved from the project root, not from
the process current working directory.

## Persistence And Checkpoints

The `runtime_modules` table records loaded and failed modules, source hashes,
entrypoints, and registration summaries. This is audit and reproducibility
metadata, not a dynamic loading authority for processes. A failure that occurs
before the manifest can be verified uses a `failed:<manifest-filename>`
placeholder id; `entrypoint`, manifest/source hashes, and `source_path` may be
empty, `registered` may contain only empty collections, and `loaded_at` is
`null`. Consumers must branch on `status` rather than treating those fields as
present identity evidence. A `module_id` can be loaded only once per `Runtime`;
duplicate load attempts are rejected without overwriting an already loaded row.

Checkpoint snapshots record the currently loaded module summaries. Restore and
fork fail if the current Python runtime has not loaded the same required module
ids and source hashes. Checkpoint restore does not load Python modules and does
not roll back the module environment.

Checkpoint-committed AgentImages also copy those module summaries into image
`required_modules`, so the committed image itself records the same startup
module prerequisite. Use `modules verify <module.yaml>` to get the
`source_sha256` value for hand-authored image manifests and packages.
