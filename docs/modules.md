# Startup Runtime Modules

Runtime Modules are trusted Python extensions loaded before `Runtime.open()`
returns. They are for extending the runtime composition root, not for giving an
AgentProcess extra authority.

## Boundary

Modules are part of the host trusted computing base:

- They run in the Python interpreter, not in the Deno/JIT sandbox.
- They are loaded only from explicit manifests.
- The entrypoint source file must match the manifest `sha256`.
- The `(module_id, source_sha256)` pair must be trusted by config or CLI.
- Loading a module never grants filesystem, shell, Object Memory, human,
  process, checkpoint, Skill, image, or JSON-RPC capabilities to a process.
- Import-string entrypoints are resolved to a concrete source file under the
  manifest directory without importing package code, then that source file is
  loaded fresh under an isolated module name so a previous `sys.modules` entry
  cannot satisfy a newer trusted hash.
- Module source files are hashed with a configured size limit
  (`AgentLibOSConfig.modules.source_max_bytes`) before import.

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
sha256: "<sha256 of example_module.py bytes>"
metadata:
  owner: local
```

The manifest may also use an import-string entrypoint such as
`example_package.module:register_module`. Import strings and file entrypoints
are resolved relative to the manifest directory; file entrypoints cannot escape
that directory with `../`. Package import strings require each package parent to
exist under the manifest directory with an `__init__.py`, which keeps manifest
resolution from drifting to arbitrary installed packages. The package parents
are not imported or executed by the module loader; the trusted hash covers the
entrypoint source file.

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
    ctx.add_startup_hook(initialize_example)
```

The module context buffers registrations first. The registry verifies that the
module registered only declared resources, checks name collisions, then applies
the registrations before the runtime is returned to the caller. If registration
or startup hooks fail, external module registrations are rolled back from the
runtime registries and the module row is recorded as failed. Hook code may still
perform arbitrary trusted host-side effects, so hooks should be kept small and
idempotent.

`Runtime.open()` runs all configured startup hooks before returning. If host code
manually calls `runtime.modules.load_module_manifest()` after startup has already
completed, that module's provider and startup hooks run immediately; failure
rolls back the newly loaded module.

## Registration Surfaces

`ctx.register_tool(tool)` registers a static model-facing tool through
`ToolBroker`. Tool visibility remains separate from resource authority.

`ctx.register_image(image)` registers an `AgentImage` through the image
primitive validation path. Image `required_capabilities` are applied only by
normal process bootstrap rules and never by module loading itself.

`ctx.register_syscall(name, handler)` adds a module syscall to the syscall
router. The handler receives the `LibOSSyscallSession` and syscall args. It
must call existing primitives for protected effects.

`ctx.register_provider_hook(kind, hook)` records one trusted provider hook for a
declared provider hook kind. Hooks are startup module code, not process
capabilities.

`ctx.add_startup_hook(fn)` runs a synchronous hook after all startup module
registrations have been applied.

## PTY Module

`modules/pty/module.yaml` is the standard trusted module for interactive
terminal sessions. When loaded and trusted, it registers the tools
`pty_create`, `pty_read`, `pty_write`, `pty_resize`, `pty_close`, and
`pty_list`, plus the `pty-agent:v0` image. The adapter, local PTY provider,
reader thread, buffer limits, and timeout/window defaults live inside this
module, not in the core Runtime or default Resource Provider Substrate.

`pty_create(argv, cwd=None, cols=80, rows=24, startup_timeout_s=0.2,
max_output_chars=4000, name=None)` launches a local PTY through the shell
primitive's argv validation, workspace cwd checks, shell policy, human approval,
resource budget, provider classification, events, and audit path. It returns a
mutable Object Memory `EXTERNAL_REF` object id as `session_oid`; the payload is
descriptive metadata only and is not an authority source.

Follow-on tools use that `session_oid` as the public handle:

- `pty_read(session_oid, timeout_s=0, max_chars=32000)` requires object `read`.
- `pty_write(session_oid, text)` requires object `write`.
- `pty_resize(session_oid, cols, rows)` requires object `write`.
- `pty_close(session_oid, force=True, timeout_s=2)` requires object `delete`,
  closes the host PTY, releases the object, and revokes its object
  capabilities.
- `pty_list()` returns active sessions whose PTY object is readable by the
  caller.

PTY sessions are memory-resident host resources. They are closed by explicit
`pty_close`, object release, process-owned memory release on process exit,
runtime shutdown, or PTY child process exit. Reopening a runtime does not
reconnect old PTY objects; stale PTY `EXTERNAL_REF` rows are released during
startup.

Tests and hosts can override module defaults by setting `substrate.pty_settings`
to a mapping before loading the module, and can inject a fake provider by
setting `substrate.pty`. Without that injection, the module constructs its own
local provider from the runtime workspace root. On Windows, the real backend
uses `pywinpty`; install it through the optional `pty` extra when real ConPTY
support is needed.

## CLI

Verify a manifest without loading it:

```bash
uv run agent-libos modules verify modules/example/module.yaml
```

Load a trusted module before a command:

```bash
uv run agent-libos \
  --module-manifest modules/example/module.yaml \
  --trusted-module example-module:v0:<source_sha256> \
  modules list
```

The weaker `--trusted-module-sha256 <sha256>` trusts any module id with that
source hash and is intended for local development only.

Every command that needs module-provided images, tools, or syscalls must pass
the same startup module configuration, or use an application-level config that
sets `AgentLibOSConfig.modules.manifest_paths` and trusted hashes.

## Persistence And Checkpoints

The `runtime_modules` table records loaded and failed modules, source hashes,
entrypoints, and registration summaries. This is audit and reproducibility
metadata, not a dynamic loading authority for processes. A `module_id` can be
loaded only once per `Runtime`; duplicate load attempts are rejected without
overwriting an already loaded row.

Checkpoint snapshots record the currently loaded module summaries. Restore and
fork fail if the current Python runtime has not loaded the same required module
ids and source hashes. Checkpoint restore does not load Python modules and does
not roll back the module environment.
