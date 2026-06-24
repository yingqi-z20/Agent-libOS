from __future__ import annotations
import pytest
import asyncio
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.api.cli import main as cli_main
from agent_libos.models.exceptions import CapabilityDenied, NotFound, ValidationError
from agent_libos.modules.loader import ModuleLoader
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import LocalResourceProviderSubstrate

class TestRuntimeModule:

    def test_trusted_startup_module_registers_tool_image_syscall_and_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                loaded = runtime.modules.inspect_module('test-module:v0')
                assert loaded['status'] == 'loaded'
                assert 'module_echo' in loaded['registered']['tools']
                assert 'module-agent:v0' in runtime.images
                assert runtime.syscalls.get('module.ping') is not None
                assert any((record.action == 'test.startup_hook' for record in runtime.audit.trace()))
                pid = runtime.process.spawn(image='module-agent:v0', goal='use module')
                tool_result = runtime.tools.call(pid, 'module_echo', {'text': 'hello'})
                assert tool_result.ok
                assert tool_result.payload['echo'] == 'hello'
                syscall_result = asyncio.run(LibOSSyscallSession(runtime, pid).handle('module.ping', {'value': 'ok'}))
                assert syscall_result['value'] == 'ok'
                assert syscall_result['pid'] == pid
            finally:
                runtime.close()

    def test_provider_hook_runs_before_runtime_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, provider_hook=True)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                actions = [record.action for record in runtime.audit.trace()]
                assert 'test.provider_hook' in actions
                assert 'module.provider_hook' in actions
            finally:
                runtime.close()

    def test_module_tool_visibility_does_not_grant_filesystem_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'secret.txt').write_text('secret', encoding='utf-8')
            manifest, source_sha = _write_module(root, expose_read_tool=True)
            runtime = Runtime.open(substrate=LocalResourceProviderSubstrate(root), module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='read')
                assert 'read_text_file' in runtime.process.get(pid).tool_table
                result = runtime.tools.call(pid, 'read_text_file', {'path': 'secret.txt'})
                assert not result.ok
                assert 'lacks read' in (result.error or '')
            finally:
                runtime.close()

    def test_untrusted_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _source_sha = _write_module(Path(temp_dir))
            with pytest.raises(CapabilityDenied):
                Runtime.open(module_manifests=(str(manifest),))

    def test_import_string_entrypoint_loads_from_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module')
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                assert 'test-module:v0' in [module['module_id'] for module in runtime.modules.list_modules()]
                assert 'module-agent:v0' in runtime.images
            finally:
                runtime.close()

    def test_import_string_resolve_does_not_execute_package_init_before_trust(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / 'pkg'
            package.mkdir()
            sentinel = root / 'import_side_effect.txt'
            (package / '__init__.py').write_text(
                f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
                encoding='utf-8',
            )
            source = package / 'test_module.py'
            source.write_text("def register_module(ctx):\n    pass\n", encoding='utf-8')
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / 'module.yaml'
            manifest.write_text(
                f"""
schema_version: 1
module_id: package-module:v0
name: Package module
entrypoint: pkg.test_module:register_module
provides: {{}}
sha256: {source_sha}
""".lstrip(),
                encoding='utf-8',
            )

            resolved = ModuleLoader().resolve(manifest)

            assert resolved.source_path == str(source.resolve())
            assert not sentinel.exists()
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'package-module:v0:{source_sha}',))
            try:
                loaded = runtime.modules.inspect_module('package-module:v0')
                assert loaded['status'] == 'loaded'
                assert not sentinel.exists()
            finally:
                runtime.close()

    def test_import_string_entrypoint_executes_current_source_not_cached_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module', marker='old')
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='old module')
                assert runtime.tools.call(pid, 'module_echo', {'text': 'hello'}).payload['marker'] == 'old'
            finally:
                runtime.close()

            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module', marker='new')
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='new module')
                assert runtime.tools.call(pid, 'module_echo', {'text': 'hello'}).payload['marker'] == 'new'
            finally:
                runtime.close()

    def test_import_string_entrypoint_restores_existing_sys_modules_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module')
            previous = sys.modules.get('test_module')
            sentinel = types.ModuleType('test_module')
            sentinel.marker = 'original'
            sys.modules['test_module'] = sentinel
            try:
                runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
                try:
                    assert 'test-module:v0' in [module['module_id'] for module in runtime.modules.list_modules()]
                    assert sys.modules['test_module'] is sentinel
                finally:
                    runtime.close()
            finally:
                if previous is None:
                    sys.modules.pop('test_module', None)
                else:
                    sys.modules['test_module'] = previous

    def test_runtime_loaded_module_after_start_runs_hooks_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, provider_hook=True)
            runtime = Runtime.open()
            try:
                loaded = runtime.modules.load_module_manifest(
                    manifest,
                    trusted_modules=(f'test-module:v0:{source_sha}',),
                )
                actions = [record.action for record in runtime.audit.trace()]
                assert loaded['status'] == 'loaded'
                assert 'test.provider_hook' in actions
                assert 'test.startup_hook' in actions
                assert 'module.provider_hook' in actions
                assert 'module.startup_hook' in actions
            finally:
                runtime.close()

    def test_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            text = manifest.read_text(encoding='utf-8').replace(source_sha, '0' * 64)
            manifest.write_text(text, encoding='utf-8')
            with pytest.raises(ValidationError):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))

    def test_module_source_swap_after_trust_does_not_execute_swapped_bytes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            source = root / 'test_module.py'
            sentinel = root / 'swapped_executed.txt'
            original_resolve = ModuleLoader.resolve

            def swapping_resolve(loader: ModuleLoader, manifest_path: str | Path):
                resolved = original_resolve(loader, manifest_path)
                source.write_text(
                    f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n"
                    "def register_module(ctx):\n    pass\n",
                    encoding='utf-8',
                )
                return resolved

            monkeypatch.setattr(ModuleLoader, 'resolve', swapping_resolve)
            with pytest.raises(ValidationError, match='source changed after verification'):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))

            assert not sentinel.exists()

    def test_module_source_size_limit_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            config = replace(
                DEFAULT_CONFIG,
                modules=replace(DEFAULT_CONFIG.modules, source_max_bytes=32),
            )
            with pytest.raises(ValidationError, match='source_max_bytes'):
                Runtime.open(config=config, module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))

    def test_failed_module_does_not_leave_partial_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, invalid_registration=True)
            runtime = Runtime.open()
            try:
                with pytest.raises(ValidationError):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(f'test-module:v0:{source_sha}',))
                with pytest.raises(NotFound):
                    runtime.tools.resolve('module_echo')
                failed = runtime.modules.inspect_module('test-module:v0')
                assert failed['status'] == 'failed'
            finally:
                runtime.close()

    def test_apply_failure_rolls_back_registered_tool_and_failed_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open()
            original_register = runtime.image_registry.register
            try:
                def fail_register(*args, **kwargs):
                    raise RuntimeError('image register exploded')

                runtime.image_registry.register = fail_register
                with pytest.raises(RuntimeError, match='image register exploded'):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(f'test-module:v0:{source_sha}',))
                assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
                assert 'module-agent:v0' not in runtime.images
                assert runtime.modules.inspect_module('test-module:v0')['status'] == 'failed'
            finally:
                runtime.image_registry.register = original_register
                runtime.close()

    def test_invalid_module_image_does_not_leave_partial_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, invalid_image=True)
            runtime = Runtime.open()
            try:
                with pytest.raises(ValidationError):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(f'test-module:v0:{source_sha}',))
                with pytest.raises(NotFound):
                    runtime.tools.resolve('module_echo')
            finally:
                runtime.close()

    def test_startup_hook_failure_rolls_back_external_module_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root, failing_startup_hook=True)
            with pytest.raises(RuntimeError, match='startup hook failed'):
                Runtime.open(db, module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))

            runtime = Runtime.open(db)
            try:
                failed = runtime.modules.inspect_module('test-module:v0')
                assert failed['status'] == 'failed'
                assert 'startup hook failed' in failed['error']
                assert 'module-agent:v0' not in runtime.images
                assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
                assert runtime.syscalls.get('module.ping') is None
            finally:
                runtime.close()

    def test_duplicate_module_id_does_not_overwrite_loaded_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                with pytest.raises(ValidationError, match='already loaded'):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(f'test-module:v0:{source_sha}',))
                loaded = runtime.modules.inspect_module('test-module:v0')
                assert loaded['status'] == 'loaded'
                assert loaded['source_sha256'] == source_sha
            finally:
                runtime.close()

    def test_checkpoint_restore_requires_same_startup_module_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(db, module_manifests=(str(manifest),), trusted_modules=(f'test-module:v0:{source_sha}',))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='checkpoint')
                checkpoint_id = runtime.checkpoint.create(pid, 'module checkpoint', actor=pid)
            finally:
                runtime.close()
            reopened_without_module = Runtime.open(db)
            try:
                with pytest.raises(ValidationError):
                    reopened_without_module.checkpoint.restore('cli', checkpoint_id, require_capability=False)
            finally:
                reopened_without_module.close()

    def test_cli_modules_verify_list_and_spawn_with_module_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root)
            trust = f'test-module:v0:{source_sha}'
            verified = _run_cli_json(['--db', str(db), 'modules', 'verify', str(manifest)])
            assert not verified['trusted']
            listed = _run_cli_json(['--db', str(db), '--module-manifest', str(manifest), '--trusted-module', trust, 'modules', 'list'])
            assert 'test-module:v0' in [module['module_id'] for module in listed]
            spawned = _run_cli_json(['--db', str(db), '--module-manifest', str(manifest), '--trusted-module', trust, 'spawn', '--image', 'module-agent:v0', '--goal', 'cli module'])
            assert spawned['image'] == 'module-agent:v0'

    def test_json_manifest_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match='duplicate module manifest JSON key'):
            ModuleLoader().parse_manifest(
                """
{
  "schema_version": 1,
  "module_id": "json-module:v0",
  "name": "first",
  "name": "second",
  "entrypoint": "./module.py:register_module",
  "provides": {},
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000"
}
""".strip()
            )

    def test_manifest_schema_version_must_be_integer(self) -> None:
        with pytest.raises(ValidationError, match='schema_version must be an integer'):
            ModuleLoader().parse_manifest(
                """
schema_version: true
module_id: bool-schema:v0
name: Bool schema
entrypoint: ./module.py:register_module
provides: {}
sha256: 0000000000000000000000000000000000000000000000000000000000000000
""".lstrip()
            )

def _write_module(root: Path, *, expose_read_tool: bool=False, invalid_registration: bool=False, invalid_image: bool=False, provider_hook: bool=False, failing_startup_hook: bool=False, entrypoint: str='./test_module.py:register_module', marker: str='module') -> tuple[Path, str]:
    source = root / 'test_module.py'
    default_tools = ['module_echo', 'read_text_file'] if expose_read_tool else ['module_echo']
    if invalid_image:
        default_tools.append('missing_module_tool')
    provider_hook_code = '\ndef mark_provider(runtime):\n    runtime.audit.record(actor="module:test-module:v0", action="test.provider_hook", target="runtime")\n'.rstrip() if provider_hook else ''
    provider_hook_registration = "ctx.register_provider_hook('test_hook', mark_provider)" if provider_hook else ''
    startup_hook_body = "raise RuntimeError('startup hook failed')" if failing_startup_hook else 'runtime.audit.record(actor="module:test-module:v0", action="test.startup_hook", target="runtime")'
    source.write_text(f"""\nfrom pydantic import BaseModel\n\nfrom agent_libos.models import AgentImage\nfrom agent_libos.tools.base import SyncAgentTool, ToolContext\n\n\nclass EchoArgs(BaseModel):\n    text: str\n\n\nclass ModuleEchoTool(SyncAgentTool[EchoArgs]):\n    name = "module_echo"\n    description = "Echo text through a startup module."\n    args_schema = EchoArgs\n\n    def run(self, args: EchoArgs, ctx: ToolContext):\n        return {{"echo": args.text, "pid": ctx.pid, "marker": {marker!r}}}\n\n\ndef module_ping(session, args):\n    return {{"pid": session.pid, "value": args.get("value")}}\n\n\ndef mark_startup(runtime):\n    {startup_hook_body}\n\n\n{provider_hook_code}\n\n\ndef register_module(ctx):\n    ctx.register_tool(ModuleEchoTool())\n    {("ctx.register_syscall('undeclared.syscall', module_ping)" if invalid_registration else "ctx.register_syscall('module.ping', module_ping)")}\n    ctx.register_image(AgentImage(\n        image_id="module-agent:v0",\n        name="module-agent",\n        default_tools={default_tools!r},\n    ))\n    {provider_hook_registration}\n    ctx.add_startup_hook(mark_startup)\n""".lstrip(), encoding='utf-8')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    syscalls = '[]\n' if invalid_registration else "['module.ping']\n"
    manifest = root / 'module.yaml'
    manifest.write_text(f"\nschema_version: 1\nmodule_id: test-module:v0\nname: Test startup module\nversion: v0\nentrypoint: {entrypoint}\nprovides:\n  tools: ['module_echo']\n  images: ['module-agent:v0']\n  syscalls: {syscalls.rstrip()}\n  provider_hooks: {(['test_hook'] if provider_hook else [])!r}\n  startup_hooks: ['mark_startup']\nsha256: {source_sha}\nmetadata:\n  test: true\n".lstrip(), encoding='utf-8')
    return (manifest, source_sha)

def _run_cli_json(argv: list[str]) -> object:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli_main(argv)
    return json.loads(buffer.getvalue())
