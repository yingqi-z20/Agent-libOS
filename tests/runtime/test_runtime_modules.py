from __future__ import annotations
import pytest
import asyncio
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import types
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from agent_libos import Runtime
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.api.cli import main as cli_main
from agent_libos.models import AgentImage
from agent_libos.models.exceptions import (
    CapabilityDenied,
    NotFound,
    ProviderHostError,
    ValidationError,
)
from agent_libos.modules.host import ModuleHookContext, ModuleHookServices
from agent_libos.modules.journal import RegistrationJournal, RegistrationRollbackError
from agent_libos.modules.loader import ModuleLoader
from agent_libos.runtime.audit_manager import AuditManager
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import LocalResourceProviderSubstrate

class TestRuntimeModule:

    def test_registration_journal_rolls_back_in_reverse_order_once(self) -> None:
        order: list[str] = []
        journal = RegistrationJournal('journal-test:v0')
        journal.record(kind='tool', target='one', undo=lambda: order.append('tool'))
        journal.record(kind='image', target='two', undo=lambda: order.append('image'))
        journal.record(kind='startup_hook', target='three', undo=lambda: order.append('startup_hook'))

        journal.rollback()
        journal.rollback()

        assert order == ['startup_hook', 'image', 'tool']
        assert journal.rolled_back

    def test_registration_journal_retains_failed_inverse_for_retry(self) -> None:
        order: list[str] = []
        attempts = 0
        journal = RegistrationJournal('journal-retry:v0')

        def transient_failure() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('transient inverse failure')
            order.append('retry_succeeded')

        journal.record(kind='tool', target='older', undo=lambda: order.append('older'))
        journal.record(kind='tool', target='retry', undo=transient_failure)

        with pytest.raises(
            RegistrationRollbackError,
            match='module_rollback_failed: RegistrationRollbackError',
        ) as caught:
            journal.rollback()

        assert 'transient inverse failure' not in str(caught.value)
        assert str(caught.value).count('correlation_id=') == 1

        assert order == ['older']
        assert journal.size == 1
        assert not journal.rolled_back
        with pytest.raises(RuntimeError, match='rollback has started'):
            journal.record(kind='tool', target='late', undo=lambda: None)

        journal.rollback()
        journal.rollback()

        assert attempts == 2
        assert order == ['older', 'retry_succeeded']
        assert journal.size == 0
        assert journal.rolled_back

    def test_tool_rollback_quarantine_is_identity_bound(self) -> None:
        runtime = Runtime.open()
        try:
            handle = runtime.tools.loaded_tool_handles()[0]
            equal_but_distinct = deepcopy(handle)

            assert equal_but_distinct == handle
            assert equal_but_distinct is not handle
            assert not runtime.tools.discard_tool_registration(equal_but_distinct)
            assert runtime.tools.resolve(handle.name) is handle
        finally:
            runtime.close()

    def test_module_registry_does_not_expose_surface_snapshot_rollback(self) -> None:
        runtime = Runtime.open()
        try:
            assert not hasattr(runtime.modules, '_snapshot_runtime_surfaces')
            assert not hasattr(runtime.modules, '_restore_runtime_surfaces')
            assert runtime.modules._registration_journals['agent-libos-core:v0'].size > 0
        finally:
            runtime.close()

    def test_failed_core_module_rehydrate_preserves_durable_images(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / 'runtime.sqlite'
        runtime = Runtime.open(db)
        registry_type = type(runtime.modules)
        original_apply = registry_type._apply_context
        durable_ids = {image.image_id for image, _metadata in runtime.store.list_images()}
        assert durable_ids
        runtime.close()

        def apply_then_fail(self: Any, ctx: Any, journal: RegistrationJournal) -> None:
            original_apply(self, ctx, journal)
            if ctx.module_id == 'agent-libos-core:v0':
                raise RuntimeError('injected failure after core image rehydrate')

        monkeypatch.setattr(registry_type, '_apply_context', apply_then_fail)
        with pytest.raises(RuntimeError, match='after core image rehydrate'):
            Runtime.open(db)

        monkeypatch.setattr(registry_type, '_apply_context', original_apply)
        reopened = Runtime.open(db)
        try:
            persisted_ids = {
                image.image_id for image, _metadata in reopened.store.list_images()
            }
            assert persisted_ids == durable_ids
            assert durable_ids <= set(reopened.images)
        finally:
            reopened.close()

    def test_module_hook_host_has_only_explicit_journaled_runtime_state(self) -> None:
        runtime = Runtime.open()
        journal = RegistrationJournal('host-contract:v0')
        host = ModuleHookContext(ModuleHookServices.from_host(runtime), 'host-contract:v0', journal)
        try:
            assert host.audit is runtime.audit
            assert not hasattr(host, 'llm')
            with pytest.raises(ValidationError, match='explicit journaled'):
                host.unregistered_state = object()  # type: ignore[attr-defined]
            with pytest.raises(ValidationError, match='prefix'):
                host.set_runtime_attribute('unscoped_state', object())

            state = object()
            host.set_runtime_attribute('_agent_libos_host_contract', state)
            assert host.get_runtime_attribute('_agent_libos_host_contract') is state
            journal.rollback()
            assert host.get_runtime_attribute('_agent_libos_host_contract') is None
        finally:
            runtime.close()

    def test_module_hook_memory_finalizer_registration_is_journaled(self) -> None:
        runtime = Runtime.open()
        journal = RegistrationJournal('memory-view-contract:v0')
        host = ModuleHookContext(ModuleHookServices.from_host(runtime), 'memory-view-contract:v0', journal)
        finalizer = lambda _obj, _actor, _reason: None
        try:
            before = list(runtime.memory._object_release_finalizers)

            host.memory.bind_object_release_finalizer(finalizer)

            assert runtime.memory._object_release_finalizers == [*before, finalizer]
            journal.rollback()
            assert runtime.memory._object_release_finalizers == before
        finally:
            host.deactivate()
            journal.rollback()
            runtime.close()

    def test_module_hook_recovery_cleanup_registration_is_explicit_and_journaled(self) -> None:
        runtime = Runtime.open()
        journal = RegistrationJournal('recovery-cleanup-contract:v0')
        host = ModuleHookContext(
            ModuleHookServices.from_host(runtime),
            'recovery-cleanup-contract:v0',
            journal,
        )

        def cleanup() -> bool:
            return True

        try:
            before = runtime.lifecycle.finalizers_snapshot()

            with pytest.raises(RuntimeError, match='recovery cleanup lease'):
                host.require_recovery_cleanup_lease()

            host.bind_recovery_cleanup(cleanup)

            assert runtime.lifecycle.finalizers_snapshot() == (*before, cleanup)
            assert runtime.lifecycle._finalizers[-1].recovery_safe is True
            journal.rollback()
            assert runtime.lifecycle.finalizers_snapshot() == before
        finally:
            host.deactivate()
            journal.rollback()
            runtime.close()

    def test_module_hook_image_view_cannot_mutate_live_images(self) -> None:
        runtime = Runtime.open()
        journal = RegistrationJournal('image-view-contract:v0')
        host = ModuleHookContext(ModuleHookServices.from_host(runtime), 'image-view-contract:v0', journal)
        try:
            image_id = runtime.config.runtime.default_image_id
            original = runtime.images[image_id]
            exposed = host.images[image_id]

            exposed.boot['kind'] = 'mutated-by-module'
            exposed.metadata['nested'] = {'mutated': True}

            assert runtime.images[image_id] == original
            assert runtime.images[image_id].boot.get('kind') != 'mutated-by-module'
            assert 'nested' not in runtime.images[image_id].metadata
        finally:
            host.deactivate()
            journal.rollback()
            runtime.close()

    def test_module_discovery_rejects_unbounded_limits(self) -> None:
        runtime = Runtime.open()
        try:
            for limit in (0, -1, True, runtime.config.modules.discover_limit + 1):
                with pytest.raises(ValidationError, match='limit'):
                    runtime.modules.list_modules(limit=limit)  # type: ignore[arg-type]
        finally:
            runtime.close()

    def test_trusted_startup_module_registers_tool_image_syscall_and_hook(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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

    def test_image_required_modules_rejects_spawn_when_module_is_not_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _manifest, source_sha = _write_module(Path(temp_dir))
            runtime = Runtime.open()
            try:
                runtime.register_image(
                    AgentImage(
                        image_id='needs-module:v0',
                        name='needs-module',
                        required_modules=[{'module_id': 'test-module:v0', 'source_sha256': source_sha}],
                    ),
                    actor='cli',
                )

                with pytest.raises(ValidationError, match='image requires startup modules'):
                    runtime.process.spawn(image='needs-module:v0', goal='blocked')
            finally:
                runtime.close()

    def test_image_required_modules_allows_spawn_when_module_hash_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                runtime.register_image(
                    AgentImage(
                        image_id='needs-module:v0',
                        name='needs-module',
                        default_tools=['module_echo'],
                        required_modules=[{'module_id': 'test-module:v0', 'source_sha256': source_sha}],
                    ),
                    actor='cli',
                )

                pid = runtime.process.spawn(image='needs-module:v0', goal='allowed')
                result = runtime.tools.call(pid, 'module_echo', {'text': 'hello'})
                assert result.ok
                assert result.payload['echo'] == 'hello'
            finally:
                runtime.close()

    def test_image_required_modules_rejects_loaded_module_with_different_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                runtime.register_image(
                    AgentImage(
                        image_id='needs-other-module:v0',
                        name='needs-other-module',
                        required_modules=[{'module_id': 'test-module:v0', 'source_sha256': '1' * 64}],
                    ),
                    actor='cli',
                )

                with pytest.raises(ValidationError, match='image requires startup modules'):
                    runtime.process.spawn(image='needs-other-module:v0', goal='wrong hash')
            finally:
                runtime.close()

    def test_provider_hook_runs_before_runtime_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, provider_hook=True)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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
            runtime = Runtime.open(substrate=LocalResourceProviderSubstrate(root), module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='read')
                assert 'read_text_file' in runtime.process.get(pid).tool_table
                result = runtime.tools.call(pid, 'read_text_file', {'path': 'secret.txt'})
                assert not result.ok
                assert (result.error or '').startswith(
                    'permission_denied: CapabilityDenied'
                )
                assert 'secret.txt' not in (result.error or '')
            finally:
                runtime.close()

    def test_untrusted_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest, _source_sha = _write_module(Path(temp_dir))
            with pytest.raises(CapabilityDenied):
                Runtime.open(module_manifests=(str(manifest),))

    def test_warn_policy_module_failure_redacts_exception_text_from_durable_evidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        secret = 'token=module-secret /opaque/private/module.py payload=opaque-value'
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(DEFAULT_CONFIG.modules, load_policy='warn'),
        )
        runtime = Runtime.open(config=config)
        try:
            def fail_resolve(
                _loader: ModuleLoader,
                _manifest_path: str | Path,
            ) -> object:
                raise RuntimeError(secret)

            monkeypatch.setattr(ModuleLoader, 'resolve', fail_resolve)
            result = runtime.modules.load_module_manifest(
                tmp_path / 'failed-module.yaml'
            )
            failed = runtime.modules.inspect_module('failed:failed-module.yaml')
            failure_audits = [
                record.decision
                for record in runtime.audit.trace()
                if record.action == 'module.load_failed'
            ]
            projected = json.dumps(
                {
                    'result': result,
                    'failed': failed,
                    'audit': failure_audits,
                },
                sort_keys=True,
            )

            assert secret not in projected
            assert '/opaque/private/module.py' not in projected
            assert 'opaque-value' not in projected
            assert result['error'].startswith('module_load_failed: RuntimeError')
            assert failed['error'] == result['error']
            assert result['correlation_id'] in projected
            assert failed['module_id'] == 'failed:failed-module.yaml'
            assert failed['entrypoint'] == ''
            assert failed['manifest_sha256'] == ''
            assert failed['source_path'] == ''
            assert failed['source_sha256'] == ''
            assert failed['loaded_at'] is None
            assert not any(failed['registered'].values())
        finally:
            runtime.close()

    def test_module_failure_does_not_trust_exception_supplied_public_envelope(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        supplied = {
            'code': 'secretModuleCode',
            'error_type': 'SecretModuleType',
            'correlation_id': 'secretCorrelation',
        }
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(DEFAULT_CONFIG.modules, load_policy='warn'),
        )
        runtime = Runtime.open(config=config)
        try:
            def fail_resolve(
                _loader: ModuleLoader,
                _manifest_path: str | Path,
            ) -> object:
                raise ProviderHostError(**supplied)

            monkeypatch.setattr(ModuleLoader, 'resolve', fail_resolve)
            result = runtime.modules.load_module_manifest(
                tmp_path / 'failed-provider-envelope.yaml'
            )
            failed = runtime.modules.inspect_module(
                'failed:failed-provider-envelope.yaml'
            )
            failure_audits = [
                record.decision
                for record in runtime.audit.trace()
                if record.action == 'module.load_failed'
            ]
            projected = json.dumps(
                {'result': result, 'failed': failed, 'audit': failure_audits},
                sort_keys=True,
            )

            assert all(value not in projected for value in supplied.values())
            assert result['error'].startswith(
                'module_load_failed: ProviderHostError'
            )
            assert result['correlation_id'].startswith('corr_')
        finally:
            runtime.close()

    def test_module_trust_binds_manifest_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            original_trust = _module_trust_key('test-module:v0', manifest, source_sha)
            manifest.write_text(
                manifest.read_text(encoding='utf-8').replace('  test: true', '  test: false'),
                encoding='utf-8',
            )

            with pytest.raises(CapabilityDenied, match='not trusted'):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(original_trust,))

    def test_module_entrypoint_cannot_access_runtime_before_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            source = root / 'escape_module.py'
            source.write_text(
                "def register_module(ctx):\n"
                "    ctx.runtime.audit.record(actor='module:escape-module:v0', action='preflight.escape', target='runtime')\n",
                encoding='utf-8',
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / 'module.yaml'
            manifest.write_text(
                f"""
schema_version: 1
module_id: escape-module:v0
name: Escape module
entrypoint: ./escape_module.py:register_module
provides: {{}}
sha256: {source_sha}
""".lstrip(),
                encoding='utf-8',
            )

            with pytest.raises(ValidationError, match='cannot access runtime.audit before module preflight'):
                Runtime.open(
                    db,
                    module_manifests=(str(manifest),),
                    trusted_modules=(_module_trust_key('escape-module:v0', manifest, source_sha),),
                )

            runtime = Runtime.open(db)
            try:
                assert all(record.action != 'preflight.escape' for record in runtime.audit.trace())
            finally:
                runtime.close()

    def test_module_entrypoint_cannot_mutate_provides_to_bypass_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / 'mutating_module.py'
            source.write_text(
                """
from pydantic import BaseModel

from agent_libos.tools.base import SyncAgentTool, ToolContext


class HiddenArgs(BaseModel):
    pass


class HiddenTool(SyncAgentTool[HiddenArgs]):
    name = "hidden_tool"
    description = "Hidden undeclared tool."
    args_schema = HiddenArgs

    def run(self, args: HiddenArgs, ctx: ToolContext):
        return {"hidden": True}


def register_module(ctx):
    try:
        ctx.manifest.provides.tools.append("hidden_tool")
    except AttributeError:
        pass
    ctx.register_tool(HiddenTool())
""".lstrip(),
                encoding='utf-8',
            )
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root / 'module.yaml'
            manifest.write_text(
                f"""
schema_version: 1
module_id: mutating-module:v0
name: Mutating module
version: v0
entrypoint: ./mutating_module.py:register_module
provides:
  tools: []
sha256: {source_sha}
""".lstrip(),
                encoding='utf-8',
            )

            with pytest.raises(ValidationError, match='registered undeclared tool: hidden_tool'):
                Runtime.open(
                    module_manifests=(str(manifest),),
                    trusted_modules=(_module_trust_key('mutating-module:v0', manifest, source_sha),),
                )

    def test_module_entrypoint_cannot_mutate_registration_buffers_before_preflight(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / 'mutating_buffers.py'
        source.write_text(
            "from agent_libos.models import AgentImage\n\n"
            "def register_module(ctx):\n"
            "    ctx.images.append(AgentImage(image_id='hidden-buffer:v0', name='hidden-buffer'))\n",
            encoding='utf-8',
        )
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = tmp_path / 'module.yaml'
        manifest.write_text(
            f"""
schema_version: 1
module_id: mutating-buffers:v0
name: Mutating buffers module
entrypoint: ./mutating_buffers.py:register_module
provides: {{}}
sha256: {source_sha}
""".lstrip(),
            encoding='utf-8',
        )
        runtime = Runtime.open()
        register_attempts = 0
        original_register = runtime.image_registry.register_module_image

        def observe_register(*args: Any, **kwargs: Any) -> Any:
            nonlocal register_attempts
            register_attempts += 1
            return original_register(*args, **kwargs)

        monkeypatch.setattr(runtime.image_registry, 'register_module_image', observe_register)
        try:
            with pytest.raises(ValidationError, match='register_image'):
                runtime.modules.load_module_manifest(
                    manifest,
                    trusted_modules=(
                        _module_trust_key(
                            'mutating-buffers:v0',
                            manifest,
                            source_sha,
                        ),
                    ),
                )

            assert register_attempts == 0
            assert 'hidden-buffer:v0' not in runtime.images
            assert runtime.store.get_image('hidden-buffer:v0') is None
        finally:
            runtime.close()

    def test_module_tool_name_cannot_change_after_declared_registration(
        self,
        tmp_path: Path,
    ) -> None:
        source = tmp_path / 'mutating_tool.py'
        source.write_text(
            "from pydantic import BaseModel\n"
            "from agent_libos.tools.base import SyncAgentTool\n\n"
            "class Args(BaseModel):\n"
            "    pass\n\n"
            "class MutatingTool(SyncAgentTool[Args]):\n"
            "    name = 'declared_tool'\n"
            "    description = 'mutates after registration'\n"
            "    args_schema = Args\n\n"
            "    def run(self, args, ctx):\n"
            "        return {}\n\n"
            "def register_module(ctx):\n"
            "    tool = MutatingTool()\n"
            "    ctx.register_tool(tool)\n"
            "    tool.name = 'hidden_tool'\n",
            encoding='utf-8',
        )
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = tmp_path / 'module.yaml'
        manifest.write_text(
            f"""
schema_version: 1
module_id: mutating-tool:v0
name: Mutating tool module
entrypoint: ./mutating_tool.py:register_module
provides:
  tools: [declared_tool]
sha256: {source_sha}
""".lstrip(),
            encoding='utf-8',
        )
        runtime = Runtime.open()
        try:
            with pytest.raises(ValidationError, match='changed after registration'):
                runtime.modules.load_module_manifest(
                    manifest,
                    trusted_modules=(
                        _module_trust_key(
                            'mutating-tool:v0',
                            manifest,
                            source_sha,
                        ),
                    ),
                )

            with pytest.raises(NotFound):
                runtime.tools.resolve('declared_tool')
            with pytest.raises(NotFound):
                runtime.tools.resolve('hidden_tool')
            assert all(
                row['name'] not in {'declared_tool', 'hidden_tool'}
                for row in runtime.store.list_tools()
            )
        finally:
            runtime.close()

    def test_configured_startup_module_paths_resolve_from_project_root_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project_root = tmp_path / 'project'
        cwd_root = tmp_path / 'cwd'
        project_root.mkdir()
        cwd_root.mkdir()
        project_manifest, source_sha = _write_module(project_root, marker='project')
        _write_module(cwd_root, marker='cwd')
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(
                DEFAULT_CONFIG.modules,
                manifest_paths=('module.yaml',),
                trusted_modules=(_module_trust_key('test-module:v0', project_manifest, source_sha),),
            ),
        )
        monkeypatch.setattr('agent_libos.modules.registry.get_project_root', lambda: project_root)
        monkeypatch.chdir(cwd_root)

        runtime = Runtime.open(config=config)
        try:
            loaded = runtime.modules.inspect_module('test-module:v0')
            assert Path(loaded['manifest_path']) == project_manifest.resolve()
            assert any(record.action == 'module.load' for record in runtime.audit.trace())
            pid = runtime.process.spawn(image='module-agent:v0', goal='project-root module')
            result = runtime.tools.call(pid, 'module_echo', {'text': 'hello'})
            assert result.payload['marker'] == 'project'
        finally:
            runtime.close()

    def test_import_string_entrypoint_loads_from_manifest_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module')
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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

            assert os.path.samefile(resolved.source_path, source)
            assert not sentinel.exists()
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('package-module:v0', manifest, source_sha),))
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
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='old module')
                assert runtime.tools.call(pid, 'module_echo', {'text': 'hello'}).payload['marker'] == 'old'
            finally:
                runtime.close()

            manifest, source_sha = _write_module(root, entrypoint='test_module:register_module', marker='new')
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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
                runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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
                    trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),),
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
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))

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
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))

            assert not sentinel.exists()

    def test_multifile_module_package_relative_import_registers_runtime_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            verified = ModuleLoader().verify(manifest)
            assert verified['source_kind'] == 'package'
            assert verified['source_sha256'] == package_sha
            assert [item['path'] for item in verified['source_files']] == [
                'pkg/__init__.py',
                'pkg/helper.py',
                'pkg/main.py',
            ]
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),))
            try:
                loaded = runtime.modules.inspect_module('multi-module:v0')
                assert loaded['status'] == 'loaded'
                assert loaded['source_sha256'] == package_sha
                pid = runtime.process.spawn(image='multi-agent:v0', goal='multi module')
                result = runtime.tools.call(pid, 'multi_echo', {'text': 'hello'})
                assert result.ok
                assert result.payload['helper_marker'] == 'helper-v1'
                syscall_result = asyncio.run(LibOSSyscallSession(runtime, pid).handle('multi.ping', {'value': 'ok'}))
                assert syscall_result['helper_marker'] == 'helper-v1'
            finally:
                runtime.close()

    def test_multifile_module_runtime_lazy_relative_imports_use_snapshot_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = root / 'lazypkg'
            package.mkdir()
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'helper.py').write_text(
                "def helper(value):\n"
                "    return f'{value}-from-helper'\n",
                encoding='utf-8',
            )
            (package / 'main.py').write_text(
                """
from pydantic import BaseModel

from agent_libos.models import AgentImage
from agent_libos.tools.base import SyncAgentTool, ToolContext


class LazyEchoArgs(BaseModel):
    text: str


class LazyEchoTool(SyncAgentTool[LazyEchoArgs]):
    name = "lazy_echo"
    description = "Echo text through a runtime relative import."
    args_schema = LazyEchoArgs

    def run(self, args: LazyEchoArgs, ctx: ToolContext):
        from .helper import helper

        return {"value": helper(args.text), "pid": ctx.pid}


def register_module(ctx):
    ctx.register_tool(LazyEchoTool())
    ctx.register_image(AgentImage(
        image_id="lazy-agent:v0",
        name="lazy-agent",
        default_tools=["lazy_echo"],
    ))
""".lstrip(),
                encoding='utf-8',
            )
            package_sha = _module_package_sha(root, package)
            manifest = root / 'module.yaml'
            manifest.write_text(
                f"""
schema_version: 1
module_id: lazy-module:v0
name: Lazy multi-file startup module
version: v0
entrypoint: lazypkg.main:register_module
provides:
  tools: ['lazy_echo']
  images: ['lazy-agent:v0']
  syscalls: []
  provider_hooks: []
  startup_hooks: []
sha256: {package_sha}
""".lstrip(),
                encoding='utf-8',
            )

            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('lazy-module:v0', manifest, package_sha),))
            try:
                pid = runtime.process.spawn(image='lazy-agent:v0', goal='lazy module')
                result = runtime.tools.call(pid, 'lazy_echo', {'text': 'hello'})

                assert result.ok, result.error
                assert result.payload['value'] == 'hello-from-helper'
            finally:
                runtime.close()

    def test_multifile_module_package_ignores_generated_cache_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            cache_dir = root / 'pkg' / '__pycache__'
            cache_dir.mkdir()
            (cache_dir / 'helper.cpython-314.pyc').write_bytes(b'generated cache')

            verified = ModuleLoader().verify(manifest)

            assert verified['source_sha256'] == package_sha
            assert all('__pycache__' not in item['path'] for item in verified['source_files'])
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),))
            try:
                assert runtime.modules.inspect_module('multi-module:v0')['status'] == 'loaded'
            finally:
                runtime.close()

    def test_multifile_module_helper_change_rejects_old_package_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            (root / 'pkg' / 'helper.py').write_text("HELPER_MARKER = 'helper-v2'\n", encoding='utf-8')
            with pytest.raises(ValidationError, match='source sha256 mismatch'):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),))

    def test_multifile_module_reloads_fresh_with_same_package_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sentinel = root / 'imports.txt'
            package = root / 'freshpkg'
            package.mkdir()
            (package / '__init__.py').write_text('', encoding='utf-8')
            (package / 'main.py').write_text(
                f"from pathlib import Path\n"
                f"Path({str(sentinel)!r}).open('a', encoding='utf-8').write('imported\\n')\n\n"
                "def register_module(ctx):\n"
                "    pass\n",
                encoding='utf-8',
            )
            package_sha = _module_package_sha(root, package)
            manifest = root / 'module.yaml'
            manifest.write_text(
                f"""
schema_version: 1
module_id: fresh-module:v0
name: Fresh multi-file module
entrypoint: freshpkg.main:register_module
provides: {{}}
sha256: {package_sha}
""".lstrip(),
                encoding='utf-8',
            )

            first = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('fresh-module:v0', manifest, package_sha),))
            first.close()
            second = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('fresh-module:v0', manifest, package_sha),))
            second.close()

            assert sentinel.read_text(encoding='utf-8') == 'imported\nimported\n'

    def test_multifile_module_package_import_state_is_cleaned_on_runtime_close(self) -> None:
        before_meta_path = tuple(sys.meta_path)
        before_module_names = {name for name in sys.modules if name.startswith('_agent_libos_module_pkg_')}

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_tiny_multifile_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('tiny-module:v0', manifest, package_sha),))
            try:
                loaded_module_names = {name for name in sys.modules if name.startswith('_agent_libos_module_pkg_')}

                assert len(sys.meta_path) == len(before_meta_path) + 1
                assert loaded_module_names - before_module_names
            finally:
                runtime.close()

        assert tuple(sys.meta_path) == before_meta_path
        assert {name for name in sys.modules if name.startswith('_agent_libos_module_pkg_')} == before_module_names

    def test_reexported_package_entrypoint_retains_manifest_import_cleanup(
        self,
        tmp_path: Path,
    ) -> None:
        package = tmp_path / 'reexported'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'helper.py').write_text(
            'def register_module(ctx):\n    pass\n',
            encoding='utf-8',
        )
        (package / 'main.py').write_text(
            'from .helper import register_module\n',
            encoding='utf-8',
        )
        package_sha = _module_package_sha(tmp_path, package)
        manifest = tmp_path / 'reexported.yaml'
        manifest.write_text(
            f'''
schema_version: 1
module_id: reexported-module:v0
name: Re-exported module
entrypoint: reexported.main:register_module
provides: {{}}
sha256: {package_sha}
'''.lstrip(),
            encoding='utf-8',
        )
        before_meta_path = tuple(sys.meta_path)
        before_module_names = {
            name
            for name in sys.modules
            if name.startswith('_agent_libos_module_pkg_')
        }
        runtime: Runtime | None = None
        try:
            runtime = Runtime.open(
                module_manifests=(str(manifest),),
                trusted_modules=(
                    _module_trust_key(
                        'reexported-module:v0',
                        manifest,
                        package_sha,
                    ),
                ),
            )

            assert 'reexported-module:v0' in runtime.modules._module_import_cleanups
            runtime.close()
            runtime = None

            assert tuple(sys.meta_path) == before_meta_path
            assert {
                name
                for name in sys.modules
                if name.startswith('_agent_libos_module_pkg_')
            } == before_module_names
        finally:
            if runtime is not None:
                runtime.close()
            for importer in tuple(sys.meta_path):
                package_name = getattr(importer, 'package_name', '')
                if (
                    importer not in before_meta_path
                    and isinstance(package_name, str)
                    and package_name.startswith('_agent_libos_module_pkg_')
                ):
                    ModuleLoader.cleanup_imported_package(
                        (package_name, importer)
                    )

    def test_package_import_identity_collision_fails_before_execution(
        self,
        tmp_path: Path,
    ) -> None:
        package = tmp_path / 'collision'
        nested = package / 'a'
        nested.mkdir(parents=True)
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'a.b.py').write_text(
            'def register_module(ctx):\n    pass\n',
            encoding='utf-8',
        )
        sentinel = tmp_path / 'colliding-module-executed'
        (nested / 'b.py').write_text(
            'from pathlib import Path\n'
            f'Path({str(sentinel)!r}).write_text("executed", encoding="utf-8")\n'
            'def register_module(ctx):\n    pass\n',
            encoding='utf-8',
        )
        package_sha = _module_package_sha(tmp_path, package)
        manifest = tmp_path / 'collision.yaml'
        manifest.write_text(
            f'''
schema_version: 1
module_id: collision-module:v0
name: Collision module
entrypoint: ./collision/a.b.py:register_module
provides: {{}}
sha256: {package_sha}
'''.lstrip(),
            encoding='utf-8',
        )
        loader = ModuleLoader()
        source = loader.resolve(manifest)
        before_meta_path = tuple(sys.meta_path)

        with pytest.raises(ValidationError, match='same import identity'):
            loader.import_entrypoint(source)

        assert tuple(sys.meta_path) == before_meta_path
        assert not sentinel.exists()

    @pytest.mark.parametrize(
        ('failure_statement', 'exception_type'),
        [
            ('raise KeyboardInterrupt()', KeyboardInterrupt),
            ('raise asyncio.CancelledError()', asyncio.CancelledError),
        ],
    )
    def test_multifile_module_base_exception_import_cleans_snapshot_namespace(
        self,
        tmp_path: Path,
        failure_statement: str,
        exception_type: type[BaseException],
    ) -> None:
        package = tmp_path / 'interrupting'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'main.py').write_text(
            'import asyncio\n'
            f'{failure_statement}\n\n'
            'def register_module(ctx):\n'
            '    pass\n',
            encoding='utf-8',
        )
        package_sha = _module_package_sha(tmp_path, package)
        manifest = tmp_path / 'module.yaml'
        manifest.write_text(
            f"""
schema_version: 1
module_id: interrupting-module:v0
name: Interrupting module
entrypoint: interrupting.main:register_module
provides: {{}}
sha256: {package_sha}
""".lstrip(),
            encoding='utf-8',
        )
        loader = ModuleLoader()
        source = loader.resolve(manifest)
        before_meta_path = tuple(sys.meta_path)
        before_modules = {
            name for name in sys.modules
            if name.startswith('_agent_libos_module_pkg_')
        }

        with pytest.raises(exception_type):
            loader.import_entrypoint(source)

        leaked_importers = [
            importer for importer in sys.meta_path
            if all(importer is not existing for existing in before_meta_path)
        ]
        leaked_modules = {
            name for name in sys.modules
            if name.startswith('_agent_libos_module_pkg_')
        } - before_modules
        try:
            assert leaked_importers == []
            assert leaked_modules == set()
        finally:
            for importer in leaked_importers:
                try:
                    sys.meta_path.remove(importer)
                except ValueError:
                    pass
            for name in leaked_modules:
                sys.modules.pop(name, None)

    def test_multifile_module_source_swap_after_trust_does_not_execute_swapped_helper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            helper = root / 'pkg' / 'helper.py'
            sentinel = root / 'swapped_helper_executed.txt'
            original_resolve = ModuleLoader.resolve

            def swapping_resolve(loader: ModuleLoader, manifest_path: str | Path):
                resolved = original_resolve(loader, manifest_path)
                helper.write_text(
                    f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n"
                    "HELPER_MARKER = 'helper-swapped'\n",
                    encoding='utf-8',
                )
                return resolved

            monkeypatch.setattr(ModuleLoader, 'resolve', swapping_resolve)
            with pytest.raises(ValidationError, match='source changed after verification'):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),))

            assert not sentinel.exists()

    def test_multifile_module_cli_verify_outputs_package_source_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, package_sha = _write_multifile_module(root)
            verified = _run_cli_json(['--db', str(db), 'modules', 'verify', str(manifest)])
            assert verified['source_kind'] == 'package'
            assert verified['source_sha256'] == package_sha
            assert any(item['path'] == 'pkg/helper.py' for item in verified['source_files'])

    def test_multifile_module_package_file_count_limit_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            config = replace(
                DEFAULT_CONFIG,
                modules=replace(DEFAULT_CONFIG.modules, max_package_files=2),
            )
            with pytest.raises(ValidationError, match='max_package_files'):
                Runtime.open(
                    config=config,
                    module_manifests=(str(manifest),),
                    trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),),
                )

    def test_multifile_module_package_total_size_limit_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_tiny_multifile_module(root)
            config = replace(
                DEFAULT_CONFIG,
                modules=replace(DEFAULT_CONFIG.modules, source_max_bytes=128, package_max_bytes=200),
            )
            with pytest.raises(ValidationError, match='package_max_bytes'):
                Runtime.open(
                    config=config,
                    module_manifests=(str(manifest),),
                    trusted_modules=(_module_trust_key('tiny-module:v0', manifest, package_sha),),
                )

    def test_multifile_module_package_hardlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, package_sha = _write_multifile_module(root)
            try:
                os.link(root / 'pkg' / 'helper.py', root / 'pkg' / 'linked_helper.py')
            except OSError as exc:
                pytest.skip(f'hard links are not available on this filesystem: {exc}')
            with pytest.raises(ValidationError, match='hard links'):
                Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('multi-module:v0', manifest, package_sha),))

    def test_module_source_size_limit_fails_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            config = replace(
                DEFAULT_CONFIG,
                modules=replace(DEFAULT_CONFIG.modules, source_max_bytes=32),
            )
            with pytest.raises(ValidationError, match='source_max_bytes'):
                Runtime.open(config=config, module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='os.fstat mutation injection is POSIX-only; Win32 target handles deny concurrent writes',
    )
    def test_module_source_growth_after_descriptor_open_uses_bounded_reads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, _source_sha = _write_module(tmp_path)
        source = tmp_path / 'test_module.py'
        source_identity = (source.stat().st_dev, source.stat().st_ino)
        source_limit = source.stat().st_size + 64
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(
                DEFAULT_CONFIG.modules,
                source_max_bytes=source_limit,
            ),
        )
        real_fstat = os.fstat
        real_fdopen = os.fdopen
        grew = False
        read_sizes: list[int] = []

        class RecordingHandle:
            def __init__(self, handle: Any):
                self._handle = handle

            def __enter__(self) -> 'RecordingHandle':
                self._handle.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self._handle.__exit__(*args)

            def fileno(self) -> int:
                return self._handle.fileno()

            def read(self, size: int = -1) -> bytes:
                read_sizes.append(size)
                return self._handle.read(size)

        def grow_after_open(fd: int) -> os.stat_result:
            nonlocal grew
            result = real_fstat(fd)
            if not grew and (result.st_dev, result.st_ino) == source_identity:
                grew = True
                with source.open('ab') as output:
                    output.write(b'x' * source_limit)
            return result

        def record_source_reads(fd: int, *args: object, **kwargs: object) -> Any:
            handle = real_fdopen(fd, *args, **kwargs)
            opened = real_fstat(fd)
            if (opened.st_dev, opened.st_ino) == source_identity:
                return RecordingHandle(handle)
            return handle

        monkeypatch.setattr(os, 'fstat', grow_after_open)
        monkeypatch.setattr(os, 'fdopen', record_source_reads)

        with pytest.raises(ValidationError, match=f'source_max_bytes={source_limit}'):
            ModuleLoader(config).resolve(manifest)

        assert grew
        assert read_sizes
        assert all(0 < size <= source_limit + 1 for size in read_sizes)

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='os.fstat mutation injection is POSIX-only; Win32 target handles deny concurrent writes',
    )
    def test_module_manifest_growth_after_descriptor_open_uses_bounded_reads(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, _source_sha = _write_module(tmp_path)
        manifest_identity = (manifest.stat().st_dev, manifest.stat().st_ino)
        manifest_limit = manifest.stat().st_size + 64
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(
                DEFAULT_CONFIG.modules,
                manifest_max_bytes=manifest_limit,
            ),
        )
        real_path_stat = Path.stat
        real_fstat = os.fstat
        real_fdopen = os.fdopen
        grew = False
        read_sizes: list[int] = []

        class RecordingHandle:
            def __init__(self, handle: Any):
                self._handle = handle

            def __enter__(self) -> 'RecordingHandle':
                self._handle.__enter__()
                return self

            def __exit__(self, *args: object) -> object:
                return self._handle.__exit__(*args)

            def fileno(self) -> int:
                return self._handle.fileno()

            def read(self, size: int = -1) -> bytes:
                read_sizes.append(size)
                return self._handle.read(size)

        def grow_manifest() -> None:
            nonlocal grew
            if grew:
                return
            grew = True
            with manifest.open('ab') as output:
                output.write(b'\n#' + (b'x' * manifest_limit))

        def grow_from_path_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
            result = real_path_stat(path, *args, **kwargs)
            if (result.st_dev, result.st_ino) == manifest_identity:
                grow_manifest()
            return result

        def grow_from_fstat(fd: int) -> os.stat_result:
            result = real_fstat(fd)
            if (result.st_dev, result.st_ino) == manifest_identity:
                grow_manifest()
            return result

        def record_manifest_reads(fd: int, *args: object, **kwargs: object) -> Any:
            handle = real_fdopen(fd, *args, **kwargs)
            opened = real_fstat(fd)
            if (opened.st_dev, opened.st_ino) == manifest_identity:
                return RecordingHandle(handle)
            return handle

        monkeypatch.setattr(Path, 'stat', grow_from_path_stat)
        monkeypatch.setattr(os, 'fstat', grow_from_fstat)
        monkeypatch.setattr(os, 'fdopen', record_manifest_reads)

        with pytest.raises(ValidationError, match=f'manifest_max_bytes={manifest_limit}'):
            ModuleLoader(config).resolve(manifest)

        assert grew
        assert read_sizes
        assert all(0 < size <= manifest_limit + 1 for size in read_sizes)

    @pytest.mark.skipif(
        os.name == 'nt',
        reason='os.fstat replacement injection is POSIX-only; Win32 target handles deny replacement',
    )
    def test_module_source_swap_during_descriptor_read_fails_in_resolve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, _source_sha = _write_module(tmp_path)
        source = tmp_path / 'test_module.py'
        original_source = tmp_path / 'original_test_module.py'
        replacement = tmp_path / 'replacement.py'
        replacement.write_text('def register_module(ctx):\n    pass\n', encoding='utf-8')
        source_identity = (source.stat().st_dev, source.stat().st_ino)
        real_fstat = os.fstat
        swapped = False

        def swap_after_open(fd: int) -> os.stat_result:
            nonlocal swapped
            result = real_fstat(fd)
            if not swapped and (result.st_dev, result.st_ino) == source_identity:
                swapped = True
                os.replace(source, original_source)
                os.replace(replacement, source)
            return result

        monkeypatch.setattr(os, 'fstat', swap_after_open)

        with pytest.raises(ValidationError, match='changed during read'):
            ModuleLoader().resolve(manifest)

        assert swapped

    @pytest.mark.skipif(os.name == 'nt', reason='requires POSIX symlink semantics')
    def test_module_manifest_rejects_symlinked_intermediate_ancestor(
        self,
        tmp_path: Path,
    ) -> None:
        real_parent = tmp_path / 'real-parent'
        real_parent.mkdir()
        manifest, _source_sha = _write_module(real_parent)
        alias_parent = tmp_path / 'alias-parent'
        alias_parent.symlink_to(real_parent, target_is_directory=True)

        with pytest.raises(ValidationError, match='symlinks are not supported'):
            ModuleLoader().resolve(alias_parent / manifest.name)

    @pytest.mark.skipif(os.name == 'nt', reason='requires POSIX symlink semantics')
    def test_module_package_root_rejects_symlinked_intermediate_ancestor(
        self,
        tmp_path: Path,
    ) -> None:
        real_parent = tmp_path / 'real-parent'
        package = real_parent / 'tiny'
        package.mkdir(parents=True)
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'main.py').write_text(
            'def register_module(ctx):\n    pass\n',
            encoding='utf-8',
        )
        alias_parent = tmp_path / 'alias-parent'
        alias_parent.symlink_to(real_parent, target_is_directory=True)

        with pytest.raises(ValidationError, match='securely open module package'):
            ModuleLoader()._read_package_source_files(
                tmp_path.resolve(),
                alias_parent / 'tiny',
            )

    @pytest.mark.skipif(os.name == 'nt', reason='requires POSIX symlink semantics')
    def test_module_resolve_rejects_symlinked_package_ancestor(
        self,
        tmp_path: Path,
    ) -> None:
        package = tmp_path / 'realpkg'
        package.mkdir()
        (package / '__init__.py').write_text('', encoding='utf-8')
        (package / 'main.py').write_text(
            'def register_module(ctx):\n    pass\n',
            encoding='utf-8',
        )
        package_sha = _module_package_sha(tmp_path, package)
        alias = tmp_path / 'aliaspkg'
        alias.symlink_to(package, target_is_directory=True)
        manifest = tmp_path / 'module.yaml'
        manifest.write_text(
            f'''\
schema_version: 1
module_id: symlinked-package:v0
name: Symlinked package
entrypoint: aliaspkg.main:register_module
provides: {{}}
sha256: {package_sha}
''',
            encoding='utf-8',
        )

        with pytest.raises(ValidationError, match='symlinks are not supported'):
            ModuleLoader().resolve(manifest)

    def test_module_package_file_limit_stops_lazy_enumeration(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, _package_sha = _write_tiny_multifile_module(tmp_path)
        package = (tmp_path / 'tiny').resolve()
        real_rglob = Path.rglob
        yielded = 0

        def counting_rglob(path: Path, pattern: str):
            iterator = real_rglob(path, pattern)
            if path != package:
                return iterator

            def counted():
                nonlocal yielded
                for item in iterator:
                    yielded += 1
                    yield item

            return counted()

        monkeypatch.setattr(Path, 'rglob', counting_rglob)
        config = replace(
            DEFAULT_CONFIG,
            modules=replace(DEFAULT_CONFIG.modules, max_package_files=1),
        )

        with pytest.raises(ValidationError, match='max_package_files=1'):
            ModuleLoader(config).resolve(manifest)

        assert yielded <= 2

    def test_module_package_directory_replacement_during_enumeration_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        if os.name == 'nt':
            pytest.skip('POSIX descriptor replacement fixture')
        manifest, _package_sha = _write_tiny_multifile_module(tmp_path)
        package = tmp_path / 'tiny'
        parked = tmp_path / 'tiny-original'
        package_identity = (package.stat().st_dev, package.stat().st_ino)
        real_fstat = os.fstat
        swapped = False

        def replace_after_directory_open(fd: int) -> os.stat_result:
            nonlocal swapped
            result = real_fstat(fd)
            if not swapped and (result.st_dev, result.st_ino) == package_identity:
                swapped = True
                os.replace(package, parked)
                shutil.copytree(parked, package)
            return result

        monkeypatch.setattr(os, 'fstat', replace_after_directory_open)

        with pytest.raises(
            ValidationError,
            match='changed during enumeration|cannot securely open module source',
        ):
            ModuleLoader().resolve(manifest)

        assert swapped

    @pytest.mark.skipif(
        sys.platform != 'win32',
        reason='requires real Win32 package handles',
    )
    def test_windows_multifile_module_reads_through_secure_handle_chain(
        self,
        tmp_path: Path,
    ) -> None:
        manifest, package_sha = _write_tiny_multifile_module(tmp_path)

        source = ModuleLoader().resolve(manifest)

        assert source.source_kind == 'package'
        assert source.source_sha256 == package_sha
        assert {record.module_path for record in source.source_files} == {
            '__init__.py',
            'a.py',
            'b.py',
            'main.py',
        }

    def test_failed_module_does_not_leave_partial_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, invalid_registration=True)
            runtime = Runtime.open()
            try:
                with pytest.raises(ValidationError):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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
            original_register = runtime.image_registry.register_module_image
            try:
                def fail_register(*args, **kwargs):
                    raise RuntimeError('image register exploded')

                runtime.image_registry.register_module_image = fail_register
                with pytest.raises(RuntimeError, match='image register exploded'):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
                assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
                assert 'module-agent:v0' not in runtime.images
                assert runtime.modules.inspect_module('test-module:v0')['status'] == 'failed'
            finally:
                runtime.image_registry.register_module_image = original_register
                runtime.close()

    def test_late_module_audit_failure_never_publishes_image_to_concurrent_readers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, source_sha = _write_module(tmp_path)
        trust = _module_trust_key('test-module:v0', manifest, source_sha)
        runtime = Runtime.open()
        audit_entered = threading.Event()
        release_audit = threading.Event()
        outcomes: list[BaseException] = []
        try:
            real_record = runtime.audit.record

            def fail_late_module_audit(*args, **kwargs):
                if kwargs.get('action') == 'module.load':
                    audit_entered.set()
                    if not release_audit.wait(timeout=5):
                        raise RuntimeError('timed out waiting to fail module load audit')
                    raise RuntimeError('injected late module load audit failure')
                return real_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', fail_late_module_audit)

            def load_module() -> None:
                try:
                    runtime.modules.load_module_manifest(
                        manifest,
                        trusted_modules=(trust,),
                    )
                except BaseException as exc:
                    outcomes.append(exc)

            thread = threading.Thread(target=load_module, daemon=True)
            thread.start()
            assert audit_entered.wait(timeout=5)

            assert 'module-agent:v0' not in runtime.images
            assert runtime.llm._images.get('module-agent:v0') is None
            with pytest.raises(NotFound):
                runtime.launch.require_image('module-agent:v0')

            release_audit.set()
            thread.join(timeout=10)

            assert not thread.is_alive()
            assert len(outcomes) == 1
            assert isinstance(outcomes[0], RuntimeError)
            assert 'injected late module load audit failure' in str(outcomes[0])
            assert 'module-agent:v0' not in runtime.images
            assert runtime.store.get_image('module-agent:v0') is None
        finally:
            release_audit.set()
            runtime.close()

    def test_module_rollback_removes_image_cache_only_after_durable_commit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, source_sha = _write_module(tmp_path)
        trust = _module_trust_key('test-module:v0', manifest, source_sha)
        runtime = Runtime.open()
        audit_entered = threading.Event()
        release_audit = threading.Event()
        outcomes: list[BaseException] = []
        try:
            runtime.modules.load_module_manifest(
                manifest,
                trusted_modules=(trust,),
            )
            original = runtime.get_image('module-agent:v0')
            real_record = runtime.audit.record

            def block_module_rollback_audit(*args, **kwargs):
                if kwargs.get('action') == 'module.rollback':
                    audit_entered.set()
                    if not release_audit.wait(timeout=5):
                        raise RuntimeError('timed out waiting to commit module rollback')
                return real_record(*args, **kwargs)

            monkeypatch.setattr(runtime.audit, 'record', block_module_rollback_audit)

            def rollback_module() -> None:
                try:
                    runtime.modules._rollback_module('test-module:v0')
                except BaseException as exc:
                    outcomes.append(exc)

            thread = threading.Thread(target=rollback_module, daemon=True)
            thread.start()
            assert audit_entered.wait(timeout=5)

            assert runtime.get_image('module-agent:v0') == original
            assert runtime.launch.require_image('module-agent:v0') == original
            assert runtime.llm._images.get('module-agent:v0') == original

            release_audit.set()
            thread.join(timeout=10)

            assert not thread.is_alive()
            assert outcomes == []
            assert 'module-agent:v0' not in runtime.images
            assert runtime.store.get_image('module-agent:v0') is None
        finally:
            release_audit.set()
            runtime.close()

    def test_invalid_module_image_does_not_leave_partial_tool_registration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root, invalid_image=True)
            runtime = Runtime.open()
            try:
                with pytest.raises(ValidationError):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
                with pytest.raises(NotFound):
                    runtime.tools.resolve('module_echo')
            finally:
                runtime.close()

    @pytest.mark.parametrize('load_policy', ['fail', 'warn'])
    def test_startup_hook_failure_rolls_back_external_module_state(
        self,
        load_policy: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root, failing_startup_hook=True)
            config = replace(
                DEFAULT_CONFIG,
                modules=replace(DEFAULT_CONFIG.modules, load_policy=load_policy),
            )
            with pytest.raises(RuntimeError, match='startup hook failed'):
                Runtime.open(
                    db,
                    config=config,
                    module_manifests=(str(manifest),),
                    trusted_modules=(
                        _module_trust_key(
                            'test-module:v0',
                            manifest,
                            source_sha,
                        ),
                    ),
                )

            runtime = Runtime.open(db)
            try:
                failed = runtime.modules.inspect_module('test-module:v0')
                assert failed['status'] == 'failed'
                assert failed['error'].startswith(
                    'module_load_failed: RuntimeError'
                )
                assert 'startup hook failed' not in failed['error']
                assert 'module-agent:v0' not in runtime.images
                assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
                assert runtime.syscalls.get('module.ping') is None
            finally:
                runtime.close()

    def test_startup_hook_failure_rolls_back_entire_external_startup_batch(
        self,
        tmp_path: Path,
    ) -> None:
        successful_root = tmp_path / 'successful'
        failing_root = tmp_path / 'failing'
        successful_root.mkdir()
        failing_root.mkdir()
        successful_manifest, successful_sha = _write_module(successful_root)
        failing_manifest, failing_sha = _write_mutating_hook_module(failing_root)
        database = tmp_path / 'runtime.sqlite'

        with pytest.raises(RuntimeError, match='mutating startup hook failed'):
            Runtime.open(
                database,
                module_manifests=(str(successful_manifest), str(failing_manifest)),
                trusted_modules=(
                    _module_trust_key(
                        'test-module:v0',
                        successful_manifest,
                        successful_sha,
                    ),
                    _module_trust_key(
                        'mutating-hook-module:v0',
                        failing_manifest,
                        failing_sha,
                    ),
                ),
            )

        runtime = Runtime.open(database)
        try:
            assert runtime.modules.inspect_module('test-module:v0')['status'] == 'failed'
            assert (
                runtime.modules.inspect_module('mutating-hook-module:v0')['status']
                == 'failed'
            )
            with pytest.raises(NotFound):
                runtime.tools.resolve('module_echo')
            assert 'module-agent:v0' not in runtime.images
            assert runtime.syscalls.get('module.ping') is None
        finally:
            runtime.close()

    @pytest.mark.parametrize(
        ('startup_hook_exception', 'exception_type', 'error_message'),
        [
            (
                "KeyboardInterrupt('startup hook interrupted token=secret /opaque/private/hook.py payload=opaque')",
                KeyboardInterrupt,
                'startup hook interrupted token=secret /opaque/private/hook.py payload=opaque',
            ),
            (
                "__import__('asyncio').CancelledError('startup hook cancelled token=secret /opaque/private/hook.py payload=opaque')",
                asyncio.CancelledError,
                'startup hook cancelled token=secret /opaque/private/hook.py payload=opaque',
            ),
        ],
        ids=('keyboard-interrupt', 'cancelled-error'),
    )
    def test_startup_hook_base_exception_rolls_back_external_module_state(
        self,
        tmp_path: Path,
        startup_hook_exception: str,
        exception_type: type[BaseException],
        error_message: str,
    ) -> None:
        db = tmp_path / 'runtime.sqlite'
        manifest, source_sha = _write_module(
            tmp_path,
            startup_hook_exception=startup_hook_exception,
        )
        trust = _module_trust_key('test-module:v0', manifest, source_sha)

        with pytest.raises(exception_type, match=error_message):
            Runtime.open(
                db,
                module_manifests=(str(manifest),),
                trusted_modules=(trust,),
            )

        runtime = Runtime.open(db)
        try:
            failed = runtime.modules.inspect_module('test-module:v0')
            assert failed['status'] == 'failed'
            assert failed['error'].startswith(
                f'module_load_failed: {exception_type.__name__}'
            )
            assert error_message not in failed['error']
            failure_audits = [
                record.decision
                for record in runtime.audit.trace()
                if record.action == 'module.load_failed'
            ]
            projected = json.dumps(
                {'failed': failed, 'audit': failure_audits},
                sort_keys=True,
            )
            assert error_message not in projected
            assert '/opaque/private/hook.py' not in projected
            assert failed['metadata']['failure']['correlation_id'] in projected
            assert 'module-agent:v0' not in runtime.images
            assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
            assert runtime.syscalls.get('module.ping') is None
        finally:
            runtime.close()

    def test_startup_hook_rollback_audit_failure_recovers_persistent_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / 'runtime.sqlite'
        manifest, source_sha = _write_module(tmp_path, failing_startup_hook=True)
        trust = _module_trust_key('test-module:v0', manifest, source_sha)
        original_record = AuditManager.record

        def fail_rollback_audit(self, *args, **kwargs):
            action = kwargs.get('action')
            if action is None and len(args) > 1:
                action = args[1]
            if action == 'module.rollback':
                raise RuntimeError('rollback audit sink failed')
            return original_record(self, *args, **kwargs)

        monkeypatch.setattr(AuditManager, 'record', fail_rollback_audit)

        with pytest.raises(RuntimeError, match='startup hook failed'):
            Runtime.open(
                db,
                module_manifests=(str(manifest),),
                trusted_modules=(trust,),
            )

        runtime = Runtime.open(db)
        try:
            failed = runtime.modules.inspect_module('test-module:v0')
            assert failed['status'] == 'failed'
            assert failed['error'].startswith(
                'module_load_failed: RuntimeError'
            )
            assert 'startup hook failed' not in failed['error']
            assert 'module-agent:v0' not in runtime.images
            assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
            assert any(
                record.action == 'module.rollback_recovered'
                for record in runtime.audit.trace()
            )
        finally:
            runtime.close()

    def test_module_rollback_commit_failure_recovers_persistent_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / 'runtime.sqlite'
        manifest, source_sha = _write_module(tmp_path)
        trust = _module_trust_key('test-module:v0', manifest, source_sha)
        runtime = Runtime.open(
            db,
            module_manifests=(str(manifest),),
            trusted_modules=(trust,),
        )
        original_transaction = runtime.modules._module_publications.transaction
        fail_commit = True

        @contextlib.contextmanager
        def fail_first_commit(*, include_object_payloads: bool = False):
            nonlocal fail_commit
            with original_transaction(
                include_object_payloads=include_object_payloads,
            ):
                yield
                if fail_commit:
                    fail_commit = False
                    raise RuntimeError('simulated rollback commit failure')

        monkeypatch.setattr(
            runtime.modules._module_publications,
            'transaction',
            fail_first_commit,
        )
        try:
            runtime.modules._rollback_module('test-module:v0')

            assert not runtime.modules.is_loaded('test-module:v0')
            assert 'module-agent:v0' not in runtime.images
            assert all(row['name'] != 'module_echo' for row in runtime.store.list_tools())
            assert runtime.modules.inspect_module('test-module:v0')['status'] == 'failed'
        finally:
            runtime.close()

        reopened = Runtime.open(db)
        try:
            assert reopened.modules.inspect_module('test-module:v0')['status'] == 'failed'
            assert 'module-agent:v0' not in reopened.images
            assert all(row['name'] != 'module_echo' for row in reopened.store.list_tools())
        finally:
            reopened.close()

    def test_rehydrated_module_image_survives_durable_rollback_recovery(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        db = tmp_path / 'runtime.sqlite'
        manifest, source_sha = _write_module(tmp_path)
        trust = _module_trust_key('test-module:v0', manifest, source_sha)
        runtime = Runtime.open(
            db,
            module_manifests=(str(manifest),),
            trusted_modules=(trust,),
        )
        try:
            persisted_before = runtime.store.get_image('module-agent:v0')
            assert persisted_before is not None
        finally:
            runtime.close()

        manifest, failing_sha = _write_module(tmp_path, failing_startup_hook=True)
        failing_trust = _module_trust_key(
            'test-module:v0',
            manifest,
            failing_sha,
        )
        original_record = AuditManager.record

        def fail_rollback_audit(self: Any, *args: Any, **kwargs: Any) -> Any:
            action = kwargs.get('action')
            if action is None and len(args) > 1:
                action = args[1]
            if action == 'module.rollback':
                raise RuntimeError('rollback audit sink failed')
            return original_record(self, *args, **kwargs)

        monkeypatch.setattr(AuditManager, 'record', fail_rollback_audit)
        with pytest.raises(RuntimeError, match='startup hook failed'):
            Runtime.open(
                db,
                module_manifests=(str(manifest),),
                trusted_modules=(failing_trust,),
            )

        monkeypatch.setattr(AuditManager, 'record', original_record)
        reopened = Runtime.open(db)
        try:
            persisted_after = reopened.store.get_image('module-agent:v0')
            assert persisted_after == persisted_before
            assert reopened.get_image('module-agent:v0') == persisted_before[0]
        finally:
            reopened.close()

    def test_failing_startup_hook_cannot_leave_direct_runtime_registrations(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_mutating_hook_module(root)
            trust = _module_trust_key('mutating-hook-module:v0', manifest, source_sha)

            with pytest.raises(RuntimeError, match='mutating startup hook failed'):
                Runtime.open(db, module_manifests=(str(manifest),), trusted_modules=(trust,))

            runtime = Runtime.open(db)
            try:
                failed = runtime.modules.inspect_module('mutating-hook-module:v0')
                assert failed['status'] == 'failed'
                assert failed['error'].startswith(
                    'module_load_failed: RuntimeError'
                )
                assert 'mutating startup hook failed' not in failed['error']
                assert 'hook-direct-image:v0' not in runtime.images
                assert all(row['name'] != 'hook_direct_tool' for row in runtime.store.list_tools())
                assert runtime.store.get_image('hook-direct-image:v0') is None
                assert runtime.syscalls.get('hook.direct') is None
            finally:
                runtime.close()

    def test_runtime_loaded_failing_hook_restores_in_memory_registries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_mutating_hook_module(root)
            trust = _module_trust_key('mutating-hook-module:v0', manifest, source_sha)
            runtime = Runtime.open()
            try:
                before_provider_hooks = {
                    kind: list(hooks)
                    for kind, hooks in runtime.provider_hooks.items()
                }
                before_shutdown_finalizers = runtime.lifecycle.finalizers_snapshot()
                before_release_finalizers = list(runtime.memory._object_release_finalizers)
                assert runtime.module_state.get('_agent_libos_pty_adapter') is None
                assert not hasattr(runtime.substrate, 'hook_direct_provider')
                with pytest.raises(RuntimeError, match='mutating startup hook failed'):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(trust,))

                assert 'hook-direct-image:v0' not in runtime.images
                assert all(row['name'] != 'hook_direct_tool' for row in runtime.store.list_tools())
                assert runtime.store.get_image('hook-direct-image:v0') is None
                assert runtime.syscalls.get('hook.direct') is None
                assert runtime.provider_hooks == before_provider_hooks
                assert runtime.lifecycle.finalizers_snapshot() == before_shutdown_finalizers
                assert runtime.memory._object_release_finalizers == before_release_finalizers
                assert runtime.module_state.get('_agent_libos_pty_adapter') is None
                assert not hasattr(runtime.substrate, 'hook_direct_provider')
                assert runtime.modules.inspect_module('mutating-hook-module:v0')['status'] == 'failed'
            finally:
                runtime.close()

    def test_failed_module_rollback_quarantines_tool_and_retries_cleanup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        manifest, source_sha = _write_mutating_hook_module(tmp_path)
        trust = _module_trust_key('mutating-hook-module:v0', manifest, source_sha)
        runtime = Runtime.open()
        original_unregister = runtime.tools.unregister_tool
        attempts = 0

        def transient_unregister_failure(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('transient tool unregister failure')
            return original_unregister(*args, **kwargs)

        monkeypatch.setattr(runtime.tools, 'unregister_tool', transient_unregister_failure)
        try:
            with pytest.raises(
                RegistrationRollbackError,
                match='module_rollback_failed: RegistrationRollbackError',
            ) as caught:
                runtime.modules.load_module_manifest(manifest, trusted_modules=(trust,))

            assert 'transient tool unregister failure' not in str(caught.value)

            with pytest.raises(NotFound):
                runtime.tools.resolve('hook_direct_tool')
            assert all(
                handle.name != 'hook_direct_tool'
                for handle in runtime.tools.loaded_tool_handles()
            )
            assert all(row['name'] != 'hook_direct_tool' for row in runtime.store.list_tools())
            assert runtime.modules.inspect_module('mutating-hook-module:v0')['status'] == 'failed'
            assert any(
                record.action == 'module.load_failed'
                and record.target == 'module:mutating-hook-module:v0'
                for record in runtime.audit.trace()
            )
            pending = runtime.modules._registration_journals['mutating-hook-module:v0']
            assert pending.size == 1
            assert not pending.rolled_back

            with pytest.raises(RuntimeError, match='mutating startup hook failed'):
                runtime.modules.load_module_manifest(manifest, trusted_modules=(trust,))

            assert attempts == 3
            assert 'mutating-hook-module:v0' not in runtime.modules._registration_journals
            with pytest.raises(NotFound):
                runtime.tools.resolve('hook_direct_tool')
        finally:
            runtime.close()

    def test_runtime_loaded_failing_hook_preserves_previously_loaded_module(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            successful_root = root / 'successful'
            failing_root = root / 'failing'
            successful_root.mkdir()
            failing_root.mkdir()
            successful_manifest, successful_sha = _write_module(successful_root)
            failing_manifest, failing_sha = _write_mutating_hook_module(failing_root)
            runtime = Runtime.open(
                module_manifests=(str(successful_manifest),),
                trusted_modules=(_module_trust_key('test-module:v0', successful_manifest, successful_sha),),
            )
            try:
                with pytest.raises(RuntimeError, match='mutating startup hook failed'):
                    runtime.modules.load_module_manifest(
                        failing_manifest,
                        trusted_modules=(_module_trust_key('mutating-hook-module:v0', failing_manifest, failing_sha),),
                    )

                assert runtime.modules.inspect_module('test-module:v0')['status'] == 'loaded'
                assert runtime.tools.resolve('module_echo').name == 'module_echo'
                assert 'module-agent:v0' in runtime.images
                assert runtime.syscalls.get('module.ping') is not None
            finally:
                runtime.close()

    def test_failed_module_rollback_cannot_clobber_concurrent_successful_module_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        failing_root = tmp_path / 'failing'
        successful_root = tmp_path / 'successful'
        failing_root.mkdir()
        successful_root.mkdir()
        failing_manifest, failing_sha = _write_mutating_hook_module(failing_root)
        successful_manifest, successful_sha = _write_module(successful_root)
        failing_trust = _module_trust_key('mutating-hook-module:v0', failing_manifest, failing_sha)
        successful_trust = _module_trust_key('test-module:v0', successful_manifest, successful_sha)
        runtime = Runtime.open()
        failing_hook_entered = threading.Event()
        release_failing_hook = threading.Event()
        successful_load_entered = threading.Event()
        original_run_hook = runtime.modules._run_module_hook
        original_require_available = runtime.modules._require_module_id_available
        failures: list[BaseException] = []

        def blocked_run_hook(
            module_id: str,
            hook_name: str,
            hook: object,
            *,
            kind: str,
        ) -> None:
            if module_id == 'mutating-hook-module:v0':
                failing_hook_entered.set()
                if not release_failing_hook.wait(timeout=5):
                    raise TimeoutError('timed out waiting to release failing module hook')
            original_run_hook(module_id, hook_name, hook, kind=kind)  # type: ignore[arg-type]

        def observed_require_available(module_id: str, source_sha256: str) -> None:
            if module_id == 'test-module:v0':
                successful_load_entered.set()
            original_require_available(module_id, source_sha256)

        monkeypatch.setattr(runtime.modules, '_run_module_hook', blocked_run_hook)
        monkeypatch.setattr(runtime.modules, '_require_module_id_available', observed_require_available)

        def load_failing() -> None:
            try:
                runtime.modules.load_module_manifest(failing_manifest, trusted_modules=(failing_trust,))
            except BaseException as exc:
                failures.append(exc)

        def load_successful() -> None:
            try:
                runtime.modules.load_module_manifest(successful_manifest, trusted_modules=(successful_trust,))
            except BaseException as exc:
                failures.append(exc)

        failing_thread = threading.Thread(target=load_failing)
        successful_thread = threading.Thread(target=load_successful)
        try:
            failing_thread.start()
            assert failing_hook_entered.wait(timeout=3)
            successful_thread.start()
            assert not successful_load_entered.wait(timeout=0.2)

            release_failing_hook.set()
            failing_thread.join(timeout=5)
            successful_thread.join(timeout=5)

            assert not failing_thread.is_alive()
            assert not successful_thread.is_alive()
            assert len(failures) == 1
            assert 'mutating startup hook failed' in str(failures[0])
            assert successful_load_entered.is_set()
            assert runtime.modules.inspect_module('test-module:v0')['status'] == 'loaded'
            assert runtime.tools.resolve('module_echo').name == 'module_echo'
            assert 'module-agent:v0' in runtime.images
            assert runtime.syscalls.get('module.ping') is not None
            assert runtime.modules.inspect_module('mutating-hook-module:v0')['status'] == 'failed'
        finally:
            release_failing_hook.set()
            failing_thread.join(timeout=5)
            successful_thread.join(timeout=5)
            runtime.close()

    def test_duplicate_module_id_does_not_overwrite_loaded_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                with pytest.raises(ValidationError, match='already loaded'):
                    runtime.modules.load_module_manifest(manifest, trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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
            runtime = Runtime.open(db, module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
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

    def test_checkpoint_committed_image_carries_required_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root)
            runtime = Runtime.open(db, module_manifests=(str(manifest),), trusted_modules=(_module_trust_key('test-module:v0', manifest, source_sha),))
            try:
                pid = runtime.process.spawn(image='module-agent:v0', goal='commit module image')
                checkpoint_id = runtime.checkpoint.create(pid, 'module image checkpoint', actor=pid)
                runtime.image_registry.grant_register(pid, 'module-committed:v0', issued_by='test')
                result = runtime.image_registry.commit_from_checkpoint(
                    actor=pid,
                    checkpoint_id=checkpoint_id,
                    image_id='module-committed:v0',
                    name='module-committed',
                )
                assert {'module_id': 'test-module:v0', 'source_sha256': source_sha} in result.image.required_modules
            finally:
                runtime.close()

            reopened_without_module = Runtime.open(db)
            try:
                inspected = reopened_without_module.image_registry.inspect('module-committed:v0')
                assert {'module_id': 'test-module:v0', 'source_sha256': source_sha} in inspected['image']['required_modules']
                with pytest.raises(ValidationError, match='image requires startup modules'):
                    reopened_without_module.process.spawn(image='module-committed:v0', goal='blocked commit')
            finally:
                reopened_without_module.close()

    def test_cli_modules_verify_list_and_spawn_with_module_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            db = root / 'runtime.sqlite'
            manifest, source_sha = _write_module(root)
            trust = _module_trust_key('test-module:v0', manifest, source_sha)
            verified = _run_cli_json(['--db', str(db), 'modules', 'verify', str(manifest)])
            assert not verified['trusted']
            assert verified['trust_key'] == trust
            assert verified['source_sha256'] == source_sha
            listed = _run_cli_json(['--db', str(db), '--module-manifest', str(manifest), '--trusted-module', trust, 'modules', 'list'])
            assert 'test-module:v0' in [module['module_id'] for module in listed]
            spawned = _run_cli_json(['--db', str(db), '--module-manifest', str(manifest), '--trusted-module', trust, 'spawn', '--image', 'module-agent:v0', '--goal', 'cli module'])
            assert spawned['image'] == 'module-agent:v0'
            digest_pair = f"{verified['manifest_sha256']}:{verified['source_sha256']}"
            listed_by_digest_pair = _run_cli_json(
                [
                    '--db',
                    str(root / 'runtime-digest-pair.sqlite'),
                    '--module-manifest',
                    str(manifest),
                    '--trusted-module-sha256',
                    digest_pair,
                    'modules',
                    'list',
                ]
            )
            assert 'test-module:v0' in [module['module_id'] for module in listed_by_digest_pair]

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

def _write_module(
    root: Path,
    *,
    expose_read_tool: bool = False,
    invalid_registration: bool = False,
    invalid_image: bool = False,
    provider_hook: bool = False,
    failing_startup_hook: bool = False,
    startup_hook_exception: str | None = None,
    entrypoint: str = './test_module.py:register_module',
    marker: str = 'module',
) -> tuple[Path, str]:
    source = root / 'test_module.py'
    default_tools = ['module_echo', 'read_text_file'] if expose_read_tool else ['module_echo']
    if invalid_image:
        default_tools.append('missing_module_tool')
    provider_hook_code = '\ndef mark_provider(runtime):\n    runtime.audit.record(actor="module:test-module:v0", action="test.provider_hook", target="runtime")\n'.rstrip() if provider_hook else ''
    provider_hook_registration = "ctx.register_provider_hook('test_hook', mark_provider)" if provider_hook else ''
    if startup_hook_exception is not None:
        startup_hook_body = f'raise {startup_hook_exception}'
    elif failing_startup_hook:
        startup_hook_body = "raise RuntimeError('startup hook failed')"
    else:
        startup_hook_body = 'runtime.audit.record(actor="module:test-module:v0", action="test.startup_hook", target="runtime")'
    source.write_text(f"""\nfrom pydantic import BaseModel\n\nfrom agent_libos.models import AgentImage\nfrom agent_libos.tools.base import SyncAgentTool, ToolContext\n\n\nclass EchoArgs(BaseModel):\n    text: str\n\n\nclass ModuleEchoTool(SyncAgentTool[EchoArgs]):\n    name = "module_echo"\n    description = "Echo text through a startup module."\n    args_schema = EchoArgs\n\n    def run(self, args: EchoArgs, ctx: ToolContext):\n        return {{"echo": args.text, "pid": ctx.pid, "marker": {marker!r}}}\n\n\ndef module_ping(session, args):\n    return {{"pid": session.pid, "value": args.get("value")}}\n\n\ndef mark_startup(runtime):\n    {startup_hook_body}\n\n\n{provider_hook_code}\n\n\ndef register_module(ctx):\n    ctx.register_tool(ModuleEchoTool())\n    {("ctx.register_syscall('undeclared.syscall', module_ping)" if invalid_registration else "ctx.register_syscall('module.ping', module_ping)")}\n    ctx.register_image(AgentImage(\n        image_id="module-agent:v0",\n        name="module-agent",\n        default_tools={default_tools!r},\n    ))\n    {provider_hook_registration}\n    ctx.add_startup_hook(mark_startup)\n""".lstrip(), encoding='utf-8')
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    syscalls = '[]\n' if invalid_registration else "['module.ping']\n"
    manifest = root / 'module.yaml'
    manifest.write_text(f"\nschema_version: 1\nmodule_id: test-module:v0\nname: Test startup module\nversion: v0\nentrypoint: {entrypoint}\nprovides:\n  tools: ['module_echo']\n  images: ['module-agent:v0']\n  syscalls: {syscalls.rstrip()}\n  provider_hooks: {(['test_hook'] if provider_hook else [])!r}\n  startup_hooks: ['mark_startup']\nsha256: {source_sha}\nmetadata:\n  test: true\n".lstrip(), encoding='utf-8')
    return (manifest, source_sha)


def _write_mutating_hook_module(root: Path) -> tuple[Path, str]:
    source = root / 'mutating_hook_module.py'
    source.write_text(
        """
from pydantic import BaseModel

from agent_libos.models import AgentImage
from agent_libos.tools.base import SyncAgentTool, ToolContext


class HookArgs(BaseModel):
    pass


class HookDirectTool(SyncAgentTool[HookArgs]):
    name = "hook_direct_tool"
    description = "Registered directly by a failing startup hook."
    args_schema = HookArgs

    def run(self, args: HookArgs, ctx: ToolContext):
        return {"ok": True}


def hook_syscall(session, args):
    return {"ok": True}


def mutating_hook(runtime):
    runtime.register_tool(HookDirectTool())
    runtime.register_image(AgentImage(image_id="hook-direct-image:v0", name="hook-direct-image"))
    runtime.register_syscall("hook.direct", hook_syscall)
    runtime.register_provider_hook("hook-direct", lambda _runtime: None)
    runtime.bind_shutdown_finalizer(lambda: True)
    runtime.memory.bind_object_release_finalizer(lambda _obj, _actor, _reason: None)
    runtime.set_runtime_attribute("_agent_libos_pty_adapter", object())
    runtime.set_substrate_attribute("hook_direct_provider", object())
    raise RuntimeError("mutating startup hook failed")


def register_module(ctx):
    ctx.add_startup_hook(mutating_hook)
""".lstrip(),
        encoding='utf-8',
    )
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = root / 'mutating-hook-module.yaml'
    manifest.write_text(
        f"""
schema_version: 1
module_id: mutating-hook-module:v0
name: Mutating hook module
version: v0
entrypoint: ./mutating_hook_module.py:register_module
provides:
  startup_hooks: ['mutating_hook']
sha256: {source_sha}
""".lstrip(),
        encoding='utf-8',
    )
    return manifest, source_sha


def _write_multifile_module(root: Path) -> tuple[Path, str]:
    package = root / 'pkg'
    package.mkdir()
    (package / '__init__.py').write_text(
        "raise RuntimeError('package init should not run for module entrypoint')\n",
        encoding='utf-8',
    )
    (package / 'helper.py').write_text(
        "HELPER_MARKER = 'helper-v1'\n\n"
        "def helper_payload(value):\n"
        "    return {'helper_marker': HELPER_MARKER, 'value': value}\n",
        encoding='utf-8',
    )
    (package / 'main.py').write_text(
        """
from pydantic import BaseModel

from agent_libos.models import AgentImage
from agent_libos.tools.base import SyncAgentTool, ToolContext

from .helper import HELPER_MARKER, helper_payload


class EchoArgs(BaseModel):
    text: str


class MultiEchoTool(SyncAgentTool[EchoArgs]):
    name = "multi_echo"
    description = "Echo text through a multi-file startup module."
    args_schema = EchoArgs

    def run(self, args: EchoArgs, ctx: ToolContext):
        payload = helper_payload(args.text)
        payload["pid"] = ctx.pid
        return payload


def multi_ping(session, args):
    payload = helper_payload(args.get("value"))
    payload["pid"] = session.pid
    return payload


def register_module(ctx):
    assert HELPER_MARKER == "helper-v1"
    ctx.register_tool(MultiEchoTool())
    ctx.register_syscall("multi.ping", multi_ping)
    ctx.register_image(AgentImage(
        image_id="multi-agent:v0",
        name="multi-agent",
        default_tools=["multi_echo"],
    ))
""".lstrip(),
        encoding='utf-8',
    )
    package_sha = _module_package_sha(root, package)
    manifest = root / 'module.yaml'
    manifest.write_text(
        f"""
schema_version: 1
module_id: multi-module:v0
name: Multi-file startup module
version: v0
entrypoint: pkg.main:register_module
provides:
  tools: ['multi_echo']
  images: ['multi-agent:v0']
  syscalls: ['multi.ping']
  provider_hooks: []
  startup_hooks: []
sha256: {package_sha}
""".lstrip(),
        encoding='utf-8',
    )
    return manifest, package_sha


def _write_tiny_multifile_module(root: Path) -> tuple[Path, str]:
    package = root / 'tiny'
    package.mkdir()
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'main.py').write_text('from .a import A\n\ndef register_module(ctx):\n    _ = A\n', encoding='utf-8')
    (package / 'a.py').write_text(f"A = {('a' * 70)!r}\n", encoding='utf-8')
    (package / 'b.py').write_text(f"B = {('b' * 70)!r}\n", encoding='utf-8')
    package_sha = _module_package_sha(root, package)
    manifest = root / 'module.yaml'
    manifest.write_text(
        f"""
schema_version: 1
module_id: tiny-module:v0
name: Tiny multi-file startup module
entrypoint: tiny.main:register_module
provides: {{}}
sha256: {package_sha}
""".lstrip(),
        encoding='utf-8',
    )
    return manifest, package_sha


def _module_package_sha(manifest_dir: Path, source_root: Path) -> str:
    loader = ModuleLoader()
    return loader._package_sha256(loader._read_package_source_files(manifest_dir.resolve(), source_root.resolve()))


def _module_trust_key(module_id: str, manifest: Path, source_sha: str) -> str:
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return ModuleLoader.trust_key(module_id, manifest_sha, source_sha)


def _run_cli_json(argv: list[str]) -> object:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        cli_main(argv)
    return json.loads(buffer.getvalue())
