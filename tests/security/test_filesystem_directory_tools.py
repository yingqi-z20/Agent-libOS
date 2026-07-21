from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
import pytest
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, NotFound, ValidationError
from agent_libos.models import (
    CapabilityRight,
    EventType,
    ExternalEffectRollbackClass,
    ExternalEffectRollbackStatus,
    HumanRequestStatus,
    ResourceBudget,
)
from agent_libos.substrate import (
    HierarchicalPathLock,
    LocalFilesystemProvider,
    LocalResourceProviderSubstrate,
    ProviderEffectNotStarted,
    ResolvedPath,
)
_TOOL_DEFAULTS = DEFAULT_CONFIG.tools
DEFAULT_FILESYSTEM_READ_HARD_LIMIT = _TOOL_DEFAULTS.filesystem_read_hard_limit_bytes
DEFAULT_DIRECTORY_ENTRY_HARD_LIMIT = _TOOL_DEFAULTS.directory_entry_hard_limit

class TestFilesystemDirectoryTool:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')
        self.human_output: list[str] = []
        self.runtime.substrate.human.output_sink = self.human_output.append

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_read_write_and_delete_directory_and_file_tools(self) -> None:
        base = f'agent_outputs/fs_ops_{uuid4().hex}'
        existing_file = self._write_fixture(f'{base}/existing.txt', 'existing')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='filesystem ops')
        self.runtime.filesystem.grant_path_list(pid, read_dirs=[base], write_dirs=[base], delete_dirs=[base], issued_by='test')
        listed = self.runtime.tools.call(pid, 'read_directory', {'path': base})
        made_dir = self.runtime.tools.call(pid, 'write_directory', {'path': f'{base}/created/nested'})
        wrote_file = self.runtime.tools.call(pid, 'write_text_file', {'path': f'{base}/created/nested/out.txt', 'content': 'created'})
        deleted_file = self.runtime.tools.call(pid, 'delete_file', {'path': existing_file})
        deleted_dir = self.runtime.tools.call(pid, 'delete_directory', {'path': f'{base}/created', 'recursive': True})
        assert listed.ok, listed.error
        assert [entry['name'] for entry in listed.payload['entries']] == ['existing.txt']
        assert made_dir.ok, made_dir.error
        assert made_dir.payload['created']
        assert wrote_file.ok, wrote_file.error
        assert deleted_file.ok, deleted_file.error
        assert not (self.runtime.workspace_root / existing_file).exists()
        assert deleted_dir.ok, deleted_dir.error
        assert not (self.runtime.workspace_root / base / 'created').exists()
        mutation_effects = [
            effect
            for effect in self.runtime.store.list_external_effects(pid=pid)
            if effect.state_mutation
        ]
        assert len(mutation_effects) == 4
        assert {effect.operation for effect in mutation_effects} == {
            'make_directory',
            'write_text',
            'delete_file',
            'delete_directory',
        }
        assert all(
            effect.rollback_class == ExternalEffectRollbackClass.IRREVERSIBLE
            and effect.rollback_status == ExternalEffectRollbackStatus.NOT_SUPPORTED
            for effect in mutation_effects
        )

    def test_filesystem_post_provider_event_failure_leaves_durable_unknown_effect_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = f'agent_outputs/effect_intent_{uuid4().hex}.txt'
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='durable filesystem effect intent')
        self.runtime.filesystem.grant_path(pid, path, [CapabilityRight.WRITE], issued_by='test')
        original_emit = self.runtime.events.emit

        def fail_write_event(event_type: EventType | str, *args: object, **kwargs: object) -> object:
            if EventType(event_type) == EventType.EXTERNAL_WRITE and kwargs.get('target') == self.runtime.filesystem.resource_for_path(path):
                raise RuntimeError('injected filesystem result event failure')
            return original_emit(event_type, *args, **kwargs)

        monkeypatch.setattr(self.runtime.events, 'emit', fail_write_event)
        with pytest.raises(RuntimeError, match='injected filesystem result event failure'):
            self.runtime.filesystem.write_text(pid, path, 'provider committed')

        assert (self.runtime.workspace_root / path).read_text(encoding='utf-8') == 'provider committed'
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
        assert effects[0].provider_metadata['effect_state'] == 'pending'

    @pytest.mark.parametrize('operation', ['read_text', 'read_bytes', 'read_directory'])
    def test_one_time_read_covers_missing_path_state_probe(self, operation: str) -> None:
        path = f'agent_outputs/missing_probe_{uuid4().hex}'
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='one-shot state probe')
        resource = (
            self.runtime.filesystem.directory_resource_for_path(path)
            if operation == 'read_directory'
            else self.runtime.filesystem.resource_for_path(path)
        )
        cap = self.runtime.capability.grant_once(pid, resource, [CapabilityRight.READ], issued_by='test')

        with pytest.raises(NotFound) as first_error:
            getattr(self.runtime.filesystem, operation)(pid, path)
        assert 'does not exist' in str(first_error.value)

        persisted = self.runtime.store.get_capability(cap.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.NOT_REQUIRED
        assert effects[0].state_mutation is False
        assert effects[0].information_flow is True
        assert effects[0].provider_metadata['outcome'] == 'rejected_after_state_observation'
        with pytest.raises(CapabilityDenied):
            getattr(self.runtime.filesystem, operation)(pid, path)

    def test_validate_directory_consumes_one_time_read_and_records_state_effect(self) -> None:
        path = f'agent_outputs/cwd_state_{uuid4().hex}'
        (self.runtime.workspace_root / path).mkdir(parents=True)
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='validate cwd state')
        resource = self.runtime.filesystem.directory_resource_for_path(path)
        cap = self.runtime.capability.grant_once(pid, resource, [CapabilityRight.READ], issued_by='test')

        assert self.runtime.filesystem.validate_directory(pid, path) == path

        persisted = self.runtime.store.get_capability(cap.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        effect = effects[0]
        assert effect.provider == 'filesystem'
        assert effect.operation == 'state'
        assert effect.target == resource
        assert effect.effect_state == 'finalized'
        assert effect.rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        assert effect.rollback_status == ExternalEffectRollbackStatus.NOT_REQUIRED
        assert effect.information_flow
        assert self.runtime.process.get(pid).resource_usage.external_read_bytes > 0

    @pytest.mark.parametrize('outcome', ['not_found', 'not_directory'])
    def test_validate_directory_finalizes_failed_state_observation(self, outcome: str) -> None:
        path = f'agent_outputs/cwd_state_{outcome}_{uuid4().hex}'
        if outcome == 'not_directory':
            self._write_fixture(path, 'not a directory')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='validate invalid cwd state')
        resource = self.runtime.filesystem.directory_resource_for_path(path)
        cap = self.runtime.capability.grant_once(pid, resource, [CapabilityRight.READ], issued_by='test')

        with pytest.raises(NotFound, match='working directory'):
            self.runtime.filesystem.validate_directory(pid, path)

        persisted = self.runtime.store.get_capability(cap.cap_id)
        assert persisted is not None and persisted.uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].rollback_class == ExternalEffectRollbackClass.NO_ROLLBACK_REQUIRED
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.NOT_REQUIRED
        assert effects[0].provider_metadata['result']['outcome'] == outcome

    def test_write_local_type_rejection_finalizes_read_only_state_effect(self) -> None:
        path = f'agent_outputs/write_directory_target_{uuid4().hex}'
        (self.runtime.workspace_root / path).mkdir(parents=True)
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='reject directory write target')
        resource = self.runtime.filesystem.resource_for_path(path)
        self.runtime.capability.issue_trusted(
            pid,
            resource,
            [CapabilityRight.WRITE],
            issued_by='test',
        )

        with pytest.raises(CapabilityDenied, match='path is not a file'):
            self.runtime.filesystem.write_text(pid, path, 'content')

        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        effect = effects[0]
        assert effect.effect_state == 'finalized'
        assert effect.transaction_state == 'committed'
        assert effect.state_mutation is False
        assert effect.information_flow is True
        assert effect.provider_metadata['outcome'] == 'rejected_after_state_observation'
        assert effect.provider_metadata['provider_phases'] == [
            {'name': 'state', 'state_mutation': False, 'information_flow': True}
        ]

    def test_validate_directory_not_started_restores_one_time_read_and_abandons_intent(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = f'agent_outputs/cwd_state_not_started_{uuid4().hex}'
        (self.runtime.workspace_root / path).mkdir(parents=True)
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='validate cwd provider failure')
        resource = self.runtime.filesystem.directory_resource_for_path(path)
        cap = self.runtime.capability.grant_once(pid, resource, [CapabilityRight.READ], issued_by='test')

        def fail_before_state(_target: ResolvedPath) -> object:
            raise ProviderEffectNotStarted('state provider did not start')

        monkeypatch.setattr(self.runtime.filesystem.provider, 'state', fail_before_state)
        with pytest.raises(ProviderEffectNotStarted, match='did not start'):
            self.runtime.filesystem.validate_directory(pid, path)

        persisted = self.runtime.store.get_capability(cap.cap_id)
        assert persisted is not None and persisted.uses_remaining == 1
        assert self.runtime.store.list_external_effects(pid=pid) == []
        assert self.runtime.process.get(pid).resource_usage.external_read_bytes == 0

    def test_invalid_text_decode_finalizes_observed_read_and_consumes_one_time_authority(self) -> None:
        path = f'agent_outputs/invalid_utf8_{uuid4().hex}.bin'
        target = self.runtime.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'\xff')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='invalid text decoding')
        cap = self.runtime.capability.grant_once(
            pid,
            self.runtime.filesystem.resource_for_path(path),
            [CapabilityRight.READ],
            issued_by='test',
        )

        with pytest.raises(UnicodeDecodeError):
            self.runtime.filesystem.read_text(pid, path, encoding='utf-8')

        assert self.runtime.store.get_capability(cap.cap_id).uses_remaining == 0
        effects = self.runtime.store.list_external_effects(pid=pid)
        assert len(effects) == 1
        assert effects[0].effect_state == 'finalized'
        assert effects[0].information_flow is True
        assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN

    def test_delete_requires_delete_capability_not_write_capability(self) -> None:
        path = self._write_fixture(f'agent_outputs/delete_denied_{uuid4().hex}.txt', 'keep')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='delete denied')
        self.runtime.filesystem.grant_path(pid, path, [CapabilityRight.WRITE], issued_by='test')
        denied = self.runtime.tools.call(pid, 'delete_file', {'path': path})
        assert not denied.ok
        assert 'lacks delete' in (denied.error or '')
        assert (self.runtime.workspace_root / path).exists()

    def test_filesystem_resource_keeps_os_distinct_path_names_separate(self) -> None:
        if os.name == 'nt':
            pytest.skip('backslash is a path separator on Windows')
        base = f'agent_outputs/path_alias_{uuid4().hex}'
        normal = self._write_fixture(f'{base}/dir/file.txt', 'normal')
        backslash = self._write_fixture(f'{base}/dir\\file.txt', 'backslash')
        trailing_space = self._write_fixture(f'{base}/space.txt ', 'space')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='path alias')
        self.runtime.filesystem.grant_path(pid, normal, [CapabilityRight.READ], issued_by='test')
        self.runtime.filesystem.grant_path(pid, f'{base}/space.txt', [CapabilityRight.READ], issued_by='test')

        assert self.runtime.filesystem.read_text(pid, normal).content == 'normal'
        with pytest.raises(CapabilityDenied):
            self.runtime.filesystem.read_text(pid, backslash)
        with pytest.raises(CapabilityDenied):
            self.runtime.filesystem.read_text(pid, trailing_space)

    def test_local_filesystem_resolve_is_lexical_and_enforces_containment(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace).resolve()
            provider = LocalFilesystemProvider(root)
            nested = root / 'nested'
            nested.mkdir()

            assert provider.resolve('nested/../target').relative == 'target'
            assert provider.resolve(nested / '..' / 'target').relative == 'target'
            assert provider.resolve('.').is_root
            with pytest.raises(CapabilityDenied, match='escapes filesystem adapter root'):
                provider.resolve('../outside')
            with pytest.raises(CapabilityDenied, match='escapes filesystem adapter root'):
                provider.resolve(Path(outside) / 'target')

    def test_delete_ask_each_time_uses_filesystem_primitive_context(self) -> None:
        path = self._write_fixture(f'agent_outputs/delete_prompt_{uuid4().hex}.txt', 'delete me')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='delete with prompt')
        resource = self.runtime.filesystem.resource_for_path(path)
        self.runtime.capability.set_permission_policy(subject=pid, resource=resource, rights=[CapabilityRight.DELETE], policy=CapabilityManager.ASK_EACH_TIME, issued_by='test')
        with pytest.raises(HumanApprovalRequired):
            self.runtime.tools.call(pid, 'delete_file', {'path': path})
        request = self.runtime.human.pending()[0]
        processed = self.runtime.human.drain_terminal_queue(auto_approve=True)
        retried = self.runtime.tools.call(pid, 'delete_file', {'path': path})
        assert request.payload['context']['primitive'] == 'runtime.filesystem.delete_file'
        assert request.payload['context']['operation'] == 'delete_file'
        assert request.payload['context']['right'] == 'delete'
        assert 'target' not in request.payload['context']
        assert request.payload['context']['target_state_observation'] == 'deferred_until_authorized'
        assert processed[0].status == HumanRequestStatus.APPROVED
        assert retried.ok, retried.error
        assert not (self.runtime.workspace_root / path).exists()

    def test_truncated_utf8_read_does_not_split_codepoint(self) -> None:
        path = self._write_fixture(f'agent_outputs/utf8_{uuid4().hex}.txt', 'éx')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='read utf8')
        self.runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by='test')
        result = self.runtime.filesystem.read_text(pid, path, max_bytes=1)
        assert result.truncated
        assert result.bytes_read == 1
        assert result.content == ''

    def test_unrelated_file_reads_do_not_share_one_global_label_io_lock(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base = f'agent_outputs/parallel_reads_{uuid4().hex}'
        first_path = self._write_fixture(f'{base}/first.txt', 'first')
        second_path = self._write_fixture(f'{base}/second.txt', 'second')
        pid = self.runtime.process.spawn(
            image='review-agent:v0',
            goal='read unrelated files concurrently',
        )
        self.runtime.filesystem.grant_path(
            pid,
            first_path,
            [CapabilityRight.READ],
            issued_by='test',
        )
        self.runtime.filesystem.grant_path(
            pid,
            second_path,
            [CapabilityRight.READ],
            issued_by='test',
        )
        provider = self.runtime.filesystem.provider
        assert isinstance(provider, LocalFilesystemProvider)
        original_before_path_sink = provider._before_path_sink
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        errors: list[BaseException] = []

        def interleaved_sink(operation: str, target: Path) -> None:
            if operation != 'read_bytes':
                return original_before_path_sink(operation, target)
            relative = target.relative_to(provider.root).as_posix()
            if relative == first_path:
                first_entered.set()
                if not release_first.wait(timeout=5):
                    raise TimeoutError('first read was not released')
            elif relative == second_path:
                second_entered.set()
            original_before_path_sink(operation, target)

        def read(path: str) -> None:
            try:
                self.runtime.filesystem.read_text(pid, path)
            except BaseException as exc:
                errors.append(exc)

        monkeypatch.setattr(
            provider,
            '_before_path_sink',
            interleaved_sink,
        )
        first = threading.Thread(target=read, args=(first_path,), daemon=True)
        second = threading.Thread(target=read, args=(second_path,), daemon=True)
        first.start()
        assert first_entered.wait(timeout=5)
        second.start()
        try:
            assert second_entered.wait(timeout=1)
        finally:
            release_first.set()
            first.join(timeout=5)
            second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert errors == []

    def test_hierarchical_path_lock_does_not_starve_queued_ancestor(self) -> None:
        path_lock = HierarchicalPathLock()
        ancestor_entered = threading.Event()
        release_ancestor = threading.Event()
        later_descendant_entered = threading.Event()

        def hold_ancestor() -> None:
            with path_lock.hold('tree'):
                ancestor_entered.set()
                assert release_ancestor.wait(timeout=5)

        def hold_later_descendant() -> None:
            with path_lock.hold('tree/second'):
                later_descendant_entered.set()

        ancestor = threading.Thread(target=hold_ancestor, daemon=True)
        later_descendant = threading.Thread(
            target=hold_later_descendant,
            daemon=True,
        )
        with path_lock.hold('tree/first'):
            ancestor.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with path_lock._condition:
                    if any(parts == ('tree',) for _, parts, _ in path_lock._waiters):
                        break
                time.sleep(0.01)
            else:
                pytest.fail('ancestor lock did not enter the waiter queue')
            later_descendant.start()
            assert not later_descendant_entered.wait(timeout=0.1)

        assert ancestor_entered.wait(timeout=5)
        assert not later_descendant_entered.wait(timeout=0.1)
        release_ancestor.set()
        ancestor.join(timeout=5)
        later_descendant.join(timeout=5)

        assert not ancestor.is_alive()
        assert not later_descendant.is_alive()
        assert later_descendant_entered.is_set()

    def test_hierarchical_path_lock_rejects_widening_upgrade(self) -> None:
        path_lock = HierarchicalPathLock()

        with path_lock.hold('tree/first'):
            with pytest.raises(RuntimeError, match='cannot widen'):
                with path_lock.hold('tree'):
                    pytest.fail('widening lock upgrade unexpectedly succeeded')

    def test_hierarchical_path_lock_creation_scope_covers_shared_missing_parents(
        self,
    ) -> None:
        path_lock = HierarchicalPathLock()
        assert path_lock.creation_scope('top-level.txt') == 'top-level.txt'
        assert path_lock.creation_scope('tree/first/output.txt') == 'tree'
        sibling_started = threading.Event()
        sibling_entered = threading.Event()

        def create_sibling() -> None:
            sibling_started.set()
            with path_lock.hold(
                path_lock.creation_scope('tree/second/output.txt')
            ):
                sibling_entered.set()

        sibling = threading.Thread(target=create_sibling, daemon=True)
        with path_lock.hold(path_lock.creation_scope('tree/first/output.txt')):
            sibling.start()
            assert sibling_started.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with path_lock._condition:
                    if path_lock._waiters:
                        break
                time.sleep(0.01)
            else:
                pytest.fail('sibling creation lock did not enter the waiter queue')
            assert not sibling_entered.wait(timeout=0.1)
        sibling.join(timeout=5)

        assert not sibling.is_alive()
        assert sibling_entered.is_set()

    def test_hierarchical_path_lock_serializes_case_and_unicode_aliases(self) -> None:
        path_lock = HierarchicalPathLock()
        alias_started = threading.Event()
        alias_entered = threading.Event()

        def hold_alias() -> None:
            alias_started.set()
            with path_lock.hold('caseprobe/e\u0301.TXT'):
                alias_entered.set()

        alias = threading.Thread(target=hold_alias, daemon=True)
        with path_lock.hold('CaseProbe/\u00e9.txt'):
            alias.start()
            assert alias_started.wait(timeout=5)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                with path_lock._condition:
                    if path_lock._waiters:
                        break
                time.sleep(0.01)
            else:
                pytest.fail('alias lock did not enter the waiter queue')
            assert not alias_entered.wait(timeout=0.1)
        alias.join(timeout=5)

        assert not alias.is_alive()
        assert alias_entered.is_set()

    def test_read_detects_growth_after_state_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            path = root / 'growing.txt'
            path.write_text('a', encoding='utf-8')
            provider = GrowingReadProvider(root, replacement=b'abcdefghij')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(
                    image='base-agent:v0',
                    goal='read growing file',
                    resource_budget=ResourceBudget(max_external_read_bytes=4),
                )
                runtime.filesystem.grant_path(pid, 'growing.txt', [CapabilityRight.READ], issued_by='test')

                result = runtime.filesystem.read_bytes(pid, 'growing.txt', max_bytes=4)

                assert result.content == b'abcd'
                assert result.bytes_read == 4
                assert result.truncated
                assert runtime.process.get(pid).resource_usage.external_read_bytes == 4
            finally:
                runtime.close()

    def test_one_time_read_capabilities_are_consumed_after_provider_read(self) -> None:
        base = f'agent_outputs/read_once_{uuid4().hex}'
        text_path = self._write_fixture(f'{base}/text.txt', 'text')
        bytes_path = self._write_fixture(f'{base}/bytes.bin', 'bytes')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='read once')
        text_resource = self.runtime.filesystem.resource_for_path(text_path)
        bytes_resource = self.runtime.filesystem.resource_for_path(bytes_path)
        directory_resource = self.runtime.filesystem.directory_resource_for_path(base)
        self.runtime.capability.grant_once(pid, text_resource, [CapabilityRight.READ], issued_by='test')
        self.runtime.capability.grant_once(pid, bytes_resource, [CapabilityRight.READ], issued_by='test')
        self.runtime.capability.grant_once(pid, directory_resource, [CapabilityRight.READ], issued_by='test')
        assert self.runtime.filesystem.read_text(pid, text_path).content == 'text'
        assert self.runtime.filesystem.read_bytes(pid, bytes_path).content == b'bytes'
        assert self.runtime.filesystem.read_directory(pid, base).count == 2
        assert not self.runtime.capability.check(pid, text_resource, CapabilityRight.READ)
        assert not self.runtime.capability.check(pid, bytes_resource, CapabilityRight.READ)
        assert not self.runtime.capability.check(pid, directory_resource, CapabilityRight.READ)

    def test_filesystem_primitive_enforces_read_limits_without_tool_schema(self) -> None:
        path = self._write_fixture(f'agent_outputs/read_limit_{uuid4().hex}.txt', 'content')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='read limit')
        self.runtime.filesystem.grant_path(pid, path, [CapabilityRight.READ], issued_by='test')
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_text(pid, path, max_bytes=0)
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_text(pid, path, max_bytes=True)
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_text(pid, path, max_bytes=DEFAULT_FILESYSTEM_READ_HARD_LIMIT + 1)

    def test_directory_primitive_enforces_limit_without_tool_schema(self) -> None:
        base = f'agent_outputs/list_limit_{uuid4().hex}'
        self._write_fixture(f'{base}/item.txt', 'content')
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='directory limit')
        self.runtime.filesystem.grant_directory(pid, base, [CapabilityRight.READ], issued_by='test')
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_directory(pid, base, limit=0)
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_directory(pid, base, limit=True)
        with pytest.raises(ValidationError):
            self.runtime.filesystem.read_directory(pid, base, limit=DEFAULT_DIRECTORY_ENTRY_HARD_LIMIT + 1)

    def test_filesystem_write_rechecks_symlink_escape_after_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            (root / 'dir').mkdir()
            try:
                os.symlink(Path(outside), root / 'probe-link', target_is_directory=True)
                (root / 'probe-link').unlink()
            except OSError:
                pytest.skip('symlink creation is not available in this environment')
            provider = SwappingSymlinkProvider(root, Path(outside))
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='symlink escape')
                runtime.filesystem.grant_path(pid, 'dir/payload.txt', [CapabilityRight.WRITE], issued_by='test')
                with pytest.raises(CapabilityDenied):
                    runtime.filesystem.write_text(pid, 'dir/payload.txt', 'escaped')
                assert not (Path(outside) / 'payload.txt').exists()
            finally:
                runtime.close()

    def test_write_directory_rejects_reparse_swap_before_mkdir_sink(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir').mkdir()
            provider = SinkSwapProvider(root, outside_root, operation='make_directory')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='mkdir reparse race')
                runtime.filesystem.grant_directory(pid, 'dir/created', [CapabilityRight.WRITE], issued_by='test')

                with pytest.raises(CapabilityDenied, match='symlink|junction|escapes filesystem adapter root'):
                    runtime.filesystem.write_directory(pid, 'dir/created', parents=True)

                assert provider.swapped
                assert not (outside_root / 'created').exists()
            finally:
                runtime.close()

    def test_write_file_rejects_reparse_swap_before_parent_mkdir_sink(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir').mkdir()
            provider = SinkSwapProvider(root, outside_root, operation='write_parent')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='write parent reparse race')
                runtime.filesystem.grant_path(pid, 'dir/nested/payload.txt', [CapabilityRight.WRITE], issued_by='test')

                with pytest.raises(CapabilityDenied, match='symlink|junction|escapes filesystem adapter root'):
                    runtime.filesystem.write_text(pid, 'dir/nested/payload.txt', 'escaped')

                assert provider.swapped
                assert not (outside_root / 'nested').exists()
                assert not (outside_root / 'payload.txt').exists()
            finally:
                runtime.close()

    def test_list_directory_rejects_reparse_swap_before_iterdir_sink(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir').mkdir()
            (root / 'dir' / 'inside.txt').write_text('inside', encoding='utf-8')
            (outside_root / 'secret.txt').write_text('outside', encoding='utf-8')
            provider = SinkSwapProvider(root, outside_root, operation='list_directory')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='list reparse race')
                runtime.filesystem.grant_directory(pid, 'dir', [CapabilityRight.READ], issued_by='test')

                with pytest.raises(CapabilityDenied, match='symlink|junction|escapes filesystem adapter root|path changed'):
                    runtime.filesystem.read_directory(pid, 'dir')

                assert provider.swapped
            finally:
                runtime.close()

    def test_descriptor_bound_parent_mkdir_closes_intermediate_directory_fds(self) -> None:
        if os.open not in os.supports_dir_fd:
            pytest.skip('descriptor-bound directory operations are not used on this platform')
        before = _open_fd_count()
        if before is None:
            pytest.skip('open fd count is not available on this platform')
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace).resolve()
            provider = LocalFilesystemProvider(root)
            for index in range(20):
                file_path = f'write/{index}/a/b/c/payload.txt'
                dir_path = f'mkdir/{index}/a/b/c'
                provider.write_text(ResolvedPath(display=str(root / file_path), relative=file_path), 'payload', 'utf-8')
                provider.make_directory(ResolvedPath(display=str(root / dir_path), relative=dir_path), parents=True, exist_ok=True)

        after = _open_fd_count()
        if after is None:
            pytest.skip('open fd count disappeared on this platform')
        assert after <= before + 3

    def test_delete_file_rejects_reparse_swap_before_unlink_sink(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir').mkdir()
            (root / 'dir' / 'victim.txt').write_text('inside', encoding='utf-8')
            outside_file = outside_root / 'victim.txt'
            outside_file.write_text('outside', encoding='utf-8')
            provider = SinkSwapProvider(root, outside_root, operation='delete_file')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='unlink reparse race')
                runtime.filesystem.grant_path(pid, 'dir/victim.txt', [CapabilityRight.DELETE], issued_by='test')

                with pytest.raises(
                    CapabilityDenied,
                    match='symlink|junction|escapes filesystem adapter root|path changed during delete',
                ):
                    runtime.filesystem.delete_file(pid, 'dir/victim.txt')

                assert provider.swapped or os.name == 'nt'
                assert outside_file.read_text(encoding='utf-8') == 'outside'
            finally:
                runtime.close()

    def test_delete_directory_rejects_reparse_swap_before_recursive_delete_sink(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir' / 'victim').mkdir(parents=True)
            (root / 'dir' / 'victim' / 'inside.txt').write_text('inside', encoding='utf-8')
            (outside_root / 'victim').mkdir()
            outside_file = outside_root / 'victim' / 'outside.txt'
            outside_file.write_text('outside', encoding='utf-8')
            provider = SinkSwapProvider(root, outside_root, operation='delete_directory')
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='rmtree reparse race')
                runtime.filesystem.grant_directory(pid, 'dir/victim', [CapabilityRight.DELETE], issued_by='test')

                with pytest.raises(
                    CapabilityDenied,
                    match='symlink|junction|escapes filesystem adapter root|path changed during delete',
                ):
                    runtime.filesystem.delete_directory(pid, 'dir/victim', recursive=True)

                assert provider.swapped or os.name == 'nt'
                assert outside_file.read_text(encoding='utf-8') == 'outside'
            finally:
                runtime.close()

    def test_write_file_rejects_reparse_swap_during_fallback_open_before_truncate(self) -> None:
        if os.open in os.supports_dir_fd:
            pytest.skip('fallback open path is not used on this platform')
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_root = Path(outside)
            (root / 'dir').mkdir()
            (root / 'dir' / 'victim.txt').write_text('inside', encoding='utf-8')
            outside_file = outside_root / 'victim.txt'
            outside_file.write_text('outside', encoding='utf-8')
            provider = FallbackOpenSwapProvider(root, outside_root)
            runtime = self._runtime_with_filesystem_provider(root, provider)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='fallback open reparse race')
                runtime.filesystem.grant_path(pid, 'dir/victim.txt', [CapabilityRight.WRITE], issued_by='test')

                with pytest.raises(CapabilityDenied, match='opened path changed|escapes filesystem adapter root|symlink|junction'):
                    runtime.filesystem.write_text(pid, 'dir/victim.txt', 'changed')

                assert provider.swap_attempted
                assert outside_file.read_text(encoding='utf-8') == 'outside'
            finally:
                runtime.close()

    def test_directory_listing_does_not_follow_child_symlink_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            (root / 'list').mkdir()
            outside_file = Path(outside) / 'outside.txt'
            outside_file.write_text('outside secret metadata', encoding='utf-8')
            try:
                os.symlink(outside_file, root / 'list' / 'outside-link')
            except OSError:
                pytest.skip('symlink creation is not available in this environment')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='list symlink')
                runtime.filesystem.grant_directory(pid, 'list', [CapabilityRight.READ], issued_by='test')

                result = runtime.filesystem.read_directory(pid, 'list')

                assert result.entries[0].name == 'outside-link'
                assert result.entries[0].kind == 'symlink'
                assert result.entries[0].size_bytes is None
            finally:
                runtime.close()

    def test_filesystem_rejects_workspace_hardlink_to_external_inode(self) -> None:
        with tempfile.TemporaryDirectory() as workspace, tempfile.TemporaryDirectory() as outside:
            root = Path(workspace)
            outside_file = Path(outside) / 'secret.txt'
            outside_file.write_text('outside secret', encoding='utf-8')
            link = root / 'linked-secret.txt'
            try:
                os.link(outside_file, link)
            except OSError:
                pytest.skip('hardlink creation is not available in this environment')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='hardlink escape')
                runtime.filesystem.grant_path(
                    pid,
                    'linked-secret.txt',
                    [CapabilityRight.READ, CapabilityRight.WRITE, CapabilityRight.DELETE],
                    issued_by='test',
                )

                with pytest.raises(CapabilityDenied):
                    runtime.filesystem.read_text(pid, 'linked-secret.txt')
                with pytest.raises(CapabilityDenied):
                    runtime.filesystem.write_text(pid, 'linked-secret.txt', 'changed')
                with pytest.raises(CapabilityDenied):
                    runtime.filesystem.delete_file(pid, 'linked-secret.txt')

                assert outside_file.read_text(encoding='utf-8') == 'outside secret'
                assert link.exists()
            finally:
                runtime.close()

    def test_one_time_mutation_capability_is_consumed_after_state_observation(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            provider = FailingMutationProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='mutation failure')
                write_cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('out.txt'),
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                with pytest.raises(ProviderEffectNotStarted, match='simulated write failure'):
                    runtime.filesystem.write_text(pid, 'out.txt', 'content')
                # The state phase already crossed an information-flow boundary,
                # so a later certified-not-started mutation cannot restore the
                # one-time authority.
                assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 0
                write_effect = runtime.store.list_external_effects(pid=pid)[0]
                assert write_effect.effect_state == 'finalized'
                assert write_effect.state_mutation is False
                assert write_effect.information_flow is True
                assert write_effect.rollback_status == ExternalEffectRollbackStatus.UNKNOWN

                (root / 'delete.txt').write_text('delete me', encoding='utf-8')
                delete_cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('delete.txt'),
                    [CapabilityRight.DELETE],
                    issued_by='test',
                )
                with pytest.raises(ProviderEffectNotStarted, match='simulated delete failure'):
                    runtime.filesystem.delete_file(pid, 'delete.txt')
                assert runtime.store.get_capability(delete_cap.cap_id).uses_remaining == 0
                assert (root / 'delete.txt').exists()
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 2
                assert all(effect.effect_state == 'finalized' for effect in effects)
                assert all(effect.state_mutation is False for effect in effects)
                assert all(effect.information_flow is True for effect in effects)
            finally:
                runtime.close()

    def test_commit_then_throw_keeps_one_time_authority_consumed_and_records_unknown_effect(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            provider = CommitThenThrowMutationProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='ambiguous mutation failure')
                cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('committed.txt'),
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )

                with pytest.raises(OSError, match='failure after write committed'):
                    runtime.filesystem.write_text(pid, 'committed.txt', 'durable content')

                assert (root / 'committed.txt').read_text(encoding='utf-8') == 'durable content'
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
                effects = runtime.store.list_external_effects(pid=pid)
                assert len(effects) == 1
                assert effects[0].operation == 'write_text'
                assert effects[0].rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert effects[0].provider_metadata['outcome'] == 'unknown_after_provider_exception'
            finally:
                runtime.close()

    def test_post_commit_classifier_failure_records_conservative_effect(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            provider = FailingClassifierMutationProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='classifier failure')
                cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('classified.txt'),
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )

                result = runtime.filesystem.write_text(pid, 'classified.txt', 'written')

                assert result.bytes_written == len('written')
                assert runtime.store.get_capability(cap.cap_id).uses_remaining == 0
                effect = runtime.store.list_external_effects(pid=pid)[0]
                assert effect.rollback_status == ExternalEffectRollbackStatus.UNKNOWN
                assert effect.provider_metadata['classification_fallback'] == 'post_effect_failure'
            finally:
                runtime.close()

    def test_concurrent_one_time_file_write_crosses_provider_once(self) -> None:
        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            provider = SlowCountingMutationProvider(root)
            substrate = LocalResourceProviderSubstrate(root)
            substrate.filesystem = provider
            runtime = Runtime.open('local', substrate=substrate)
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='filesystem write race')
                write_cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('race.txt'),
                    [CapabilityRight.WRITE],
                    issued_by='test',
                )
                workers = 2
                barrier = threading.Barrier(workers)

                def write(index: int) -> bool:
                    barrier.wait(timeout=2)
                    try:
                        runtime.filesystem.write_text(pid, 'race.txt', f'content {index}')
                        return True
                    except CapabilityDenied:
                        return False

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    outcomes = list(executor.map(write, range(workers)))

                assert outcomes.count(True) == 1
                assert outcomes.count(False) == 1
                assert provider.write_attempts == 1
                assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 0
            finally:
                runtime.close()

    def _write_fixture(self, path: str, content: str) -> str:
        target = self.runtime.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return path

    def _runtime_with_filesystem_provider(self, root: Path, provider: LocalFilesystemProvider) -> Runtime:
        substrate = LocalResourceProviderSubstrate(root)
        substrate.filesystem = provider
        return Runtime.open('local', substrate=substrate)


class SwappingSymlinkProvider(LocalFilesystemProvider):
    def __init__(self, root: Path, outside: Path):
        super().__init__(root)
        self.outside = outside
        self.swapped = False

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = '\n',
        *,
        overwrite: bool = True,
    ) -> None:
        if not self.swapped:
            shutil.rmtree(Path(self.root_display) / 'dir')
            os.symlink(self.outside, Path(self.root_display) / 'dir', target_is_directory=True)
            self.swapped = True
        super().write_text(path, text, encoding=encoding, newline=newline, overwrite=overwrite)


class SinkSwapProvider(LocalFilesystemProvider):
    def __init__(self, root: Path, outside: Path, *, operation: str):
        super().__init__(root)
        self.outside = outside
        self.operation = operation
        self.swapped = False

    def _before_path_sink(self, operation: str, target: Path) -> None:
        if self.swapped or operation != self.operation:
            return
        link = Path(self.root_display) / 'dir'
        _remove_directory_for_swap(link)
        _create_directory_reparse_link(link, self.outside)
        self.swapped = True


class FallbackOpenSwapProvider(LocalFilesystemProvider):
    def __init__(self, root: Path, outside: Path):
        super().__init__(root)
        self.outside = outside
        self.swapped = False
        self.swap_attempted = False

    def _before_fallback_open(self, target: Path, flags: int) -> None:
        if self.swapped:
            return
        self.swap_attempted = True
        link = Path(self.root_display) / 'dir'
        _remove_directory_for_swap(link)
        _create_directory_reparse_link(link, self.outside)
        self.swapped = True


class FailingMutationProvider(LocalFilesystemProvider):
    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = '\n',
        *,
        overwrite: bool = True,
    ) -> None:
        raise ProviderEffectNotStarted('simulated write failure')

    def delete_file(self, path: ResolvedPath) -> None:
        raise ProviderEffectNotStarted('simulated delete failure')


class GrowingReadProvider(LocalFilesystemProvider):
    def __init__(self, root: Path, *, replacement: bytes):
        super().__init__(root)
        self.replacement = replacement

    def read_bytes(self, path: ResolvedPath, *, max_bytes: int | None = None) -> bytes:
        Path(path.display).write_bytes(self.replacement)
        return super().read_bytes(path, max_bytes=max_bytes)


class CommitThenThrowMutationProvider(LocalFilesystemProvider):
    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = '\n',
        *,
        overwrite: bool = True,
    ) -> None:
        super().write_text(path, text, encoding=encoding, newline=newline, overwrite=overwrite)
        raise OSError('failure after write committed')


class FailingClassifierMutationProvider(LocalFilesystemProvider):
    def classify_external_effect(self, operation: str, context: dict, result: object):
        raise RuntimeError('classifier unavailable after commit')


class SlowCountingMutationProvider(LocalFilesystemProvider):
    def __init__(self, root: Path):
        super().__init__(root)
        self._lock = threading.Lock()
        self.write_attempts = 0

    def write_text(
        self,
        path: ResolvedPath,
        text: str,
        encoding: str,
        newline: str | None = '\n',
        *,
        overwrite: bool = True,
    ) -> None:
        with self._lock:
            self.write_attempts += 1
        time.sleep(0.05)
        super().write_text(path, text, encoding=encoding, newline=newline, overwrite=overwrite)


def _remove_directory_for_swap(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
        return
    is_junction = getattr(path, 'is_junction', None)
    if callable(is_junction) and is_junction():
        path.rmdir()
        return
    shutil.rmtree(path)


def _create_directory_reparse_link(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except OSError:
        if os.name != 'nt':
            pytest.skip('symlink creation is not available in this environment')
    result = subprocess.run(
        ['cmd', '/c', 'mklink', '/J', str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f'junction creation is not available in this environment: {result.stderr or result.stdout}')


def _open_fd_count() -> int | None:
    for candidate in (Path('/proc/self/fd'), Path('/dev/fd')):
        try:
            return len(list(candidate.iterdir()))
        except OSError:
            continue
    return None
