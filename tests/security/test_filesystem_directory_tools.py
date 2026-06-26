from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import os
import pytest
import shutil
import tempfile
import threading
import time
from pathlib import Path
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.config import DEFAULT_CONFIG
from agent_libos.models.exceptions import CapabilityDenied, HumanApprovalRequired, ValidationError
from agent_libos.models import CapabilityRight, HumanRequestStatus
from agent_libos.substrate import LocalFilesystemProvider, LocalResourceProviderSubstrate, ResolvedPath
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
        assert 'target' in request.payload['context']
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

    def test_one_time_mutation_capability_survives_provider_failure(self) -> None:
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
                with pytest.raises(OSError, match='simulated write failure'):
                    runtime.filesystem.write_text(pid, 'out.txt', 'content')
                assert runtime.store.get_capability(write_cap.cap_id).uses_remaining == 1

                (root / 'delete.txt').write_text('delete me', encoding='utf-8')
                delete_cap = runtime.capability.grant_once(
                    pid,
                    runtime.filesystem.resource_for_path('delete.txt'),
                    [CapabilityRight.DELETE],
                    issued_by='test',
                )
                with pytest.raises(OSError, match='simulated delete failure'):
                    runtime.filesystem.delete_file(pid, 'delete.txt')
                assert runtime.store.get_capability(delete_cap.cap_id).uses_remaining == 1
                assert (root / 'delete.txt').exists()
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


class SwappingSymlinkProvider(LocalFilesystemProvider):
    def __init__(self, root: Path, outside: Path):
        super().__init__(root)
        self.outside = outside
        self.swapped = False

    def write_text(self, path: ResolvedPath, text: str, encoding: str, newline: str | None = '\n') -> None:
        if not self.swapped:
            shutil.rmtree(Path(self.root_display) / 'dir')
            os.symlink(self.outside, Path(self.root_display) / 'dir', target_is_directory=True)
            self.swapped = True
        super().write_text(path, text, encoding=encoding, newline=newline)


class FailingMutationProvider(LocalFilesystemProvider):
    def write_text(self, path: ResolvedPath, text: str, encoding: str, newline: str | None = '\n') -> None:
        raise OSError('simulated write failure')

    def delete_file(self, path: ResolvedPath) -> None:
        raise OSError('simulated delete failure')


class SlowCountingMutationProvider(LocalFilesystemProvider):
    def __init__(self, root: Path):
        super().__init__(root)
        self._lock = threading.Lock()
        self.write_attempts = 0

    def write_text(self, path: ResolvedPath, text: str, encoding: str, newline: str | None = '\n') -> None:
        with self._lock:
            self.write_attempts += 1
        time.sleep(0.05)
        super().write_text(path, text, encoding=encoding, newline=newline)
