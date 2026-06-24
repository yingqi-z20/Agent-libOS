from __future__ import annotations
import pytest
from uuid import uuid4
from agent_libos import Runtime
from agent_libos.models import CapabilityRight

class TestGranularPermission:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')

    def teardown_method(self) -> None:
        self.runtime.close()

    def test_filesystem_file_and_directory_allow_lists(self) -> None:
        allowed_file = self._write_fixture('allowed file')
        denied_file = self._write_fixture('denied file')
        allowed_dir = f'agent_outputs/granular_dir_{uuid4().hex}'
        allowed_dir_file = self._write_fixture('allowed directory file', f'{allowed_dir}/nested/readme.txt')
        exact_write = f'agent_outputs/granular_write_{uuid4().hex}/exact.txt'
        write_dir = f'agent_outputs/granular_write_dir_{uuid4().hex}'
        denied_write = f'agent_outputs/granular_denied_write_{uuid4().hex}.txt'
        pid = self.runtime.process.spawn(image='review-agent:v0', goal='granular filesystem')
        self.runtime.filesystem.grant_path_list(pid, read_files=[allowed_file], read_dirs=[allowed_dir], write_files=[exact_write], write_dirs=[write_dir], issued_by='test')
        read_file = self.runtime.tools.call(pid, 'read_text_file', {'path': allowed_file})
        read_dir_file = self.runtime.tools.call(pid, 'read_text_file', {'path': allowed_dir_file})
        read_denied = self.runtime.tools.call(pid, 'read_text_file', {'path': denied_file})
        write_exact = self.runtime.tools.call(pid, 'write_text_file', {'path': exact_write, 'content': 'exact'})
        write_dir_file = self.runtime.tools.call(pid, 'write_text_file', {'path': f'{write_dir}/nested/out.txt', 'content': 'dir'})
        write_denied = self.runtime.tools.call(pid, 'write_text_file', {'path': denied_write, 'content': 'no'})
        assert read_file.ok, read_file.error
        assert read_file.payload['content'] == 'allowed file'
        assert read_dir_file.ok, read_dir_file.error
        assert read_dir_file.payload['content'] == 'allowed directory file'
        assert not read_denied.ok
        assert 'lacks read' in (read_denied.error or '')
        assert write_exact.ok, write_exact.error
        assert write_dir_file.ok, write_dir_file.error
        assert not write_denied.ok
        assert 'lacks write' in (write_denied.error or '')

    def test_child_inherits_only_explicit_filesystem_subset(self) -> None:
        allowed_dir = f'agent_outputs/inherit_allowed_{uuid4().hex}'
        allowed_file = self._write_fixture('allowed child read', f'{allowed_dir}/data.txt')
        denied_file = self._write_fixture('denied child read')
        parent_write = f'agent_outputs/inherit_parent_write_{uuid4().hex}.txt'
        parent = self.runtime.process.spawn(image='review-agent:v0', goal='parent')
        self.runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        self.runtime.filesystem.grant_path_list(parent, read_dirs=[allowed_dir], write_files=[parent_write], issued_by='test')
        forked = self.runtime.tools.call(parent, 'fork_child_process', {'goal': 'child', 'inherit_read_dirs': [allowed_dir]})
        child = forked.payload['child_pid']
        child_read = self.runtime.tools.call(child, 'read_text_file', {'path': allowed_file})
        child_read_denied = self.runtime.tools.call(child, 'read_text_file', {'path': denied_file})
        child_write_denied = self.runtime.tools.call(child, 'write_text_file', {'path': parent_write, 'content': 'child should not write'})
        parent_write_allowed = self.runtime.tools.call(parent, 'write_text_file', {'path': parent_write, 'content': 'parent can write'})
        assert forked.ok, forked.error
        assert child_read.ok, child_read.error
        assert not child_read_denied.ok
        assert 'lacks read' in (child_read_denied.error or '')
        assert not child_write_denied.ok
        assert 'lacks write' in (child_write_denied.error or '')
        assert parent_write_allowed.ok, parent_write_allowed.error

    def test_child_cannot_inherit_broader_permission_than_parent_has(self) -> None:
        allowed_file = self._write_fixture('one file')
        parent = self.runtime.process.spawn(image='review-agent:v0', goal='parent')
        self.runtime.capability.grant(parent, 'process:spawn', [CapabilityRight.WRITE], issued_by='test')
        self.runtime.filesystem.grant_path_list(parent, read_files=[allowed_file], issued_by='test')
        requested_dir = '/'.join(allowed_file.split('/')[:-1])
        forked = self.runtime.tools.call(parent, 'fork_child_process', {'goal': 'child', 'inherit_read_dirs': [requested_dir]})
        children = self.runtime.process.list_children(parent)
        assert not forked.ok
        assert 'cannot inherit' in (forked.error or '')
        assert children == []

    def _write_fixture(self, content: str, path: str | None=None) -> str:
        relative = path or f'agent_outputs/granular_fixture_{uuid4().hex}.txt'
        target = self.runtime.workspace_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        return relative
