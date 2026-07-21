from __future__ import annotations

import ast
import contextlib
import pytest
import asyncio
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models import (
    AgentImage,
    CapabilityRight,
    JIT_MULTIPLEXER_TOOL_NAME,
    JIT_TOOL_EXPOSURE_MULTIPLEXED,
    ObjectOwnerKind,
    ProcessStatus,
    ResourceBudget,
    ToolSpec,
    ValidationResult,
)
from agent_libos.models.exceptions import (
    CapabilityDenied,
    ResourceLimitExceeded,
    ValidationError,
)
from agent_libos.runtime.syscalls import LibOSSyscallSession
from agent_libos.substrate import CommandMetrics, LocalResourceProviderSubstrate, SubprocessLimits, WindowsJobObject
from agent_libos.tools.broker import ToolBroker
from agent_libos.tools.sandbox import DenoTypescriptSandbox, SandboxBackend
from agent_libos.utils.serde import dumps
from tests.support.deno import (
    BAD_OUTPUT_SOURCE,
    COUNT_CHARS_SOURCE,
    EXEC_AFTER_RESULT_SOURCE,
    EXIT_AFTER_RESULT_SOURCE,
    MISSING_EXEC_AFTER_RESULT_SOURCE,
    READ_FILE_SOURCE,
    WRITE_FILE_SOURCE,
)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True

def test_tool_broker_sort_key_uses_persisted_spec_without_loaded_handle() -> None:
    broker = object.__new__(ToolBroker)
    broker.registry = SimpleNamespace(handle=lambda _tool_id: None)
    broker.extensions = SimpleNamespace(
        get_tool_spec=lambda _tool_id: ToolSpec(
            name="persisted_name",
            description="persisted-only tool",
        )
    )

    assert broker._tool_sort_key("tool_persisted") == (
        "persisted_name",
        "tool_persisted",
    )


class TestJitSecurity:

    def setup_method(self) -> None:
        self.runtime = Runtime.open('local')

    def teardown_method(self) -> None:
        self.runtime.close()

    @pytest.mark.real_deno
    def test_deno_jit_tool_is_visible_only_to_registering_process(self) -> None:
        owner = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='make parser')
        other = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='unrelated process')
        candidate = self.runtime.tools.propose(owner, {'name': 'count_chars', 'description': 'Count characters in text.', 'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}}}, 'output_schema': {'type': 'object'}}, source_code=COUNT_CHARS_SOURCE, tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}])
        validation = self.runtime.tools.validate(candidate)
        assert validation.ok, validation.errors
        self.runtime.tools.register(owner, candidate)
        owner_schema_names = self._schema_names(owner)
        other_schema_names = self._schema_names(other)
        owner_call = self.runtime.tools.call(owner, 'count_chars', {'text': 'abcd'})
        other_call = self.runtime.tools.call(other, 'count_chars', {'text': 'abcd'})
        assert 'count_chars' in owner_schema_names
        assert 'count_chars' not in other_schema_names
        assert owner_call.ok
        assert owner_call.payload == {'count': 4}
        assert not other_call.ok
        assert 'not in process tool table' in (other_call.error or '')

    @pytest.mark.real_deno
    def test_jit_candidate_tools_are_owned_by_proposing_process(self) -> None:
        owner = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='make private tool')
        other = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='try private candidate')
        candidate = self.runtime.tools.propose(owner, {'name': 'owned_count_chars', 'description': 'Count characters in text.', 'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}}}, 'output_schema': {'type': 'object'}}, source_code=COUNT_CHARS_SOURCE, tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}])
        denied_validation = self.runtime.tools.call(other, 'validate_jit_tool', {'candidate_id': candidate})
        denied_registration = self.runtime.tools.call(other, 'register_jit_tool', {'candidate_id': candidate})
        allowed_validation = self.runtime.tools.call(owner, 'validate_jit_tool', {'candidate_id': candidate})
        allowed_registration = self.runtime.tools.call(owner, 'register_jit_tool', {'candidate_id': candidate})
        assert not denied_validation.ok
        assert 'belongs to process' in (denied_validation.error or '')
        assert not denied_registration.ok
        assert 'belongs to process' in (denied_registration.error or '')
        assert 'owned_count_chars' not in self.runtime.process.get(other).tool_table
        assert allowed_validation.ok, allowed_validation.error
        assert allowed_registration.ok, allowed_registration.error
        assert 'owned_count_chars' in self.runtime.process.get(owner).tool_table

    def test_jit_candidate_specs_are_conservative_side_effects(self) -> None:
        owner = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='make conservative tool')
        candidate_id = self.runtime.tools.propose(
            owner,
            {
                'name': 'maybe_reads',
                'description': 'Could use libOS syscalls.',
                'input_schema': {'type': 'object'},
                'output_schema': {'type': 'object'},
                'policy': {'side_effects': False, 'declared_permissions': []},
            },
            source_code='export function run(args, libos) { return {}; }',
        )

        candidate = self.runtime.store.get_tool_candidate(candidate_id)

        assert candidate is not None
        assert candidate.spec.policy['side_effects'] is True
        assert candidate.spec.policy['idempotent'] is False
        assert 'libos.syscall' in candidate.spec.side_effects
        assert 'filesystem.write' in candidate.spec.side_effects
        assert 'jsonrpc.call' in candidate.spec.side_effects

    def test_jit_registration_publishes_durable_alias_and_executable_handle_together(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.runtime.tools.sandbox = NoLimitValidationSandbox()
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='atomic JIT publication')
        candidate_id = self.runtime.tools.propose(
            pid,
            {
                'name': 'atomic_jit_publication',
                'description': 'Exercise the JIT publication boundary.',
                'input_schema': {'type': 'object'},
                'output_schema': {'type': 'object'},
            },
            source_code='export function run(args, libos) { return { ok: true }; }',
        )
        assert self.runtime.tools.validate(candidate_id).ok
        registration_before_commit = threading.Event()
        release_registration = threading.Event()
        resolver_done = threading.Event()
        original_record = self.runtime.audit.record
        registered: list[Any] = []
        resolved: list[Any] = []
        errors: list[BaseException] = []

        def delayed_record(*args: Any, **kwargs: Any) -> Any:
            decision = kwargs.get('decision')
            if (
                kwargs.get('action') == 'tool.register'
                and isinstance(decision, dict)
                and decision.get('candidate_id') == candidate_id
            ):
                registration_before_commit.set()
                if not release_registration.wait(timeout=5):
                    raise TimeoutError('timed out waiting to release JIT registration')
            return original_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', delayed_record)

        def register() -> None:
            try:
                registered.append(self.runtime.tools.register(pid, candidate_id))
            except BaseException as exc:
                errors.append(exc)

        def resolve() -> None:
            try:
                resolved.append(self.runtime.tools.resolve('atomic_jit_publication', pid=pid))
            except BaseException as exc:
                errors.append(exc)
            finally:
                resolver_done.set()

        registration_thread = threading.Thread(target=register)
        resolver_thread = threading.Thread(target=resolve)
        try:
            registration_thread.start()
            assert registration_before_commit.wait(timeout=3)
            resolver_thread.start()
            assert not resolver_done.wait(timeout=0.2)

            release_registration.set()
            registration_thread.join(timeout=5)
            resolver_thread.join(timeout=5)

            assert errors == []
            assert registered and resolved
            assert registered[0].tool_id == resolved[0].tool_id
            tool_id = registered[0].tool_id
            assert self.runtime.process.get(pid).tool_table['atomic_jit_publication'] == tool_id
            assert self.runtime.tools.is_jit_tool_id(tool_id)
            assert self.runtime.tools.loaded_tool_handle(tool_id) is not None
            assert any(row['tool_id'] == tool_id for row in self.runtime.store.list_tools())
        finally:
            release_registration.set()
            registration_thread.join(timeout=5)
            resolver_thread.join(timeout=5)

    def test_jit_registration_audit_failure_rolls_back_alias_source_and_handle(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.runtime.tools.sandbox = NoLimitValidationSandbox()
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='JIT publication rollback')
        candidate_id = self.runtime.tools.propose(
            pid,
            {
                'name': 'rollback_jit_publication',
                'description': 'Exercise JIT publication rollback.',
                'input_schema': {'type': 'object'},
                'output_schema': {'type': 'object'},
            },
            source_code='export function run(args, libos) { return { ok: true }; }',
        )
        assert self.runtime.tools.validate(candidate_id).ok
        before_handles = self.runtime.tools.loaded_tool_ids()
        before_sources = self.runtime.tools.loaded_jit_tool_ids()
        original_record = self.runtime.audit.record

        def fail_registration_audit(*args: Any, **kwargs: Any) -> Any:
            decision = kwargs.get('decision')
            if (
                kwargs.get('action') == 'tool.register'
                and isinstance(decision, dict)
                and decision.get('candidate_id') == candidate_id
            ):
                raise RuntimeError('injected JIT registration audit failure')
            return original_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', fail_registration_audit)

        with pytest.raises(RuntimeError, match='injected JIT registration audit failure'):
            self.runtime.tools.register(pid, candidate_id)

        candidate = self.runtime.store.get_tool_candidate(candidate_id)
        assert candidate is not None
        assert candidate.status.value == 'validated'
        assert candidate.registered_tool_id is None
        assert 'rollback_jit_publication' not in self.runtime.process.get(pid).tool_table
        assert self.runtime.tools.loaded_tool_ids() == before_handles
        assert self.runtime.tools.loaded_jit_tool_ids() == before_sources
        assert all(row['name'] != 'rollback_jit_publication' for row in self.runtime.store.list_tools())

    def test_jit_proposal_failure_rolls_back_candidate_and_descriptor(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        owner = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='atomic proposal')
        before_oids = {
            obj.oid
            for obj in self.runtime.store.list_objects_owned_by(ObjectOwnerKind.PROCESS, owner)
        }
        before_candidates = self.runtime.store.select_table_rows('tool_candidates', 'pid = ?', [owner])
        real_record = self.runtime.audit.record

        def fail_proposal_audit(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get('action') == 'tool.propose':
                raise RuntimeError('proposal audit failed')
            return real_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', fail_proposal_audit)
        with pytest.raises(RuntimeError, match='proposal audit failed'):
            self.runtime.tools.propose(
                owner,
                {'name': 'atomic_proposal', 'description': 'Atomic.', 'input_schema': {'type': 'object'}},
                source_code='export function run(args, libos) { return {}; }',
            )

        assert self.runtime.store.select_table_rows('tool_candidates', 'pid = ?', [owner]) == before_candidates
        assert {
            obj.oid
            for obj in self.runtime.store.list_objects_owned_by(ObjectOwnerKind.PROCESS, owner)
        } == before_oids

    def test_jit_validation_and_registration_metadata_commit_atomically(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self.runtime.tools.sandbox = RecordingValidationSandbox()
        owner = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='atomic jit lifecycle')
        candidate_id = self.runtime.tools.propose(
            owner,
            {'name': 'atomic_jit', 'description': 'Atomic.', 'input_schema': {'type': 'object'}},
            source_code='export function run(args, libos) { return {}; }',
        )
        real_record = self.runtime.audit.record

        def fail_validation_audit(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get('action') == 'tool.validate':
                raise RuntimeError('validation audit failed')
            return real_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', fail_validation_audit)
        with pytest.raises(RuntimeError, match='validation audit failed'):
            self.runtime.tools.validate(candidate_id, pid=owner)
        candidate = self.runtime.store.get_tool_candidate(candidate_id)
        assert candidate is not None
        assert candidate.status.value == 'proposed'
        assert candidate.validation is None

        monkeypatch.setattr(self.runtime.audit, 'record', real_record)
        assert self.runtime.tools.validate(candidate_id, pid=owner).ok

        def fail_registration_audit(*args: Any, **kwargs: Any) -> Any:
            if kwargs.get('action') == 'tool.register':
                raise RuntimeError('registration audit failed')
            return real_record(*args, **kwargs)

        monkeypatch.setattr(self.runtime.audit, 'record', fail_registration_audit)
        with pytest.raises(RuntimeError, match='registration audit failed'):
            self.runtime.tools.register(owner, candidate_id)

        candidate = self.runtime.store.get_tool_candidate(candidate_id)
        assert candidate is not None
        assert candidate.status.value == 'validated'
        assert candidate.registered_tool_id is None
        assert 'atomic_jit' not in self.runtime.process.get(owner).tool_table
        assert not [row for row in self.runtime.store.list_tools() if row['name'] == 'atomic_jit']
        assert not [
            handle
            for handle in self.runtime.tools.loaded_tool_handles()
            if handle.name == 'atomic_jit'
        ]

    def test_deno_runtime_execution_uses_cached_only_while_validation_can_resolve_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        commands: list[list[str]] = []
        launch_kwargs: list[dict[str, Any]] = []
        jobs: list[Any] = []

        class FakeJob:
            def __init__(self) -> None:
                self.assigned: list[int] = []
                self.closed = False

            def assign_pid(self, pid: int) -> None:
                self.assigned.append(pid)

            def close(self) -> None:
                self.closed = True

        async def fake_create_subprocess_exec(*command: str, **kwargs: Any) -> Any:
            commands.append(list(command))
            launch_kwargs.append(dict(kwargs))
            return SimpleNamespace(pid=4242)

        async def fake_monitor_process(_proc: Any, _limits: Any) -> CommandMetrics:
            return CommandMetrics()

        async def fake_serve_process(_proc: Any, _args: dict[str, Any], _syscall_handler: Any) -> dict[str, bool]:
            return {'ok': True}

        async def fake_kill_process(_proc: Any) -> None:
            return None

        sandbox = DenoTypescriptSandbox()
        monkeypatch.setattr(sandbox, '_resolve_deno', lambda: 'deno')
        monkeypatch.setattr(sandbox, 'deno_version', lambda: 'deno 2.0.0')
        monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
        monkeypatch.setattr(sandbox, '_monitor_process', fake_monitor_process)
        monkeypatch.setattr(sandbox, '_serve_process', fake_serve_process)
        monkeypatch.setattr(sandbox, '_kill_process', fake_kill_process)
        if os.name == 'nt':
            def create_job() -> FakeJob:
                job = FakeJob()
                jobs.append(job)
                return job

            monkeypatch.setattr(WindowsJobObject, 'create', create_job)

        source = 'export function run(args, libos) { return {ok: true}; }'

        assert sandbox.run_source(source, {}) == {'ok': True}
        validation = sandbox.run_tests(source, [{'args': {}, 'expected': {'ok': True}}])

        assert validation.ok, validation.errors
        assert Path(commands[0][2]).name == '_process_supervisor.py'
        assert commands[0][-5:] == ['deno', 'run', '--no-prompt', '--cached-only', 'runner.ts']
        assert commands[1][-4:] == ['deno', 'run', '--no-prompt', 'runner.ts']
        group_key = 'creationflags' if os.name == 'nt' else 'start_new_session'
        assert all(kwargs.get(group_key) for kwargs in launch_kwargs)
        if os.name == 'posix':
            assert all(kwargs.get('pass_fds') for kwargs in launch_kwargs)
        else:
            assert len(jobs) == 2
            assert all(job.assigned == [4242] and job.closed for job in jobs)

    def test_cancelled_deno_execution_kills_process_and_drains_workers(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sandbox = DenoTypescriptSandbox()
        started: asyncio.Event
        killed: asyncio.Event

        async def scenario() -> None:
            nonlocal started, killed
            started = asyncio.Event()
            killed = asyncio.Event()
            proc = SimpleNamespace(pid=4242, returncode=None)

            class FakeJob:
                def assign_pid(self, process_pid: int) -> None:
                    assert process_pid == proc.pid

                def close(self) -> None:
                    return None

            async def fake_create_subprocess_exec(*_command: str, **_kwargs: Any) -> Any:
                return proc

            async def block_worker(*_args: Any, **_kwargs: Any) -> Any:
                started.set()
                await asyncio.Event().wait()

            async def kill_process(selected: Any) -> None:
                assert selected is proc
                selected.returncode = -9
                killed.set()

            monkeypatch.setattr(sandbox, '_resolve_deno', lambda: 'deno')
            monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_create_subprocess_exec)
            monkeypatch.setattr(sandbox, '_serve_process', block_worker)
            monkeypatch.setattr(sandbox, '_monitor_process', block_worker)
            monkeypatch.setattr(sandbox, '_kill_process', kill_process)
            if os.name == 'nt':
                monkeypatch.setattr(WindowsJobObject, 'create', FakeJob)

            task = asyncio.create_task(
                sandbox.arun_source('export function run(args, libos) { return {}; }', {}),
                name='cancelled-deno-execution',
            )
            await asyncio.wait_for(started.wait(), timeout=1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert killed.is_set()
            assert not [
                worker
                for worker in asyncio.all_tasks()
                if worker is not asyncio.current_task()
                and worker.get_name() in {'deno-resource-monitor', 'deno-syscall-server'}
                and not worker.done()
            ]

        asyncio.run(scenario())

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX process-group semantics')
    def test_deno_cleanup_kills_process_group_and_descendants(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sandbox = DenoTypescriptSandbox()
        killed_groups: list[tuple[int, int]] = []

        class Child:
            killed = False

            def kill(self) -> None:
                self.killed = True

        class Root:
            def __init__(self, child: Child) -> None:
                self.child = child

            def children(self, recursive: bool = False) -> list[Child]:
                assert recursive is True
                return [self.child]

        class Proc:
            pid = 7331
            returncode: int | None = None
            killed = False
            waited = False

            def kill(self) -> None:
                self.killed = True

            async def wait(self) -> int:
                self.waited = True
                self.returncode = -9
                return self.returncode

        child = Child()
        proc = Proc()
        monkeypatch.setattr('agent_libos.tools.sandbox.psutil.Process', lambda _pid: Root(child))
        monkeypatch.setattr(
            'agent_libos.tools.sandbox.os.killpg',
            lambda pid, sig: killed_groups.append((pid, sig)),
        )

        asyncio.run(sandbox._kill_process(proc))

        assert killed_groups == [(proc.pid, signal.SIGKILL)]
        assert child.killed
        assert proc.killed
        assert proc.waited

    @pytest.mark.real_deno
    def test_jit_tool_names_are_process_local(self) -> None:
        first = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='local tool one')
        second = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='local tool two')
        first_candidate = self._register_count_tool(first, 'local_count_chars')
        second_candidate = self._register_count_tool(second, 'local_count_chars')
        first_call = self.runtime.tools.call(first, 'local_count_chars', {'text': 'aa'})
        second_call = self.runtime.tools.call(second, 'local_count_chars', {'text': 'bbbbb'})
        assert first_candidate.tool_id != second_candidate.tool_id
        assert first_call.ok, first_call.error
        assert second_call.ok, second_call.error
        assert first_call.payload == {'count': 2}
        assert second_call.payload == {'count': 5}
        assert 'local_count_chars' in self._schema_names(first)
        assert 'local_count_chars' in self._schema_names(second)

    @pytest.mark.real_deno
    def test_multiplexed_jit_schema_uses_single_protocol_tool(self) -> None:
        pid = self._spawn_multiplexed_process()
        self._register_count_tool(pid, 'count_chars')

        schema_names = self._schema_names(pid)
        model_names = {row['name'] for row in self.runtime.tools.model_visible_tools(pid)}
        real_names = {row['name'] for row in self.runtime.tools.visible_tools(pid)}

        assert JIT_MULTIPLEXER_TOOL_NAME in schema_names
        assert 'count_chars' not in schema_names
        assert JIT_MULTIPLEXER_TOOL_NAME in model_names
        assert 'count_chars' not in model_names
        assert 'count_chars' in real_names

    def test_multiplexed_jit_schema_omits_protocol_tool_without_visible_jit(self) -> None:
        pid = self._spawn_multiplexed_process()

        assert JIT_MULTIPLEXER_TOOL_NAME not in self._schema_names(pid)
        assert JIT_MULTIPLEXER_TOOL_NAME not in {
            row['name'] for row in self.runtime.tools.model_visible_tools(pid)
        }

    @pytest.mark.real_deno
    def test_multiplexer_cannot_dispatch_static_or_other_process_tool(self) -> None:
        owner = self._spawn_multiplexed_process()
        other = self._spawn_multiplexed_process()
        self._register_count_tool(owner, 'owner_count')

        with pytest.raises(ValueError, match='only dispatch process-local JIT tools'):
            self.runtime.tools.normalize_model_action(
                owner,
                {'action': JIT_MULTIPLEXER_TOOL_NAME, 'tool_name': 'process_exit', 'arguments': {}},
            )
        with pytest.raises(ValueError, match='(?:not available in this process|not in process tool table)'):
            self.runtime.tools.normalize_model_action(
                other,
                {'action': JIT_MULTIPLEXER_TOOL_NAME, 'tool_name': 'owner_count', 'arguments': {'text': 'x'}},
            )

    def test_multiplexed_jit_rejects_reserved_protocol_tool_name(self) -> None:
        pid = self._spawn_multiplexed_process()

        with pytest.raises(ValidationError, match=JIT_MULTIPLEXER_TOOL_NAME):
            self.runtime.tools.propose(
                pid,
                {
                    'name': JIT_MULTIPLEXER_TOOL_NAME,
                    'description': 'Try to shadow the JIT multiplexer.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args, libos) { return {}; }',
            )

    def test_jit_proposal_rejects_provider_invalid_name_and_schema(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='bad jit spec')

        with pytest.raises(ValidationError, match='OpenAI tool name syntax'):
            self.runtime.tools.propose(
                pid,
                {
                    'name': 'bad name with spaces',
                    'description': 'Invalid provider-facing tool name.',
                    'input_schema': {'type': 'object'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args, libos) { return {}; }',
            )
        with pytest.raises(ValidationError, match='valid JSON schema'):
            self.runtime.tools.propose(
                pid,
                {
                    'name': 'bad_schema',
                    'description': 'Invalid provider-facing schema.',
                    'input_schema': {'type': 'definitely-not-a-json-schema-type'},
                    'output_schema': {'type': 'object'},
                },
                source_code='export function run(args, libos) { return {}; }',
            )

    @pytest.mark.real_deno
    def test_deno_jit_syscall_bypasses_tool_table_but_not_capabilities(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / 'pkg').mkdir()
            (root / 'pkg' / 'data.txt').write_text('secret', encoding='utf-8')
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='read via syscall')
                runtime.filesystem.grant_directory(pid, 'pkg', [CapabilityRight.READ], issued_by='test')
                assert 'read_text_file' not in runtime.process.get(pid).tool_table
                assert runtime.tools.call(pid, 'set_working_directory', {'path': 'pkg'}).ok is False
                candidate = runtime.tools.propose(pid, {'name': 'read_via_syscall', 'description': 'Read file.', 'input_schema': {'type': 'object'}}, source_code=READ_FILE_SOURCE)
                assert runtime.tools.validate(candidate).ok
                runtime.tools.register(pid, candidate)
                result = runtime.tools.call(pid, 'read_via_syscall', {'path': 'pkg/data.txt'})
                assert result.ok, result.error
                assert result.payload['content'] == 'secret'
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_deno_jit_syscall_denies_missing_capability(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='read denied')
        candidate = self.runtime.tools.propose(pid, {'name': 'read_denied', 'description': 'Read file.', 'input_schema': {'type': 'object'}}, source_code=READ_FILE_SOURCE)
        assert self.runtime.tools.validate(candidate).ok
        self.runtime.tools.register(pid, candidate)
        result = self.runtime.tools.call(pid, 'read_denied', {'path': 'README.md'})
        assert not result.ok
        assert 'lacks read' in (result.error or '')

    def test_shell_syscall_rejects_non_list_argv_before_policy(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='bad shell argv')
        session = LibOSSyscallSession(self.runtime, pid)
        with pytest.raises(ValidationError):
            asyncio.run(session.handle('shell.run', {'argv': 'git status'}))

    def test_capability_inspect_revalidates_subject_after_concurrent_mutation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pid = self.runtime.process.spawn(goal="inspect own capability")
        other = self.runtime.process.spawn(goal="must not receive inspected authority")
        capability = self.runtime.capability.issue_trusted(
            pid,
            "test:syscall-inspect",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        original_inspect = self.runtime.capability.inspect

        def transfer_before_authoritative_read(capability_id: str) -> dict[str, Any]:
            current = self.runtime.uow.authority.get_capability(capability_id)
            assert current is not None
            self.runtime.uow.authority.update_capability(
                replace(current, subject=other)
            )
            return original_inspect(capability_id)

        monkeypatch.setattr(
            self.runtime.capability,
            "inspect",
            transfer_before_authoritative_read,
        )
        session = LibOSSyscallSession(self.runtime, pid)

        with pytest.raises(
            CapabilityDenied,
            match="inspect only their own capabilities",
        ):
            session._capability_inspect({"capability_id": capability.cap_id})

    def test_capability_inspect_rejects_noncanonical_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pid = self.runtime.process.spawn(goal="reject malformed capability status")
        capability = self.runtime.capability.issue_trusted(
            pid,
            "test:syscall-status",
            [CapabilityRight.READ],
            issued_by="test.host",
        )
        monkeypatch.setattr(
            self.runtime.capability,
            "inspect",
            lambda capability_id: {
                "cap_id": capability_id,
                "subject": pid,
                "status": "future-status",
            },
        )
        session = LibOSSyscallSession(self.runtime, pid)

        with pytest.raises(ValidationError, match="invalid capability status"):
            session._capability_inspect({"capability_id": capability.cap_id})

    @pytest.mark.real_deno
    def test_deno_jit_human_approval_is_internal_to_syscall(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(root))
            try:
                pid = runtime.process.spawn(image='toolmaker-agent:v0', goal='write with approval')
                resource = runtime.filesystem.resource_for_path('out.txt')
                runtime.capability.set_permission_policy(pid, resource, [CapabilityRight.WRITE], CapabilityManager.ASK_EACH_TIME, issued_by='test')
                candidate = runtime.tools.propose(pid, {'name': 'write_via_syscall', 'description': 'Write file.', 'input_schema': {'type': 'object'}}, source_code=WRITE_FILE_SOURCE)
                assert runtime.tools.validate(candidate).ok
                runtime.tools.register(pid, candidate)
                with runtime.human_run_context(human_auto_approve=True):
                    result = runtime.tools.call(pid, 'write_via_syscall', {'path': 'out.txt', 'content': 'ok'})
                assert result.ok, result.error
                assert (root / 'out.txt').read_text(encoding='utf-8') == 'ok'
                assert 'human.response' in [record.action for record in runtime.audit.trace()]
            finally:
                runtime.close()

    @pytest.mark.real_deno
    def test_deno_jit_process_exit_is_applied_after_tool_result(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='exit after deno result')
        candidate = self.runtime.tools.propose(pid, {'name': 'exit_after_result', 'description': 'Exit.', 'input_schema': {'type': 'object'}}, source_code=EXIT_AFTER_RESULT_SOURCE)
        assert self.runtime.tools.validate(candidate).ok
        self.runtime.tools.register(pid, candidate)
        result = self.runtime.tools.call(pid, 'exit_after_result', {})
        assert result.ok, result.error
        assert result.payload == {'returned_after_exit_syscall': True}
        assert self.runtime.process.get(pid).status == ProcessStatus.EXITED

    @pytest.mark.real_deno
    def test_deno_jit_process_exec_is_applied_after_tool_result(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='exec after deno result')
        self.runtime.capability.grant(pid, 'image:base-agent:v0', [CapabilityRight.READ], issued_by='test')
        candidate = self.runtime.tools.propose(pid, {'name': 'exec_after_result', 'description': 'Exec.', 'input_schema': {'type': 'object'}}, source_code=EXEC_AFTER_RESULT_SOURCE)
        assert self.runtime.tools.validate(candidate).ok
        self.runtime.tools.register(pid, candidate)
        result = self.runtime.tools.call(pid, 'exec_after_result', {})
        process = self.runtime.process.get(pid)
        assert result.ok, result.error
        assert result.payload == {'returned_after_exec_syscall': True}
        assert process.image_id == 'base-agent:v0'
        assert process.status == ProcessStatus.RUNNABLE

    @pytest.mark.real_deno
    def test_deno_jit_deferred_exec_failure_does_not_persist_success_result(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='bad deferred exec')
        candidate = self.runtime.tools.propose(
            pid,
            {'name': 'bad_deferred_exec', 'description': 'Bad deferred exec.', 'input_schema': {'type': 'object'}},
            source_code=MISSING_EXEC_AFTER_RESULT_SOURCE,
        )
        assert self.runtime.tools.validate(candidate).ok
        self.runtime.tools.register(pid, candidate)

        result = self.runtime.tools.call(pid, 'bad_deferred_exec', {})

        assert not result.ok
        assert result.result_handle is None
        assert result.error == 'JIT tool failed while applying deferred lifecycle.'
        assert result.payload['policy_decision'] == 'lifecycle_error'
        assert [obj for obj in self.runtime.store.list_objects() if obj.type.value == 'tool_result'] == []
        tool_audits = [
            record
            for record in self.runtime.audit.trace()
            if record.action == 'tool.call' and record.decision.get('tool') == 'bad_deferred_exec'
        ]
        assert tool_audits[-1].decision['policy_decision'] == 'lifecycle_error'
        assert not any(record.decision.get('ok') is True for record in tool_audits)

    @pytest.mark.real_deno
    def test_direct_deno_jit_call_validates_input_schema(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='strict direct jit')
        candidate = self.runtime.tools.propose(
            pid,
            {
                'name': 'strict_direct_count',
                'description': 'Strict count.',
                'input_schema': {
                    'type': 'object',
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                    'additionalProperties': False,
                },
                'output_schema': {'type': 'object'},
            },
            source_code=COUNT_CHARS_SOURCE,
            tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}],
        )
        assert self.runtime.tools.validate(candidate).ok
        self.runtime.tools.register(pid, candidate)

        missing = self.runtime.tools.call(pid, 'strict_direct_count', {})
        extra = self.runtime.tools.call(pid, 'strict_direct_count', {'text': 'abc', 'extra': True})

        assert not missing.ok
        assert 'do not match input_schema' in (missing.error or '')
        assert not extra.ok
        assert 'Additional properties are not allowed' in (extra.error or '')
        with pytest.raises(ValueError, match='do not match input_schema'):
            self.runtime.tools.normalize_model_action(pid, {'action': 'strict_direct_count'})

    @pytest.mark.real_deno
    def test_deno_jit_call_validates_output_schema(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='bad output jit')
        candidate = self.runtime.tools.propose(
            pid,
            {
                'name': 'bad_output_jit',
                'description': 'Bad output.',
                'input_schema': {'type': 'object'},
                'output_schema': {
                    'type': 'object',
                    'properties': {'count': {'type': 'integer'}},
                    'required': ['count'],
                    'additionalProperties': False,
                },
            },
            source_code=BAD_OUTPUT_SOURCE,
        )
        self.runtime.tools.register(pid, candidate)

        result = self.runtime.tools.call(pid, 'bad_output_jit', {})

        assert not result.ok
        assert result.result_handle is None
        assert 'output_schema' in (result.error or '')
        assert [obj for obj in self.runtime.store.list_objects() if obj.type.value == 'tool_result'] == []

    def test_deno_static_check_is_format_and_dependency_lint_not_security_blacklist(self) -> None:
        checker = DenoTypescriptSandbox(deno_executable='deno')
        computed_deno = checker.static_check(
            'export async function run(args, libos) { '
            'const d = (globalThis as Record<string, any>)["De" + "no"]; '
            'return d.readTextFileSync("x"); }'
        )
        denied_import = checker.static_check('import x from "npm:left-pad";\nexport async function run(args, libos) { return {}; }')
        ordinary_export = checker.static_check(
            'export function run(args, libos) { '
            'const obj = { from: "npm:left-pad", import() { return "local"; } }; '
            'const label = `hello ${args.name ?? "world"}`; '
            'return { value: obj.import(), label }; }'
        )

        assert computed_deno.ok, computed_deno.errors
        assert ordinary_export.ok, ordinary_export.errors
        assert not denied_import.ok
        assert any(('imports are not allowed in JIT tool source: npm:left-pad' in error for error in denied_import.errors))

    def test_deno_static_check_import_allowlist(self) -> None:
        checker = DenoTypescriptSandbox(deno_executable='deno')
        pinned_jsr = checker.static_check('import { join } from "jsr:@std/path@1.0.0";\nexport function run(args, libos) { return { path: join("a", "b") }; }')
        denied = checker.static_check('import fs from "node:fs";\nimport x from "https://deno.land/std/path/mod.ts";\nimport y from "file:///tmp/tool.ts";\nimport z from "jsr:@bad/pkg@1.0.0";\nimport { join } from "jsr:@std/path";\nexport function run(args, libos) { return {}; }')
        assert not pinned_jsr.ok
        assert any(('imports are not allowed in JIT tool source: jsr:@std/path@1.0.0' in error for error in pinned_jsr.errors))
        assert not denied.ok
        assert any(('imports are not allowed in JIT tool source: node:fs' in error for error in denied.errors))
        assert any(('imports are not allowed in JIT tool source: https://deno.land/std/path/mod.ts' in error for error in denied.errors))
        assert any(('imports are not allowed in JIT tool source: file:///tmp/tool.ts' in error for error in denied.errors))
        assert any(('imports are not allowed in JIT tool source: jsr:@bad/pkg@1.0.0' in error for error in denied.errors))
        assert any(('imports are not allowed in JIT tool source: jsr:@std/path' in error for error in denied.errors))

    def test_deno_static_check_rejects_mutable_jsr_versions(self) -> None:
        checker = DenoTypescriptSandbox(deno_executable='deno')
        pinned = checker.static_check('import { join } from "jsr:@std/path@1.0.0";\nexport function run(args, libos) { return { path: join("a", "b") }; }')
        denied = checker.static_check('import { join } from "jsr:@std/path@1";\nimport { normalize } from "jsr:@std/path@latest";\nimport { dirname } from "jsr:@std/path@^1.0.0";\nexport function run(args, libos) { return {}; }')

        assert not pinned.ok
        assert any('imports are not allowed in JIT tool source: jsr:@std/path@1.0.0' in error for error in pinned.errors)
        assert not denied.ok
        assert any('imports are not allowed in JIT tool source: jsr:@std/path@1' in error for error in denied.errors)
        assert any('imports are not allowed in JIT tool source: jsr:@std/path@latest' in error for error in denied.errors)
        assert any('imports are not allowed in JIT tool source: jsr:@std/path@^1.0.0' in error for error in denied.errors)

    def test_deno_static_check_rejects_comment_split_imports(self) -> None:
        checker = DenoTypescriptSandbox(deno_executable='deno')
        dynamic_import = checker.static_check('export async function run(args, libos) { return await import/*comment*/("https://example.com/tool.ts"); }')
        template_import = checker.static_check('export async function run(args, libos) { return `${await import("npm:left-pad")}`; }')
        npm_import = checker.static_check('import x from /*comment*/ "npm:left-pad";\nexport function run(args, libos) { return {}; }')
        exported_import = checker.static_check('export { join } from /*comment*/ "npm:left-pad";\nexport function run(args, libos) { return {}; }')
        jsr_import = checker.static_check('import { join } from /*comment*/ "jsr:@std/path@1.0.0";\nexport function run(args, libos) { return { path: join("a", "b") }; }')

        assert not dynamic_import.ok
        assert any('dynamic import() is not allowed' in error for error in dynamic_import.errors)
        assert not template_import.ok
        assert any('dynamic import() is not allowed' in error for error in template_import.errors)
        assert not npm_import.ok
        assert any('imports are not allowed in JIT tool source: npm:left-pad' in error for error in npm_import.errors)
        assert not exported_import.ok
        assert any('imports are not allowed in JIT tool source: npm:left-pad' in error for error in exported_import.errors)
        assert not jsr_import.ok
        assert any('imports are not allowed in JIT tool source: jsr:@std/path@1.0.0' in error for error in jsr_import.errors)

    def test_deno_static_check_rejects_runtime_code_generation_import_bypasses(self) -> None:
        checker = DenoTypescriptSandbox(deno_executable='deno')
        eval_import = checker.static_check('export async function run(args, libos) { return await eval("import(args.spec)"); }')
        new_function_import = checker.static_check('export async function run(args, libos) { const loader = new Function("s", "return import(s)"); return await loader(args.spec); }')
        global_function_import = checker.static_check('export async function run(args, libos) { const loader = globalThis.Function("s", "return import(s)"); return await loader(args.spec); }')
        global_function_call_import = checker.static_check('export async function run(args, libos) { return globalThis.Function.call(null, "return import(args.spec)")(); }')
        bracket_function_import = checker.static_check('export async function run(args, libos) { const loader = globalThis["Function"]("s", "return import(s)"); return await loader(args.spec); }')
        bracket_function_call_import = checker.static_check('export async function run(args, libos) { return globalThis["Function"].call(null, "return import(args.spec)")(); }')
        bracket_eval_import = checker.static_check('export async function run(args, libos) { return await window["eval"]("import(args.spec)"); }')
        optional_eval_import = checker.static_check('export async function run(args, libos) { return await eval?.("import(args.spec)"); }')
        indirect_eval_import = checker.static_check('export async function run(args, libos) { return await (0, eval)("import(args.spec)"); }')
        computed_eval_import = checker.static_check('export async function run(args, libos) { return await globalThis["ev" + "al"]("import(args.spec)"); }')
        optional_function_import = checker.static_check('export async function run(args, libos) { const loader = Function?.("s", "return import(s)"); return await loader(args.spec); }')
        function_call_import = checker.static_check('export async function run(args, libos) { const loader = Function.call(null, "s", "return import(s)"); return await loader(args.spec); }')
        async_function_import = checker.static_check('export async function run(args, libos) { const AsyncFunction = (async function() {}).constructor; return await AsyncFunction("return import(args.spec)")(); }')
        generator_function_import = checker.static_check('export function run(args, libos) { const GeneratorFunction = (function* () {}).constructor; return GeneratorFunction("yield 1")().next(); }')
        direct_constructor_import = checker.static_check('export async function run(args, libos) { return await (async function() {}).constructor("return import(args.spec)")(); }')
        bracket_constructor_import = checker.static_check('export async function run(args, libos) { return await (async function() {})["constructor"]("return import(args.spec)")(); }')
        optional_constructor_import = checker.static_check('export async function run(args, libos) { return await (async function() {}).constructor?.("return import(args.spec)")(); }')
        bracket_constructor_call_import = checker.static_check('export async function run(args, libos) { return await (async function() {})["constructor"].call(null, "return import(args.spec)")(); }')
        constructor_alias_import = checker.static_check('export async function run(args, libos) { const C = (function(){}).constructor; return await C("return import(args.spec)")(); }')
        bracket_constructor_alias_import = checker.static_check('export async function run(args, libos) { const C = (function(){})["constructor"]; return await C("return import(args.spec)")(); }')
        optional_constructor_alias_import = checker.static_check('export async function run(args, libos) { const C = (function(){}).constructor; return await C?.("return import(args.spec)")(); }')
        local_methods = checker.static_check(
            'export function run(args, libos) { '
            'const obj = { eval() { return 1; }, Function() { return 2; } }; '
            'return { value: obj.eval() + obj.Function() + obj["eval"]() + obj["Function"]() + obj.Function.call(obj) }; }'
        )

        assert not eval_import.ok
        assert any('runtime code generation is not allowed' in error for error in eval_import.errors)
        assert not new_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in new_function_import.errors)
        assert not global_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in global_function_import.errors)
        assert not global_function_call_import.ok
        assert any('runtime code generation is not allowed' in error for error in global_function_call_import.errors)
        assert not bracket_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_function_import.errors)
        assert not bracket_function_call_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_function_call_import.errors)
        assert not bracket_eval_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_eval_import.errors)
        assert not optional_eval_import.ok
        assert any('runtime code generation is not allowed' in error for error in optional_eval_import.errors)
        assert not indirect_eval_import.ok
        assert any('runtime code generation is not allowed' in error for error in indirect_eval_import.errors)
        assert not computed_eval_import.ok
        assert any('runtime code generation is not allowed' in error for error in computed_eval_import.errors)
        assert not optional_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in optional_function_import.errors)
        assert not function_call_import.ok
        assert any('runtime code generation is not allowed' in error for error in function_call_import.errors)
        assert not async_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in async_function_import.errors)
        assert not generator_function_import.ok
        assert any('runtime code generation is not allowed' in error for error in generator_function_import.errors)
        assert not direct_constructor_import.ok
        assert any('runtime code generation is not allowed' in error for error in direct_constructor_import.errors)
        assert not bracket_constructor_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_constructor_import.errors)
        assert not optional_constructor_import.ok
        assert any('runtime code generation is not allowed' in error for error in optional_constructor_import.errors)
        assert not bracket_constructor_call_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_constructor_call_import.errors)
        assert not constructor_alias_import.ok
        assert any('runtime code generation is not allowed' in error for error in constructor_alias_import.errors)
        assert not bracket_constructor_alias_import.ok
        assert any('runtime code generation is not allowed' in error for error in bracket_constructor_alias_import.errors)
        assert not optional_constructor_alias_import.ok
        assert any('runtime code generation is not allowed' in error for error in optional_constructor_alias_import.errors)
        assert local_methods.ok, local_methods.errors

    def test_deno_executable_resolution_rejects_workspace_path_hijack(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        hijack_deno = tmp_path / ('deno.exe' if os.name == 'nt' else 'deno')
        hijack_deno.write_text('#!/bin/sh\necho hijack-deno\n', encoding='utf-8')
        hijack_deno.chmod(0o755)
        monkeypatch.setenv('PATH', str(tmp_path))
        sandbox = DenoTypescriptSandbox(deno_executable='deno', forbidden_executable_roots=[tmp_path])

        metadata = sandbox.metadata_for_source('export function run(args, libos) { return {}; }')

        assert 'deno_version_error' in metadata
        assert 'forbidden root' in metadata['deno_version_error']

    @pytest.mark.real_deno
    def test_deno_candidate_tests_fail_when_expected_syscall_is_not_performed(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='deno')
        validation = sandbox.run_tests('export function run(args, libos) { return { ok: true }; }', [{'args': {}, 'syscalls': [{'name': 'filesystem.read_text', 'args': {'path': 'README.md'}}], 'expected': {'ok': True}}])
        assert not validation.ok
        assert any(('expected syscall(s) not performed' in error for error in validation.errors))

    def test_jit_proposal_limits_source_and_tests_before_persistence(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='jit limits')
        with pytest.raises(ValidationError):
            self.runtime.tools.propose(
                pid,
                {'name': 'huge_source', 'description': 'Huge source.', 'input_schema': {'type': 'object'}},
                source_code='x' * (self.runtime.config.tools.jit_source_max_chars + 1),
            )
        with pytest.raises(ValidationError):
            self.runtime.tools.propose(
                pid,
                {'name': 'too_many_tests', 'description': 'Too many tests.', 'input_schema': {'type': 'object'}},
                source_code='export function run(args, libos) { return {}; }',
                tests=[{} for _ in range(self.runtime.config.tools.jit_tests_max_count + 1)],
            )
        with pytest.raises(ValidationError):
            self.runtime.tools.propose(
                pid,
                {'name': 'huge_test', 'description': 'Huge test.', 'input_schema': {'type': 'object'}},
                source_code='export function run(args, libos) { return {}; }',
                tests=[{'args': {'blob': 'x' * self.runtime.config.tools.jit_test_case_max_bytes}}],
            )

    def test_jit_validation_fails_closed_without_budget_limit_support(self) -> None:
        self.runtime.tools.sandbox = NoLimitValidationSandbox()
        pid = self.runtime.process.spawn(
            image='toolmaker-agent:v0',
            goal='budgeted jit validation',
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
        )
        candidate = self.runtime.tools.propose(
            pid,
            {'name': 'budgeted_no_limits', 'description': 'No limits.', 'input_schema': {'type': 'object'}},
            source_code='export function run(args, libos) { return {}; }',
        )

        with pytest.raises(ValidationError):
            self.runtime.tools.validate(candidate)

    def test_jit_validation_charges_returned_subprocess_metrics(self) -> None:
        sandbox = RecordingValidationSandbox()
        self.runtime.tools.sandbox = sandbox
        pid = self.runtime.process.spawn(
            image='toolmaker-agent:v0',
            goal='budgeted jit validation metrics',
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=1.0),
        )
        candidate = self.runtime.tools.propose(
            pid,
            {'name': 'budgeted_metrics', 'description': 'Metrics.', 'input_schema': {'type': 'object'}},
            source_code='export function run(args, libos) { return {}; }',
        )

        validation = self.runtime.tools.validate(candidate)

        assert validation.ok, validation.errors
        assert isinstance(sandbox.last_limits, SubprocessLimits)
        assert self.runtime.process.get(pid).resource_usage.subprocess_wall_seconds == pytest.approx(0.25)

    def test_jit_validation_recomputes_budget_between_test_cases(self) -> None:
        sandbox = PerCaseValidationSandbox(wall_seconds=0.25)
        self.runtime.tools.sandbox = sandbox
        pid = self.runtime.process.spawn(
            image='toolmaker-agent:v0',
            goal='per-case validation budget',
            resource_budget=ResourceBudget(max_subprocess_wall_seconds=0.25),
        )
        candidate = self.runtime.tools.propose(
            pid,
            {'name': 'per_case_budget', 'description': 'Metrics.', 'input_schema': {'type': 'object'}},
            source_code='export function run(args, libos) { return {}; }',
            tests=[{'args': {'case': 1}}, {'args': {'case': 2}}],
        )

        with pytest.raises(ResourceLimitExceeded):
            self.runtime.tools.validate(candidate)

        assert sandbox.calls == 1
        assert self.runtime.process.get(pid).resource_usage.subprocess_wall_seconds == pytest.approx(0.25)

    def test_tool_events_and_audit_do_not_store_raw_sensitive_tool_args(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(temp_dir))
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='redact tool args')
                secret = 'SECRET_TOOL_ARG_SHOULD_NOT_APPEAR'
                runtime.tools.configure_process_tools(pid, ['write_text_file', 'send_process_message'], assigned_by='test')
                runtime.filesystem.grant_path(pid, 'secret.txt', [CapabilityRight.WRITE], issued_by='test')

                written = runtime.tools.call(
                    pid,
                    'write_text_file',
                    {'path': 'secret.txt', 'content': secret, 'overwrite': True},
                )
                sent = runtime.tools.call(
                    pid,
                    'send_process_message',
                    {'recipient_pid': pid, 'subject': 'secret', 'body': secret, 'payload': {'token': secret}},
                )

                assert written.ok, written.error
                assert sent.ok, sent.error
                observed = dumps(
                    {
                        'events': [event.payload for event in runtime.events.list(target=pid)],
                        'audit': [record.decision for record in runtime.audit.trace()],
                    }
                )
                assert secret not in observed
                assert 'sha256' in observed
            finally:
                runtime.close()

    def test_jit_validation_observability_redacts_sensitive_errors(self) -> None:
        secret = 'SECRET_JIT_VALIDATION_LEAK'
        self.runtime.tools.sandbox = LeakyValidationSandbox(secret)
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='redact jit validation')
        candidate = self.runtime.tools.propose(
            pid,
            {'name': 'leaky_validation', 'description': 'Leaky validation.', 'input_schema': {'type': 'object'}},
            source_code='export function run(args, libos) { return {}; }',
        )

        validation = self.runtime.tools.validate(candidate)
        persisted = self.runtime.store.get_tool_candidate(candidate).validation
        observed = dumps(
            {
                'candidate': persisted,
                'audit': [record.decision for record in self.runtime.audit.trace()],
            }
        )

        assert not validation.ok
        assert secret in validation.errors[0]
        assert secret not in observed
        assert 'sha256' in observed

    def test_jit_schema_error_observability_redacts_sensitive_values(self) -> None:
        secret = 'SECRET_JIT_SCHEMA_LEAK'
        self.runtime.tools.sandbox = SecretOutputSandbox(secret)
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='redact jit schema')
        candidate = self.runtime.tools.propose(
            pid,
            {
                'name': 'bad_secret_output',
                'description': 'Returns a sensitive invalid output.',
                'input_schema': {'type': 'object'},
                'output_schema': {
                    'type': 'object',
                    'properties': {'token': {'type': 'number'}},
                    'required': ['token'],
                },
            },
            source_code='export function run(args, libos) { return {}; }',
        )
        self.runtime.tools.register(pid, candidate)

        result = self.runtime.tools.call(pid, 'bad_secret_output', {})
        observed = dumps(
            {
                'events': [event.payload for event in self.runtime.events.list(target=pid)],
                'audit': [record.decision for record in self.runtime.audit.trace()],
            }
        )

        assert not result.ok
        assert secret in (result.error or '')
        assert secret not in observed
        assert 'sha256' in observed

    @pytest.mark.real_deno
    def test_real_deno_tool_runs_and_has_no_host_read_permission(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='deno', default_timeout_s=10.0)
        result = sandbox.run_source('export function run(args, libos) { return { doubled: args.value * 2 }; }', {'value': 21})
        with pytest.raises(Exception) as raised:
            sandbox.run_source('export function run(args, libos) { const d = (globalThis as Record<string, any>)["De" + "no"]; return d.readTextFileSync("secret.txt"); }', {})
        assert result == {'doubled': 42}
        assert 'read' in str(raised.value).lower() or 'permission' in str(raised.value).lower()

    @pytest.mark.real_deno
    @pytest.mark.skipif(os.name != 'posix', reason='POSIX death-pipe integration')
    def test_deno_supervisor_kills_untrusted_process_after_host_sigkill(self) -> None:
        candidate = '''
            export async function run(args, libos) {
              const deno = (globalThis as Record<string, any>)["Deno"];
              await libos.syscall("test.report_pid", { pid: deno.pid });
              while (true) {}
            }
        '''
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_file = Path(temp_dir) / 'deno.pid'
            helper_source = (
                'from pathlib import Path\n'
                'from agent_libos.substrate import CommandMetrics\n'
                'from agent_libos.tools.sandbox import DenoTypescriptSandbox\n'
                'class ParentDeathSandbox(DenoTypescriptSandbox):\n'
                '    async def _monitor_process(self, proc, limits):\n'
                '        await proc.wait()\n'
                '        return CommandMetrics()\n'
                f'pid_file = Path({str(pid_file)!r})\n'
                'def handler(name, args):\n'
                '    assert name == "test.report_pid"\n'
                '    pid_file.write_text(str(args["pid"]), encoding="utf-8")\n'
                '    return {}\n'
                f'ParentDeathSandbox(deno_executable="deno", default_timeout_s=60.0).run_source({candidate!r}, {{}}, syscall_handler=handler, timeout=60.0)\n'
            )
            host = subprocess.Popen(
                [sys.executable, '-c', helper_source],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deno_pid: int | None = None
            try:
                deadline = time.monotonic() + 10.0
                while time.monotonic() < deadline:
                    if host.poll() is not None:
                        pytest.fail(f'Deno host exited before the parent-death check: {host.returncode}')
                    if pid_file.exists():
                        deno_pid = int(pid_file.read_text(encoding='utf-8'))
                        break
                    time.sleep(0.05)
                assert deno_pid is not None, 'Deno child did not report its pid'

                os.kill(host.pid, signal.SIGKILL)
                host.wait(timeout=5.0)
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and _pid_is_alive(deno_pid):
                    time.sleep(0.05)
                assert not _pid_is_alive(deno_pid), f'Deno process survived host SIGKILL: {deno_pid}'
            finally:
                if host.poll() is None:
                    host.kill()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        host.wait(timeout=2.0)
                if deno_pid is not None and _pid_is_alive(deno_pid):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.kill(deno_pid, signal.SIGKILL)

    @pytest.mark.real_deno
    def test_real_deno_result_frame_completes_even_with_live_handles(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='deno', default_timeout_s=1.0)
        result = sandbox.run_source('\n            export function run(args, libos) {\n              setInterval(() => {}, 1000);\n              return { ok: true };\n            }\n            ', {})
        assert result == {'ok': True}

    def test_deno_missing_is_clear_validation_error(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='agent-libos-deno-definitely-missing')
        validation = sandbox.run_tests('export function run(args, libos) { return {}; }', [])
        assert not validation.ok
        assert any(('Deno executable not found' in error for error in validation.errors))

    @pytest.mark.real_deno
    def test_deno_validation_logs_are_bounded(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='deno', max_validation_log_chars=64)
        validation = sandbox.run_tests(
            'export function run(args, libos) { return { blob: "x".repeat(1000) }; }',
            [{'args': {}}],
        )

        assert validation.ok
        assert len(validation.logs) < 256
        assert 'validation logs truncated' in validation.logs
        assert 'sha256=' in validation.logs

    @pytest.mark.real_deno
    def test_deno_validation_mismatch_errors_are_bounded(self) -> None:
        sandbox = DenoTypescriptSandbox(deno_executable='deno', max_validation_log_chars=64)
        validation = sandbox.run_tests(
            'export function run(args, libos) { return { blob: "x".repeat(1000) }; }',
            [{'args': {}, 'expected': {'ok': True}}],
        )

        assert not validation.ok
        assert len(validation.errors[0]) < 256
        assert 'truncated validation result repr' in validation.errors[0]
        assert 'sha256=' in validation.errors[0]
        assert 'x' * 128 not in validation.errors[0]

    @pytest.mark.real_deno
    def test_jit_tool_cannot_shadow_existing_tool_name(self) -> None:
        pid = self.runtime.process.spawn(image='toolmaker-agent:v0', goal='shadow builtin')
        candidate = self.runtime.tools.propose(pid, {'name': 'process_exit', 'description': 'Try to shadow a builtin.', 'input_schema': {'type': 'object'}, 'output_schema': {'type': 'object'}}, source_code='export function run(args, libos) { return { ok: true }; }', tests=[{'args': {}, 'expected': {'ok': True}}])
        validation = self.runtime.tools.validate(candidate)
        assert validation.ok, validation.errors
        with pytest.raises(ValidationError):
            self.runtime.tools.register(pid, candidate)

    def test_builtin_tools_do_not_directly_touch_host_boundaries(self) -> None:
        builtins_dir = Path('agent_libos/tools/builtin')
        forbidden = {'subprocess', 'urllib', 'socket', 'requests'}
        for path in builtins_dir.glob('*.py'):
            source = path.read_text(encoding='utf-8')
            imports: set[str] = set()
            for node in ast.walk(ast.parse(source, filename=str(path))):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split('.', 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split('.', 1)[0])
            assert not (imports & forbidden), f'{path} should not import host boundary libraries directly'

    def _schema_names(self, pid: str) -> set[str]:
        return {schema['function']['name'] for schema in self.runtime.tools.openai_tool_schemas(pid)}

    def _register_count_tool(self, pid: str, name: str) -> Any:
        candidate = self.runtime.tools.propose(pid, {'name': name, 'description': 'Count characters in text.', 'input_schema': {'type': 'object', 'properties': {'text': {'type': 'string'}}}, 'output_schema': {'type': 'object'}}, source_code=COUNT_CHARS_SOURCE, tests=[{'args': {'text': 'abc'}, 'expected': {'count': 3}}])
        assert self.runtime.tools.validate(candidate).ok
        return self.runtime.tools.register(pid, candidate)

    def _spawn_multiplexed_process(self) -> str:
        image_id = 'multiplexed-toolmaker:v0'
        if image_id not in self.runtime.images:
            self.runtime.register_image(
                AgentImage(
                    image_id=image_id,
                    name='multiplexed-toolmaker',
                    default_tools=['process_exit'],
                    jit_tool_exposure=JIT_TOOL_EXPOSURE_MULTIPLEXED,
                ),
                actor='test',
            )
        return self.runtime.process.spawn(image=image_id, goal='multiplexed jit')


class NoLimitValidationSandbox(SandboxBackend):
    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {"ok": True}

    def run_tests(self, source_code: str, tests: list[dict[str, Any]], timeout: float | None = None) -> ValidationResult:
        return ValidationResult(ok=True)


class RecordingValidationSandbox(SandboxBackend):
    def __init__(self) -> None:
        self.last_limits: SubprocessLimits | None = None

    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {"ok": True}

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        self.last_limits = limits
        metadata = {}
        if return_metrics:
            metadata["metrics"] = {
                "wall_seconds": 0.25,
                "cpu_seconds": 0.05,
                "peak_memory_bytes": 1024,
                "killed": False,
                "limit_kind": None,
            }
        return ValidationResult(ok=True, metadata=metadata)


class PerCaseValidationSandbox(SandboxBackend):
    def __init__(self, *, wall_seconds: float) -> None:
        self.wall_seconds = wall_seconds
        self.calls = 0

    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {"ok": True}

    def run_tests(
        self,
        source_code: str,
        tests: list[dict[str, Any]],
        timeout: float | None = None,
        *,
        limits: SubprocessLimits | None = None,
        return_metrics: bool = False,
    ) -> ValidationResult:
        self.calls += 1
        assert len(tests) <= 1
        metadata = {}
        if return_metrics:
            metadata["metrics"] = {
                "wall_seconds": self.wall_seconds,
                "cpu_seconds": 0.0,
                "peak_memory_bytes": 0,
                "killed": False,
                "limit_kind": None,
            }
        return ValidationResult(ok=True, logs=f"case {self.calls}", metadata=metadata)


class LeakyValidationSandbox(SandboxBackend):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {"ok": True}

    def run_tests(self, source_code: str, tests: list[dict[str, Any]], timeout: float | None = None) -> ValidationResult:
        return ValidationResult(
            ok=False,
            errors=[f"validation failed with token={self.secret}"],
            logs=f"Authorization: Bearer {self.secret}",
        )


class SecretOutputSandbox(SandboxBackend):
    def __init__(self, secret: str) -> None:
        self.secret = secret

    def static_check(self, source_code: str) -> ValidationResult:
        return ValidationResult(ok=True)

    async def arun_source(self, source_code: str, args: dict[str, Any], **kwargs: Any) -> Any:
        return {"token": self.secret}

    def run_tests(self, source_code: str, tests: list[dict[str, Any]], timeout: float | None = None) -> ValidationResult:
        return ValidationResult(ok=True)
