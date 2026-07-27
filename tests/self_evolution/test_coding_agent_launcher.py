from __future__ import annotations
import importlib
import pytest
import asyncio
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from agent_libos import Runtime
from agent_libos.capability.manager import CapabilityManager
from agent_libos.models import CapabilityRight
from agent_libos.runtime import RuntimeAssemblyCleanupRequired
from agent_libos.substrate import LocalResourceProviderSubstrate
from scripts import run_coding_agent, runtime_assembly


ASYNC_RUNTIME_ENTRYPOINTS = (
    'ask_file_then_show.py',
    'async_clock_interleave_smoke.py',
    'human_llm_chat.py',
    'llm_summarize_document.py',
    'llm_write_goal_smoke.py',
    'object_memory_file_copy_smoke.py',
    'run_coding_agent.py',
)

class TestCodingAgentLauncher:

    def test_default_runtime_database_is_outside_writable_workspace(
        self,
        tmp_path: Path,
    ) -> None:
        args = run_coding_agent.build_parser().parse_args(
            ["--goal", "edit", "--workspace", str(tmp_path), "--no-run"]
        )

        selected = run_coding_agent._resolve_db_path(args, tmp_path)

        assert not run_coding_agent._path_is_within(selected, tmp_path)

    def test_explicit_runtime_database_inside_workspace_is_rejected(
        self,
        tmp_path: Path,
    ) -> None:
        args = run_coding_agent.build_parser().parse_args(
            [
                "--goal",
                "edit",
                "--workspace",
                str(tmp_path),
                "--db",
                str(tmp_path / "runtime.sqlite"),
                "--no-run",
            ]
        )

        with pytest.raises(SystemExit, match="must stay outside"):
            run_coding_agent._resolve_db_path(args, tmp_path)

    def test_default_edit_preset_pregrants_workspace_write_not_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = run_coding_agent.build_parser().parse_args(['--goal', 'edit the workspace', '--workspace', tmp, '--no-run'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(Path(tmp)))
            try:
                pid = runtime.process.spawn(image='coding-agent:v0', goal=args.goal)
                grants = run_coding_agent.configure_coding_agent_permissions(runtime, pid, args)
                assert grants
                assert runtime.capability.permission_policy(pid, runtime.filesystem.workspace_resource(), CapabilityRight.WRITE) == CapabilityManager.ALWAYS_ALLOW
                assert runtime.capability.permission_policy(pid, runtime.filesystem.workspace_resource(), CapabilityRight.DELETE) == CapabilityManager.MISSING
            finally:
                runtime.close()

    def test_read_only_preset_can_add_specific_write_and_delete_grants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = run_coding_agent.build_parser().parse_args(['--goal', 'edit selected paths', '--workspace', tmp, '--permission-preset', 'read-only', '--write-file', 'src/main.py', '--delete-dir', 'build', '--no-run'])
            runtime = Runtime.open('local', substrate=LocalResourceProviderSubstrate(Path(tmp)))
            try:
                pid = runtime.process.spawn(image='coding-agent:v0', goal=args.goal)
                run_coding_agent.configure_coding_agent_permissions(runtime, pid, args)
                assert runtime.capability.permission_policy(pid, runtime.filesystem.resource_for_path('src/main.py'), CapabilityRight.WRITE) == CapabilityManager.ALWAYS_ALLOW
                assert runtime.capability.permission_policy(pid, runtime.filesystem.directory_resource_for_path('build'), CapabilityRight.DELETE) == CapabilityManager.ALWAYS_ALLOW
                assert runtime.capability.permission_policy(pid, runtime.filesystem.workspace_resource(), CapabilityRight.WRITE) == CapabilityManager.MISSING
            finally:
                runtime.close()

    def test_launcher_loads_project_env_before_workspace_mount(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            env_file = project_root / '.env'
            env_file.write_text('OPENAI_MODEL=test\n', encoding='utf-8')
            args = run_coding_agent.build_parser().parse_args(['--goal', 'inspect', '--workspace', '.', '--no-run'])
            with patch.object(run_coding_agent, 'PROJECT_ROOT', project_root), patch.object(run_coding_agent, 'load_dotenv') as load:
                run_coding_agent._load_env(args)
        load.assert_called_once_with(env_file)

    def test_launcher_does_not_change_host_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            before = Path.cwd()
            args = run_coding_agent.build_parser().parse_args(['--goal', 'inspect', '--workspace', tmp, '--ephemeral-db', '--no-run'])
            with patch.object(run_coding_agent, 'load_dotenv'), patch.object(
                Runtime,
                'open',
                side_effect=AssertionError('async launcher must not call Runtime.open'),
            ):
                asyncio.run(run_coding_agent.amain(args))
            assert Path.cwd() == before

    def test_aopen_runtime_returns_successful_runtime(self) -> None:
        async def exercise() -> None:
            args = run_coding_agent.build_parser().parse_args(
                ['--goal', 'inspect', '--workspace', '.', '--ephemeral-db', '--no-run']
            )
            runtime = object()
            with patch.object(Runtime, 'aopen', new=AsyncMock(return_value=runtime)) as aopen:
                opened = await run_coding_agent._aopen_runtime(args, Path.cwd())
            assert opened is runtime
            assert aopen.await_count == 1
            assert aopen.await_args.args == ('local',)
            assert isinstance(
                aopen.await_args.kwargs['substrate'],
                LocalResourceProviderSubstrate,
            )

        asyncio.run(exercise())

    def test_aopen_runtime_releases_assembly_cleanup_handle_before_reraising(self) -> None:
        async def exercise() -> None:
            args = run_coding_agent.build_parser().parse_args(
                ['--goal', 'inspect', '--workspace', '.', '--ephemeral-db', '--no-run']
            )
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=None,
                store=object(),  # type: ignore[arg-type]
                cleanup_errors=[{'component': 'scheduler'}],
            )
            release = AsyncMock()
            handle.arelease = release  # type: ignore[method-assign]
            assembly_error = BaseExceptionGroup(
                'injected assembly failure',
                [RuntimeError('assembly failed'), handle],
            )
            with patch.object(Runtime, 'aopen', new=AsyncMock(side_effect=assembly_error)):
                with pytest.raises(BaseExceptionGroup) as caught:
                    await run_coding_agent._aopen_runtime(args, Path.cwd())
            assert caught.value is assembly_error
            release.assert_awaited_once_with()

        asyncio.run(exercise())

    def test_aopen_runtime_preserves_primary_cleanup_and_cancellation_errors(self) -> None:
        async def exercise() -> None:
            args = run_coding_agent.build_parser().parse_args(
                ['--goal', 'inspect', '--workspace', '.', '--ephemeral-db', '--no-run']
            )
            handle = RuntimeAssemblyCleanupRequired(
                partial_runtime=None,
                store=object(),  # type: ignore[arg-type]
                cleanup_errors=[{'component': 'scheduler'}],
            )
            cleanup_error = BaseExceptionGroup(
                'injected cleanup failure',
                [OSError('cleanup failed'), asyncio.CancelledError()],
            )
            release = AsyncMock(side_effect=cleanup_error)
            handle.arelease = release  # type: ignore[method-assign]
            primary = RuntimeError('assembly failed')
            assembly_error = BaseExceptionGroup(
                'injected assembly failure',
                [primary, handle],
            )
            with patch.object(Runtime, 'aopen', new=AsyncMock(side_effect=assembly_error)):
                with pytest.raises(BaseExceptionGroup) as caught:
                    await run_coding_agent._aopen_runtime(args, Path.cwd())
            assert caught.value is not assembly_error
            assert caught.value.exceptions[0] is assembly_error
            assert caught.value.subgroup(RuntimeError) is not None
            assert caught.value.subgroup(OSError) is not None
            assert caught.value.subgroup(asyncio.CancelledError) is not None
            assert RuntimeAssemblyCleanupRequired.extract(caught.value) == (handle,)
            release.assert_awaited_once_with()

        asyncio.run(exercise())

    @pytest.mark.parametrize('script_name', ASYNC_RUNTIME_ENTRYPOINTS)
    def test_async_entrypoint_uses_shared_runtime_assembly_helper(
        self,
        script_name: str,
    ) -> None:
        module_name = script_name.removesuffix('.py')
        module = importlib.import_module(f'scripts.{module_name}')
        assert module.aopen_runtime is runtime_assembly.aopen_runtime
        source = (run_coding_agent.PROJECT_ROOT / 'scripts' / script_name).read_text(
            encoding='utf-8'
        )
        assert 'await Runtime.aopen(' not in source

    @pytest.mark.parametrize('script_name', ASYNC_RUNTIME_ENTRYPOINTS)
    def test_async_entrypoint_can_load_as_a_direct_script(
        self,
        script_name: str,
        tmp_path: Path,
    ) -> None:
        script = run_coding_agent.PROJECT_ROOT / 'scripts' / script_name
        completed = subprocess.run(
            [sys.executable, str(script), '--help'],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

    def test_explicit_missing_env_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / 'missing.env'
            args = run_coding_agent.build_parser().parse_args(['--goal', 'inspect', '--workspace', '.', '--env-file', str(missing), '--no-run'])
            with pytest.raises(SystemExit):
                run_coding_agent._load_env(args)

    def test_audit_counts_are_scoped_to_launched_process(self) -> None:
        records = [SimpleNamespace(actor='pid_current', action='llm.request'), SimpleNamespace(actor='pid_other', action='llm.action_repair_requested'), SimpleNamespace(actor='pid_current', action='llm.action_repair_requested')]
        counts = run_coding_agent._audit_counts_for_process(records, 'pid_current')
        assert counts['audit_records'] == 2
        assert counts['audit_records_total'] == 3
        assert counts['llm_repair_attempts'] == 1
