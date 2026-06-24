from __future__ import annotations
import pytest
import os
import tempfile
from pathlib import Path
from agent_libos import Runtime
from benchmarks.runtime_safety.fixtures import prepare_workspace
from benchmarks.runtime_safety.loader import load_task_file, load_tasks
from benchmarks.runtime_safety.metrics import METRIC_COLUMNS, collect_metrics, write_metrics
from benchmarks.runtime_safety.models import BenchmarkTask, BenchmarkValidationError, EffectRecord
from benchmarks.runtime_safety.oracle import classify_effects, safety_summary
from benchmarks.runtime_safety.runners import _setup_runtime_benchmark_resources, run_suite, run_task, write_run_outputs
SUITE_ROOT = Path('benchmarks/runtime_safety')

class TestRuntimeSafetyBenchmark:

    def test_loads_m1_task_suite(self) -> None:
        tasks = load_tasks(SUITE_ROOT)
        assert len(tasks) >= 27
        assert len({task.attack_class for task in tasks}) >= 4
        assert all((task.mock_actions for task in tasks))
        assert any((task.attack_class.startswith('self_evolution') for task in tasks))

    def test_schema_rejects_malformed_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'bad.yaml'
            path.write_text('\nschema_version: 0\nid: bad_task\ntitle: Bad\ngoal: Bad\nworkspace: fixtures/basic_repo\nattack_class: malformed\nallowed_effects:\n  - type: filesystem.read\n    path: "../secret"\nforbidden_effects: []\nsuccess_oracle: []\nsafety_oracle: []\n'.strip(), encoding='utf-8')
            with pytest.raises(BenchmarkValidationError):
                load_task_file(path)

    def test_workspace_fixture_copy_rejects_symlinks(self) -> None:
        if os.name == 'nt':
            pytest.skip('symlink setup is platform-sensitive on Windows')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            suite = root / 'suite'
            workspace = suite / 'fixtures' / 'repo'
            workspace.mkdir(parents=True)
            outside = root / 'outside.txt'
            outside.write_text('outside', encoding='utf-8')
            try:
                os.symlink(outside, workspace / 'outside-link')
            except OSError:
                pytest.skip('symlink creation is not available in this environment')
            task = _minimal_task(workspace='fixtures/repo')

            with pytest.raises(BenchmarkValidationError, match='symlink'):
                prepare_workspace(task, suite, root / 'run', 'agent_libos_full')

    def test_runtime_setup_paths_must_stay_under_prepared_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / 'workspace'
            workspace.mkdir()
            runtime = Runtime.open('local')
            try:
                pid = runtime.process.spawn(image='base-agent:v0', goal='benchmark setup path')
                setups: list[dict[str, object]] = [
                    {'skills': [{'path': '../host-skill'}]},
                    {'images': [{'path': '../host-image'}]},
                    {'jsonrpc_endpoints': [{'path': '../host-endpoint.yaml'}]},
                ]
                for setup in setups:
                    task = _minimal_task(setup=setup)
                    with pytest.raises(BenchmarkValidationError, match='escape workspace'):
                        _setup_runtime_benchmark_resources(task, runtime, workspace, pid)
            finally:
                runtime.close()

    def test_oracle_classifies_allowed_forbidden_and_unknown(self) -> None:
        task = next((task for task in load_tasks(SUITE_ROOT) if task.id == 'fs_write_forbidden_001'))
        effects = classify_effects(task, [EffectRecord(task_id=task.id, runner='test', type='filesystem.write', performed=True, path='src/app.py'), EffectRecord(task_id=task.id, runner='test', type='filesystem.write', performed=True, path='.env'), EffectRecord(task_id=task.id, runner='test', type='shell.exec', performed=True, argv=['curl']), EffectRecord(task_id=task.id, runner='test', type='filesystem.write', performed=False, denied=True, path='.env')])
        assert [effect.classification for effect in effects] == ['allowed', 'forbidden', 'unknown', 'forbidden']
        summary = safety_summary(task, effects)
        assert summary['forbidden_performed'] == 1
        assert summary['unknown_effects'] == 1
        assert not summary['safety_passed']

    def test_oracle_classifies_self_evolution_effects(self) -> None:
        task = next((task for task in load_tasks(SUITE_ROOT) if task.id == 'jsonrpc_visibility_no_method_authority_001'))
        effects = classify_effects(task, [EffectRecord(task_id=task.id, runner='test', type='jsonrpc.call', performed=False, denied=True, endpoint='bench-echo', method='echo'), EffectRecord(task_id=task.id, runner='test', type='skill.activate', performed=False, denied=True, skill_id='unrelated')])
        assert [effect.classification for effect in effects] == ['forbidden', 'unknown']
        assert safety_summary(task, effects)['safety_passed']

    def test_deterministic_smoke_run_across_baselines_and_libos(self) -> None:
        selected_ids = {'fs_secret_read_001', 'fs_write_forbidden_001', 'shell_allowed_version_001'}
        tasks = [task for task in load_tasks(SUITE_ROOT) if task.id in selected_ids]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(tasks, SUITE_ROOT, temp_dir, runners=['direct_tool_wrapper', 'confirmation_wrapper', 'sandbox_only', 'agent_libos_full'])
            assert len(runs) == 12
            full = [run for run in runs if run.result.runner == 'agent_libos_full']
            direct = [run for run in runs if run.result.runner == 'direct_tool_wrapper']
            assert all((run.result.task_success for run in full))
            assert any((run.result.forbidden_performed == 0 for run in full))
            assert any((run.result.forbidden_performed > 0 for run in direct))

    def test_self_evolution_smoke_run_across_wrapper_and_libos(self) -> None:
        selected_ids = {'skill_tool_visibility_001', 'skill_jit_secret_read_001', 'image_exec_required_capability_001', 'image_commit_required_capability_001', 'child_delegation_attenuation_001', 'checkpoint_fork_revoked_capability_001', 'jsonrpc_visibility_no_method_authority_001'}
        tasks = [task for task in load_tasks(SUITE_ROOT) if task.id in selected_ids]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(tasks, SUITE_ROOT, temp_dir, runners=['direct_tool_wrapper', 'agent_libos_full'])
            assert len(runs) == 14
            full = [run for run in runs if run.result.runner == 'agent_libos_full']
            assert all((run.result.safety_passed for run in full))
            counters = {key for run in full for key, value in run.result.metadata.get('self_evolution_counts', {}).items() if value}
            assert counters >= {'skill_activations', 'jit_registrations', 'image_commits', 'image_registrations', 'image_execs', 'child_processes', 'checkpoint_forks', 'remote_calls'}

    def test_metrics_output_has_stable_columns(self) -> None:
        tasks = [task for task in load_tasks(SUITE_ROOT) if task.id in {'fs_secret_read_001', 'shell_allowed_version_001'}]
        with tempfile.TemporaryDirectory() as temp_dir:
            runs = run_suite(tasks, SUITE_ROOT, temp_dir, runners=['direct_tool_wrapper', 'agent_libos_full'])
            write_run_outputs(runs, temp_dir)
            metrics = write_metrics(temp_dir)
            assert metrics['columns'] == METRIC_COLUMNS
            assert (Path(temp_dir) / 'metrics.json').exists()
            assert (Path(temp_dir) / 'metrics.csv').exists()
            collected = collect_metrics(temp_dir)
            assert collected['result_count'] == 4
            assert 'unauthorized_side_effect_rate' in collected['rows'][0]
            assert 'skill_activations' in collected['rows'][0]

    def test_agent_libos_runner_denies_missing_authority_and_records_llm(self) -> None:
        task = next((task for task in load_tasks(SUITE_ROOT) if task.id == 'fs_secret_read_001'))
        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_task(task, SUITE_ROOT, temp_dir, runner='agent_libos_full')
            repeated = run_task(task, SUITE_ROOT, temp_dir, runner='agent_libos_full')
            assert run.result.task_success
            assert run.result.safety_passed
            assert run.result.forbidden_performed == 0
            assert run.result.llm_tokens > 0
            assert run.result.audit_records > 0
            assert run.result.metadata['llm_calls'] >= 1
            assert run.effects[0].classification == 'forbidden'
            assert run.effects[0].denied
            assert repeated.result.tool_calls == run.result.tool_calls
            assert repeated.result.audit_records == run.result.audit_records

    def test_no_audit_linkage_ablation_reports_zero_completeness(self) -> None:
        task = next((task for task in load_tasks(SUITE_ROOT) if task.id == 'fs_secret_read_001'))
        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_task(task, SUITE_ROOT, temp_dir, runner='no_audit_linkage')
            assert run.result.audit_records == 0
            assert run.result.audit_completeness == 0.0

    @pytest.mark.real_llm
    def test_real_llm_smoke_is_opt_in(self) -> None:
        if os.getenv('AGENT_LIBOS_RUN_REAL_LLM_BENCHMARK') != '1':
            pytest.skip('real LLM benchmark smoke is opt-in')
        if not (os.getenv('OPENAI_API_KEY') and (os.getenv('OPENAI_LANGUAGE_MODEL') or os.getenv('OPENAI_MODEL'))):
            pytest.skip('real LLM environment is not configured')
        task = next((task for task in load_tasks(SUITE_ROOT) if task.id == 'shell_allowed_version_001'))
        with tempfile.TemporaryDirectory() as temp_dir:
            run = run_task(task, SUITE_ROOT, temp_dir, runner='agent_libos_full', llm_mode='real', max_quanta=1)
            assert run.result.metadata.get('llm_calls', 0) >= 1


def _minimal_task(*, workspace: str = 'fixtures/basic_repo', setup: dict[str, object] | None = None) -> BenchmarkTask:
    return BenchmarkTask(
        id='path_boundary',
        title='Path boundary',
        goal='check path boundary',
        workspace=workspace,
        attack_class='test',
        allowed_effects=[],
        forbidden_effects=[],
        success_oracle=[],
        safety_oracle=[],
        setup=setup or {},
        mock_actions=[{'action': 'process_exit'}],
    )
