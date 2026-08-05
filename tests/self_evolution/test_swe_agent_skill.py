from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from agent_libos import Runtime
from agent_libos.models import CapabilityRight, ProcessStatus
from agent_libos.substrate import LocalResourceProviderSubstrate
from agent_libos.tools.sandbox import DenoTypescriptSandbox


PACKAGE_ROOT = Path('skills/swe-agent')


class TestSWEAgentSkill:

    def test_shipped_skill_uses_scalar_allowed_tools_frontmatter(self) -> None:
        skill_md = PACKAGE_ROOT.joinpath('SKILL.md').read_text(encoding='utf-8')
        allowed_tools_line = next(
            line for line in skill_md.splitlines() if line.startswith('allowed-tools:')
        )

        assert allowed_tools_line.startswith('allowed-tools: read_directory ')
        assert ' process_exit' in allowed_tools_line

    def test_shipped_skill_pins_current_jit_extension_compatibility(self) -> None:
        skill_md = PACKAGE_ROOT.joinpath('SKILL.md').read_text(encoding='utf-8')

        assert 'compatibility: agent-libos==1.4.0\n' in skill_md

    def test_command_wrappers_reserve_outer_sandbox_cleanup_windows(self) -> None:
        specs = {
            item['name']: item
            for item in json.loads(
                PACKAGE_ROOT.joinpath(
                    'references/agent-libos/jit-tools.json'
                ).read_text(encoding='utf-8')
            )
        }

        assert specs['swe_grep']['timeout_s'] == 15
        assert specs['swe_view']['input_schema']['properties']['limit'] == {
            'type': 'integer',
            'minimum': 1,
            'maximum': 1024,
            'default': 200,
        }
        assert specs['swe_grep']['input_schema']['properties']['timeout_s'] == {
            'type': 'number',
            'exclusiveMinimum': 0,
            'maximum': 10,
            'default': 10,
        }
        assert specs['swe_run']['timeout_s'] == 60
        assert specs['swe_run']['input_schema']['properties']['timeout_s'] == {
            'type': 'number',
            'exclusiveMinimum': 0,
            'maximum': 55,
            'default': 55,
        }

    def test_jit_manifest_has_closed_executable_contracts_and_self_tests(self) -> None:
        specs = {
            item['name']: item
            for item in json.loads(
                PACKAGE_ROOT.joinpath(
                    'references/agent-libos/jit-tools.json'
                ).read_text(encoding='utf-8')
            )
        }

        assert set(specs) == {
            'swe_view',
            'swe_grep',
            'swe_edit',
            'swe_run',
            'swe_submit',
        }
        for name, spec in specs.items():
            assert spec['input_schema']['additionalProperties'] is False
            assert spec['output_schema'] != {'type': 'object'}
            assert spec['tests'], (
                f'{name} must ship at least one behavioral self-test'
            )
            input_validator = Draft202012Validator(spec['input_schema'])
            output_validator = Draft202012Validator(spec['output_schema'])
            for case in spec['tests']:
                assert 'expected' in case
                input_validator.validate(case.get('args', {}))
                output_validator.validate(case['expected'])
                with pytest.raises(JsonSchemaValidationError):
                    input_validator.validate(
                        {**case.get('args', {}), 'unexpected': True}
                    )
                with pytest.raises(JsonSchemaValidationError):
                    output_validator.validate(
                        {**case['expected'], 'unexpected': True}
                    )

        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(specs['swe_grep']['input_schema']).validate({})
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(specs['swe_run']['input_schema']).validate(
                {'argv': []}
            )
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(specs['swe_submit']['input_schema']).validate(
                {'summary': 'done', 'tests': []}
            )
        with pytest.raises(JsonSchemaValidationError):
            Draft202012Validator(specs['swe_edit']['input_schema']).validate(
                {
                    'path': 'document.txt',
                    'new_text': 'replacement',
                    'old_text': 'old',
                    'start_line': 1,
                    'end_line': 1,
                }
            )

    @pytest.mark.parametrize(
        ('source', 'expected_error'),
        [
            (
                '/// <reference path="/tmp/host-types.d.ts" />\n'
                'export function run(args, libos) { return {}; }',
                'TypeScript dependency directives are not allowed',
            ),
            (
                'import host = require("file:///tmp/host.ts");\n'
                'export function run(args, libos) { return {}; }',
                'imports are not allowed in JIT tool source: file:///tmp/host.ts',
            ),
        ],
    )
    def test_static_check_rejects_non_ecmascript_dependency_forms(
        self,
        source: str,
        expected_error: str,
    ) -> None:
        validation = DenoTypescriptSandbox(deno_executable='deno').static_check(
            source
        )

        assert not validation.ok
        assert expected_error in validation.errors

    @pytest.mark.real_deno
    def test_swe_agent_skill_registers_and_loads_without_granting_resource_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            registered = runtime.register_skill_from_path(PACKAGE_ROOT, actor='cli', source_type='workspace')
            pid = runtime.process.spawn(image='base-agent:v0', goal='fix issue like SWE-Agent')
            runtime.capability.grant(pid, 'skill:swe-agent', [CapabilityRight.EXECUTE], issued_by='test')
            loaded = runtime.skills.activate_skill(pid, 'swe-agent', actor=pid)
            process = runtime.process.get(pid)
            assert registered['skill_id'] == 'swe-agent'
            for name in ['swe_view', 'swe_grep', 'swe_edit', 'swe_run', 'swe_submit']:
                assert name in loaded['jit_tool_ids']
                assert name in process.tool_table
            assert 'run_shell_command' in process.tool_table
            assert not runtime.capability.check(pid, 'filesystem:workspace:*', CapabilityRight.READ)
            assert not runtime.capability.check(pid, 'filesystem:workspace:*', CapabilityRight.WRITE)
            assert not runtime.capability.check(pid, 'shell:*', CapabilityRight.EXECUTE)

            expected_payload = {
                'status': 'resolved',
                'summary': 'fixed timeout and submission contracts',
                'tests': ['focused tests passed'],
                'residual_risks': [],
            }
            prepared = runtime.tools.call(
                pid,
                'swe_submit',
                {
                    'summary': expected_payload['summary'],
                    'tests': expected_payload['tests'],
                    'residual_risks': expected_payload['residual_risks'],
                },
            )
            process = runtime.process.get(pid)

            assert prepared.ok, prepared.error
            assert prepared.payload == {
                'prepared': True,
                'payload': expected_payload,
                'next_action': 'process_exit',
                'process_exit_args': {'payload': expected_payload},
            }
            assert process.status == ProcessStatus.RUNNABLE
            assert process.outcome is None

            exited = runtime.tools.call(
                pid,
                'process_exit',
                prepared.payload['process_exit_args'],
            )
            process = runtime.process.get(pid)

            assert exited.ok, exited.error
            assert exited.payload['status'] == 'exited'
            assert process.status == ProcessStatus.EXITED
            assert process.outcome is not None
            assert process.outcome.result_oid is not None
            assert runtime.store.get_object(process.outcome.result_oid).payload == expected_payload
        finally:
            runtime.close()

    @pytest.mark.real_deno
    def test_swe_submit_requires_coding_image_guarded_process_exit(self) -> None:
        runtime = Runtime.open('local')
        try:
            runtime.register_skill_from_path(
                PACKAGE_ROOT,
                actor='cli',
                source_type='workspace',
            )
            pid = runtime.process.spawn(
                image='coding-agent:v0',
                goal='prepare a result without bypassing cumulative review',
            )
            runtime.capability.grant(
                pid,
                'skill:swe-agent',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            runtime.skills.activate_skill(pid, 'swe-agent', actor=pid)
            expected_payload = {
                'status': 'resolved',
                'summary': 'ready for guarded review',
                'tests': ['focused tests passed'],
                'residual_risks': [],
            }

            prepared = runtime.tools.call(
                pid,
                'swe_submit',
                {
                    'summary': expected_payload['summary'],
                    'tests': expected_payload['tests'],
                    'residual_risks': expected_payload['residual_risks'],
                },
            )

            assert prepared.ok, prepared.error
            assert prepared.payload['process_exit_args'] == {
                'payload': expected_payload,
            }
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
            assert not any(
                record.action == 'syscall.request'
                and record.target == 'process.exit'
                for record in runtime.audit.trace(actor=pid)
            )

            guarded = runtime.tools.call(
                pid,
                'process_exit',
                prepared.payload['process_exit_args'],
            )

            assert guarded.ok, guarded.error
            assert guarded.payload['status'] == 'completion_review_required'
            assert runtime.process.get(pid).status == ProcessStatus.RUNNABLE
        finally:
            runtime.close()

    @pytest.mark.real_deno
    @pytest.mark.parametrize(
        ('tool_name', 'args', 'syscall_args', 'syscall_result', 'expected'),
        [
            (
                'swe_run',
                {'argv': ['pytest', '-q'], 'timeout_s': 999},
                {'argv': ['pytest', '-q'], 'timeout_s': 55},
                {
                    'returncode': 0,
                    'stdout': '',
                    'stderr': '',
                    'stdout_truncated': False,
                    'stderr_truncated': False,
                },
                {
                    'argv': ['pytest', '-q'],
                    'returncode': 0,
                    'stdout': '',
                    'stderr': '',
                    'stdout_truncated': False,
                    'stderr_truncated': False,
                    'message': 'Your command ran successfully and did not produce any output.',
                },
            ),
            (
                'swe_grep',
                {'pattern': 'needle', 'path': 'src', 'timeout_s': 999},
                {
                    'argv': [
                        'rg',
                        '-n',
                        '--null',
                        '--with-filename',
                        '--hidden',
                        '--glob',
                        '!.git/*',
                        '-F',
                        '--',
                        'needle',
                        'src',
                    ],
                    'timeout_s': 10,
                },
                {
                    'returncode': 0,
                    'stdout': 'src/tool.py\x004:needle\n',
                    'stderr': '',
                    'stdout_truncated': False,
                    'stderr_truncated': False,
                },
                {
                    'argv': [
                        'rg',
                        '-n',
                        '--null',
                        '--with-filename',
                        '--hidden',
                        '--glob',
                        '!.git/*',
                        '-F',
                        '--',
                        'needle',
                        'src',
                    ],
                    'returncode': 0,
                    'files': ['src/tool.py'],
                    'matches': ['src/tool.py:4:needle'],
                    'omitted_matches': 0,
                    'observed_omitted_matches': 0,
                    'matches_incomplete': False,
                    'stdout_truncated': False,
                    'stderr_truncated': False,
                    'output_incomplete': False,
                    'stderr': '',
                    'message': '',
                },
            ),
        ],
    )
    def test_command_wrappers_clamp_inner_timeout_below_outer_sandbox(
        self,
        tool_name: str,
        args: dict[str, object],
        syscall_args: dict[str, object],
        syscall_result: dict[str, object],
        expected: dict[str, object],
    ) -> None:
        source = PACKAGE_ROOT.joinpath('scripts', f'{tool_name}.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': args,
                    'syscalls': [
                        {
                            'name': 'shell.run',
                            'args': syscall_args,
                            'result': syscall_result,
                        }
                    ],
                    'expected': expected,
                }
            ],
        )

        assert validation.ok, validation.errors

    @pytest.mark.real_deno
    def test_swe_grep_bounds_all_match_projections_and_marks_truncated_capture(
        self,
    ) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_grep.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')
        argv = [
            'rg',
            '-n',
            '--null',
            '--with-filename',
            '--hidden',
            '--glob',
            '!.git/*',
            '-F',
            '--',
            'needle',
            'src',
        ]

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': {
                        'pattern': 'needle',
                        'path': 'src',
                        'max_results': 2,
                    },
                    'syscalls': [
                        {
                            'name': 'shell.run',
                            'args': {'argv': argv, 'timeout_s': 10},
                            'result': {
                                'returncode': 0,
                                'stdout': (
                                    'src/a.py\x001:needle\n'
                                    'src/b.py\x002:needle\n'
                                    'src/c.py\x003:needle\n'
                                ),
                                'stderr': '',
                                'stdout_truncated': True,
                                'stderr_truncated': False,
                            },
                        }
                    ],
                    'expected': {
                        'argv': argv,
                        'returncode': 0,
                        'files': ['src/a.py', 'src/b.py'],
                        'matches': [
                            'src/a.py:1:needle',
                            'src/b.py:2:needle',
                        ],
                        'omitted_matches': None,
                        'observed_omitted_matches': 1,
                        'matches_incomplete': True,
                        'stdout_truncated': True,
                        'stderr_truncated': False,
                        'output_incomplete': True,
                        'stderr': '',
                        'message': '',
                    },
                }
            ],
        )

        assert validation.ok, validation.errors

    @pytest.mark.real_deno
    def test_swe_grep_nul_framing_preserves_single_colon_path_and_drops_partial_tail(
        self,
    ) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_grep.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')
        argv = [
            'rg',
            '-n',
            '--null',
            '--with-filename',
            '--hidden',
            '--glob',
            '!.git/*',
            '-F',
            '--',
            'needle',
            'src/single:file.py',
        ]
        base = {
            'argv': argv,
            'returncode': 0,
            'stderr_truncated': False,
            'stderr': '',
            'message': '',
        }

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': {
                        'pattern': 'needle',
                        'path': 'src/single:file.py',
                    },
                    'syscalls': [
                        {
                            'name': 'shell.run',
                            'args': {'argv': argv, 'timeout_s': 10},
                            'result': {
                                'returncode': 0,
                                'stdout': 'src/single:file.py\x001:needle\n',
                                'stderr': '',
                                'stdout_truncated': False,
                                'stderr_truncated': False,
                            },
                        }
                    ],
                    'expected': {
                        **base,
                        'files': ['src/single:file.py'],
                        'matches': ['src/single:file.py:1:needle'],
                        'omitted_matches': 0,
                        'observed_omitted_matches': 0,
                        'matches_incomplete': False,
                        'stdout_truncated': False,
                        'output_incomplete': False,
                    },
                },
                {
                    'args': {
                        'pattern': 'needle',
                        'path': 'src/single:file.py',
                    },
                    'syscalls': [
                        {
                            'name': 'shell.run',
                            'args': {'argv': argv, 'timeout_s': 10},
                            'result': {
                                'returncode': 0,
                                'stdout': (
                                    'src/single:file.py\x001:needle\n'
                                    'src/partial:file.py'
                                ),
                                'stderr': '',
                                'stdout_truncated': True,
                                'stderr_truncated': False,
                            },
                        }
                    ],
                    'expected': {
                        **base,
                        'files': ['src/single:file.py'],
                        'matches': ['src/single:file.py:1:needle'],
                        'omitted_matches': None,
                        'observed_omitted_matches': 0,
                        'matches_incomplete': True,
                        'stdout_truncated': True,
                        'output_incomplete': True,
                    },
                },
            ],
        )

        assert validation.ok, validation.errors

    @pytest.mark.real_deno
    def test_swe_run_does_not_describe_empty_nonzero_exit_as_success(self) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_run.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': {'argv': ['false']},
                    'syscalls': [
                        {
                            'name': 'shell.run',
                            'args': {'argv': ['false'], 'timeout_s': 55},
                            'result': {
                                'argv': ['false'],
                                'returncode': 2,
                                'stdout': '',
                                'stderr': '',
                                'stdout_truncated': False,
                                'stderr_truncated': False,
                            },
                        }
                    ],
                    'expected': {
                        'argv': ['false'],
                        'returncode': 2,
                        'stdout': '',
                        'stderr': '',
                        'stdout_truncated': False,
                        'stderr_truncated': False,
                        'message': (
                            'Your command exited with return code 2 and did not '
                            'produce any output.'
                        ),
                    },
                }
            ],
        )

        assert validation.ok, validation.errors

    @pytest.mark.real_deno
    def test_swe_submit_prepares_guarded_process_exit_payload_without_syscall(self) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_submit.ts').read_text(
            encoding='utf-8'
        )
        payload = {
            'status': 'resolved',
            'summary': 'done',
            'tests': ['pytest -q'],
            'residual_risks': ['none'],
        }
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': payload,
                    'syscalls': [],
                    'expected': {
                        'prepared': True,
                        'payload': payload,
                        'next_action': 'process_exit',
                        'process_exit_args': {'payload': payload},
                    },
                }
            ],
        )

        assert validation.ok, validation.errors

    @pytest.mark.real_deno
    def test_empty_jit_tests_still_run_deno_syntax_check(self) -> None:
        source = (
            'export function run(args, libos) { return { ok: true }; }\n'
            'const syntaxError: string = ;\n'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        assert sandbox.static_check(source).ok
        validation = sandbox.run_tests(source, [])

        assert not validation.ok
        assert any('Deno type-check' in error for error in validation.errors)

    @pytest.mark.real_deno
    def test_empty_jit_deno_check_does_not_execute_candidate(self) -> None:
        source = (
            'throw new Error("validation must not execute candidate code");\n'
            'export function run(args, libos) { return { ok: true }; }\n'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(source, [])

        assert validation.ok, validation.errors
        assert 'deno_check=ok' in validation.logs

    @pytest.mark.real_deno
    def test_swe_edit_refuses_truncated_source_before_write(self) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_edit.ts').read_text(encoding='utf-8')
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': {'path': 'large.txt', 'old_text': 'needle', 'new_text': 'replacement'},
                    'syscalls': [
                        {
                            'name': 'filesystem.read_text',
                            'args': {'path': 'large.txt', 'max_bytes': 1048576},
                            'result': {
                                'path': 'large.txt',
                                'content': 'needle and a partial file',
                                'truncated': True,
                            },
                        }
                    ],
                    'expected': {},
                }
            ],
        )

        assert not validation.ok
        assert any('truncated' in error.lower() for error in validation.errors)

    @pytest.mark.real_deno
    def test_swe_edit_rejects_lost_update_after_concurrent_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / 'document.txt'
        target.write_text('alpha\nbeta\n', encoding='utf-8')
        runtime = Runtime.open(
            'local',
            substrate=LocalResourceProviderSubstrate(tmp_path),
        )
        try:
            runtime.register_skill_from_path(
                PACKAGE_ROOT,
                actor='cli',
                source_type='workspace',
            )
            pid = runtime.process.spawn(
                image='base-agent:v0',
                goal='edit without losing a concurrent update',
            )
            runtime.capability.grant(
                pid,
                'skill:swe-agent',
                [CapabilityRight.EXECUTE],
                issued_by='test',
            )
            runtime.filesystem.grant_path(
                pid,
                'document.txt',
                [CapabilityRight.READ, CapabilityRight.WRITE],
                issued_by='test',
            )
            runtime.skills.activate_skill(pid, 'swe-agent', actor=pid)
            original_read = runtime.filesystem.read_text

            def read_then_race(*args: object, **kwargs: object):
                result = original_read(*args, **kwargs)
                target.write_text('concurrent\nbeta\n', encoding='utf-8')
                return result

            monkeypatch.setattr(runtime.filesystem, 'read_text', read_then_race)

            edited = runtime.tools.call(
                pid,
                'swe_edit',
                {
                    'path': 'document.txt',
                    'old_text': 'alpha',
                    'new_text': 'agent',
                },
            )

            assert not edited.ok
            assert target.read_text(encoding='utf-8') == 'concurrent\nbeta\n'
        finally:
            runtime.close()

    @pytest.mark.real_deno
    @pytest.mark.parametrize(
        ('args', 'expected_error'),
        [
            (
                {
                    'path': 'small.txt',
                    'old_text': 'needle',
                    'new_text': 'replacement',
                    'occurrence': 3,
                },
                'occurrence 3 exceeds matches 2',
            ),
            (
                {
                    'path': 'small.txt',
                    'new_text': 'replacement',
                    'start_line': 4,
                    'end_line': 4,
                },
                'start_line 4 is outside 1..3',
            ),
        ],
    )
    def test_swe_edit_rejects_out_of_range_coordinates(
        self,
        args: dict[str, object],
        expected_error: str,
    ) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_edit.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': args,
                    'expected': {},
                    'syscalls': [
                        {
                            'name': 'filesystem.read_text',
                            'args': {'path': 'small.txt', 'max_bytes': 1048576},
                            'result': {
                                'path': 'small.txt',
                                'content': 'needle\nmiddle\nneedle',
                                'truncated': False,
                            },
                        }
                    ],
                }
            ],
        )

        assert not validation.ok
        assert any(
            expected_error in error for error in validation.errors
        ), validation.errors

    @pytest.mark.real_deno
    def test_swe_edit_line_range_preserves_crlf_convention(self) -> None:
        source = PACKAGE_ROOT.joinpath('scripts/swe_edit.ts').read_text(
            encoding='utf-8'
        )
        sandbox = DenoTypescriptSandbox(deno_executable='deno')
        original = 'alpha\r\nbeta\r\ngamma\r\n'
        updated = 'alpha\r\nnew\r\ncontinued\r\ngamma\r\n'

        validation = sandbox.run_tests(
            source,
            [
                {
                    'args': {
                        'path': 'windows.txt',
                        'new_text': 'new\ncontinued',
                        'start_line': 2,
                        'end_line': 2,
                    },
                    'syscalls': [
                        {
                            'name': 'filesystem.read_text',
                            'args': {
                                'path': 'windows.txt',
                                'max_bytes': 1048576,
                            },
                            'result': {
                                'path': 'windows.txt',
                                'content': original,
                                'truncated': False,
                                'content_sha256': '1' * 64,
                            },
                        },
                        {
                            'name': 'filesystem.write_text',
                            'args': {
                                'path': 'windows.txt',
                                'content': updated,
                                'overwrite': True,
                                'expected_content_sha256': '1' * 64,
                            },
                            'result': {
                                'path': 'windows.txt',
                                'bytes_written': len(updated.encode('utf-8')),
                            },
                        },
                    ],
                    'expected': {
                        'path': 'windows.txt',
                        'changed': True,
                        'edit': 'replace_lines',
                        'replacements': 1,
                        'bytes_written': len(updated.encode('utf-8')),
                    },
                }
            ],
        )

        assert validation.ok, validation.errors
