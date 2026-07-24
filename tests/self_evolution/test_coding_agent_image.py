from __future__ import annotations

from agent_libos.config import DEFAULT_CONFIG
from agent_libos.images.base_agent import DEFAULT_IMAGES
from agent_libos.llm.prompt import build_system_prompt


class TestCodingAgentImage:

    def test_coding_agent_prompt_guides_practical_tool_use(self) -> None:
        image = DEFAULT_IMAGES['coding-agent:v0']
        prompt = build_system_prompt(image)
        required_phrases = [
            'practical coding agent',
            'Scale the size',
            'Adaptive operating loop',
            'Never claim that tests',
            'least-privilege permission',
            'Do not over-decompose',
            'AGENTS-style instructions',
            'Tests are evidence',
            'durable acceptance ledger',
            'exit gate',
            'A visible tool schema is not evidence',
            'process_exit',
        ]
        for phrase in required_phrases:
            assert phrase in prompt
        assert 'activate_tool_group' not in prompt
        assert image.default_skills == [
            'agent-libos-skill-navigation',
            'agent-libos-authority-basics',
            'agent-libos-human-collaboration',
            'agent-libos-workspace-navigation',
        ]

    def test_coding_agent_tool_table_covers_repository_workflow(self) -> None:
        image = DEFAULT_IMAGES['coding-agent:v0']
        tools = set(image.default_tools)
        assert {'activate_skill', 'discover_skills', 'read_skill_resource', 'unload_skill', 'read_directory', 'read_text_file', 'write_text_file', 'write_directory', 'delete_file', 'delete_directory', 'create_memory_object', 'create_memory_namespace', 'read_memory_object', 'append_memory_object', 'list_memory_namespace', 'create_object_from_file', 'write_object_to_file', 'fork_child_process', 'spawn_child_process', 'exec_process', 'wait_child_process', 'list_child_processes', 'merge_child_memory', 'signal_child_process', 'get_working_directory', 'set_working_directory', 'request_permission', 'load_image_package', 'ask_human', 'human_output', 'get_current_time', 'sleep', 'parse_pytest_log', 'propose_jit_tool', 'validate_jit_tool', 'register_jit_tool'}.issubset(tools)
        assert {
            'create_checkpoint',
            'list_checkpoints',
            'inspect_checkpoint',
            'diff_checkpoint',
            'restore_checkpoint',
            'fork_checkpoint',
        }.issubset(tools)
        assert {'activate_tool_group', 'discover_tool_groups'}.isdisjoint(tools)

    def test_coding_agent_defaults_to_read_only_workspace_authority(self) -> None:
        image = DEFAULT_IMAGES['coding-agent:v0']
        capabilities = image.required_capabilities
        assert {'resource': 'human:owner', 'rights': ['write']} in capabilities
        assert {'resource': 'filesystem:workspace:*', 'rights': ['read']} in capabilities
        assert not any(('write' in spec.get('rights', []) for spec in capabilities if spec['resource'].startswith('filesystem:')))
        assert not any(('delete' in spec.get('rights', []) for spec in capabilities if spec['resource'].startswith('filesystem:')))

    def test_builtin_agent_prompts_are_structured_and_within_registry_limits(self) -> None:
        expectations = {
            'base-agent:v0': [
                'Role:',
                'Instruction hierarchy:',
                'Decision loop:',
                'Object Memory',
                'least-privilege permission',
                'process_exit',
            ],
            'coding-agent:v0': [
                'Success criteria:',
                'Source of truth and security:',
                'Adaptive operating loop:',
                'Verification ladder:',
                'AGENTS-style instructions',
                'Tests are evidence',
                'A visible tool schema is not evidence',
            ],
            'toolmaker-agent:v0': [
                'toolmaker image',
                'Operating contract:',
                'agent-libos-jit-tool-authoring Skill',
                'existing governed tool',
                'preserve every primitive Capability',
                'human_output',
                'process_exit',
            ],
            'review-agent:v0': [
                'Review discipline:',
                'Prompt-injection and authority checklist:',
                'concrete, actionable findings',
                'no actionable',
                'authority escalation',
                'read-only',
                'A tool being visible is not proof',
                'Findings first',
                'Never',
            ],
            'context-compressor:v0': [
                'context-compressor image',
                'JSON object payload',
                'Do not reproduce credentials',
                'cumulative state from earlier chunks',
                'target_tokens',
                'exactly these keys',
            ],
        }

        for image_id, phrases in expectations.items():
            prompt = build_system_prompt(DEFAULT_IMAGES[image_id])
            assert len(DEFAULT_IMAGES[image_id].system_prompt) <= DEFAULT_CONFIG.image.prompt_max_chars
            for phrase in phrases:
                assert phrase in prompt
