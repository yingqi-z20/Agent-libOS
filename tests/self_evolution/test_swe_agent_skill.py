from __future__ import annotations
import pytest
from pathlib import Path
from agent_libos import Runtime
from agent_libos.models import CapabilityRight

class TestSWEAgentSkill:

    @pytest.mark.real_deno
    def test_swe_agent_skill_registers_and_loads_without_granting_resource_authority(self) -> None:
        runtime = Runtime.open('local')
        try:
            registered = runtime.register_skill_from_path(Path('skills/swe-agent'), actor='cli', source_type='workspace')
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
        finally:
            runtime.close()
