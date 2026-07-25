from __future__ import annotations
import pytest
import asyncio
from pathlib import Path
from uuid import uuid4
from scripts.ask_file_then_show import run_file_viewer

class TestAskFileThenShowScript:

    def test_script_asks_for_file_and_outputs_content(self) -> None:
        relative = f'agent_outputs/ask_file_then_show_{uuid4().hex}.txt'
        target = Path(relative)
        content = 'human selected this file\n'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding='utf-8')
        report = asyncio.run(run_file_viewer(auto_answer=relative, max_bytes=1024, max_quanta=10, echo=False))
        assert report['process_status'] == 'exited'
        assert report['selected_path'] == relative
        assert report['displayed']
        assert report['error'] is None
        assert content.strip() in report['outputs'][-1]
        assert report['actions'] == [
            'discover_skills',
            'activate_skill',
            None,
            'ask_human',
            'discover_skills',
            'activate_skill',
            'read_text_file',
            'human_output',
            'process_exit',
            'process_exit',
        ]
