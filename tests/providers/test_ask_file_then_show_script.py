from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from agent_libos.config import DEFAULT_CONFIG
import scripts.ask_file_then_show as ask_file_then_show
from scripts.ask_file_then_show import run_file_viewer
from scripts.runtime_assembly import aopen_runtime


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

    def test_script_collects_terminal_result_at_exact_quantum_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = replace(
            DEFAULT_CONFIG,
            scheduler=replace(DEFAULT_CONFIG.scheduler, drain_window_s=0.0),
        )

        async def open_runtime_with_zero_drain(target: str):
            return await aopen_runtime(target, config=config)

        monkeypatch.setattr(
            ask_file_then_show,
            'aopen_runtime',
            open_runtime_with_zero_drain,
        )
        relative = f'agent_outputs/ask_file_then_show_{uuid4().hex}.txt'
        target = Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('exact boundary\n', encoding='utf-8')

        report = asyncio.run(
            ask_file_then_show.run_file_viewer(
                auto_answer=relative,
                max_bytes=1024,
                max_quanta=10,
                echo=False,
            )
        )

        assert report['process_status'] == 'exited'
        assert len(report['actions']) == 10
        assert report['actions'][-2:] == ['process_exit', 'process_exit']
