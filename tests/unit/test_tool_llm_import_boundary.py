from __future__ import annotations

import subprocess
import sys


def test_tool_contract_and_llm_actions_import_in_fresh_interpreter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import agent_libos.tools.base; "
                "import agent_libos.llm.actions; "
                "from agent_libos.llm.openai_schema import openai_chat_tool_schema; "
                "assert openai_chat_tool_schema('echo', 'Echo.', {})"
                "['function']['name'] == 'echo'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
