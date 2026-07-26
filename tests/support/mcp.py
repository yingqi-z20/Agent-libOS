from __future__ import annotations

import json
import os
import sys


MCP_TEST_STDIO_COMMAND = sys.executable if os.name == "nt" else "python3"
MCP_TEST_STDIO_COMMAND_YAML = json.dumps(MCP_TEST_STDIO_COMMAND)
