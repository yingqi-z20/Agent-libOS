#!/usr/bin/env python3
from __future__ import annotations

from typing import Sequence

from agent_libos.storage.tool_skill_migration import cli


def main(argv: Sequence[str] | None = None) -> int:
    return cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
