from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

repo_root = Path(SPECPATH).resolve().parents[1]
entrypoint = repo_root / "scripts" / "desktop_gui_server_entry.py"
hook_root = repo_root / "desktop" / "pyinstaller" / "hooks"

datas = collect_data_files("agent_libos")
datas.append((str(repo_root / "agent_libos" / "modules" / "core.py"), "agent_libos/modules"))
for distribution in ("agent-libos", "keyring", "mcp"):
    datas += copy_metadata(distribution)

hiddenimports = []
for package in (
    "agent_libos",
    "httpcore2",
    "keyring.backends",
    "opentelemetry",
):
    hiddenimports += collect_submodules(package)

# The Python MCP SDK's optional developer CLI imports typer and rich, which are
# intentionally outside Agent libOS' desktop runtime dependency closure.  The
# governed client, transport, shared protocol, and type modules are frozen; the
# unrelated `mcp dev`/`mcp run` commands are not.
hiddenimports += collect_submodules(
    "mcp",
    filter=lambda name: not name.startswith("mcp.cli"),
)

# httpx2 exposes an optional WebSocket module backed by wsproto.  Agent libOS'
# exact MCP transports are stdio and Streamable HTTP, so collecting that
# uninstalled extra would make the frozen build depend on an unused feature.
hiddenimports += collect_submodules(
    "httpx2",
    filter=lambda name: name != "httpx2.websockets",
)

analysis = Analysis(
    [str(entrypoint)],
    pathex=[str(repo_root)],
    binaries=[],
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[str(hook_root)],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "PIL",
        "matplotlib",
        "numpy",
        "psycopg",
        "pytest",
        "tkinter",
    ],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="agent-libos-gui-server",
    debug=False,
    bootloader_ignore_signals=True,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="agent-libos-gui-server",
)
