from PyInstaller.utils.hooks import collect_submodules, copy_metadata

datas = copy_metadata("keyring")
hiddenimports = collect_submodules("keyring.backends")

# The MCP credential broker verifies the exact keyring backend source bytes and
# distribution-owned source path before trusting an OS keychain.  Keeping this
# package as external source is part of that production security contract.
module_collection_mode = {"keyring": "py"}
