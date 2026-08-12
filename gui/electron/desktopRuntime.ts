import * as fs from "node:fs";
import * as path from "node:path";
import { runtimeServerEnv } from "./env.js";

export type RuntimeServerCommand = {
  command: string;
  args: string[];
};

export type PackagedRuntimeLayout = {
  backendExecutable: string;
  configFile: string;
  databaseFile: string;
  denoBinDirectory: string;
  llmProfilesFile: string;
  rendererRoot: string;
  runtimeDirectory: string;
};

export const packagedBackendDirectoryName = "backend";
export const packagedDenoDirectoryName = "bin";
export const packagedRendererDirectoryName = "renderer";

export function packagedRuntimeLayout(
  resourcesPath: string,
  userDataPath: string,
  platform: NodeJS.Platform = process.platform
): PackagedRuntimeLayout {
  requireAbsoluteDirectoryRoot(resourcesPath, "Electron resources");
  requireAbsoluteDirectoryRoot(userDataPath, "Electron user data");
  const runtimeDirectory = path.join(userDataPath, "runtime");
  return {
    backendExecutable: path.join(
      resourcesPath,
      packagedBackendDirectoryName,
      platform === "win32" ? "agent-libos-gui-server.exe" : "agent-libos-gui-server"
    ),
    configFile: path.join(userDataPath, "config.yaml"),
    databaseFile: path.join(runtimeDirectory, "agent-libos.sqlite"),
    denoBinDirectory: path.join(resourcesPath, packagedDenoDirectoryName),
    llmProfilesFile: path.join(userDataPath, "llm-profiles.json"),
    rendererRoot: path.join(resourcesPath, packagedRendererDirectoryName),
    runtimeDirectory
  };
}

export function resolveRuntimeServerCommand({
  packaged,
  platform = process.platform,
  repoRoot,
  resourcesPath,
  userDataPath,
  explicit = process.env.AGENT_LIBOS_GUI_SERVER_BIN
}: {
  packaged: boolean;
  platform?: NodeJS.Platform;
  repoRoot: string;
  resourcesPath: string;
  userDataPath: string;
  explicit?: string;
}): RuntimeServerCommand {
  if (packaged) {
    const command = packagedRuntimeLayout(resourcesPath, userDataPath, platform).backendExecutable;
    requireRegularExecutable(command, platform);
    return { command, args: [] };
  }
  if (explicit && explicit.trim()) {
    return { command: explicit.trim(), args: [] };
  }
  const venvScript = platform === "win32"
    ? path.join(repoRoot, ".venv", "Scripts", "agent-libos-gui-server.exe")
    : path.join(repoRoot, ".venv", "bin", "agent-libos-gui-server");
  if (fs.existsSync(venvScript)) {
    return { command: venvScript, args: [] };
  }
  return { command: "uv", args: ["run", "agent-libos-gui-server"] };
}

export function runtimeChildEnvironment({
  packaged,
  platform = process.platform,
  repoRoot,
  resourcesPath,
  userDataPath,
  baseEnv = process.env
}: {
  packaged: boolean;
  platform?: NodeJS.Platform;
  repoRoot: string;
  resourcesPath: string;
  userDataPath: string;
  baseEnv?: NodeJS.ProcessEnv;
}): NodeJS.ProcessEnv {
  if (!packaged) return runtimeServerEnv(repoRoot, baseEnv);
  const selected = { ...baseEnv };
  const layout = packagedRuntimeLayout(resourcesPath, userDataPath, platform);
  const pathKey = environmentPathKey(selected, platform);
  const inheritedPath = selected[pathKey];
  selected[pathKey] = inheritedPath
    ? `${layout.denoBinDirectory}${path.delimiter}${inheritedPath}`
    : layout.denoBinDirectory;
  selected.DENO_NO_UPDATE_CHECK = "1";
  return selected;
}

export function ensurePrivateRuntimeDirectory(selectedPath: string, platform: NodeJS.Platform = process.platform): void {
  requireAbsoluteDirectoryRoot(selectedPath, "Runtime data");
  fs.mkdirSync(selectedPath, { recursive: true, mode: 0o700 });
  const selected = fs.lstatSync(selectedPath);
  if (!selected.isDirectory() || selected.isSymbolicLink()) {
    throw new Error("Packaged Runtime data path must be a real directory.");
  }
  if (platform !== "win32") fs.chmodSync(selectedPath, 0o700);
}

export function packagedRuntimeArguments(layout: PackagedRuntimeLayout, selectedDatabase?: string): string[] {
  const args = [
    "--db",
    selectedDatabase ?? layout.databaseFile,
    "--port",
    "0",
    "--llm-profiles-file",
    layout.llmProfilesFile
  ];
  if (fs.existsSync(layout.configFile)) {
    const selected = fs.lstatSync(layout.configFile);
    if (!selected.isFile() || selected.isSymbolicLink()) {
      throw new Error("Packaged Runtime config must be a regular file.");
    }
    args.push("--config", layout.configFile);
  }
  return args;
}

function environmentPathKey(env: NodeJS.ProcessEnv, platform: NodeJS.Platform): string {
  if (platform !== "win32") return "PATH";
  return Object.keys(env).find((key) => key.toLowerCase() === "path") ?? "Path";
}

function requireAbsoluteDirectoryRoot(value: string, label: string): void {
  if (!value || !path.isAbsolute(value)) throw new Error(`${label} path must be absolute.`);
}

function requireRegularExecutable(command: string, platform: NodeJS.Platform): void {
  let selected: fs.Stats;
  try {
    selected = fs.lstatSync(command);
  } catch (error) {
    throw new Error(`Packaged GUI server is unavailable at ${command}.`, { cause: error });
  }
  if (!selected.isFile() || selected.isSymbolicLink()) {
    throw new Error("Packaged GUI server must be a regular file.");
  }
  if (platform !== "win32" && (selected.mode & 0o111) === 0) {
    throw new Error("Packaged GUI server is not executable.");
  }
}
