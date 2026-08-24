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
  workspaceDirectory: string;
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
    runtimeDirectory,
    workspaceDirectory: path.join(userDataPath, "workspace")
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
  ensurePrivateDirectory(selectedPath, "Runtime data", platform);
}

export function ensurePrivateWorkspaceDirectory(selectedPath: string, platform: NodeJS.Platform = process.platform): void {
  ensurePrivateDirectory(selectedPath, "Runtime workspace", platform);
}

export function assertDatabaseOutsideWorkspace(
  selectedDatabase: string,
  workspaceDirectory: string,
  platform: NodeJS.Platform = process.platform
): void {
  requireAbsoluteDirectoryRoot(workspaceDirectory, "Runtime workspace");
  const databasePath = persistentSqlitePath(selectedDatabase, workspaceDirectory);
  if (databasePath === null) return;
  rejectMutablePathComponents(databasePath, platform);
  const canonicalWorkspace = canonicalPath(workspaceDirectory);
  const canonicalDatabase = canonicalPath(databasePath);
  if (pathContains(canonicalWorkspace, canonicalDatabase)) {
    throw new Error("Selected database must be outside the Runtime workspace.");
  }
}

export function developmentRuntimeArguments(llmProfilesFile: string, selectedDatabase?: string): string[] {
  return [
    "--db",
    selectedDatabase ?? "user",
    "--port",
    "0",
    "--llm-profiles-file",
    llmProfilesFile
  ];
}

function ensurePrivateDirectory(selectedPath: string, label: string, platform: NodeJS.Platform): void {
  requireAbsoluteDirectoryRoot(selectedPath, label);
  fs.mkdirSync(selectedPath, { recursive: true, mode: 0o700 });
  const selected = fs.lstatSync(selectedPath);
  if (!selected.isDirectory() || selected.isSymbolicLink()) {
    throw new Error(`Packaged ${label} path must be a real directory.`);
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

function canonicalPath(selectedPath: string): string {
  let existing = path.resolve(selectedPath);
  const missingComponents: string[] = [];
  while (!fs.existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) {
      throw new Error("Selected database path cannot be resolved safely.");
    }
    missingComponents.unshift(path.basename(existing));
    existing = parent;
  }
  return path.join(fs.realpathSync.native(existing), ...missingComponents);
}

function pathContains(parent: string, selected: string): boolean {
  const parentComponents = pathComponents(parent);
  const selectedComponents = pathComponents(selected);
  return parentComponents.length <= selectedComponents.length
    && parentComponents.every((component, index) => component === selectedComponents[index]);
}

function pathComponents(selectedPath: string): string[] {
  const parsed = path.parse(selectedPath);
  return [parsed.root, ...selectedPath.slice(parsed.root.length).split(path.sep).filter(Boolean)];
}

function persistentSqlitePath(selectedDatabase: string, workspaceDirectory: string): string | null {
  if (selectedDatabase === "user" || selectedDatabase === "local" || selectedDatabase === ":memory:") {
    return null;
  }
  const scheme = /^([A-Za-z][A-Za-z0-9+.-]*):/.exec(selectedDatabase)?.[1].toLowerCase();
  if (scheme === "postgres" || scheme === "postgresql") {
    if (!selectedDatabase.includes("://")) throw new Error("Unsupported Runtime store target.");
    return null;
  }
  if (scheme === "sqlite") {
    const sqlitePath = sqliteUriPath(selectedDatabase);
    if (sqlitePath === null) return null;
    return path.isAbsolute(sqlitePath) ? sqlitePath : path.resolve(workspaceDirectory, sqlitePath);
  }
  if (selectedDatabase.includes("://")) throw new Error("Unsupported Runtime store target.");
  if (/(?:^|\s)(?:dbname|host|hostaddr|options|password|port|service|sslmode|target_session_attrs|user)\s*=/i.test(selectedDatabase)) {
    throw new Error("Unsupported Runtime store target.");
  }
  return path.isAbsolute(selectedDatabase)
    ? selectedDatabase
    : path.resolve(workspaceDirectory, selectedDatabase);
}

function sqliteUriPath(selectedDatabase: string): string | null {
  const schemeEnd = selectedDatabase.indexOf(":") + 1;
  let remainder = selectedDatabase.slice(schemeEnd);
  let authority = "";
  let rawPath = "";
  if (remainder.startsWith("//")) {
    remainder = remainder.slice(2);
    const suffixIndex = firstIndex(remainder, "?", "#");
    const authorityAndPath = suffixIndex === -1 ? remainder : remainder.slice(0, suffixIndex);
    const pathIndex = authorityAndPath.indexOf("/");
    if (pathIndex !== -1) {
      authority = authorityAndPath.slice(0, pathIndex);
      rawPath = authorityAndPath.slice(pathIndex);
    }
  } else {
    const suffixIndex = firstIndex(remainder, "?", "#");
    rawPath = suffixIndex === -1 ? remainder : remainder.slice(0, suffixIndex);
  }
  if (!rawPath) return null;
  let decodedPath: string;
  try {
    decodedPath = decodeURIComponent(rawPath);
  } catch (error) {
    throw new Error("Unsupported Runtime store target.", { cause: error });
  }
  if (authority) return `//${authority}${decodedPath}`;
  if (decodedPath.startsWith("//")) decodedPath = `/${decodedPath.replace(/^\/+/, "")}`;
  if (decodedPath.startsWith("/") && decodedPath.length > 2 && decodedPath[2] === ":") {
    return decodedPath.slice(1);
  }
  return decodedPath;
}

function firstIndex(value: string, ...needles: string[]): number {
  const selected = needles.map((needle) => value.indexOf(needle)).filter((index) => index !== -1);
  return selected.length === 0 ? -1 : Math.min(...selected);
}

function rejectMutablePathComponents(selectedPath: string, platform: NodeJS.Platform): void {
  const absolute = path.resolve(selectedPath);
  const parsed = path.parse(absolute);
  let current = parsed.root;
  for (const component of absolute.slice(parsed.root.length).split(path.sep).filter(Boolean)) {
    current = path.join(current, component);
    let metadata: fs.Stats;
    try {
      metadata = fs.lstatSync(current);
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw new Error("Selected database path cannot be resolved safely.", { cause: error });
    }
    if (metadata.isSymbolicLink() || (platform === "win32" && isWindowsReparsePoint(current, metadata))) {
      throw new Error("Selected database path must not contain a symlink or reparse point.");
    }
  }
}

function isWindowsReparsePoint(selectedPath: string, metadata: fs.Stats): boolean {
  const attributes = (metadata as fs.Stats & { fileAttributes?: number; st_file_attributes?: number });
  if (((attributes.fileAttributes ?? attributes.st_file_attributes ?? 0) & 0x400) !== 0) return true;
  try {
    const followed = fs.statSync(selectedPath);
    return followed.dev !== metadata.dev || followed.ino !== metadata.ino;
  } catch (error) {
    throw new Error("Selected database path cannot be resolved safely.", { cause: error });
  }
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
