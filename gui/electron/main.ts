import { app, BrowserWindow, dialog, ipcMain, shell, type OpenDialogOptions } from "electron";
import { ChildProcess, ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { databaseTargetFromRenderer } from "./database.js";
import { readImagePackageFiles } from "./imagePackage.js";

type ServerConnection = {
  url: string;
  token: string;
  db: string;
};

type RuntimeServerCommand = {
  command: string;
  args: string[];
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const smokeMode = process.env.AGENT_LIBOS_GUI_SMOKE === "1";
const smokeLogPath = process.env.AGENT_LIBOS_GUI_SMOKE_LOG;
const imageManifestMaxBytes = 1_048_576;
const allowedExternalProtocols = new Set(["http:", "https:", "mailto:"]);

if (smokeMode) {
  const smokeUserDataPath = path.join(repoRoot, "gui", ".smoke-user-data");
  fs.mkdirSync(smokeUserDataPath, { recursive: true });
  app.setPath("userData", smokeUserDataPath);
  app.setPath("sessionData", smokeUserDataPath);
  app.disableHardwareAcceleration();
  app.commandLine.appendSwitch("disable-gpu");
  app.commandLine.appendSwitch("disable-gpu-compositing");
  app.commandLine.appendSwitch("disable-gpu-sandbox");
  app.commandLine.appendSwitch("in-process-gpu");
}

let mainWindow: BrowserWindow | null = null;
let serverProcess: ChildProcessWithoutNullStreams | null = null;
let connection: ServerConnection | null = null;
let stoppingServer: Promise<void> | null = null;
let quittingAfterServerStop = false;

function smokeLog(stage: string, details: Record<string, unknown> = {}) {
  if (!smokeMode) return;
  const line = JSON.stringify({ stage, ...details }) + "\n";
  process.stdout.write(line);
  if (smokeLogPath) fs.appendFileSync(smokeLogPath, line, "utf8");
}

async function stopRuntimeServer({ graceful = true, timeoutMs = 2500 }: { graceful?: boolean; timeoutMs?: number } = {}) {
  if (stoppingServer) return stoppingServer;
  stoppingServer = doStopRuntimeServer({ graceful, timeoutMs }).finally(() => {
    stoppingServer = null;
  });
  return stoppingServer;
}

async function doStopRuntimeServer({ graceful = true, timeoutMs = 2500 }: { graceful?: boolean; timeoutMs?: number } = {}) {
  const child = serverProcess;
  const currentConnection = connection;
  if (!child) return;
  if (graceful && currentConnection) {
    await requestServerShutdown(currentConnection, timeoutMs);
    await waitForExit(child, timeoutMs);
  }
  if (child.exitCode === null && !child.killed) {
    await killProcessTree(child, timeoutMs);
  }
  serverProcess = null;
  connection = null;
}

async function requestServerShutdown(selected: ServerConnection, timeoutMs: number) {
  try {
    await requestServer(selected, "/api/shutdown", "POST", timeoutMs);
  } catch {
    // If the server is already exiting or the request races process teardown,
    // the follow-up wait/kill path below still provides bounded shutdown.
  }
}

function requestServer(
  selected: ServerConnection,
  pathname: string,
  method: "GET" | "POST",
  timeoutMs: number
): Promise<{ ok: boolean; status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const url = new URL(pathname, selected.url);
    const request = http.request(
      url,
      {
        method,
        headers: { Authorization: `Bearer ${selected.token}` },
        timeout: timeoutMs
      },
      (response) => {
        let body = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          body += chunk;
        });
        response.on("end", () => {
          const status = response.statusCode ?? 0;
          resolve({ ok: status >= 200 && status < 300, status, body });
        });
      }
    );
    request.on("timeout", () => {
      request.destroy(new Error(`${method} ${url.href} timed out after ${timeoutMs}ms`));
    });
    request.on("error", reject);
    request.end();
  });
}

function waitForExit(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
  if (child.exitCode !== null || child.killed) return Promise.resolve();
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

async function killProcessTree(child: ChildProcessWithoutNullStreams, timeoutMs: number) {
  if (child.exitCode !== null || child.killed) return;
  if (process.platform === "win32" && child.pid !== undefined) {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore"
    });
    await waitForChildExit(killer, timeoutMs);
  } else {
    child.kill();
  }
  await waitForExit(child, timeoutMs);
  if (child.exitCode === null && !child.killed) child.kill();
}

function waitForChildExit(child: ChildProcess, timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      }
    );
  });
}

async function startRuntimeServer(db = "local"): Promise<ServerConnection> {
  smokeLog("server.start", { db });
  if (serverProcess) {
    await stopRuntimeServer();
  }
  connection = null;
  const serverCommand = resolveRuntimeServerCommand();
  smokeLog("server.command", { command: serverCommand.command, args: serverCommand.args });
  const child = spawn(serverCommand.command, [...serverCommand.args, "--db", db, "--port", "0"], {
    cwd: repoRoot,
    windowsHide: true
  });
  serverProcess = child;
  const startup = await new Promise<ServerConnection>((resolve, reject) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    const fail = (error: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    };
    const succeed = (value: ServerConnection) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => fail(new Error(`GUI server did not start. ${stderr}`)), 15000);
    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString("utf8");
      smokeLog("server.stdout", { preview: chunk.toString("utf8").slice(0, 200) });
      const line = stdout.split(/\r?\n/).find((item) => item.trim().startsWith("{"));
      if (!line) return;
      try {
        succeed(JSON.parse(line) as ServerConnection);
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)));
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString("utf8");
      smokeLog("server.stderr", { preview: chunk.toString("utf8").slice(0, 200) });
      console.error(chunk.toString("utf8"));
    });
    child.on("exit", (code) => {
      fail(new Error(`GUI server exited before startup with code ${code}. ${stderr}`));
    });
    child.on("error", (error) => {
      fail(error);
    });
  }).catch(async (error) => {
    await killProcessTree(child, 3000);
    if (serverProcess === child) serverProcess = null;
    throw error;
  });
  try {
    await waitForServerHealth(startup, 15000);
  } catch (error) {
    await killProcessTree(child, 3000);
    if (serverProcess === child) serverProcess = null;
    throw error;
  }
  connection = startup;
  return startup;
}

async function waitForServerHealth(selected: ServerConnection, timeoutMs: number) {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown = null;
  while (Date.now() < deadline) {
    try {
      const health = await requestServer(selected, "/api/health", "GET", 500);
      if (health.ok) {
        smokeLog("server.health.ready", { status: health.status });
        return health;
      }
      lastError = new Error(`health returned HTTP ${health.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(100);
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError ?? "GUI server health check timed out"));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function resolveRuntimeServerCommand(): RuntimeServerCommand {
  const explicit = process.env.AGENT_LIBOS_GUI_SERVER_BIN;
  if (explicit && explicit.trim()) {
    return { command: explicit.trim(), args: [] };
  }
  const venvScript =
    process.platform === "win32"
      ? path.join(repoRoot, ".venv", "Scripts", "agent-libos-gui-server.exe")
      : path.join(repoRoot, ".venv", "bin", "agent-libos-gui-server");
  if (fs.existsSync(venvScript)) {
    return { command: venvScript, args: [] };
  }
  return { command: "uv", args: ["run", "agent-libos-gui-server"] };
}

async function createWindow() {
  smokeLog("window.create.start");
  connection = await startRuntimeServer();
  smokeLog("window.server.ready", { db: connection.db, url: connection.url });
  if (smokeMode && process.env.AGENT_LIBOS_GUI_SMOKE_WINDOW !== "1") {
    const health = await withTimeout(requestServer(connection, "/api/health", "GET", 5000), 5000, "server health");
    smokeLog("server.health.checked", { ok: health.ok, status: health.status });
    await stopRuntimeServer({ graceful: true, timeoutMs: 3000 });
    smokeLog("smoke.exiting", { code: health.ok ? 0 : 2 });
    app.exit(health.ok ? 0 : 2);
    process.exit(health.ok ? 0 : 2);
  }
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1100,
    minHeight: 720,
    title: "Agent libOS Console",
    show: !smokeMode,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedExternalUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    const current = mainWindow?.webContents.getURL();
    if (url === current) return;
    event.preventDefault();
    if (isAllowedExternalUrl(url)) void shell.openExternal(url);
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  if (smokeMode) {
    await withTimeout(
      mainWindow.loadURL("data:text/html;charset=utf-8,<html lang=\"zh-Hans\"><body>Agent libOS smoke</body></html>"),
      15000,
      "renderer smoke loadURL"
    );
  } else if (process.env.VITE_DEV_SERVER_URL) {
    await withTimeout(mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL), 15000, "renderer loadURL");
  } else {
    await withTimeout(mainWindow.loadFile(path.join(repoRoot, "gui", "dist", "index.html")), 15000, "renderer loadFile");
  }
  smokeLog("window.loaded");
  if (smokeMode) {
    const preloadReady = await withTimeout(
      mainWindow.webContents.executeJavaScript(
        "Boolean(window.libosApi) && window.libosApi.getConnection().then((connection) => Boolean(connection && connection.url && connection.token))"
      ),
      5000,
      "preload bridge"
    );
    smokeLog("window.preload.checked", { preloadReady });
    smokeLog("smoke.complete", { preloadReady, db: connection?.db ?? null, pid: process.pid });
    await stopRuntimeServer({ graceful: true, timeoutMs: 3000 });
    smokeLog("smoke.exiting", { code: preloadReady ? 0 : 2 });
    app.exit(preloadReady ? 0 : 2);
    process.exit(preloadReady ? 0 : 2);
  }
}

ipcMain.handle("libos:getConnection", () => connection);

ipcMain.handle("libos:chooseDatabase", async () => {
  const options: OpenDialogOptions = {
    title: "Open Agent libOS SQLite database",
    properties: ["openFile"],
    filters: [{ name: "SQLite database", extensions: ["sqlite", "db"] }, { name: "All files", extensions: ["*"] }]
  };
  const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
  if (result.canceled || result.filePaths.length === 0) return connection;
  return startRuntimeServer(result.filePaths[0]);
});

ipcMain.handle("libos:chooseImagePackage", async () => {
  const options: OpenDialogOptions = {
    title: "Open AgentImage package",
    properties: ["openDirectory"]
  };
  const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
  if (result.canceled || result.filePaths.length === 0) return null;
  const selected = result.filePaths[0];
  const stats = fs.lstatSync(selected);
  if (!stats.isDirectory()) throw new Error("Selected image package is not a directory.");
  const files = readImagePackageFiles(selected);
  const manifest = Buffer.from(files["IMAGE.yaml"].base64, "base64");
  if (manifest.length > imageManifestMaxBytes) {
    throw new Error(`Image manifest exceeds ${imageManifestMaxBytes} bytes.`);
  }
  return {
    path: selected,
    name: path.basename(selected),
    manifest: manifest.toString("utf8"),
    files
  };
});

ipcMain.handle("libos:useDatabase", async (_event, db: string) => {
  return startRuntimeServer(databaseTargetFromRenderer(db));
});

ipcMain.handle("libos:openExternal", async (_event, url: string) => {
  if (!isAllowedExternalUrl(url)) return false;
  await shell.openExternal(url);
  return true;
});

function isAllowedExternalUrl(url: string): boolean {
  try {
    return allowedExternalProtocols.has(new URL(url).protocol);
  } catch {
    return false;
  }
}

app.whenReady().then(createWindow).catch((error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  void stopRuntimeServer({ graceful: false });
  app.exit(1);
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) void createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", (event) => {
  if (!serverProcess || quittingAfterServerStop) return;
  event.preventDefault();
  quittingAfterServerStop = true;
  void stopRuntimeServer().finally(() => app.quit());
});
