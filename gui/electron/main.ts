import { app, BrowserWindow, dialog, ipcMain, protocol, shell, type IpcMainInvokeEvent, type OpenDialogOptions } from "electron";
import { ChildProcess, ChildProcessWithoutNullStreams, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import * as fs from "node:fs";
import * as http from "node:http";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { databaseTargetFromRenderer } from "./database.js";
import { redactGuiServerOutput, requireLoopbackDevServerUrl, runtimeServerEnv } from "./env.js";
import { readImagePackageFiles } from "./imagePackage.js";
import { appendStartupOutput, cleanupBeforeExit, consumeStartupOutput, isChildAlive, withStartupFailureCleanup, type ServerConnection } from "./processLifecycle.js";
import {
  productionRendererEntryUrl,
  productionRendererOrigin,
  productionRendererScheme,
  readProductionRendererAsset
} from "./rendererProtocol.js";
import { isCompletedShutdownResponse } from "./shutdown.js";
import {
  assertTrustedIpcSender,
  installDefaultDenyPermissions,
  installTrustedRendererNavigationGuard
} from "./security.js";
import { mainWindowBounds, shouldCreateBrowserWindow } from "./windowBounds.js";

type RuntimeServerCommand = {
  command: string;
  args: string[];
};

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, "..", "..");
const smokeMode = process.env.AGENT_LIBOS_GUI_SMOKE === "1";
const smokeWindowMode = smokeMode && process.env.AGENT_LIBOS_GUI_SMOKE_WINDOW === "1";
const smokeLogPath = process.env.AGENT_LIBOS_GUI_SMOKE_LOG;
const imageManifestMaxBytes = 1_048_576;
const startupOutputMaxBytes = 65_536;
const allowedExternalProtocols = new Set(["http:", "https:", "mailto:"]);
const productionCsp = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "font-src 'self' data:",
  "connect-src 'self' http://127.0.0.1:* http://localhost:* http://[::1]:*",
  "object-src 'none'",
  "base-uri 'none'",
  "frame-ancestors 'none'"
].join("; ");

protocol.registerSchemesAsPrivileged([
  {
    scheme: productionRendererScheme,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true
    }
  }
]);

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
let startingServerProcess: ChildProcessWithoutNullStreams | null = null;
let connection: ServerConnection | null = null;
let stoppingServer: Promise<void> | null = null;
let startingServer: Promise<ServerConnection> | null = null;
let creatingWindow: Promise<void> | null = null;
let quittingAfterServerStop = false;
let productionRendererProtocolInstalled = false;

function smokeLog(stage: string, details: Record<string, unknown> = {}) {
  if (!smokeMode) return;
  const line = JSON.stringify({ stage, ...details }) + "\n";
  process.stdout.write(line);
  if (smokeLogPath) fs.appendFileSync(smokeLogPath, line, "utf8");
}

async function stopRuntimeServer({ graceful = true, timeoutMs = 2500 }: { graceful?: boolean; timeoutMs?: number } = {}) {
  if (stoppingServer) return stoppingServer;
  const child = serverProcess;
  const startingChild = startingServerProcess === child ? null : startingServerProcess;
  const currentConnection = connection;
  stoppingServer = Promise.all([
    stopRuntimeServerInstance(child, currentConnection, { graceful, timeoutMs }),
    stopRuntimeServerInstance(startingChild, null, { graceful: false, timeoutMs })
  ]).then(() => undefined).finally(() => {
    if (serverProcess === child) serverProcess = null;
    if (startingServerProcess === startingChild) startingServerProcess = null;
    if (connection === currentConnection) connection = null;
    stoppingServer = null;
  });
  return stoppingServer;
}

async function stopRuntimeServerInstance(
  child: ChildProcessWithoutNullStreams | null,
  currentConnection: ServerConnection | null,
  { graceful = true, timeoutMs = 2500 }: { graceful?: boolean; timeoutMs?: number } = {}
) {
  if (!child) return;
  let gracefulAcknowledged = false;
  if (graceful && currentConnection) {
    gracefulAcknowledged = await requestServerShutdown(currentConnection, timeoutMs);
    if (gracefulAcknowledged) await waitForExit(child, timeoutMs);
  }
  const forced = isChildAlive(child);
  if (graceful && currentConnection && !gracefulAcknowledged) {
    console.warn("Agent libOS GUI server did not confirm completed teardown; forcing process termination.");
  }
  if (isChildAlive(child)) {
    await killProcessTree(child, timeoutMs);
  }
  smokeLog("server.stop.completed", { gracefulAcknowledged, forced });
}

async function requestServerShutdown(selected: ServerConnection, timeoutMs: number): Promise<boolean> {
  try {
    return isCompletedShutdownResponse(await requestServer(selected, "/api/shutdown", "POST", timeoutMs));
  } catch {
    // If the server is already exiting or the request races process teardown,
    // the follow-up wait/kill path below still provides bounded shutdown.
    return false;
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
  if (!isChildAlive(child)) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      child.removeListener("exit", finish);
      resolve();
    };
    const timer = setTimeout(finish, timeoutMs);
    child.once("exit", finish);
    if (!isChildAlive(child)) finish();
  });
}

async function killProcessTree(child: ChildProcessWithoutNullStreams, timeoutMs: number) {
  if (!isChildAlive(child)) return;
  if (process.platform === "win32" && child.pid !== undefined) {
    const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
      windowsHide: true,
      stdio: "ignore"
    });
    await waitForChildExit(killer, timeoutMs);
  } else {
    signalProcessGroup(child, "SIGTERM");
  }
  await waitForExit(child, timeoutMs);
  if (isChildAlive(child)) {
    signalProcessGroup(child, "SIGKILL");
    await waitForExit(child, timeoutMs);
  }
}

function signalProcessGroup(child: ChildProcessWithoutNullStreams, signal: NodeJS.Signals) {
  if (child.pid === undefined) {
    child.kill(signal);
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch {
    child.kill(signal);
  }
}

function waitForChildExit(child: ChildProcess, timeoutMs: number): Promise<void> {
  if (!isChildAlive(child)) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      child.removeListener("exit", finish);
      resolve();
    };
    const timer = setTimeout(finish, timeoutMs);
    child.once("exit", finish);
    if (!isChildAlive(child)) finish();
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

async function startRuntimeServer(db?: string): Promise<ServerConnection> {
  if (quittingAfterServerStop) throw new Error("GUI server cannot start while the application is quitting.");
  if (stoppingServer) await stoppingServer;
  if (quittingAfterServerStop) throw new Error("GUI server cannot start while the application is quitting.");
  if (startingServer) {
    try {
      await startingServer;
    } catch {
      // The next start below gets its own process and error surface.
    }
  }
  if (stoppingServer) await stoppingServer;
  if (quittingAfterServerStop) throw new Error("GUI server cannot start while the application is quitting.");
  if (isChildAlive(serverProcess) && connection && (db === undefined || connection.db === db)) {
    return connection;
  }
  startingServer = doStartRuntimeServer(db).finally(() => {
    startingServer = null;
  });
  return startingServer;
}

async function doStartRuntimeServer(db?: string): Promise<ServerConnection> {
  smokeLog("server.start", { db: db ?? null });
  const previousProcess = serverProcess;
  const previousConnection = connection;
  const serverCommand = resolveRuntimeServerCommand();
  smokeLog("server.command", { command: serverCommand.command, args: serverCommand.args });
  const llmProfilesFile = path.join(app.getPath("userData"), "llm-profiles.json");
  fs.mkdirSync(path.dirname(llmProfilesFile), { recursive: true });
  const serverArgs = db === undefined
    ? [...serverCommand.args, "--port", "0", "--llm-profiles-file", llmProfilesFile]
    : [...serverCommand.args, "--db", db, "--port", "0", "--llm-profiles-file", llmProfilesFile];
  const child = spawn(serverCommand.command, serverArgs, {
    cwd: repoRoot,
    env: runtimeServerEnv(repoRoot),
    detached: process.platform !== "win32",
    windowsHide: true
  });
  startingServerProcess = child;
  try {
    return await withStartupFailureCleanup(async () => {
      const startup = await new Promise<ServerConnection>((resolve, reject) => {
        let stdoutState = { text: "", scanOffset: 0 };
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
          if (settled) return;
          try {
            const consumed = consumeStartupOutput(stdoutState, chunk, startupOutputMaxBytes);
            stdoutState = consumed.state;
            if (!consumed.connection) return;
            smokeLog("server.stdout", { preview: redactGuiServerOutput(consumed.frame ?? "").slice(0, 200) });
            succeed(consumed.connection);
          } catch (error) {
            fail(error instanceof Error ? error : new Error(String(error)));
          }
        });
        child.stderr.on("data", (chunk: Buffer) => {
          if (!settled) {
            try {
              stderr = appendStartupOutput(stderr, chunk, startupOutputMaxBytes, "stderr");
            } catch (error) {
              fail(error instanceof Error ? error : new Error(String(error)));
              return;
            }
          }
          smokeLog("server.stderr", { preview: chunk.toString("utf8").slice(0, 200) });
          console.error(chunk.toString("utf8"));
        });
        child.on("exit", (code, signal) => {
          fail(new Error(`GUI server exited before startup with code ${code} and signal ${signal}. ${stderr}`));
        });
        child.on("error", (error) => {
          fail(error);
        });
      });
      await waitForServerHealth(startup, 15000);
      if (!isChildAlive(child)) throw new Error("GUI server exited during its health check.");
      if (stoppingServer || quittingAfterServerStop) {
        throw new Error("GUI server startup was cancelled during application shutdown.");
      }
      serverProcess = child;
      connection = startup;
      if (startingServerProcess === child) startingServerProcess = null;
      if (previousProcess && previousProcess !== child) {
        await stopRuntimeServerInstance(previousProcess, previousConnection, { graceful: true, timeoutMs: 2500 });
      }
      return startup;
    }, async () => {
      await killProcessTree(child, 3000);
      if (serverProcess === child) serverProcess = null;
    });
  } finally {
    if (startingServerProcess === child) startingServerProcess = null;
  }
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
  smokeLog("startup.begin");
  // Smoke validation must never open or mutate an operator's default
  // persistent database merely because the command runs from the repo root.
  connection = await startRuntimeServer(smokeMode ? "local" : undefined);
  smokeLog("window.server.ready", { db: connection.db, url: connection.url });
  if (!shouldCreateBrowserWindow(smokeMode, smokeWindowMode)) {
    const health = await withTimeout(requestServer(connection, "/api/health", "GET", 5000), 5000, "server health");
    smokeLog("server.health.checked", { ok: health.ok, status: health.status });
    await stopRuntimeServer({ graceful: true, timeoutMs: 3000 });
    smokeLog("smoke.exiting", { code: health.ok ? 0 : 2 });
    app.exit(health.ok ? 0 : 2);
    process.exit(health.ok ? 0 : 2);
    return;
  }
  smokeLog("window.create.start");
  mainWindow = new BrowserWindow({
    ...mainWindowBounds,
    title: "Agent libOS Console",
    show: !smokeMode,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true
    }
  });
  installDefaultDenyPermissions(mainWindow.webContents.session);
  installTrustedRendererNavigationGuard(mainWindow.webContents, trustedRendererUrl());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedExternalUrl(url)) void shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  if (!smokeWindowMode && process.env.VITE_DEV_SERVER_URL) {
    await withTimeout(mainWindow.loadURL(requireLoopbackDevServerUrl(process.env.VITE_DEV_SERVER_URL)), 15000, "renderer loadURL");
  } else {
    installProductionRendererProtocol();
    installProductionCsp(mainWindow);
    await withTimeout(mainWindow.loadURL(productionRendererEntryUrl), 15000, "renderer loadURL");
  }
  smokeLog("window.loaded");
  if (smokeMode) {
    const rendererProbe = await withTimeout(
      mainWindow.webContents.executeJavaScript(
        `(async () => {
          const api = window.libosApi;
          if (!api) return { preloadReady: false, apiReady: false, origin: window.location.origin };
          const selected = await api.getConnection();
          const preloadReady = Boolean(selected && selected.url && selected.token);
          if (!preloadReady) return { preloadReady, apiReady: false, origin: window.location.origin };
          const response = await fetch(selected.url + "/api/health", {
            headers: { Authorization: "Bearer " + selected.token }
          });
          const body = await response.json();
          return {
            preloadReady,
            apiReady: response.ok && body && body.ok === true,
            origin: window.location.origin
          };
        })()`
      ),
      5000,
      "production renderer probe"
    );
    const preloadReady = rendererProbe?.preloadReady === true;
    const apiReady = rendererProbe?.apiReady === true;
    const originReady = rendererProbe?.origin === productionRendererOrigin;
    const smokePassed = preloadReady && apiReady && originReady;
    smokeLog("window.renderer.checked", { preloadReady, apiReady, origin: rendererProbe?.origin, originReady });
    smokeLog("smoke.complete", { preloadReady, apiReady, originReady, db: connection?.db ?? null, pid: process.pid });
    await stopRuntimeServer({ graceful: true, timeoutMs: 3000 });
    smokeLog("smoke.exiting", { code: smokePassed ? 0 : 2 });
    app.exit(smokePassed ? 0 : 2);
    process.exit(smokePassed ? 0 : 2);
  }
}

function ensureWindow(): Promise<void> {
  if (creatingWindow) return creatingWindow;
  const request = createWindow().finally(() => {
    if (creatingWindow === request) creatingWindow = null;
  });
  creatingWindow = request;
  return request;
}

function assertIpcSender(event: IpcMainInvokeEvent): void {
  assertTrustedIpcSender(event, mainWindow?.webContents ?? null, trustedRendererUrl());
}

function trustedRendererUrl(): string {
  if (!smokeWindowMode && process.env.VITE_DEV_SERVER_URL) {
    return requireLoopbackDevServerUrl(process.env.VITE_DEV_SERVER_URL);
  }
  return productionRendererOrigin;
}

ipcMain.handle("libos:getConnection", (event) => {
  assertIpcSender(event);
  return connection;
});

ipcMain.handle("libos:chooseDatabase", async (event) => {
  assertIpcSender(event);
  const options: OpenDialogOptions = {
    title: "Open Agent libOS SQLite database",
    properties: ["openFile"],
    filters: [{ name: "SQLite database", extensions: ["sqlite", "db"] }, { name: "All files", extensions: ["*"] }]
  };
  const result = mainWindow ? await dialog.showOpenDialog(mainWindow, options) : await dialog.showOpenDialog(options);
  if (result.canceled || result.filePaths.length === 0) return connection;
  return startRuntimeServer(result.filePaths[0]);
});

ipcMain.handle("libos:chooseImagePackage", async (event) => {
  assertIpcSender(event);
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
    name: path.basename(selected),
    manifest: manifest.toString("utf8"),
    manifest_sha256: createHash("sha256").update(manifest).digest("hex"),
    files
  };
});

ipcMain.handle("libos:useDatabase", async (_event, db: string) => {
  assertIpcSender(_event);
  return startRuntimeServer(databaseTargetFromRenderer(db));
});

ipcMain.handle("libos:openExternal", async (_event, url: string) => {
  assertIpcSender(_event);
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

function installProductionCsp(window: BrowserWindow) {
  window.webContents.session.webRequest.onHeadersReceived({ urls: [`${productionRendererScheme}://*/*`] }, (details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        "Content-Security-Policy": [productionCsp]
      }
    });
  });
}

function installProductionRendererProtocol() {
  if (productionRendererProtocolInstalled) return;
  const distRoot = path.join(repoRoot, "gui", "dist");
  protocol.handle(productionRendererScheme, async (request) => {
    const asset = await readProductionRendererAsset(distRoot, request.url);
    if (asset === null) {
      return new Response("Not found", { status: 404, headers: { "Content-Type": "text/plain; charset=utf-8" } });
    }
    return new Response(asset.body, { headers: { "Content-Type": asset.contentType } });
  });
  productionRendererProtocolInstalled = true;
}

app.whenReady().then(ensureWindow).catch(async (error) => {
  console.error(error instanceof Error ? error.stack : String(error));
  await cleanupBeforeExit(
    () => stopRuntimeServer({ graceful: false }),
    () => app.exit(1)
  );
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    void ensureWindow().catch((error) => {
      console.error(error instanceof Error ? error.stack : String(error));
    });
  }
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("before-quit", (event) => {
  if ((!isChildAlive(serverProcess) && !isChildAlive(startingServerProcess)) || quittingAfterServerStop) return;
  event.preventDefault();
  quittingAfterServerStop = true;
  void stopRuntimeServer().finally(() => app.quit());
});
