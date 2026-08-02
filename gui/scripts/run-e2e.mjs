import { randomBytes } from "node:crypto";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

const guiRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = resolve(guiRoot, "..");
const tempRoot = mkdtempSync(join(tmpdir(), "agent-libos-gui-e2e-"));
const expectedDb = join(tempRoot, "runtime.db");
const controlTokenFile = join(tempRoot, "control-token");
const guiToken = secret("e2e_gui_secret_");
const providerKey = secret("e2e_provider_secret_");
const controlToken = secret("e2e_control_secret_");
const secrets = [guiToken, providerKey, controlToken];
const children = [];
const baseEnv = allowlistedEnvironment();
let interrupted = false;

for (const [name, code] of [["SIGINT", 130], ["SIGTERM", 143]]) {
  process.once(name, () => {
    interrupted = true;
    process.exitCode = code;
    for (const child of children) {
      if (child.exitCode === null) child.kill("SIGTERM");
    }
  });
}

try {
  const python = virtualenvPython();
  writeFileSync(controlTokenFile, controlToken, { encoding: "utf8", mode: 0o600 });
  const fixture = start(python, [
    join(guiRoot, "e2e", "fixture_server.py"),
    "--db",
    expectedDb
  ], {
    cwd: repoRoot,
    env: {
      ...baseEnv,
      AGENT_LIBOS_E2E_GUI_TOKEN: guiToken,
      AGENT_LIBOS_E2E_PROVIDER_KEY: providerKey,
      AGENT_LIBOS_E2E_CONTROL_TOKEN: controlToken,
      PYTHONUNBUFFERED: "1"
    }
  });
  const startup = await readJsonStartup(fixture);
  validateStartup(startup);

  const vitePort = await freeLoopbackPort();
  const viteUrl = `http://127.0.0.1:${vitePort}`;
  const vite = start(process.execPath, [
    join(guiRoot, "node_modules", "vite", "bin", "vite.js"),
    "--host",
    "127.0.0.1",
    "--port",
    String(vitePort),
    "--strictPort"
  ], {
    cwd: guiRoot,
    env: {
      ...baseEnv,
      VITE_AGENT_LIBOS_GUI_URL: startup.url,
      VITE_AGENT_LIBOS_GUI_TOKEN: guiToken,
      VITE_AGENT_LIBOS_GUI_DB: startup.db
    }
  });
  await waitForUrl(viteUrl, vite);

  const playwright = start(process.execPath, [
    join(guiRoot, "node_modules", "@playwright", "test", "cli.js"),
    "test",
    "--config",
    join(guiRoot, "playwright.config.ts"),
    ...process.argv.slice(2)
  ], {
    cwd: guiRoot,
    env: {
      ...baseEnv,
      AGENT_LIBOS_E2E_BASE_URL: viteUrl,
      AGENT_LIBOS_E2E_GUI_URL: startup.url,
      AGENT_LIBOS_E2E_CONTROL_URL: startup.control_url,
      AGENT_LIBOS_E2E_CONTROL_TOKEN_FILE: controlTokenFile,
      AGENT_LIBOS_E2E_LIVE_CALL_ID: startup.live_call_id,
      AGENT_LIBOS_E2E_SUMMARY_CALL_ID: startup.summary_call_id,
      AGENT_LIBOS_E2E_HASH_CALL_ID: startup.hash_call_id,
      AGENT_LIBOS_E2E_CONFLICT_CALL_ID: startup.conflict_call_id,
      AGENT_LIBOS_E2E_LIMITED_CALL_ID: startup.limited_call_id,
      AGENT_LIBOS_E2E_OLDEST_CALL_ID: startup.oldest_call_id,
      AGENT_LIBOS_E2E_EXPECTED_CALL_COUNT: String(startup.expected_call_count)
    },
    inherit: true
  });
  const exitCode = await waitForExit(playwright);
  if (!interrupted && exitCode !== 0) process.exitCode = exitCode;
} catch (error) {
  if (!interrupted) {
    process.exitCode = 1;
    process.stderr.write(`${redact(error instanceof Error ? error.stack || error.message : error)}\n`);
  }
} finally {
  for (const child of children.reverse()) await stop(child);
  rmSync(tempRoot, { recursive: true, force: true });
}

function secret(prefix) {
  return `${prefix}${randomBytes(32).toString("hex")}`;
}

function allowlistedEnvironment() {
  const safeNames = [
    "PATH",
    "Path",
    "PATHEXT",
    "SystemRoot",
    "SYSTEMROOT",
    "WINDIR",
    "ComSpec",
    "HOME",
    "USERPROFILE",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "CI",
    "NO_COLOR",
    "FORCE_COLOR",
    "PLAYWRIGHT_BROWSERS_PATH",
    "PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD",
    "XDG_CACHE_HOME"
  ];
  const selected = {};
  for (const name of safeNames) {
    if (typeof process.env[name] === "string") selected[name] = process.env[name];
  }
  return selected;
}

function virtualenvPython() {
  const path = process.platform === "win32"
    ? join(repoRoot, ".venv", "Scripts", "python.exe")
    : join(repoRoot, ".venv", "bin", "python");
  if (!existsSync(path)) {
    throw new Error(
      `Python environment not found at ${path}; run uv sync --frozen first`
    );
  }
  return path;
}

function start(command, args, options) {
  const child = spawn(command, args, {
    cwd: options.cwd,
    env: options.env,
    stdio: options.inherit ? "inherit" : ["ignore", "pipe", "pipe"]
  });
  child._stderr = "";
  if (!options.inherit) {
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk) => {
      child._stderr = `${child._stderr}${chunk}`.slice(-16_384);
    });
  }
  children.push(child);
  return child;
}

async function readJsonStartup(child) {
  child.stdout.setEncoding("utf8");
  let buffer = "";
  return await new Promise((resolveStartup, reject) => {
    const timer = setTimeout(
      () => reject(new Error("GUI E2E fixture did not start in time")),
      45_000
    );
    child.stdout.on("data", (chunk) => {
      buffer += chunk;
      const newline = buffer.indexOf("\n");
      if (newline < 0) return;
      const line = buffer.slice(0, newline);
      try {
        const value = JSON.parse(line);
        clearTimeout(timer);
        resolveStartup(value);
      } catch (error) {
        clearTimeout(timer);
        reject(new Error(`GUI E2E fixture startup failed: ${String(error)}`));
      }
    });
    child.once("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`GUI E2E fixture exited (${code}): ${child._stderr}`));
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
  });
}

function validateStartup(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("invalid fixture startup payload");
  }
  const urlFields = ["url", "control_url"];
  for (const field of urlFields) {
    if (typeof value[field] !== "string") throw new Error(`fixture ${field} is missing`);
    const parsed = new URL(value[field]);
    if (parsed.protocol !== "http:" || parsed.hostname !== "127.0.0.1" || !parsed.port) {
      throw new Error(`fixture ${field} must be an explicit loopback HTTP URL`);
    }
  }
  if (resolve(String(value.db || "")) !== resolve(expectedDb)) {
    throw new Error("fixture database escaped the E2E temporary directory");
  }
  for (const field of [
    "pid",
    "live_call_id",
    "summary_call_id",
    "hash_call_id",
    "conflict_call_id",
    "limited_call_id",
    "oldest_call_id"
  ]) {
    if (typeof value[field] !== "string" || !/^[A-Za-z0-9_.:-]+$/.test(value[field])) {
      throw new Error(`fixture ${field} is invalid`);
    }
  }
  if (!Number.isSafeInteger(value.expected_call_count) || value.expected_call_count <= 50) {
    throw new Error("fixture must expose more calls than one GUI API page");
  }
}

async function waitForUrl(url, child) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) {
      throw new Error(`Vite exited (${child.exitCode}): ${child._stderr}`);
    }
    try {
      const response = await fetch(url, { redirect: "error" });
      if (response.ok) return;
    } catch {
      // The loopback server is still starting.
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }
  throw new Error(`Vite did not become ready: ${child._stderr}`);
}

function freeLoopbackPort() {
  return new Promise((resolvePort, reject) => {
    const server = createServer();
    server.unref();
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        reject(new Error("failed to reserve a Vite port"));
        return;
      }
      const port = address.port;
      server.close((error) => error ? reject(error) : resolvePort(port));
    });
  });
}

function waitForExit(child) {
  if (child.exitCode !== null) return Promise.resolve(child.exitCode);
  return new Promise((resolveExit, reject) => {
    child.once("exit", (code) => resolveExit(code ?? 1));
    child.once("error", reject);
  });
}

async function stop(child) {
  if (!child || child.exitCode !== null) return;
  child.kill("SIGTERM");
  await Promise.race([
    waitForExit(child),
    new Promise((resolveWait) => setTimeout(resolveWait, 2_000))
  ]);
  if (child.exitCode === null) {
    child.kill("SIGKILL");
    await Promise.race([
      waitForExit(child),
      new Promise((resolveWait) => setTimeout(resolveWait, 1_000))
    ]);
  }
}

function redact(value) {
  let selected = String(value || "");
  for (const secretValue of secrets) {
    selected = selected.split(secretValue).join("[redacted]");
  }
  return selected;
}
