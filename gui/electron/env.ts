import * as fs from "node:fs";
import * as path from "node:path";

export function runtimeServerEnv(repoRoot: string, baseEnv: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  return mergeDotenvIntoEnv(baseEnv, readDotenv(path.join(repoRoot, ".env")));
}

export function requireLoopbackDevServerUrl(rawUrl: string): string {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch (error) {
    throw new Error(`Invalid VITE_DEV_SERVER_URL: ${rawUrl}`, { cause: error });
  }
  if (!["http:", "https:"].includes(url.protocol) || !isLoopbackHost(url.hostname)) {
    throw new Error("VITE_DEV_SERVER_URL must use a loopback HTTP(S) host.");
  }
  if (url.username || url.password) {
    throw new Error("VITE_DEV_SERVER_URL must not include URL credentials.");
  }
  return url.toString();
}

export function redactGuiServerOutput(value: string): string {
  return value.replace(/("token"\s*:\s*)"[^"\r\n]*"/g, '$1"[redacted]"');
}

export function readDotenv(filePath: string): Record<string, string> {
  if (!fs.existsSync(filePath)) return {};
  const values: Record<string, string> = {};
  for (const rawLine of fs.readFileSync(filePath, "utf8").split(/\r?\n/)) {
    let line = rawLine.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    if (line.startsWith("export ")) line = line.slice("export ".length).trim();
    const separator = line.indexOf("=");
    const key = line.slice(0, separator).trim();
    const value = stripOuterQuotes(line.slice(separator + 1).trim());
    if (key) values[key] = value;
  }
  return values;
}

export function mergeDotenvIntoEnv(
  baseEnv: NodeJS.ProcessEnv,
  dotenv: Record<string, string>,
  platform: NodeJS.Platform = process.platform
): NodeJS.ProcessEnv {
  const merged: NodeJS.ProcessEnv = { ...baseEnv };
  for (const [key, value] of Object.entries(dotenv)) {
    if (!hasEnvKey(merged, key, platform)) merged[key] = value;
  }
  return merged;
}

function hasEnvKey(env: NodeJS.ProcessEnv, key: string, platform: NodeJS.Platform): boolean {
  if (Object.prototype.hasOwnProperty.call(env, key)) return env[key] !== undefined;
  if (platform !== "win32") return false;
  const folded = key.toLowerCase();
  return Object.keys(env).some((candidate) => candidate.toLowerCase() === folded && env[candidate] !== undefined);
}

function stripOuterQuotes(value: string): string {
  return value.replace(/^"+|"+$/g, "").replace(/^'+|'+$/g, "");
}

function isLoopbackHost(hostname: string): boolean {
  const normalized = hostname.toLowerCase();
  return normalized === "localhost" || normalized === "127.0.0.1" || normalized === "::1" || normalized === "[::1]";
}
