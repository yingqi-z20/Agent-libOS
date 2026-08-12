import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  ensurePrivateRuntimeDirectory,
  packagedRuntimeArguments,
  packagedRuntimeLayout,
  resolveRuntimeServerCommand,
  runtimeChildEnvironment
} from "./desktopRuntime.js";

const roots: string[] = [];

function tempRoot(): string {
  const selected = fs.mkdtempSync(path.join(os.tmpdir(), "agent-libos-desktop-runtime-"));
  roots.push(selected);
  return selected;
}

afterEach(() => {
  while (roots.length > 0) fs.rmSync(roots.pop()!, { recursive: true, force: true });
});

describe("packaged desktop Runtime layout", () => {
  it("binds backend, renderer, Deno, database, profiles and config to packaged roots", () => {
    const resources = path.join(tempRoot(), "resources");
    const userData = path.join(tempRoot(), "user-data");
    const layout = packagedRuntimeLayout(resources, userData, "darwin");

    expect(layout).toEqual({
      backendExecutable: path.join(resources, "backend", "agent-libos-gui-server"),
      configFile: path.join(userData, "config.yaml"),
      databaseFile: path.join(userData, "runtime", "agent-libos.sqlite"),
      denoBinDirectory: path.join(resources, "bin"),
      llmProfilesFile: path.join(userData, "llm-profiles.json"),
      rendererRoot: path.join(resources, "renderer"),
      runtimeDirectory: path.join(userData, "runtime")
    });
    expect(packagedRuntimeLayout(resources, userData, "win32").backendExecutable)
      .toBe(path.join(resources, "backend", "agent-libos-gui-server.exe"));
  });

  it("never honors a development backend override in a packaged app", () => {
    const resources = path.join(tempRoot(), "resources");
    const userData = path.join(tempRoot(), "user-data");
    const backendDirectory = path.join(resources, "backend");
    fs.mkdirSync(backendDirectory, { recursive: true });
    const backend = path.join(backendDirectory, "agent-libos-gui-server");
    fs.writeFileSync(backend, "#!/bin/sh\n", { mode: 0o700 });

    expect(resolveRuntimeServerCommand({
      packaged: true,
      platform: "darwin",
      repoRoot: tempRoot(),
      resourcesPath: resources,
      userDataPath: userData,
      explicit: "/tmp/untrusted-override"
    })).toEqual({ command: backend, args: [] });
  });

  it("keeps the existing override and venv behavior in development", () => {
    const root = tempRoot();
    expect(resolveRuntimeServerCommand({
      packaged: false,
      repoRoot: root,
      resourcesPath: root,
      userDataPath: root,
      explicit: "/opt/host/gui-server"
    })).toEqual({ command: "/opt/host/gui-server", args: [] });
  });

  it("does not read checkout dotenv in packaged mode and prepends only bundled Deno", () => {
    const repoRoot = tempRoot();
    const resources = path.join(tempRoot(), "resources");
    const userData = path.join(tempRoot(), "user-data");
    fs.writeFileSync(path.join(repoRoot, ".env"), "OPENAI_API_KEY=must-not-load\n", "utf8");

    const selected = runtimeChildEnvironment({
      packaged: true,
      platform: "darwin",
      repoRoot,
      resourcesPath: resources,
      userDataPath: userData,
      baseEnv: { PATH: "/usr/bin", INHERITED: "yes" }
    });

    expect(selected.OPENAI_API_KEY).toBeUndefined();
    expect(selected.PATH).toBe(`${path.join(resources, "bin")}${path.delimiter}/usr/bin`);
    expect(selected.DENO_NO_UPDATE_CHECK).toBe("1");
    expect(selected.INHERITED).toBe("yes");
  });

  it("uses the inherited Windows Path key without creating a conflicting duplicate", () => {
    const root = tempRoot();
    const selected = runtimeChildEnvironment({
      packaged: true,
      platform: "win32",
      repoRoot: root,
      resourcesPath: path.join(root, "resources"),
      userDataPath: path.join(root, "data"),
      baseEnv: { Path: "C:\\Windows\\System32" }
    });
    expect(selected.Path).toContain(`resources${path.sep}bin${path.delimiter}`);
    expect(selected.PATH).toBeUndefined();
  });

  it("creates an owner-only real runtime directory and rejects a symlink", () => {
    const root = tempRoot();
    const runtime = path.join(root, "runtime");
    ensurePrivateRuntimeDirectory(runtime, "darwin");
    expect(fs.statSync(runtime).mode & 0o777).toBe(0o700);

    const target = path.join(root, "target");
    const link = path.join(root, "link");
    fs.mkdirSync(target);
    fs.symlinkSync(target, link, "dir");
    expect(() => ensurePrivateRuntimeDirectory(link, "darwin")).toThrow(/real directory/);
  });

  it("uses the persistent database and an optional regular user config", () => {
    const resources = path.join(tempRoot(), "resources");
    const userData = path.join(tempRoot(), "user-data");
    const layout = packagedRuntimeLayout(resources, userData, "darwin");
    fs.mkdirSync(userData, { recursive: true });

    expect(packagedRuntimeArguments(layout)).toEqual([
      "--db", layout.databaseFile,
      "--port", "0",
      "--llm-profiles-file", layout.llmProfilesFile
    ]);
    fs.writeFileSync(layout.configFile, "runtime: {}\n", "utf8");
    expect(packagedRuntimeArguments(layout).slice(-2)).toEqual(["--config", layout.configFile]);
  });
});
