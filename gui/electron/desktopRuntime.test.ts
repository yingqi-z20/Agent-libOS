import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  assertDatabaseOutsideWorkspace,
  developmentRuntimeArguments,
  ensurePrivateRuntimeDirectory,
  ensurePrivateWorkspaceDirectory,
  packagedRuntimeArguments,
  packagedRuntimeLayout,
  resolveRuntimeServerCommand,
  runtimeChildEnvironment
} from "./desktopRuntime.js";

const roots: string[] = [];

function tempRoot(): string {
  const selected = fs.realpathSync.native(
    fs.mkdtempSync(path.join(os.tmpdir(), "agent-libos-desktop-runtime-"))
  );
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
      runtimeDirectory: path.join(userData, "runtime"),
      workspaceDirectory: path.join(userData, "workspace")
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

  it("creates a separate owner-only packaged workspace", () => {
    const userData = path.join(tempRoot(), "user-data");
    const layout = packagedRuntimeLayout(path.join(tempRoot(), "resources"), userData, "darwin");

    ensurePrivateRuntimeDirectory(layout.runtimeDirectory, "darwin");
    ensurePrivateWorkspaceDirectory(layout.workspaceDirectory, "darwin");

    expect(fs.realpathSync(path.dirname(layout.databaseFile)))
      .not.toBe(fs.realpathSync(layout.workspaceDirectory));
    expect(fs.statSync(layout.workspaceDirectory).mode & 0o777).toBe(0o700);
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

  it("passes the user database target explicitly in development", () => {
    const profiles = path.join(tempRoot(), "llm-profiles.json");
    expect(developmentRuntimeArguments(profiles)).toEqual([
      "--db", "user",
      "--port", "0",
      "--llm-profiles-file", profiles
    ]);
    expect(developmentRuntimeArguments(profiles, ":memory:").slice(0, 2)).toEqual(["--db", ":memory:"]);
  });
});

describe("desktop Runtime database isolation", () => {
  it("allows an external custom SQLite database", () => {
    const root = tempRoot();
    const workspace = path.join(root, "workspace");
    const database = path.join(root, "database", "agent-libos.sqlite");
    fs.mkdirSync(workspace);
    fs.mkdirSync(path.dirname(database));
    fs.writeFileSync(database, "", "utf8");

    expect(() => assertDatabaseOutsideWorkspace(database, workspace, "darwin")).not.toThrow();
    expect(() => assertDatabaseOutsideWorkspace("user", workspace, "darwin")).not.toThrow();
    expect(() => assertDatabaseOutsideWorkspace("local", workspace, "darwin")).not.toThrow();
    expect(() => assertDatabaseOutsideWorkspace("sqlite://", workspace, "darwin")).not.toThrow();
    expect(() => assertDatabaseOutsideWorkspace("postgresql://runtime.example/agent", workspace, "darwin"))
      .not.toThrow();
  });

  it("rejects a custom SQLite database inside the effective workspace", () => {
    const workspace = path.join(tempRoot(), "workspace");
    const database = path.join(workspace, "state", "agent-libos.sqlite");
    fs.mkdirSync(path.dirname(database), { recursive: true });
    fs.writeFileSync(database, "", "utf8");

    expect(() => assertDatabaseOutsideWorkspace(database, workspace, "darwin"))
      .toThrow("Selected database must be outside the Runtime workspace.");
    expect(() => assertDatabaseOutsideWorkspace(path.join("state", "new.sqlite"), workspace, "darwin"))
      .toThrow("Selected database must be outside the Runtime workspace.");
    expect(fs.existsSync(`${database}-wal`)).toBe(false);
    expect(fs.existsSync(`${database}-shm`)).toBe(false);
  });

  it("rejects a symlink alias to a database inside the effective workspace", () => {
    const root = tempRoot();
    const workspace = path.join(root, "workspace");
    const database = path.join(workspace, "agent-libos.sqlite");
    const aliasDirectory = path.join(root, "selected");
    fs.mkdirSync(workspace);
    fs.writeFileSync(database, "", "utf8");
    fs.symlinkSync(workspace, aliasDirectory, process.platform === "win32" ? "junction" : "dir");

    expect(() => assertDatabaseOutsideWorkspace(path.join(aliasDirectory, path.basename(database)), workspace))
      .toThrow("Selected database path must not contain a symlink or reparse point.");
  });

  it("rejects a dangling database symlink to a missing workspace leaf", () => {
    const root = tempRoot();
    const workspace = path.join(root, "workspace");
    const aliasDatabase = path.join(root, "selected.sqlite");
    fs.mkdirSync(workspace);
    fs.symlinkSync(path.join(workspace, "missing.sqlite"), aliasDatabase, "file");

    expect(() => assertDatabaseOutsideWorkspace(aliasDatabase, workspace))
      .toThrow("Selected database path must not contain a symlink or reparse point.");
  });

  it("decodes SQLite file URIs and checks their filesystem paths", () => {
    const root = tempRoot();
    const workspace = path.join(root, "workspace");
    const external = path.join(root, "external database", "agent-libos.sqlite");
    const internal = path.join(workspace, "state", "agent-libos.sqlite");
    fs.mkdirSync(workspace);
    fs.mkdirSync(path.dirname(external), { recursive: true });
    fs.writeFileSync(external, "", "utf8");

    expect(() => assertDatabaseOutsideWorkspace(`sqlite://${encodeURI(external)}`, workspace, "darwin"))
      .not.toThrow();
    expect(() => assertDatabaseOutsideWorkspace(
      `sqlite:////${encodeURI(internal).replace(/^\/+/, "")}`,
      workspace,
      "darwin"
    )).toThrow("Selected database must be outside the Runtime workspace.");
  });

  it("requires exact reserved targets and rejects unsupported or malformed URIs", () => {
    const workspace = path.join(tempRoot(), "workspace");
    fs.mkdirSync(workspace);

    expect(() => assertDatabaseOutsideWorkspace(" user ", workspace, "darwin"))
      .toThrow("Selected database must be outside the Runtime workspace.");
    expect(() => assertDatabaseOutsideWorkspace("https://example.test/runtime.sqlite", workspace, "darwin"))
      .toThrow("Unsupported Runtime store target.");
    expect(() => assertDatabaseOutsideWorkspace("sqlite:///%ZZ/runtime.sqlite", workspace, "darwin"))
      .toThrow("Unsupported Runtime store target.");
  });

  it("does not fold case-distinct future sibling components during Windows preflight", () => {
    const root = tempRoot();
    const workspace = path.join(root, "CaseSensitive");
    const siblingDatabase = path.join(root, "casesensitive", "agent-libos.sqlite");

    expect(() => assertDatabaseOutsideWorkspace(siblingDatabase, workspace, "win32")).not.toThrow();
  });
});
