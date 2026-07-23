import { describe, expect, it } from "vitest";
import {
  buildGuiTaskAuthorityManifest,
  normalizeWorkspaceDirectory,
  workspaceResourceForDirectory
} from "./taskAuthority";

describe("GUI task authority", () => {
  it("normalizes the task directory with the runtime's workspace-relative rules", () => {
    expect(normalizeWorkspaceDirectory(" src/./feature/../app ")).toBe("src/app");
    expect(normalizeWorkspaceDirectory("src\\app")).toBe("src/app");
    expect(normalizeWorkspaceDirectory("")).toBe(".");
    expect(() => normalizeWorkspaceDirectory("../../outside")).toThrow(/escapes workspace root/);
    expect(() => normalizeWorkspaceDirectory("/private/tmp")).toThrow(/workspace-relative/);
    expect(() => normalizeWorkspaceDirectory("C:\\temp")).toThrow(/workspace-relative/);
  });

  it("encodes a least-privilege subtree resource for a nested cwd", () => {
    expect(workspaceResourceForDirectory("work items/review#1")).toBe(
      "filesystem:workspace:work%20items/review%231/*"
    );
    expect(workspaceResourceForDirectory(".")).toBe("filesystem:workspace:*");
  });

  it("pregrants only Human communication and keeps filesystem access request-only", () => {
    const manifest = buildGuiTaskAuthorityManifest({
      workingDirectory: "packages/gui",
      workspaceAccess: "edit",
      allowGitRequests: false
    });

    expect(manifest.authorized_capabilities).toEqual([
      { resource: "human:owner", rights: ["write"], delegable: false }
    ]);
    expect(manifest.approval_policy).toEqual({
      requestable_capabilities: [
        {
          resource: "filesystem:workspace:packages/gui/*",
          rights: ["read", "write"],
          delegable: false
        }
      ]
    });
  });

  it("adds deletion and local Git request ceilings without granting them", () => {
    const manifest = buildGuiTaskAuthorityManifest({
      workingDirectory: ".",
      workspaceAccess: "manage",
      allowGitRequests: true
    });
    const requestable = (manifest.approval_policy as { requestable_capabilities: unknown[] })
      .requestable_capabilities;

    expect(requestable).toEqual([
      {
        resource: "filesystem:workspace:*",
        rights: ["read", "write", "delete"],
        delegable: false
      },
      { resource: "shell:git", rights: ["execute"], delegable: false },
      { resource: "git:workspace", rights: ["read", "diff", "write"], delegable: false }
    ]);
    expect(manifest.authorized_capabilities).not.toContainEqual(
      expect.objectContaining({ resource: "filesystem:workspace:*" })
    );
  });

  it("supports a communication-only task with an empty request ceiling", () => {
    const manifest = buildGuiTaskAuthorityManifest({
      workingDirectory: ".",
      workspaceAccess: "none",
      allowGitRequests: false
    });
    expect(manifest.approval_policy).toEqual({ requestable_capabilities: [] });
  });
});
