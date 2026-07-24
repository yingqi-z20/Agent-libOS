import { describe, expect, it } from "vitest";
import {
  buildGuiTaskAuthorityManifest,
  DEFAULT_CONTEXT_MAINTENANCE,
  normalizeWorkspaceDirectory,
  workspaceResourceForDirectory
} from "./taskAuthority";

describe("GUI task authority", () => {
  it("keeps persistent context enrichment disabled by default", () => {
    expect(DEFAULT_CONTEXT_MAINTENANCE).toBe(false);
  });

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
      allowGitRequests: false,
      commandAccess: "none",
      contextMaintenance: false
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
      allowGitRequests: true,
      commandAccess: "none",
      contextMaintenance: false
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
      allowGitRequests: false,
      commandAccess: "none",
      contextMaintenance: false
    });
    expect(manifest.approval_policy).toEqual({ requestable_capabilities: [] });
  });

  it("adds only the constrained reviewed-command policy when explicitly enabled", () => {
    const manifest = buildGuiTaskAuthorityManifest({
      workingDirectory: ".",
      workspaceAccess: "none",
      allowGitRequests: false,
      commandAccess: "reviewed",
      contextMaintenance: false
    });

    expect(manifest.authorized_capabilities).toEqual([
      { resource: "human:owner", rights: ["write"], delegable: false },
      {
        resource: "shell:*",
        rights: ["execute"],
        delegable: false,
        constraints: { shell_policy_level: "allowlist_auto_else_ask" }
      }
    ]);
    expect(manifest.approval_policy).toEqual({ requestable_capabilities: [] });
  });

  it("binds context maintenance to the compressor child image and spawn operation", () => {
    const manifest = buildGuiTaskAuthorityManifest({
      workingDirectory: ".",
      workspaceAccess: "none",
      allowGitRequests: false,
      commandAccess: "none",
      contextMaintenance: true
    });

    expect(manifest.authorized_capabilities).toEqual([
      { resource: "human:owner", rights: ["write"], delegable: false },
      {
        resource: "context:enrichment",
        rights: ["execute"],
        delegable: false
      },
      {
        resource: "context:maintenance",
        rights: ["execute"],
        delegable: false
      },
      {
        resource: "process:spawn",
        rights: ["write"],
        delegable: false,
        constraints: {
          authority_rules: [
            {
              rule_id: "gui.context-maintenance.spawn",
              operation: "process.spawn_child",
              effect: "allow",
              risk: "low",
              conditions: { image_id: "context-compressor:v0" },
              description: "allow only the built-in context compressor child"
            }
          ]
        }
      },
      {
        resource: "image:context-compressor:v0",
        rights: ["read"],
        delegable: false
      }
    ]);
  });
});
