export type WorkspaceAccess = "none" | "read" | "edit" | "manage";
export type CommandAccess = "none" | "reviewed";
export const DEFAULT_CONTEXT_MAINTENANCE = false;

type CapabilitySpec = {
  resource: string;
  rights: string[];
  delegable: false;
  constraints?: Record<string, unknown>;
};

export type GuiTaskAuthorityOptions = {
  workingDirectory: string;
  workspaceAccess: WorkspaceAccess;
  allowGitRequests: boolean;
  commandAccess: CommandAccess;
  contextMaintenance: boolean;
};

const WORKSPACE_RIGHTS: Record<Exclude<WorkspaceAccess, "none">, string[]> = {
  read: ["read"],
  edit: ["read", "write"],
  manage: ["read", "write", "delete"]
};

export function normalizeWorkspaceDirectory(path: string): string {
  const raw = path.replaceAll("\\", "/").trim();
  if (!raw || raw === ".") return ".";
  if (raw.startsWith("/") || /^[A-Za-z]:\//.test(raw)) {
    throw new Error(`Working directory must be workspace-relative: ${path}`);
  }

  const parts: string[] = [];
  for (const part of raw.split("/")) {
    if (!part || part === ".") continue;
    if (part === "..") {
      if (parts.length === 0) {
        throw new Error(`Working directory escapes workspace root: ${path}`);
      }
      parts.pop();
      continue;
    }
    parts.push(part);
  }
  return parts.join("/") || ".";
}

export function workspaceResourceForDirectory(path: string): string {
  const normalized = normalizeWorkspaceDirectory(path);
  if (normalized === ".") return "filesystem:workspace:*";
  const encoded = normalized.split("/").map(encodeResourceSegment).join("/");
  return `filesystem:workspace:${encoded}/*`;
}

export function buildGuiTaskAuthorityManifest({
  workingDirectory,
  workspaceAccess,
  allowGitRequests,
  commandAccess,
  contextMaintenance
}: GuiTaskAuthorityOptions): Record<string, unknown> {
  const authorizedCapabilities: CapabilitySpec[] = [
    { resource: "human:owner", rights: ["write"], delegable: false }
  ];
  if (commandAccess === "reviewed") {
    authorizedCapabilities.push({
      resource: "shell:*",
      rights: ["execute"],
      delegable: false,
      constraints: { shell_policy_level: "allowlist_auto_else_ask" }
    });
  }
  if (contextMaintenance) {
    authorizedCapabilities.push(
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
    );
  }
  const requestableCapabilities: CapabilitySpec[] = [];
  if (workspaceAccess !== "none") {
    requestableCapabilities.push({
      resource: workspaceResourceForDirectory(workingDirectory),
      rights: [...WORKSPACE_RIGHTS[workspaceAccess]],
      delegable: false
    });
  }
  if (allowGitRequests) {
    requestableCapabilities.push(
      { resource: "shell:git", rights: ["execute"], delegable: false },
      { resource: "git:workspace", rights: ["read", "diff", "write"], delegable: false }
    );
  }

  return {
    authorized_capabilities: authorizedCapabilities,
    approval_policy: {
      requestable_capabilities: requestableCapabilities
    },
    metadata: {
      policy: "gui-task-v2",
      workspace_access: workspaceAccess,
      git_requests: allowGitRequests,
      command_access: commandAccess,
      context_maintenance: contextMaintenance
    }
  };
}

function encodeResourceSegment(segment: string): string {
  return encodeURIComponent(segment).replace(/[!'()*]/g, (character) => (
    `%${character.charCodeAt(0).toString(16).toUpperCase()}`
  ));
}
