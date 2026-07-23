export type WorkspaceAccess = "none" | "read" | "edit" | "manage";

type CapabilitySpec = {
  resource: string;
  rights: string[];
  delegable: false;
};

export type GuiTaskAuthorityOptions = {
  workingDirectory: string;
  workspaceAccess: WorkspaceAccess;
  allowGitRequests: boolean;
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
  allowGitRequests
}: GuiTaskAuthorityOptions): Record<string, unknown> {
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
    authorized_capabilities: [
      { resource: "human:owner", rights: ["write"], delegable: false }
    ],
    approval_policy: {
      requestable_capabilities: requestableCapabilities
    },
    metadata: {
      policy: "gui-task-v1",
      workspace_access: workspaceAccess,
      git_requests: allowGitRequests
    }
  };
}

function encodeResourceSegment(segment: string): string {
  return encodeURIComponent(segment).replace(/[!'()*]/g, (character) => (
    `%${character.charCodeAt(0).toString(16).toUpperCase()}`
  ));
}
