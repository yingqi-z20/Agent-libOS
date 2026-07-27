import type { IpcMainInvokeEvent, Session, WebContents } from "electron";

export function installDefaultDenyPermissions(session: Session): void {
  session.setPermissionCheckHandler(() => false);
  session.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false));
  session.setDevicePermissionHandler(() => false);
}

type RendererNavigationEvent = {
  preventDefault(): void;
  url: string;
};

export function installTrustedRendererNavigationGuard(
  webContents: WebContents,
  trustedRendererUrl: string
): void {
  const guard = (event: RendererNavigationEvent) => {
    enforceTrustedRendererNavigation(event, event.url, trustedRendererUrl);
  };
  webContents.on("will-navigate", guard);
  webContents.on("will-redirect", guard);
}

export function enforceTrustedRendererNavigation(
  event: Pick<RendererNavigationEvent, "preventDefault">,
  candidateUrl: string,
  trustedRendererUrl: string
): boolean {
  if (sameRendererOrigin(candidateUrl, trustedRendererUrl)) return true;
  event.preventDefault();
  return false;
}

export function assertTrustedIpcSender(
  event: IpcMainInvokeEvent,
  expectedWebContents: WebContents | null,
  trustedRendererUrl: string
): void {
  const senderFrame = event.senderFrame;
  if (
    !expectedWebContents ||
    event.sender !== expectedWebContents ||
    !senderFrame ||
    senderFrame !== event.sender.mainFrame ||
    !sameRendererOrigin(senderFrame.url, trustedRendererUrl)
  ) {
    throw new Error("Untrusted IPC sender.");
  }
}

export function sameRendererOrigin(candidate: string, trusted: string): boolean {
  const candidateOrigin = rendererOrigin(candidate);
  const trustedOrigin = rendererOrigin(trusted);
  return candidateOrigin !== null && trustedOrigin !== null && candidateOrigin === trustedOrigin;
}

function rendererOrigin(rawUrl: string): string | null {
  try {
    const parsed = new URL(rawUrl);
    if (parsed.username || parsed.password || !parsed.hostname) return null;
    return `${parsed.protocol.toLowerCase()}//${parsed.hostname.toLowerCase()}${parsed.port ? `:${parsed.port}` : ""}`;
  } catch {
    return null;
  }
}
