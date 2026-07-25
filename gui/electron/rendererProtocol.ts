import * as fs from "node:fs";
import * as path from "node:path";

export const productionRendererScheme = "agent-libos";
export const productionRendererHost = "app";
export const productionRendererOrigin = `${productionRendererScheme}://${productionRendererHost}`;
export const productionRendererEntryUrl = `${productionRendererOrigin}/index.html`;

export type ProductionRendererAsset = {
  body: ArrayBuffer;
  contentType: string;
};

export async function readProductionRendererAsset(
  distRoot: string,
  requestUrl: string
): Promise<ProductionRendererAsset | null> {
  const candidate = productionRendererCandidate(distRoot, requestUrl);
  if (candidate === null) return null;

  let canonicalRoot: string;
  let canonicalCandidate: string;
  try {
    canonicalRoot = await fs.promises.realpath(path.resolve(distRoot));
    if (!(await fs.promises.stat(canonicalRoot)).isDirectory()) return null;
    canonicalCandidate = await fs.promises.realpath(candidate);
    if (!isPathInside(canonicalRoot, canonicalCandidate)) return null;
    if (!(await fs.promises.stat(canonicalCandidate)).isFile()) return null;
  } catch {
    return null;
  }

  let handle: fs.promises.FileHandle | null = null;
  try {
    const noFollow = fs.constants.O_NOFOLLOW ?? 0;
    const nonBlocking = fs.constants.O_NONBLOCK ?? 0;
    handle = await fs.promises.open(canonicalCandidate, fs.constants.O_RDONLY | noFollow | nonBlocking);
    const openedStats = await handle.stat({ bigint: true });
    if (!openedStats.isFile()) return null;

    // Re-resolve after opening and compare the pathname with the descriptor.
    // The response is read from the descriptor, so a later pathname swap cannot
    // redirect the custom protocol outside the distribution root.
    const verifiedCandidate = await fs.promises.realpath(canonicalCandidate);
    if (!isPathInside(canonicalRoot, verifiedCandidate)) return null;
    const verifiedStats = await fs.promises.stat(verifiedCandidate, { bigint: true });
    if (openedStats.dev !== verifiedStats.dev || openedStats.ino !== verifiedStats.ino) return null;

    const bytes = await handle.readFile();
    const body = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(body).set(bytes);
    return {
      body,
      contentType: productionRendererContentType(verifiedCandidate)
    };
  } catch {
    return null;
  } finally {
    await handle?.close().catch(() => undefined);
  }
}

function productionRendererCandidate(distRoot: string, requestUrl: string): string | null {
  const authorityMarker = "://";
  const authorityIndex = requestUrl.indexOf(authorityMarker);
  if (authorityIndex < 0) return null;
  const rawPathIndex = requestUrl.indexOf("/", authorityIndex + authorityMarker.length);
  const rawPath = (rawPathIndex < 0 ? "/" : requestUrl.slice(rawPathIndex)).split(/[?#]/, 1)[0];
  try {
    if (decodeURIComponent(rawPath).replace(/\\/g, "/").split("/").includes("..")) return null;
  } catch {
    return null;
  }
  let parsed: URL;
  try {
    parsed = new URL(requestUrl);
  } catch {
    return null;
  }
  if (
    parsed.protocol !== `${productionRendererScheme}:` ||
    parsed.hostname !== productionRendererHost ||
    parsed.username !== "" ||
    parsed.password !== "" ||
    parsed.port !== ""
  ) {
    return null;
  }
  let pathname: string;
  try {
    pathname = decodeURIComponent(parsed.pathname);
  } catch {
    return null;
  }
  if (pathname.includes("\0")) return null;
  const relative = pathname === "/" ? "index.html" : pathname.replace(/^\/+/, "");
  const root = path.resolve(distRoot);
  const candidate = path.resolve(root, relative);
  if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) return null;
  return candidate;
}

function isPathInside(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative !== "" && relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function productionRendererContentType(filePath: string): string {
  switch (path.extname(filePath).toLowerCase()) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".js":
    case ".mjs":
      return "text/javascript; charset=utf-8";
    case ".json":
    case ".map":
      return "application/json; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".gif":
      return "image/gif";
    case ".webp":
      return "image/webp";
    case ".ico":
      return "image/x-icon";
    case ".woff":
      return "font/woff";
    case ".woff2":
      return "font/woff2";
    case ".ttf":
      return "font/ttf";
    case ".wasm":
      return "application/wasm";
    default:
      return "application/octet-stream";
  }
}
