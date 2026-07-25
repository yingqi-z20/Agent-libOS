import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import {
  productionRendererEntryUrl,
  productionRendererOrigin,
  readProductionRendererAsset
} from "./rendererProtocol.js";

const roots: string[] = [];

function tempRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-libos-renderer-protocol-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length > 0) {
    fs.rmSync(roots.pop()!, { recursive: true, force: true });
  }
});

describe("production renderer protocol", () => {
  it("uses the exact origin allowed by the GUI server", () => {
    expect(productionRendererOrigin).toBe("agent-libos://app");
    expect(productionRendererEntryUrl).toBe("agent-libos://app/index.html");
  });

  it("reads regular app assets inside dist with their browser content types", async () => {
    const root = tempRoot();
    fs.mkdirSync(path.join(root, "assets"));
    fs.writeFileSync(path.join(root, "index.html"), "<main>Agent libOS</main>", "utf8");
    fs.writeFileSync(path.join(root, "assets", "app.js"), "export const ready = true;", "utf8");

    const entry = await readProductionRendererAsset(root, "agent-libos://app/");
    const script = await readProductionRendererAsset(root, "agent-libos://app/assets/app.js");

    expect(Buffer.from(entry!.body).toString("utf8")).toBe("<main>Agent libOS</main>");
    expect(entry!.contentType).toBe("text/html; charset=utf-8");
    expect(Buffer.from(script!.body).toString("utf8")).toBe("export const ready = true;");
    expect(script!.contentType).toBe("text/javascript; charset=utf-8");
  });

  it("rejects other origins, traversal, missing files, and directories", async () => {
    const root = tempRoot();
    fs.mkdirSync(path.join(root, "assets"));
    fs.writeFileSync(path.join(root, "index.html"), "ok", "utf8");

    await expect(readProductionRendererAsset(root, "agent-libos://untrusted/index.html")).resolves.toBeNull();
    await expect(readProductionRendererAsset(root, "agent-libos://user@app/index.html")).resolves.toBeNull();
    await expect(readProductionRendererAsset(root, "agent-libos://app:123/index.html")).resolves.toBeNull();
    await expect(readProductionRendererAsset(root, "agent-libos://app/%2e%2e/secret.txt")).resolves.toBeNull();
    await expect(readProductionRendererAsset(root, "agent-libos://app/missing.js")).resolves.toBeNull();
    await expect(readProductionRendererAsset(root, "agent-libos://app/assets")).resolves.toBeNull();
  });

  it("rejects a dist symlink that resolves outside the canonical root", async () => {
    const workspace = tempRoot();
    const root = path.join(workspace, "dist");
    const outside = path.join(workspace, "outside");
    fs.mkdirSync(root);
    fs.mkdirSync(outside);
    fs.writeFileSync(path.join(outside, "secret.txt"), "must-not-be-served", "utf8");
    fs.symlinkSync(outside, path.join(root, "escape"), process.platform === "win32" ? "junction" : "dir");

    await expect(readProductionRendererAsset(root, "agent-libos://app/escape/secret.txt")).resolves.toBeNull();
  });
});
