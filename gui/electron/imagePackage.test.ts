import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { imagePackageMaxBytes, imagePackageMaxDepth, readImagePackageFiles } from "./imagePackage.js";

const roots: string[] = [];

function tempRoot() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "agent-libos-image-package-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length > 0) {
    fs.rmSync(roots.pop()!, { recursive: true, force: true });
  }
});

describe("readImagePackageFiles", () => {
  it("reads normal package files", () => {
    const root = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");

    const files = readImagePackageFiles(root);

    expect(Buffer.from(files["IMAGE.yaml"].base64, "base64").toString("utf8")).toContain("test:v0");
  });

  it("rejects case-insensitive .git segments before reading package contents", () => {
    const root = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    const gitDirectory = path.join(root, ".GIT");
    fs.mkdirSync(gitDirectory);
    fs.writeFileSync(path.join(gitDirectory, "config"), "credential=secret", "utf8");

    expect(() => readImagePackageFiles(root)).toThrow(/must not include \.git/i);
  });

  it("rejects excessively deep package trees before reading contents", () => {
    const root = tempRoot();
    let current = root;
    for (let index = 0; index <= imagePackageMaxDepth + 1; index += 1) {
      current = path.join(current, `d${index}`);
      fs.mkdirSync(current);
    }
    fs.writeFileSync(path.join(current, "leaf.txt"), "leaf", "utf8");

    expect(() => readImagePackageFiles(root)).toThrow(/directory depth/);
  });

  it("rejects symlinked package files", () => {
    const root = tempRoot();
    const outside = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    fs.writeFileSync(path.join(outside, "secret.txt"), "secret", "utf8");
    try {
      fs.symlinkSync(path.join(outside, "secret.txt"), path.join(root, "linked-secret.txt"));
    } catch {
      return;
    }

    expect(() => readImagePackageFiles(root)).toThrow(/symlinks/);
  });

  it("rejects hardlinked package files", () => {
    const root = tempRoot();
    const outside = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    fs.writeFileSync(path.join(outside, "secret.txt"), "secret", "utf8");
    try {
      fs.linkSync(path.join(outside, "secret.txt"), path.join(root, "linked-secret.txt"));
    } catch {
      return;
    }

    expect(() => readImagePackageFiles(root)).toThrow(/hard links/);
  });

  it("rejects a symlink used as the package root", () => {
    const root = tempRoot();
    const linkedRoot = path.join(tempRoot(), "linked-package");
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    try {
      fs.symlinkSync(root, linkedRoot, "dir");
    } catch {
      return;
    }

    expect(() => readImagePackageFiles(linkedRoot)).toThrow(/root.*symbolic link/i);
  });

  it("rejects an oversized sparse file before allocating its contents", () => {
    const root = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    const oversized = path.join(root, "oversized.bin");
    fs.closeSync(fs.openSync(oversized, "w"));
    fs.truncateSync(oversized, imagePackageMaxBytes + 1);

    expect(() => readImagePackageFiles(root)).toThrow(/bytes/);
  });

  it("returns reserved file names as own data properties", () => {
    const root = tempRoot();
    fs.writeFileSync(path.join(root, "IMAGE.yaml"), "image:\n  image_id: test:v0\n", "utf8");
    fs.writeFileSync(path.join(root, "__proto__"), "package data", "utf8");

    const files = readImagePackageFiles(root);

    expect(Object.prototype.hasOwnProperty.call(files, "__proto__")).toBe(true);
    expect(Buffer.from(files["__proto__"].base64, "base64").toString("utf8")).toBe("package data");
  });
});
