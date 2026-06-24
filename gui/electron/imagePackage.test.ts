import { afterEach, describe, expect, it } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { imagePackageMaxDepth, readImagePackageFiles } from "./imagePackage.js";

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
});
