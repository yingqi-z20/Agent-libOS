import * as fs from "node:fs";
import * as path from "node:path";

export const imagePackageMaxBytes = 16_777_216;
export const imagePackageMaxFiles = 512;
export const imagePackageMaxDirectories = 512;
export const imagePackageMaxDepth = 32;

export function readImagePackageFiles(root: string) {
  const files: Record<string, { base64: string }> = {};
  let totalBytes = 0;
  let totalDirectories = 0;

  function visit(directory: string, depth: number) {
    totalDirectories += 1;
    if (totalDirectories > imagePackageMaxDirectories) {
      throw new Error(`Image package exceeds ${imagePackageMaxDirectories} directories.`);
    }
    if (depth > imagePackageMaxDepth) {
      throw new Error(`Image package exceeds directory depth ${imagePackageMaxDepth}.`);
    }
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const fullPath = path.join(directory, entry.name);
      const relative = path.relative(root, fullPath).split(path.sep).join("/");
      if (relative.split("/").includes(".git")) {
        throw new Error("Image packages must not include .git directories.");
      }
      if (entry.isSymbolicLink()) throw new Error(`Image package symlinks are not supported: ${relative}`);
      if (entry.isDirectory()) {
        visit(fullPath, depth + 1);
        continue;
      }
      if (!entry.isFile()) throw new Error(`Image package path is not a regular file: ${relative}`);
      const stats = fs.statSync(fullPath);
      totalBytes += stats.size;
      if (Object.keys(files).length + 1 > imagePackageMaxFiles) {
        throw new Error(`Image package exceeds ${imagePackageMaxFiles} files.`);
      }
      if (totalBytes > imagePackageMaxBytes) {
        throw new Error(`Image package exceeds ${imagePackageMaxBytes} bytes.`);
      }
      const content = fs.readFileSync(fullPath);
      files[relative] = { base64: content.toString("base64") };
    }
  }

  visit(root, 0);
  return files;
}
