import * as fs from "node:fs";
import * as path from "node:path";

export const imagePackageMaxBytes = 16_777_216;
export const imagePackageMaxFiles = 512;
export const imagePackageMaxDirectories = 512;
export const imagePackageMaxDepth = 32;
export const imagePackageMaxEntries = imagePackageMaxFiles + imagePackageMaxDirectories;
export const imagePackageMaxEntryNameBytes = imagePackageMaxEntries * 512;

export function readImagePackageFiles(root: string) {
  const files = Object.create(null) as Record<string, { base64: string }>;
  let totalBytes = 0n;
  let totalFiles = 0;
  let totalDirectories = 0;
  let totalEntries = 0;
  let totalEntryNameBytes = 0;
  const selectedRoot = fs.lstatSync(root, { bigint: true });
  if (selectedRoot.isSymbolicLink()) {
    throw new Error("Image package root must not be a symbolic link.");
  }
  if (!selectedRoot.isDirectory()) throw new Error("Image package root is not a directory.");
  const canonicalRoot = fs.realpathSync(root);
  const canonicalRootStats = fs.statSync(canonicalRoot, { bigint: true });
  if (!sameIdentity(selectedRoot, canonicalRootStats) || !canonicalRootStats.isDirectory()) {
    throw new Error("Image package root changed during validation.");
  }

  function visit(directory: string, depth: number) {
    totalDirectories += 1;
    if (totalDirectories > imagePackageMaxDirectories) {
      throw new Error(`Image package exceeds ${imagePackageMaxDirectories} directories.`);
    }
    if (depth > imagePackageMaxDepth) {
      throw new Error(`Image package exceeds directory depth ${imagePackageMaxDepth}.`);
    }
    const canonicalDirectory = fs.realpathSync(directory);
    if (!isPathInsideOrEqual(canonicalRoot, canonicalDirectory)) {
      throw new Error("Image package directory resolves outside the package root.");
    }
    const beforeDirectory = fs.lstatSync(directory, { bigint: true });
    const resolvedDirectory = fs.statSync(canonicalDirectory, { bigint: true });
    if (
      beforeDirectory.isSymbolicLink() ||
      !beforeDirectory.isDirectory() ||
      !resolvedDirectory.isDirectory() ||
      !sameIdentity(beforeDirectory, resolvedDirectory)
    ) {
      throw new Error("Image package directory changed during validation.");
    }
    const entries = fs.opendirSync(directory);
    try {
      for (let entry = entries.readSync(); entry !== null; entry = entries.readSync()) {
        totalEntries += 1;
        if (totalEntries > imagePackageMaxEntries) {
          throw new Error(`Image package exceeds ${imagePackageMaxEntries} directory entries.`);
        }
        totalEntryNameBytes += Buffer.byteLength(entry.name, "utf8");
        if (totalEntryNameBytes > imagePackageMaxEntryNameBytes) {
          throw new Error(`Image package entry names exceed ${imagePackageMaxEntryNameBytes} bytes.`);
        }
        const fullPath = path.join(directory, entry.name);
        const relative = path.relative(canonicalRoot, fullPath).split(path.sep).join("/");
        if (relative.split("/").some((segment) => segment.toLowerCase() === ".git")) {
          throw new Error("Image packages must not include .git directories.");
        }
        if (entry.isSymbolicLink()) throw new Error(`Image package symlinks are not supported: ${relative}`);
        if (entry.isDirectory()) {
          visit(fullPath, depth + 1);
          continue;
        }
        if (!entry.isFile()) throw new Error(`Image package path is not a regular file: ${relative}`);
        const stats = fs.lstatSync(fullPath, { bigint: true });
        if (stats.nlink > 1) throw new Error(`Image package hard links are not supported: ${relative}`);
        totalFiles += 1;
        if (totalFiles > imagePackageMaxFiles) {
          throw new Error(`Image package exceeds ${imagePackageMaxFiles} files.`);
        }
        const remainingBytes = BigInt(imagePackageMaxBytes) - totalBytes;
        if (stats.size > remainingBytes) {
          throw new Error(`Image package exceeds ${imagePackageMaxBytes} bytes.`);
        }
        const content = readPackageFile(fullPath, relative, stats, remainingBytes, canonicalRoot);
        totalBytes += BigInt(content.byteLength);
        files[relative] = { base64: content.toString("base64") };
      }
    } finally {
      entries.closeSync();
    }
    const afterDirectory = fs.lstatSync(directory, { bigint: true });
    if (!sameStableSnapshot(beforeDirectory, afterDirectory)) {
      throw new Error("Image package directory changed during read.");
    }
  }

  visit(canonicalRoot, 0);
  if (!files["IMAGE.yaml"]) {
    throw new Error("Image package is missing IMAGE.yaml.");
  }
  return files;
}

function readPackageFile(
  fullPath: string,
  relative: string,
  before: fs.BigIntStats,
  remainingBytes: bigint,
  canonicalRoot: string
) {
  const noFollow = "O_NOFOLLOW" in fs.constants ? fs.constants.O_NOFOLLOW : 0;
  const nonBlocking = "O_NONBLOCK" in fs.constants ? fs.constants.O_NONBLOCK : 0;
  const fd = fs.openSync(fullPath, fs.constants.O_RDONLY | noFollow | nonBlocking);
  try {
    const opened = fs.fstatSync(fd, { bigint: true });
    if (!opened.isFile()) throw new Error(`Image package path is not a regular file: ${relative}`);
    if (opened.nlink > 1) throw new Error(`Image package hard links are not supported: ${relative}`);
    if (!sameStableSnapshot(before, opened)) {
      throw new Error(`Image package file changed during read: ${relative}`);
    }
    if (opened.size > remainingBytes) {
      throw new Error(`Image package exceeds ${imagePackageMaxBytes} bytes.`);
    }
    verifyDescriptorPath(fullPath, relative, canonicalRoot, opened);
    const content = readDescriptorExactly(fd, Number(opened.size), relative);
    const after = fs.fstatSync(fd, { bigint: true });
    if (!sameStableSnapshot(opened, after)) {
      throw new Error(`Image package file changed during read: ${relative}`);
    }
    verifyDescriptorPath(fullPath, relative, canonicalRoot, after);
    return content;
  } finally {
    fs.closeSync(fd);
  }
}

function readDescriptorExactly(fd: number, expectedBytes: number, relative: string): Buffer {
  const content = Buffer.allocUnsafe(expectedBytes);
  let offset = 0;
  while (offset < expectedBytes) {
    const read = fs.readSync(fd, content, offset, expectedBytes - offset, offset);
    if (read === 0) throw new Error(`Image package file changed during read: ${relative}`);
    offset += read;
  }
  const extra = Buffer.allocUnsafe(1);
  if (fs.readSync(fd, extra, 0, 1, expectedBytes) !== 0) {
    throw new Error(`Image package file changed during read: ${relative}`);
  }
  return content;
}

function verifyDescriptorPath(
  fullPath: string,
  relative: string,
  canonicalRoot: string,
  descriptor: fs.BigIntStats
): void {
  const resolved = fs.realpathSync(fullPath);
  if (!isPathInsideOrEqual(canonicalRoot, resolved) || resolved !== path.resolve(fullPath)) {
    throw new Error(`Image package file resolves outside its validated directory: ${relative}`);
  }
  const pathname = fs.statSync(resolved, { bigint: true });
  if (!sameStableSnapshot(descriptor, pathname)) {
    throw new Error(`Image package file changed during read: ${relative}`);
  }
}

function sameIdentity(left: fs.BigIntStats, right: fs.BigIntStats): boolean {
  return left.dev === right.dev && left.ino === right.ino;
}

function sameStableSnapshot(left: fs.BigIntStats, right: fs.BigIntStats): boolean {
  return sameIdentity(left, right) &&
    left.size === right.size &&
    left.mtimeNs === right.mtimeNs &&
    left.ctimeNs === right.ctimeNs;
}

function isPathInsideOrEqual(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (relative !== ".." && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative));
}
