function integerValue(value: unknown, fallback: number | undefined, field: string): number {
  const selected = value === undefined ? fallback : Number(value);
  if (selected === undefined || !Number.isFinite(selected) || !Number.isInteger(selected)) {
    throw new Error(`${field} must be an integer`);
  }
  return selected;
}

function positionsOf(text: string, needle: string): number[] {
  const result: number[] = [];
  let index = 0;
  while (true) {
    const found = text.indexOf(needle, index);
    if (found < 0) return result;
    result.push(found);
    index = found + needle.length;
  }
}

function lineLayout(text: string): { starts: number[]; ends: number[]; newline: string } {
  const starts = [0];
  const ends: number[] = [];
  const pattern = /\r\n|\n/g;
  let newline = "\n";
  let first = true;
  while (true) {
    const match = pattern.exec(text);
    if (match === null) break;
    if (first) {
      newline = match[0];
      first = false;
    }
    ends.push(match.index);
    starts.push(match.index + match[0].length);
  }
  ends.push(text.length);
  return { starts, ends, newline };
}

function normalizeNewlines(text: string, newline: string): string {
  return text.replace(/\r\n|\r|\n/g, "\n").replace(/\n/g, newline);
}

export async function run(args: Record<string, unknown>, libos: { syscall(name: string, args: unknown): Promise<any> }) {
  const path = String(args.path ?? "");
  if (!path) throw new Error("path is required");
  const newText = String(args.new_text ?? "");
  const create = Boolean(args.create_if_missing);
  const oldProvided = typeof args.old_text === "string" && args.old_text.length > 0;
  const hasStart = args.start_line !== undefined;
  const hasEnd = args.end_line !== undefined;
  if (hasStart !== hasEnd) {
    throw new Error("start_line and end_line must be supplied together");
  }
  const hasRange = hasStart && hasEnd;
  if (create && !oldProvided && !hasRange) {
    const write = await libos.syscall("filesystem.write_text", {
      path,
      content: newText,
      overwrite: false,
      expected_content_sha256: "missing",
    });
    return { path: write.path ?? path, created: Boolean(write.created), edit: "create" };
  }
  const file = await libos.syscall("filesystem.read_text", { path, max_bytes: 1048576 });
  if (Boolean(file.truncated)) {
    throw new Error(
      "swe_edit refuses to overwrite a truncated source file; use a bounded editor that preserves the complete file",
    );
  }
  const content = String(file.content ?? "");
  let updated = content;
  let edit = "replace_text";
  let replacements = 0;
  if (hasRange) {
    const layout = lineLayout(content);
    const start = integerValue(args.start_line, undefined, "start_line");
    const end = integerValue(args.end_line, undefined, "end_line");
    if (start < 1 || start > layout.starts.length) {
      throw new Error(`start_line ${start} is outside 1..${layout.starts.length}`);
    }
    if (end < start || end > layout.starts.length) {
      throw new Error(`end_line ${end} is outside ${start}..${layout.starts.length}`);
    }
    const replacement = normalizeNewlines(newText, layout.newline);
    updated = content.slice(0, layout.starts[start - 1])
      + replacement
      + content.slice(layout.ends[end - 1]);
    edit = "replace_lines";
    replacements = end - start + 1;
  } else {
    const oldText = String(args.old_text ?? "");
    if (!oldText) throw new Error("old_text or start_line/end_line is required");
    const positions = positionsOf(content, oldText);
    const occurrence = integerValue(args.occurrence, 1, "occurrence");
    if (occurrence < 1) throw new Error("occurrence must be at least 1");
    if (positions.length === 0) throw new Error("old_text was not found");
    if (occurrence > positions.length) throw new Error(`occurrence ${occurrence} exceeds matches ${positions.length}`);
    const at = positions[occurrence - 1];
    updated = content.slice(0, at) + newText + content.slice(at + oldText.length);
    replacements = 1;
  }
  if (updated === content) {
    return { path: file.path ?? path, changed: false, edit, replacements };
  }
  const expectedContentSha256 = file.content_sha256;
  if (
    typeof expectedContentSha256 !== "string"
    || !/^[0-9a-f]{64}$/.test(expectedContentSha256)
  ) {
    throw new Error(
      "filesystem.read_text did not return a complete content_sha256 version token",
    );
  }
  const write = await libos.syscall("filesystem.write_text", {
    path,
    content: updated,
    overwrite: true,
    expected_content_sha256: expectedContentSha256,
  });
  return {
    path: write.path ?? path,
    changed: true,
    edit,
    replacements,
    bytes_written: write.bytes_written,
  };
}
